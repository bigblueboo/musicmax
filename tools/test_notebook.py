#!/usr/bin/env python3
"""Dry-run tests for MiniMax_Music3_Colab.ipynb — no GPU, no network, no Colab.

    python3 tools/test_notebook.py

Layers:

0. Builder sync: regenerates the notebook and fails if the checked-in .ipynb differs.
1. Static checks. A coarse lint, not a sound execution-order analysis: module-level
   names loaded in cell N must be bound by some cell <= N (same-cell binding counts
   regardless of position, conditional bindings count as unconditional); names loaded
   inside function bodies must be bound somewhere. Plus a content check on the pinned
   install line.
2. Scenario dry-runs. Executes every code cell in notebook order in one namespace
   against a strict fake runtime (fake torch/diffusers/soundfile/google.colab/openai/
   gradio/IPython, real numpy). GPU/RAM branch matrix plus form-param variants applied
   via source overrides (Drive off, sequential seeds, no previews, custom guidance,
   FORCE_LM_STREAMING, composer provider failover, drive_folder traversal). Asserts
   branch selection including that the enabled ComponentsManager is actually passed to
   from_pretrained, pipeline call counts/kwargs, per-call guidance, saved WAV+JSON
   pairs and their sidecar contents, save_song collision behavior, composer overwrite,
   and Gradio click wiring arity.
3. Fresh-kernel and partial-namespace runs. Each inference cell executed alone (empty
   namespace, and a partially-populated one) must fail with the notebook's own guard
   message — the Colab reconnect case — never a NameError.

What this deliberately cannot validate: real package installs, real diffusers/Gradio
behavior, CUDA memory, or Drive FUSE durability. Those need the first actual Colab run.
"""

import ast
import builtins
import contextlib
import glob
import importlib.metadata
import io
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import types
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
NOTEBOOK = ROOT / "MiniMax_Music3_Colab.ipynb"
DIFFUSERS_COMMIT = "2da7040be1a2e5f2fcbc8b985083342a308f5a86"

FAILURES = []


def check(condition, label):
    status = "ok" if condition else "FAIL"
    print(f"  {status:4} {label}")
    if not condition:
        FAILURES.append(label)


def code_cells():
    nb = json.loads(NOTEBOOK.read_text())
    return [(i, "".join(c["source"])) for i, c in enumerate(nb["cells"]) if c["cell_type"] == "code"]


# --- layer 0: builder sync ----------------------------------------------------

def builder_sync_check():
    print("[builder] notebook matches build_notebook.py output")
    before = NOTEBOOK.read_bytes()
    subprocess.run([sys.executable, str(ROOT / "tools" / "build_notebook.py")],
                   check=True, capture_output=True)
    check(NOTEBOOK.read_bytes() == before, "checked-in notebook is regenerated from the builder")


# --- layer 1: static checks ---------------------------------------------------

def module_bindings(tree):
    """Names bound at module level, not descending into function/class bodies."""
    bound = set()

    def targets(node):
        for n in ast.walk(node):
            if isinstance(n, ast.Name) and isinstance(n.ctx, (ast.Store, ast.Del)):
                bound.add(n.id)

    def visit(stmts):
        for s in stmts:
            if isinstance(s, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                bound.add(s.name)
                continue
            if isinstance(s, (ast.Import, ast.ImportFrom)):
                for a in s.names:
                    bound.add((a.asname or a.name).split(".")[0])
                continue
            if isinstance(s, ast.Assign):
                for t in s.targets:
                    targets(t)
            elif isinstance(s, (ast.AugAssign, ast.AnnAssign)):
                targets(s.target)
            elif isinstance(s, (ast.For, ast.AsyncFor)):
                targets(s.target)
            elif isinstance(s, (ast.With, ast.AsyncWith)):
                for item in s.items:
                    if item.optional_vars:
                        targets(item.optional_vars)
            elif isinstance(s, ast.Try):
                for h in s.handlers:
                    if h.name:
                        bound.add(h.name)
            for field in ("body", "orelse", "finalbody"):
                visit(getattr(s, field, []) or [])
            for h in getattr(s, "handlers", []) or []:
                visit(h.body)

    visit(tree.body)
    return bound


def comprehension_and_lambda_locals(tree):
    names = set()
    for n in ast.walk(tree):
        if isinstance(n, (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)):
            for gen in n.generators:
                for t in ast.walk(gen.target):
                    if isinstance(t, ast.Name):
                        names.add(t.id)
        if isinstance(n, ast.Lambda):
            a = n.args
            names.update(x.arg for x in a.args + a.posonlyargs + a.kwonlyargs)
            for extra in (a.vararg, a.kwarg):
                if extra:
                    names.add(extra.arg)
        if isinstance(n, ast.NamedExpr) and isinstance(n.target, ast.Name):
            names.add(n.target.id)
    return names


def loads(tree, skip_function_bodies):
    found = set()

    class V(ast.NodeVisitor):
        def visit_FunctionDef(self, node):
            if not skip_function_bodies:
                self.generic_visit(node)

        visit_AsyncFunctionDef = visit_FunctionDef

        def visit_Name(self, node):
            if isinstance(node.ctx, ast.Load):
                found.add(node.id)

    V().visit(tree)
    return found


def function_body_check(cells_ast, all_bound):
    missing = set()
    for _, tree in cells_ast:
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            a = node.args
            local = {x.arg for x in a.args + a.posonlyargs + a.kwonlyargs}
            for extra in (a.vararg, a.kwarg):
                if extra:
                    local.add(extra.arg)
            body = ast.Module(body=node.body, type_ignores=[])
            local |= module_bindings(body) | comprehension_and_lambda_locals(body)
            for name in loads(body, skip_function_bodies=False):
                if name not in local and name not in all_bound and not hasattr(builtins, name):
                    missing.add(f"{node.name}: {name}")
    return missing


def static_check():
    print("[static] cross-cell name resolution (coarse lint)")
    cells_ast = [(i, ast.parse(strip_magics(src))) for i, src in code_cells()]
    bound_so_far = set()
    order_errors = []
    for i, tree in cells_ast:
        cell_locals = comprehension_and_lambda_locals(tree)
        needed = loads(tree, skip_function_bodies=True)
        cell_bound = module_bindings(tree)
        for name in sorted(needed):
            if name in cell_bound or name in cell_locals or name in bound_so_far:
                continue
            if hasattr(builtins, name):
                continue
            order_errors.append(f"cell {i}: '{name}' used before any earlier cell binds it")
        bound_so_far |= cell_bound
    check(not order_errors, f"module-level loads resolve in cell order {order_errors or ''}")
    dangling = function_body_check(cells_ast, bound_so_far)
    check(not dangling, f"function bodies reference only bound names {sorted(dangling) or ''}")

    # The dry-run strips %pip, so at least pin the install line's content statically.
    installs = [src for _, src in code_cells() if "%pip install" in src]
    check(len(installs) == 1, "exactly one pip install cell")
    if installs:
        line = next(l for l in installs[0].splitlines() if l.strip().startswith("%pip"))
        expected = [DIFFUSERS_COMMIT, "transformers", "accelerate", "soundfile", "openai", "gradio", "anthropic"]
        missing = [t for t in expected if t not in line]
        check(not missing, f"install line pins the diffusers commit and all packages {missing or ''}")


# --- fake runtime ---------------------------------------------------------------

def strip_magics(src):
    return "\n".join(l for l in src.splitlines() if not l.strip().startswith(("%", "!")))


class Scenario:
    def __init__(self, name, vram_gb, bf16=True, ram_gb=83, disk_gb=200, cuda=True,
                 overrides=None, composer_script=None, files=None, anthropic_key=False):
        self.name, self.vram_gb, self.bf16, self.ram_gb, self.disk_gb, self.cuda = (
            name, vram_gb, bf16, ram_gb, disk_gb, cuda)
        self.overrides = overrides or {}
        self.composer_script = composer_script
        self.files = files or {}
        self.anthropic_key = anthropic_key


class Registry:
    def __init__(self):
        self.pipes = []
        self.managers = []
        self.group_offload_calls = []
        self.gradio_clicks = []
        self.gradio_launches = []
        self.audio_displays = 0
        self.composer_calls = []
        self.anthropic_calls = []


class FakeGuidance:
    def __init__(self, guidance_scale):
        self.guidance_scale = float(guidance_scale)


class FakePipe:
    CALL_KWARGS = {"prompt", "lyrics", "audio_duration", "num_inference_steps", "generator", "output_type", "output"}
    sampling_rate = 44100
    frame_rate = 25.0

    def __init__(self, components_manager):
        self.components_manager = components_manager
        self.calls = []
        self.components_loaded = False
        self.device = None
        self.guider = None
        self.language_model = object()

    def load_components(self, dtype=None):
        assert dtype == "bf16-sentinel", f"load_components dtype: {dtype!r}"
        self.components_loaded = True

    def to(self, device):
        assert device == "cuda"
        self.device = device
        return self

    def update_components(self, **kw):
        assert set(kw) == {"guider"}, f"update_components kwargs: {set(kw)}"
        assert isinstance(kw["guider"], FakeGuidance)
        self.guider = kw["guider"]

    def __call__(self, **kw):
        unknown = set(kw) - self.CALL_KWARGS
        assert not unknown, f"unknown pipeline kwargs: {unknown}"
        for req in ("prompt", "lyrics", "audio_duration", "num_inference_steps", "generator"):
            assert req in kw, f"missing pipeline kwarg: {req}"
        assert kw["output"] == "audios" and kw["output_type"] == "np"
        assert isinstance(kw["prompt"], str) and kw["prompt"].strip()
        assert isinstance(kw["lyrics"], str) and kw["lyrics"].strip()
        assert kw["generator"].device == "cuda" and isinstance(kw["generator"].seed, int)
        assert 0 <= kw["generator"].seed < 2**32, f"seed out of range: {kw['generator'].seed}"
        assert self.components_loaded, "pipeline called before load_components"
        assert self.guider is not None, "pipeline called before a guider was set"
        self.calls.append({**kw, "guidance_at_call": self.guider.guidance_scale})
        n = int(float(kw["audio_duration"]) * self.sampling_rate)
        return np.zeros((1, 2, n), dtype=np.float32)


COMPOSED = {
    "lyrics": "[verse]\nDry-run lyrics line one\n[chorus]\nDry-run chorus",
    "global_metadata": "Basic Attributes: bpm is 90. key is A, and scale is minor. Desert blues.",
    "vocal_details": "Vocal Gender & Timbre: Singer A (Male), gravelly baritone.",
    "arrangement": "Instrument Lifecycle Description: Primary: baritone guitar throughout.",
}


def make_fake_modules(scenario, reg, sandbox):
    mods = {}

    def module(name, **attrs):
        m = types.ModuleType(name)
        for k, v in attrs.items():
            setattr(m, k, v)
        mods[name] = m
        return m

    class Generator:
        def __init__(self, device):
            self.device = device
            self.seed = None

        def manual_seed(self, s):
            self.seed = int(s)
            return self

    props = types.SimpleNamespace(name=f"Fake GPU ({scenario.name})", total_memory=int(scenario.vram_gb * 1024**3))
    cuda = types.SimpleNamespace(
        is_available=lambda: scenario.cuda,
        get_device_properties=lambda i: props,
        is_bf16_supported=lambda: scenario.bf16,
        reset_peak_memory_stats=lambda: None,
        max_memory_allocated=lambda: 20 * 1024**3,
    )
    module("torch", cuda=cuda, bfloat16="bf16-sentinel", Generator=Generator, device=lambda d: d)

    class ComponentsManager:
        def __init__(self):
            self.enabled_device = None
            reg.managers.append(self)

        def enable_auto_cpu_offload(self, device):
            assert device == "cuda"
            self.enabled_device = device

    class ModularPipeline:
        @staticmethod
        def from_pretrained(repo, components_manager=None):
            assert repo == "MiniMaxAI/MiniMax-Music3", repo
            if components_manager is not None:
                assert components_manager in reg.managers, "unknown ComponentsManager instance"
                assert components_manager.enabled_device == "cuda", "manager passed before enable_auto_cpu_offload"
            pipe = FakePipe(components_manager)
            reg.pipes.append(pipe)
            return pipe

    diffusers = module("diffusers", ModularPipeline=ModularPipeline, ComponentsManager=ComponentsManager)
    diffusers.guiders = module("diffusers.guiders", ClassifierFreeGuidance=FakeGuidance)

    def apply_group_offloading(mod, onload_device=None, offload_type=None, use_stream=None):
        assert offload_type == "leaf_level" and use_stream is True and onload_device == "cuda"
        reg.group_offload_calls.append(mod)

    diffusers.hooks = module("diffusers.hooks", apply_group_offloading=apply_group_offloading)

    def sf_write(path, data, sr, format=None, subtype=None):
        resolved = Path(path).resolve()
        assert resolved.is_relative_to(Path(sandbox).resolve()), f"write escapes sandbox: {path}"
        assert str(path).endswith(".wav.part"), f"expected atomic .wav.part write, got {path}"
        assert format == "WAV" and subtype == "PCM_16", f"format/subtype: {format}/{subtype}"
        assert data.ndim == 2 and data.shape[1] == 2, f"sf.write expects (samples, 2), got {data.shape}"
        assert sr == 44100
        Path(path).write_bytes(b"RIFF-dryrun")

    module("soundfile", write=sf_write)

    def userdata_get(name):
        if name == "HF_TOKEN" or (name == "ANTHROPIC_API_KEY" and scenario.anthropic_key):
            return "dryrun-token"
        raise KeyError(f"Secret {name} does not exist (dry-run)")

    colab = module("google.colab")
    colab.userdata = types.SimpleNamespace(get=userdata_get)
    colab.drive = types.SimpleNamespace(mount=lambda p: os.makedirs(p, exist_ok=True))
    module("google", colab=colab)
    mods["google.colab"] = colab

    def fake_album_json(n):
        return json.dumps({"album": "Static Bloom", "songs": [
            {"title": f"Signal {i}", "lyrics": "[verse]\nNeon rain on glass",
             "global_metadata": f"gm {i}", "vocal_details": f"vd {i}",
             "arrangement": f"arr {i}", "duration_seconds": 150 if i % 2 else 45}
            for i in range(1, n + 1)]})

    class _AnthropicStream:
        def __init__(self, text):
            self._text = text

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        @property
        def text_stream(self):
            return iter([self._text[:40], self._text[40:]])

        def get_final_message(self):
            return types.SimpleNamespace(
                stop_reason="end_turn",
                content=[types.SimpleNamespace(type="text", text=self._text)],
            )

    class _AnthropicMessages:
        @staticmethod
        def stream(model=None, max_tokens=None, system=None, output_config=None, messages=None):
            assert isinstance(model, str) and model, "model required"
            assert max_tokens >= 32000, f"long JSON output needs headroom, got max_tokens={max_tokens}"
            fmt = output_config["format"]
            assert fmt["type"] == "json_schema", fmt
            schema = fmt["schema"]
            song_schema = schema["properties"]["songs"]["items"]
            assert schema["additionalProperties"] is False and song_schema["additionalProperties"] is False, \
                "structured outputs require additionalProperties: false on every object"
            assert set(song_schema["required"]) == {"title", "lyrics", "global_metadata",
                                                    "vocal_details", "arrangement", "duration_seconds"}
            assert "MiniMax Music 3" in system and "Album craft" in system
            n = int(re.search(r"Number of tracks: (\d+)", messages[0]["content"]).group(1))
            reg.anthropic_calls.append({"kind": "album", "model": model, "max_tokens": max_tokens, "n": n})
            return _AnthropicStream(fake_album_json(n))

        @staticmethod
        def create(model=None, max_tokens=None, system=None, output_config=None, messages=None):
            assert isinstance(model, str) and model, "model required"
            assert max_tokens >= 8000, f"song + thinking need headroom, got max_tokens={max_tokens}"
            fmt = output_config["format"]
            assert fmt["type"] == "json_schema", fmt
            schema = fmt["schema"]
            assert schema["additionalProperties"] is False
            assert set(schema["required"]) == {"lyrics", "global_metadata", "vocal_details", "arrangement"}
            assert "MiniMax Music 3" in system and "# Revisions" in system
            user = messages[0]["content"]
            revised = "Revision notes:" in user
            if revised:
                assert "Current song:" in user, "revision request must carry the current song"
            reg.anthropic_calls.append({"kind": "song", "model": model, "revised": revised,
                                        "saw_current_lyrics": "Riding on a beam" in user})
            payload = {
                "lyrics": "[chorus]\nRevised chorus line" if revised else "[verse]\nFresh drafted line",
                "global_metadata": "song gm", "vocal_details": "song vd", "arrangement": "song arr",
            }
            return types.SimpleNamespace(
                stop_reason="end_turn",
                content=[types.SimpleNamespace(type="text", text=json.dumps(payload))],
            )

    class FakeAnthropicClient:
        messages = _AnthropicMessages

        def __init__(self, api_key=None):
            assert api_key, "anthropic client constructed without a key"

    module("anthropic", Anthropic=FakeAnthropicClient)

    # Composer behaviors, consumed per create() call: "ok", "raise", "missing_keys".
    script = list(scenario.composer_script) if scenario.composer_script else None

    class OpenAI:
        def __init__(self, base_url=None, api_key=None, max_retries=None):
            assert base_url == "https://router.huggingface.co/v1" and api_key
            assert max_retries == 0, "composer must disable SDK retries (it manages its own)"

        def with_options(self, timeout=None):
            assert timeout, "composer calls must set a timeout"
            return self

        class _Completions:
            @staticmethod
            def create(model=None, messages=None):
                assert ":" in model, f"composer must use provider-suffixed model ids, got {model!r}"
                assert messages[0]["role"] == "system" and messages[1]["role"] == "user"
                reg.composer_calls.append(model)
                behavior = script.pop(0) if script else "ok"
                if behavior == "raise":
                    raise RuntimeError("provider down (dry-run)")
                payload = dict(COMPOSED)
                if behavior == "missing_keys":
                    del payload["arrangement"]
                content = f"Here you go:\n```json\n{json.dumps(payload)}\n```"
                msg = types.SimpleNamespace(content=content)
                return types.SimpleNamespace(choices=[types.SimpleNamespace(message=msg)])

        chat = types.SimpleNamespace(completions=_Completions)

    module("openai", OpenAI=OpenAI)

    class Ctx:
        def __init__(self, **kw):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    class Component:
        def __init__(self, *a, **kw):
            self.kwargs = kw

    class Button(Component):
        def click(self, fn, inputs, outputs):
            reg.gradio_clicks.append((fn, inputs, outputs))

    class Blocks(Ctx):
        def queue(self):
            return self

        def launch(self, **kw):
            reg.gradio_launches.append(kw)

    module(
        "gradio",
        Blocks=Blocks, Row=Ctx, Column=Ctx, Markdown=Component, Textbox=Component,
        Slider=Component, Number=Component, Checkbox=Component, Button=Button, Audio=Component,
    )

    def fake_audio(data, rate=None):
        assert rate == 44100
        assert data.ndim == 2 and data.shape[0] == 2, f"Audio expects (2, samples), got {data.shape}"
        reg.audio_displays += 1
        return object()

    ipython = module("IPython")
    ipython.display = module("IPython.display", Audio=fake_audio, display=lambda *a, **k: None)

    return mods


@contextlib.contextmanager
def fake_environment(scenario, reg, sandbox):
    saved_modules = {}
    fake_names = ("torch", "diffusers", "diffusers.guiders", "diffusers.hooks", "soundfile",
                  "google", "google.colab", "openai", "gradio", "IPython", "IPython.display",
                  "anthropic")
    for name in fake_names:
        saved_modules[name] = sys.modules.pop(name, None)
    sys.modules.update(make_fake_modules(scenario, reg, sandbox))

    # keep the host's real API keys out of the dry run (cells check os.environ first)
    saved_env = {v: os.environ.pop(v, None) for v in ("ANTHROPIC_API_KEY", "HF_TOKEN")}

    real_sysconf, real_disk, real_version = os.sysconf, shutil.disk_usage, importlib.metadata.version
    page = 16384

    def sysconf(name):
        if name == "SC_PAGE_SIZE":
            return page
        if name == "SC_PHYS_PAGES":
            return int(scenario.ram_gb * 1024**3 / page)
        return real_sysconf(name)

    os.sysconf = sysconf
    shutil.disk_usage = lambda p: types.SimpleNamespace(free=int(scenario.disk_gb * 1024**3), total=0, used=0)
    importlib.metadata.version = lambda p: "0.0-dryrun"
    try:
        yield
    finally:
        os.sysconf, shutil.disk_usage, importlib.metadata.version = real_sysconf, real_disk, real_version
        for var, value in saved_env.items():
            os.environ.pop(var, None)
            if value is not None:
                os.environ[var] = value
        for name in fake_names:
            sys.modules.pop(name, None)
            if saved_modules[name] is not None:
                sys.modules[name] = saved_modules[name]


def transform(src, sandbox, overrides):
    src = strip_magics(src)
    for old, new in overrides.items():
        if old in src:
            src = src.replace(old, new)
    return src.replace("/content", f"{sandbox}/content")


def exec_cells(cells, sandbox, ns=None, overrides=None):
    ns = ns if ns is not None else {"__name__": "__main__"}
    for i, src in cells:
        exec(compile(transform(src, sandbox, overrides or {}), f"<cell {i}>", "exec"), ns)
    return ns


# --- layer 2: scenario dry-runs -------------------------------------------------

def run_scenario(scenario, expect_error=None):
    print(f"[scenario] {scenario.name}")
    reg = Registry()
    sandbox = tempfile.mkdtemp(prefix="mm3-dryrun-")
    for rel, content in scenario.files.items():
        p = Path(sandbox) / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)
    buf = io.StringIO()
    error = None
    ns = {"__name__": "__main__"}
    try:
        with fake_environment(scenario, reg, sandbox), contextlib.redirect_stdout(buf):
            exec_cells(code_cells(), sandbox, ns, scenario.overrides)
    except Exception as e:  # noqa: BLE001 — scenarios assert on the exception
        error = e
    # applied overrides must actually match something, or the variant is testing nothing
    for old in scenario.overrides:
        if not any(old in src for _, src in code_cells()):
            check(False, f"override target not found in any cell: {old!r}")
    if expect_error:
        kind, fragment = expect_error
        check(isinstance(error, kind) and fragment in str(error),
              f"refuses with {kind.__name__} mentioning {fragment!r} (got: {type(error).__name__}: {error})")
        shutil.rmtree(sandbox)
        return None
    if error is not None:
        print(buf.getvalue()[-2000:])
        import traceback
        traceback.print_exception(error)
        check(False, f"full run completes ({type(error).__name__}: {error})")
        shutil.rmtree(sandbox)
        return None
    check(True, "full run completes")
    return ns, reg, sandbox


SIDECAR_KEYS = {"seed", "duration_requested", "num_inference_steps", "guidance_scale",
                "audio_seconds", "generation_seconds", "global_metadata", "vocal_details",
                "arrangement", "lyrics"}
ALBUM_SIDECAR_KEYS = SIDECAR_KEYS | {"album", "title", "take"}

ALBUM_FIXTURE = {
    "album": "Midnight Static",
    "songs": [
        {"title": "Neon Rain", "lyrics": "[verse]\nRain on glass", "global_metadata": "gm one",
         "vocal_details": "vd one", "arrangement": "arr one"},
        {"title": "Last Exit", "lyrics": "[chorus]\nLast exit glow", "global_metadata": "gm two",
         "vocal_details": "vd two", "arrangement": "arr two",
         "duration_seconds": 30, "num_inference_steps": 12, "guidance_scale": 2.0},
    ],
}


def assert_full_run(ns, reg, sandbox, expect_manager, expect_group_offload, drive=True,
                    expected_displays=6, sweep_seeds=None, sweep_guidance=1.7, album=False):
    pipe = reg.pipes[-1]
    check(len(reg.pipes) == 1, "exactly one pipeline constructed")
    check((pipe.components_manager is not None) == expect_manager,
          f"enabled ComponentsManager {'passed to' if expect_manager else 'absent from'} from_pretrained")
    check(bool(reg.group_offload_calls) == expect_group_offload,
          f"group offloading {'applied' if expect_group_offload else 'not applied'}")
    if expect_group_offload:
        check(reg.group_offload_calls == [pipe.language_model], "group offloading targets pipe.language_model")
    check((pipe.device == "cuda") == (not expect_manager), "pipe.to('cuda') only on the full-GPU path")

    # smoke(1) + generate(1) + sweep(5) [+ album 2 songs x 3 takes] before the UI
    expected_calls = 7 + (6 if album else 0)
    check(len(pipe.calls) == expected_calls,
          f"pipeline called {expected_calls} times in cell order (got {len(pipe.calls)})")
    smoke, gen = pipe.calls[0], pipe.calls[1]
    check(smoke["audio_duration"] == 5.0 and smoke["num_inference_steps"] == 4, "smoke test uses 5s / 4 steps")
    check(gen["audio_duration"] == 60.0 and gen["num_inference_steps"] == 30, "generate uses form defaults")
    check(gen["guidance_at_call"] == 1.7, "generate ran at guidance 1.7")
    got_sweep_seeds = [c["generator"].seed for c in pipe.calls[2:7]]
    check(len(set(got_sweep_seeds)) == 5, "sweep used 5 distinct seeds")
    if sweep_seeds is not None:
        check(got_sweep_seeds == sweep_seeds, f"sequential sweep seeds are base..base+4 (got {got_sweep_seeds})")
    check(all(c["guidance_at_call"] == sweep_guidance for c in pipe.calls[2:7]),
          f"sweep ran at guidance {sweep_guidance}")

    if album:
        song1, song2 = pipe.calls[7:10], pipe.calls[10:13]
        check(all(c["audio_duration"] == 120.0 and c["num_inference_steps"] == 30
                  and c["guidance_at_call"] == 1.7 for c in song1),
              "album song 1 used the cell-level defaults")
        check(all(c["audio_duration"] == 30.0 and c["num_inference_steps"] == 12
                  and c["guidance_at_call"] == 2.0 for c in song2),
              "album song 2 used its per-song overrides")

    wavs = sorted(glob.glob(f"{sandbox}/**/*.wav", recursive=True))
    sidecars = sorted(glob.glob(f"{sandbox}/**/*.json", recursive=True))
    parts = glob.glob(f"{sandbox}/**/*.part", recursive=True)
    album_wavs = [w for w in wavs if "/midnight-static/" in w]
    base_wavs = [w for w in wavs if "/midnight-static/" not in w]
    # sidecar count excludes the input album.json fixture, which lives outside SONGS_DIR
    song_sidecars = [s for s in sidecars if not s.endswith("album.json")]
    check(len(base_wavs) == 6, f"generate+sweep saved 6 wavs (got {len(base_wavs)})")
    check(len(song_sidecars) == len(wavs), "one sidecar per wav")
    if album:
        check(len(album_wavs) == 6, f"album saved 2 songs x 3 takes (got {len(album_wavs)})")
        check(sum("neon-rain_take" in w for w in album_wavs) == 3
              and sum("last-exit_take" in w for w in album_wavs) == 3,
              "album filenames carry slugged titles and take numbers")
    else:
        check(not album_wavs, "no album output without an album file")
    check(not parts, "no leftover .part files after atomic renames")
    expected_dir = "/content/drive/MyDrive/MiniMax-Music3/" if drive else "/content/songs/"
    check(all(expected_dir in w for w in wavs), f"songs saved under {expected_dir}")
    check({w[:-4] for w in wavs} == {s[:-5] for s in song_sidecars}, "every wav has a same-basename json")
    for sidecar in song_sidecars:
        meta = json.loads(Path(sidecar).read_text())
        expected_keys = ALBUM_SIDECAR_KEYS if "/midnight-static/" in sidecar else SIDECAR_KEYS
        if set(meta) != expected_keys or f"seed{meta['seed']}" not in Path(sidecar).stem:
            check(False, f"sidecar {Path(sidecar).name}: keys/seed mismatch (got {sorted(meta)})")
            break
    else:
        check(True, "all sidecars carry full metadata and their filename seed")

    check(reg.audio_displays == expected_displays,
          f"{expected_displays} inline players displayed (got {reg.audio_displays})")

    check(ns["lyrics"] == COMPOSED["lyrics"], "composer cell overwrote the song inputs")

    # same-seed saves must not overwrite (microsecond stamp)
    a = ns["save_song"](np.zeros((2, 100), dtype=np.float32), 1, dict.fromkeys(SIDECAR_KEYS, 1))
    b = ns["save_song"](np.zeros((2, 100), dtype=np.float32), 1, dict.fromkeys(SIDECAR_KEYS, 1))
    check(a != b and Path(a).exists() and Path(b).exists(), "same-seed saves get distinct filenames")

    check(len(reg.gradio_clicks) == 1 and reg.gradio_launches and reg.gradio_launches[0].get("share") is True,
          "gradio UI wired one click handler and launched with share=True")
    fn, inputs, outputs = reg.gradio_clicks[0]
    check(len(inputs) == 9, f"click handler wired 9 inputs (got {len(inputs)})")
    result = fn(ns["global_metadata"], ns["vocal_details"], ns["arrangement"], ns["lyrics"],
                5, 4, 2.5, 123, False)
    check(isinstance(result, tuple) and len(result) == len(outputs),
          f"ui_generate returns {len(result)} values for {len(outputs)} wired outputs")
    (sr, pcm), seed_out, path_out = result
    check(sr == 44100 and pcm.dtype == np.int16 and pcm.ndim == 2 and pcm.shape[1] == 2,
          "UI audio is (samples, 2) int16 at 44.1 kHz")
    check(seed_out == 123 and str(path_out).endswith(".wav") and Path(path_out).exists(),
          "UI respects the fixed seed and reports a real saved path")
    check(pipe.calls[-1]["guidance_at_call"] == 2.5, "UI guidance slider reaches the pipeline")
    shutil.rmtree(sandbox)


# --- layer 3: fresh-kernel and partial-namespace guard checks --------------------

GUARDED_MARKERS = ["one-minute smoke test", "#@title Generate", "#@title Seed sweep",
                   "#@title Album mode", "ui_generate"]


def guard_check(marker, cells, preset, expect_fragment, label):
    scenario = Scenario("guard", vram_gb=40)
    matches = [(i, src) for i, src in cells if marker in src]
    check(len(matches) == 1, f"exactly one cell matches marker {marker!r}")
    if len(matches) != 1:
        return
    reg = Registry()
    sandbox = tempfile.mkdtemp(prefix="mm3-guard-")
    ns = {"__name__": "__main__", **preset}
    try:
        with fake_environment(scenario, reg, sandbox), contextlib.redirect_stdout(io.StringIO()):
            exec_cells(matches, sandbox, ns)
    except AssertionError as e:
        check("run the cells above first" in str(e) and expect_fragment in str(e),
              f"{label}: fails with guard message naming {expect_fragment!r}")
    except Exception as e:  # noqa: BLE001
        check(False, f"{label}: raised {type(e).__name__} instead of the guard: {e}")
    else:
        check(False, f"{label}: ran without its prerequisites")
    finally:
        shutil.rmtree(sandbox)


def fresh_kernel_check():
    print("[fresh-kernel] inference cells alone in an empty namespace")
    cells = code_cells()
    for marker in GUARDED_MARKERS:
        guard_check(marker, cells, {}, "", f"{marker!r} in empty namespace")

    print("[partial-namespace] loaded pipe but missing caption fields")
    partial = {"PIPE_READY": True, "pipe": object(), "save_song": lambda *a: None, "lyrics": "x"}
    for marker in ["#@title Generate", "#@title Seed sweep", "ui_generate"]:
        guard_check(marker, cells, dict(partial), "global_metadata", f"{marker!r} with partial namespace")

    print("[partial-namespace] names present but pipeline load incomplete")
    unready = {"pipe": object(), "save_song": lambda *a: None, "lyrics": "x",
               "global_metadata": "x", "vocal_details": "x", "arrangement": "x"}
    guard_check("#@title Generate", cells, dict(unready), "pipeline load incomplete",
                "'#@title Generate' without PIPE_READY")


def main():
    builder_sync_check()
    static_check()

    result = run_scenario(Scenario("A100 40GB", vram_gb=40))
    if result:
        assert_full_run(*result, expect_manager=False, expect_group_offload=False)

    result = run_scenario(Scenario("L4 24GB", vram_gb=24, ram_gb=53))
    if result:
        assert_full_run(*result, expect_manager=True, expect_group_offload=False)

    result = run_scenario(Scenario("21GB boundary", vram_gb=21, ram_gb=51))
    if result:
        assert_full_run(*result, expect_manager=True, expect_group_offload=True)

    result = run_scenario(Scenario("16GB Ampere", vram_gb=16, ram_gb=51))
    if result:
        assert_full_run(*result, expect_manager=True, expect_group_offload=True)

    result = run_scenario(Scenario("A100 + FORCE_LM_STREAMING", vram_gb=40,
                                   overrides={"FORCE_LM_STREAMING = False": "FORCE_LM_STREAMING = True"}))
    if result:
        assert_full_run(*result, expect_manager=True, expect_group_offload=True)

    result = run_scenario(Scenario("Drive off", vram_gb=40,
                                   overrides={"save_to_drive = True": "save_to_drive = False"}))
    if result:
        assert_full_run(*result, expect_manager=False, expect_group_offload=False, drive=False)

    result = run_scenario(Scenario("sweep variants", vram_gb=40, overrides={
        'seed_mode = "random"': 'seed_mode = "sequential from base_seed"',
        "base_seed = 0  #@param": "base_seed = 7  #@param",
        "preview_seconds = 30": "preview_seconds = 0",
        "sweep_guidance = 1.7": "sweep_guidance = 2.5",
    }))
    if result:
        assert_full_run(*result, expect_manager=False, expect_group_offload=False,
                        expected_displays=1, sweep_seeds=[7, 8, 9, 10, 11], sweep_guidance=2.5)

    result = run_scenario(Scenario("album mode", vram_gb=40,
                                   files={"content/album.json": json.dumps(ALBUM_FIXTURE)}))
    if result:
        assert_full_run(*result, expect_manager=False, expect_group_offload=False, album=True)

    result = run_scenario(Scenario("claude album composer", vram_gb=40, anthropic_key=True,
                                   overrides={"num_tracks = 6": "num_tracks = 2",
                                              "takes_per_song = 3": "takes_per_song = 1"}))
    if result:
        ns, reg, sandbox = result
        song_calls = [c for c in reg.anthropic_calls if c["kind"] == "song"]
        album_calls = [c for c in reg.anthropic_calls if c["kind"] == "album"]
        check(len(song_calls) == 1 and not song_calls[0]["revised"],
              "song composer made one fresh-compose call")
        check(reg.pipes[-1].calls[1]["lyrics"] == "[verse]\nFresh drafted line",
              "generate ran with the claude-drafted song")
        check(len(album_calls) == 1 and album_calls[0]["n"] == 2,
              f"album composer made one schema-constrained streaming call for 2 tracks (got {album_calls})")
        album_data = json.loads((Path(sandbox) / "content/album.json").read_text())
        check(album_data["album"] == "Static Bloom" and len(album_data["songs"]) == 2,
              "composer wrote the album JSON the render cell reads")
        album_wavs = sorted(glob.glob(f"{sandbox}/**/static-bloom/*.wav", recursive=True))
        check(len(album_wavs) == 2 and any("signal-1_take1" in w for w in album_wavs),
              f"claude-drafted album rendered with slugged titles (got {[Path(w).name for w in album_wavs]})")
        pipe = reg.pipes[-1]
        check([c["audio_duration"] for c in pipe.calls[7:9]] == [150.0, 45.0],
              "per-track durations from the drafted tracklist reached the pipeline")
        shutil.rmtree(sandbox)

    result = run_scenario(Scenario("claude song revision", vram_gb=40, anthropic_key=True,
                                   overrides={'revision_notes = ""': 'revision_notes = "punchier chorus"',
                                              "num_tracks = 6": "num_tracks = 2",
                                              "takes_per_song = 3": "takes_per_song = 1"}))
    if result:
        ns, reg, sandbox = result
        song_calls = [c for c in reg.anthropic_calls if c["kind"] == "song"]
        check(len(song_calls) == 1 and song_calls[0]["revised"] and song_calls[0]["saw_current_lyrics"],
              "revision request carried the current song inputs to Claude")
        check(reg.pipes[-1].calls[1]["lyrics"] == "[chorus]\nRevised chorus line",
              "generate ran with the revised song")
        shutil.rmtree(sandbox)

    result = run_scenario(Scenario("composer failover", vram_gb=40,
                                   composer_script=["raise", "missing_keys", "ok"]))
    if result:
        ns, reg, sandbox = result
        check(len(reg.composer_calls) == 3 and len({m.rsplit(":", 1)[1] for m in reg.composer_calls}) == 3,
              f"composer walked three distinct providers (got {reg.composer_calls})")
        check(ns["lyrics"] == COMPOSED["lyrics"], "composer recovered and set the song inputs")
        shutil.rmtree(sandbox)

    run_scenario(Scenario("T4 no-bf16", vram_gb=16, bf16=False, ram_gb=51),
                 expect_error=(RuntimeError, "bfloat16"))
    run_scenario(Scenario("low host RAM", vram_gb=16, ram_gb=13),
                 expect_error=(AssertionError, "host RAM"))
    run_scenario(Scenario("low disk", vram_gb=40, disk_gb=10),
                 expect_error=(AssertionError, "free disk"))
    run_scenario(Scenario("no GPU", vram_gb=40, cuda=False),
                 expect_error=(AssertionError, "No GPU"))
    run_scenario(Scenario("drive_folder traversal", vram_gb=40,
                          overrides={'drive_folder = "MiniMax-Music3"': 'drive_folder = "../../escape"'}),
                 expect_error=(AssertionError, "inside My Drive"))
    run_scenario(Scenario("album with missing fields", vram_gb=40,
                          files={"content/album.json": json.dumps(
                              {"album": "Broken", "songs": [{"title": "Half a Song", "lyrics": "x"}]})}),
                 expect_error=(AssertionError, "Fix the album JSON"))

    fresh_kernel_check()

    print()
    if FAILURES:
        print(f"{len(FAILURES)} FAILURE(S):")
        for f in FAILURES:
            print(f"  - {f}")
        sys.exit(1)
    print("all notebook dry-run tests passed")


if __name__ == "__main__":
    main()
