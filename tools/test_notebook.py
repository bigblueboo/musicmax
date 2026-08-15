#!/usr/bin/env python3
"""Dry-run tests for MiniMax_Music3_Colab.ipynb — no GPU, no network, no Colab.

    python3 tools/test_notebook.py

Three layers:

1. Static name check. Module-level names loaded in cell N must be bound by some cell
   <= N (catches "pipe used before the load cell defines it" orderings); names loaded
   inside function bodies must be bound somewhere in the notebook.
2. Scenario dry-runs. Executes every code cell in notebook order in one namespace
   against a strict fake runtime (fake torch/diffusers/soundfile/google.colab/openai/
   gradio/IPython, real numpy), across GPU scenarios: A100 on-GPU path, L4 auto-offload
   path, 16 GB Ampere group-offload path, T4-without-bf16 (must refuse), low host RAM
   (must refuse). The full run asserts saved WAV+JSON counts, sidecar contents, pipeline
   call counts, branch selection, and that the Gradio click wiring matches what
   ui_generate returns.
3. Fresh-kernel runs. Each inference cell is executed alone in an empty namespace and
   must fail with the notebook's own "run the cells above first" message — the Colab
   reconnect case — never a bare NameError.

The fakes are deliberately strict: unknown pipeline kwargs, wrong dtypes, missing
provider suffixes, or bad audio shapes raise instead of being absorbed.
"""

import ast
import builtins
import contextlib
import glob
import importlib.metadata
import io
import json
import os
import shutil
import sys
import tempfile
import types
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
NOTEBOOK = ROOT / "MiniMax_Music3_Colab.ipynb"

FAILURES = []


def check(condition, label):
    status = "ok" if condition else "FAIL"
    print(f"  {status:4} {label}")
    if not condition:
        FAILURES.append(label)


def code_cells():
    nb = json.loads(NOTEBOOK.read_text())
    return [(i, "".join(c["source"])) for i, c in enumerate(nb["cells"]) if c["cell_type"] == "code"]


# --- layer 1: static name check ---------------------------------------------

def module_bindings(tree):
    """Names bound at module level (imports, assigns, defs, loop/with/except targets),
    not descending into function/class bodies."""
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
            names.update(a.arg for a in n.args.args)
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
    """Names loaded inside any function body must be bound somewhere (globals, params,
    or the function's own locals)."""
    missing = set()
    for _, tree in cells_ast:
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            local = {a.arg for a in node.args.args}
            body = ast.Module(body=node.body, type_ignores=[])
            local |= module_bindings(body) | comprehension_and_lambda_locals(body)
            for name in loads(body, skip_function_bodies=False):
                if name not in local and name not in all_bound and not hasattr(builtins, name):
                    missing.add(f"{node.name}: {name}")
    return missing


def static_check():
    print("[static] cross-cell name resolution")
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


# --- fake runtime ------------------------------------------------------------

def strip_magics(src):
    return "\n".join(l for l in src.splitlines() if not l.strip().startswith(("%", "!")))


class Scenario:
    def __init__(self, name, vram_gb, bf16=True, ram_gb=83, disk_gb=200):
        self.name, self.vram_gb, self.bf16, self.ram_gb, self.disk_gb = name, vram_gb, bf16, ram_gb, disk_gb


class Registry:
    """Call log shared between the fakes and the assertions."""

    def __init__(self):
        self.pipes = []
        self.group_offload_calls = []
        self.gradio_clicks = []
        self.gradio_launches = []


class FakeGuidance:
    def __init__(self, guidance_scale):
        self.guidance_scale = float(guidance_scale)


class FakePipe:
    CALL_KWARGS = {"prompt", "lyrics", "audio_duration", "num_inference_steps", "generator", "output_type", "output"}
    sampling_rate = 44100
    frame_rate = 25.0

    def __init__(self):
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
        assert self.components_loaded, "pipeline called before load_components"
        self.calls.append(kw)
        n = int(float(kw["audio_duration"]) * self.sampling_rate)
        return np.zeros((1, 2, n), dtype=np.float32)


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
        is_available=lambda: True,
        get_device_properties=lambda i: props,
        is_bf16_supported=lambda: scenario.bf16,
        reset_peak_memory_stats=lambda: None,
        max_memory_allocated=lambda: 20 * 1024**3,
    )
    module("torch", cuda=cuda, bfloat16="bf16-sentinel", Generator=Generator, device=lambda d: d)

    class ModularPipeline:
        @staticmethod
        def from_pretrained(repo, components_manager=None):
            assert repo == "MiniMaxAI/MiniMax-Music3", repo
            pipe = FakePipe()
            reg.pipes.append(pipe)
            return pipe

    class ComponentsManager:
        def enable_auto_cpu_offload(self, device):
            assert device == "cuda"

    diffusers = module("diffusers", ModularPipeline=ModularPipeline, ComponentsManager=ComponentsManager)
    diffusers.guiders = module("diffusers.guiders", ClassifierFreeGuidance=FakeGuidance)

    def apply_group_offloading(mod, onload_device=None, offload_type=None, use_stream=None):
        assert offload_type == "leaf_level" and use_stream is True and onload_device == "cuda"
        reg.group_offload_calls.append(mod)

    diffusers.hooks = module("diffusers.hooks", apply_group_offloading=apply_group_offloading)

    def sf_write(path, data, sr):
        assert str(path).endswith(".wav") and str(path).startswith(sandbox), path
        assert data.ndim == 2 and data.shape[1] == 2, f"sf.write expects (samples, 2), got {data.shape}"
        assert sr == 44100
        Path(path).write_bytes(b"RIFF-dryrun")

    module("soundfile", write=sf_write)

    colab = module("google.colab")
    colab.userdata = types.SimpleNamespace(get=lambda name: "dryrun-token")
    colab.drive = types.SimpleNamespace(mount=lambda p: os.makedirs(p, exist_ok=True))
    module("google", colab=colab)
    mods["google.colab"] = colab

    composed = json.dumps({
        "lyrics": "[verse]\nDry-run lyrics line one\n[chorus]\nDry-run chorus",
        "global_metadata": "Basic Attributes: bpm is 90. key is A, and scale is minor. Desert blues.",
        "vocal_details": "Vocal Gender & Timbre: Singer A (Male), gravelly baritone.",
        "arrangement": "Instrument Lifecycle Description: Primary: baritone guitar throughout.",
    })

    class OpenAI:
        def __init__(self, base_url=None, api_key=None, max_retries=None):
            assert base_url == "https://router.huggingface.co/v1" and api_key

        def with_options(self, timeout=None):
            assert timeout, "composer calls must set a timeout"
            return self

        class _Completions:
            @staticmethod
            def create(model=None, messages=None):
                assert ":" in model, f"composer must use provider-suffixed model ids, got {model!r}"
                assert messages[0]["role"] == "system" and messages[1]["role"] == "user"
                content = f"Here you go:\n```json\n{composed}\n```"
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
        return object()

    ipython = module("IPython")
    ipython.display = module("IPython.display", Audio=fake_audio, display=lambda *a, **k: None)

    return mods


@contextlib.contextmanager
def fake_environment(scenario, reg, sandbox):
    saved_modules = {}
    fake_names = ("torch", "diffusers", "diffusers.guiders", "diffusers.hooks", "soundfile",
                  "google", "google.colab", "openai", "gradio", "IPython", "IPython.display")
    for name in fake_names:
        saved_modules[name] = sys.modules.pop(name, None)
    sys.modules.update(make_fake_modules(scenario, reg, sandbox))

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
        for name in fake_names:
            sys.modules.pop(name, None)
            if saved_modules[name] is not None:
                sys.modules[name] = saved_modules[name]


def exec_cells(cells, sandbox, ns=None):
    ns = ns if ns is not None else {"__name__": "__main__"}
    for i, src in cells:
        src = strip_magics(src).replace("/content", f"{sandbox}/content")
        exec(compile(src, f"<cell {i}>", "exec"), ns)
    return ns


# --- layer 2: scenario dry-runs ----------------------------------------------

def run_scenario(scenario, expect_error=None):
    print(f"[scenario] {scenario.name}")
    reg = Registry()
    sandbox = tempfile.mkdtemp(prefix="mm3-dryrun-")
    buf = io.StringIO()
    error = None
    ns = {"__name__": "__main__"}
    try:
        with fake_environment(scenario, reg, sandbox), contextlib.redirect_stdout(buf):
            exec_cells(code_cells(), sandbox, ns)
    except Exception as e:  # noqa: BLE001 — scenarios assert on the exception
        error = e
    if expect_error:
        kind, fragment = expect_error
        check(isinstance(error, kind) and fragment in str(error),
              f"refuses with {kind.__name__} mentioning {fragment!r} (got: {type(error).__name__}: {error})")
        shutil.rmtree(sandbox)
        return
    if error is not None:
        print(buf.getvalue()[-2000:])
        import traceback
        traceback.print_exception(error)
        check(False, f"full run completes ({type(error).__name__}: {error})")
        shutil.rmtree(sandbox)
        return
    check(True, "full run completes")
    return ns, reg, sandbox


def assert_full_run(ns, reg, sandbox, expect_manager, expect_group_offload):
    pipe = reg.pipes[-1]
    check(len(reg.pipes) == 1, "exactly one pipeline constructed")
    check(("manager" in ns) == expect_manager, f"offload branch selection (manager present={expect_manager})")
    check(bool(reg.group_offload_calls) == expect_group_offload,
          f"group offloading {'applied' if expect_group_offload else 'not applied'}")
    if expect_group_offload:
        check(reg.group_offload_calls == [pipe.language_model], "group offloading targets pipe.language_model")
    check((pipe.device == "cuda") == (not expect_manager), "pipe.to('cuda') only on the full-GPU path")

    # smoke(1) + generate(1) + sweep(5) = 7 pipeline calls before the UI
    check(len(pipe.calls) == 7, f"pipeline called 7 times in cell order (got {len(pipe.calls)})")
    smoke, gen = pipe.calls[0], pipe.calls[1]
    check(smoke["audio_duration"] == 5.0 and smoke["num_inference_steps"] == 4, "smoke test uses 5s / 4 steps")
    check(gen["audio_duration"] == 60.0 and gen["num_inference_steps"] == 30, "generate uses form defaults")
    sweep_seeds = [c["generator"].seed for c in pipe.calls[2:7]]
    check(len(set(sweep_seeds)) == 5, "sweep used 5 distinct seeds")

    wavs = sorted(glob.glob(f"{sandbox}/**/*.wav", recursive=True))
    sidecars = sorted(glob.glob(f"{sandbox}/**/*.json", recursive=True))
    check(len(wavs) == 6 and len(sidecars) == 6, f"generate+sweep saved 6 wav + 6 json (got {len(wavs)}/{len(sidecars)})")
    check(all("/content/drive/MyDrive/MiniMax-Music3/" in w for w in wavs), "songs saved under the Drive folder")
    meta = json.loads(Path(sidecars[0]).read_text())
    expected_keys = {"seed", "duration_requested", "num_inference_steps", "guidance_scale",
                     "audio_seconds", "generation_seconds", "global_metadata", "vocal_details",
                     "arrangement", "lyrics"}
    check(set(meta) == expected_keys, f"sidecar carries full reproduction metadata (got {sorted(meta)})")

    check(ns["lyrics"].startswith("[verse]\nDry-run"), "composer cell overwrote the song inputs")

    check(len(reg.gradio_clicks) == 1 and reg.gradio_launches and reg.gradio_launches[0].get("share") is True,
          "gradio UI wired one click handler and launched with share=True")
    fn, inputs, outputs = reg.gradio_clicks[0]
    check(len(inputs) == 9, f"click handler wired 9 inputs (got {len(inputs)})")
    result = fn(ns["global_metadata"], ns["vocal_details"], ns["arrangement"], ns["lyrics"],
                5, 4, 1.7, 123, False)
    check(isinstance(result, tuple) and len(result) == len(outputs),
          f"ui_generate returns {len(result)} values for {len(outputs)} wired outputs")
    (sr, pcm), seed_out, path_out = result
    check(sr == 44100 and pcm.dtype == np.int16 and pcm.ndim == 2 and pcm.shape[1] == 2,
          "UI audio is (samples, 2) int16 at 44.1 kHz")
    check(seed_out == 123 and str(path_out).endswith(".wav") and Path(path_out).exists(),
          "UI respects the fixed seed and reports a real saved path")
    check(len(pipe.calls) == 8, "UI generation went through the pipeline")
    shutil.rmtree(sandbox)


# --- layer 3: fresh-kernel guard checks --------------------------------------

GUARDED_MARKERS = ["one-minute smoke test", "#@title Generate", "#@title Seed sweep", "ui_generate"]


def fresh_kernel_check():
    print("[fresh-kernel] inference cells alone in an empty namespace")
    cells = code_cells()
    scenario = Scenario("fresh", vram_gb=40)
    for marker in GUARDED_MARKERS:
        matches = [(i, src) for i, src in cells if marker in src]
        check(len(matches) == 1, f"exactly one cell matches marker {marker!r}")
        if len(matches) != 1:
            continue
        reg = Registry()
        sandbox = tempfile.mkdtemp(prefix="mm3-fresh-")
        try:
            with fake_environment(scenario, reg, sandbox), contextlib.redirect_stdout(io.StringIO()):
                exec_cells(matches, sandbox)
        except AssertionError as e:
            check("run the cells above first" in str(e), f"{marker!r} fails with the guard message, not NameError")
        except Exception as e:  # noqa: BLE001
            check(False, f"{marker!r} raised {type(e).__name__} instead of the guard: {e}")
        else:
            check(False, f"{marker!r} ran without its prerequisites")
        finally:
            shutil.rmtree(sandbox)


def main():
    static_check()

    result = run_scenario(Scenario("A100 40GB", vram_gb=40))
    if result:
        assert_full_run(*result, expect_manager=False, expect_group_offload=False)

    result = run_scenario(Scenario("L4 24GB", vram_gb=24, ram_gb=53))
    if result:
        assert_full_run(*result, expect_manager=True, expect_group_offload=False)

    result = run_scenario(Scenario("16GB Ampere", vram_gb=16, ram_gb=51))
    if result:
        assert_full_run(*result, expect_manager=True, expect_group_offload=True)

    run_scenario(Scenario("T4 no-bf16", vram_gb=16, bf16=False, ram_gb=51),
                 expect_error=(RuntimeError, "bfloat16"))
    run_scenario(Scenario("low host RAM", vram_gb=16, ram_gb=13),
                 expect_error=(AssertionError, "host RAM"))

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
