#!/usr/bin/env python3
"""Generate MiniMax_Music3_Colab.ipynb.

Source of truth for the notebook. Regenerate with:
    python3 tools/build_notebook.py

Facts baked in below (verified 2026-08-13):
- Weights: MiniMaxAI/MiniMax-Music3 (11B total: 8B LM + 0.6B local LM + 2.4B DiT + 123M vocoder)
- diffusers support merged in PR #14456 on 2026-08-13; merge commit pinned until a PyPI release ships it
- Pipeline API: pipe(prompt, lyrics, audio_duration, num_inference_steps, generator, output="audios")
- Guidance lives in the ClassifierFreeGuidance guider component (default 1.7)
- Default song inputs copied verbatim from the official Space app.py
"""

import json
from pathlib import Path

DIFFUSERS_COMMIT = "2da7040be1a2e5f2fcbc8b985083342a308f5a86"

# --- Space defaults, verbatim from MiniMaxAI/MiniMax-Music3 app.py ---------

DEFAULT_LYRICS = """[intro]

[verse]
Riding on a beam of light tonight
Every little star is burning bright
[pre-chorus]
Hold your breath, the sky is opening
[chorus]
We are made of sound and time
Every heartbeat keeps the rhyme
[outro]"""

DEFAULT_GLOBAL = (
    "Basic Attributes: bpm is 120. key is C, and scale is major. Synth-Pop / Electropop. Global Emotional "
    "Progression: The track opens in shimmering anticipation, a filtered pulse like city lights coming on at dusk. "
    "The verse glides forward with hopeful momentum, the pre-chorus holds its breath as the arrangement tightens "
    "and rises, and the chorus bursts open into wide-screen euphoria — bright, weightless, celebratory. The outro "
    "drifts back down into a starry afterglow, ending on air and quiet wonder. Application Scenarios & Imagery: a "
    "night drive under neon overpasses with the windows down; a planetarium dome igniting as the lights dim; a "
    "rooftop countdown at midnight. Sonics & Production Profile: a polished, modern pop mix with a wide stereo "
    "image — airy sparkling highs, present mid-range vocals, and a tight, punchy low end; side-chained compression "
    "gives the chorus a gentle pumping lift, and the outro dissolves into long reverb tails."
)
DEFAULT_VOCALS = (
    "Vocal Gender & Timbre: Singer A (Female), a warm mezzo-soprano with an intimate, breathy texture in her low "
    "register and a clear, ringing brightness when she lifts. Vocal Style: soft and close-miked through the verse, "
    "phrasing like a secret; the pre-chorus rises with held, urgent notes, and the chorus opens into a confident, "
    "soaring belt with sustained tones riding the beat; over the outro she dissolves into wordless, airy ad-libs "
    "echoing the chorus melody. Harmony/Backing Vocals: a single ghost double shadows the pre-chorus; stacked "
    "parallel harmonies in thirds widen the chorus into a glowing wall; the verse stays solo and intimate. Vocal "
    "FX: light plate reverb throughout, tempo-synced delay throws on chorus line endings, subtle saturation for "
    "chorus presence, and a longer, washier reverb on the outro ad-libs."
)
DEFAULT_ARRANGEMENT = (
    "Instrument Lifecycle Description (Primary/Secondary Layering): Primary: a round, side-chained analog-style "
    "synth bass anchors the harmony from the first verse through the chorus, under a soft pad bed that opens the "
    "intro and never fully leaves. Secondary: a shimmering arpeggio enters at the pre-chorus and runs through the "
    "chorus; wide analog pads and a bright synth counter-melody appear only in the chorus to lift it; a sparse felt "
    "piano takes over the outro as the synths fall away. Groove & Foundation Progression: the intro pulses on a "
    "filtered four-on-the-floor kick; the verse keeps drums minimal — kick, soft clap, ticking closed hat; the "
    "pre-chorus adds open hats and a rising snare build, and the chorus lands with the full kit: punchy kick on "
    "every beat, layered claps, driving crash accents. After the chorus the drums drop out entirely, leaving piano, "
    "pad, and air for the outro. Embellishments, Textures & Spatial FX: a white-noise riser and reverse swell "
    "launch the chorus; glittering bell accents answer the vocal there; and the final piano chord rings into a "
    "long, starlit reverb wash."
)

CAPTION_CONTRACT = """The three caption fields follow the exact labeled style the model was trained on. Be concrete and musical; describe an energy arc and instrument lifecycles, never a static equipment list or decorative adjectives. Never contradict an explicit user constraint: instrumental stays instrumental; never reverse a required vocal gender, tempo limit, required instrument, or exclusion. Do not quote or paraphrase lyric lines inside the caption. Total caption length roughly 250-400 words.

global_metadata: one paragraph, in order: "Basic Attributes: bpm is <number>. key is <letter>, and scale is <major|minor>. <Genre / Subgenre>." then "Global Emotional Progression: <how the emotion evolves from the opening through the final section>." then "Application Scenarios & Imagery: <two or three vivid listening scenarios>." then "Sonics & Production Profile: <soundstage, frequency balance, dynamics, production character>."

vocal_details: one paragraph: "Vocal Gender & Timbre: Singer A (<Male|Female>), <timbre and register>." then "Vocal Style: <delivery, and how it shifts per section>." then "Harmony/Backing Vocals: <where harmonies or doubles appear and their character>." then "Vocal FX: <restrained treatment: reverb, delay, light compression>." For instrumental pieces write "Instrumental, no vocals." and name the instrument or texture carrying the lead melodic role.

arrangement: one paragraph: "Instrument Lifecycle Description (Primary/Secondary Layering): Primary: <core instruments present start to finish and their role>. Secondary: <instruments that enter, exit or intensify, and in which sections>." then "Groove & Foundation Progression: <how drums, bass and groove develop across sections>." then "Embellishments, Textures & Spatial FX: <fills, textures, transitional gestures, stereo and space treatment where relevant>." State what enters, exits, changes or intensifies for every section of the song, aligned with the lyric section tags."""

LYRICS_RULES = """lyrics: singable lyrics using ONLY these section tags, each ALWAYS ALONE on its own line: [intro] [verse] [pre-chorus] [chorus] [post-chorus] [bridge] [instrumental] [solo] [outro]. Never put words on the same line as a tag. Size the structure to the duration: <=30s: one verse + one chorus; ~60s: verse/pre-chorus/chorus/verse/chorus; >=120s: full structure with bridge and outro. Roughly 12-16 sung words per 10 seconds. Musical instructions (tempo, instruments, dynamics) never belong in the lyrics. If the song is instrumental, use [instrumental] sections with no words."""

# --- cells ------------------------------------------------------------------

cells = []


def md(source):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": source})


def code(source, form=False):
    metadata = {"cellView": "form"} if form else {}
    cells.append({
        "cell_type": "code",
        "metadata": metadata,
        "execution_count": None,
        "outputs": [],
        "source": source,
    })


md(f"""# MiniMax Music 3 — Colab

Generates full songs (up to 5 minutes, 44.1 kHz stereo, sung vocals) from lyrics plus a
structured text description, using the open [MiniMaxAI/MiniMax-Music3](https://huggingface.co/MiniMaxAI/MiniMax-Music3)
weights through the diffusers `ModularPipeline` — the same stack behind the official
[MiniMax Music 3 Studio Space](https://huggingface.co/spaces/MiniMaxAI/MiniMax-Music3).

**Before you run anything:** `Runtime → Change runtime type → GPU`.

| Runtime | Fits? | Notes |
|---|---|---|
| A100 (40 GB) | best | everything stays on the GPU |
| L4 (24 GB) | should fit | components hop between CPU and GPU automatically, per the model card's offload recipe; start with 60 s songs and work up gradually |
| T4 (16 GB) | no | pre-Ampere GPU without native bfloat16 — the notebook refuses it |

Loading also passes the ~22 GB of weights through host RAM, so the runtime needs about
30 GB of it. A100 and L4 shapes come with plenty; the standard T4 shape does not.

The model is ~11B parameters across four stages (8B language model, 0.6B frame decoder,
2.4B flow-matching transformer, vocoder). First load downloads ~22 GB of weights, which
takes a few minutes. Budget a few minutes of GPU time per minute of audio on an A100,
more on smaller cards.

Songs save to a Google Drive folder you pick (or `/content/songs` if you skip Drive),
each with a JSON sidecar of the seed and inputs that made it. A seed-sweep cell
batch-generates the same song across many seeds so you can keep the best take.""")

code("""# Confirm the runtime can hold the model before the long install and download.
import os
import shutil

import torch

assert torch.cuda.is_available(), "No GPU. Runtime -> Change runtime type -> GPU, then re-run."
props = torch.cuda.get_device_properties(0)
vram_gb = props.total_memory / 1024**3
ram_gb = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1024**3
disk_gb = shutil.disk_usage("/").free / 1024**3
print(f"{props.name} — {vram_gb:.0f} GB VRAM | host RAM {ram_gb:.0f} GB | free disk {disk_gb:.0f} GB")

assert disk_gb >= 30, "Need ~30 GB of free disk for the weights — use a fresh runtime."
assert ram_gb >= 30, (
    "The ~22 GB of weights pass through host RAM while loading; this shape is too small. "
    "Pick an A100 or L4 runtime."
)
if not torch.cuda.is_bf16_supported():
    raise RuntimeError(
        "This GPU has no native bfloat16 (pre-Ampere, e.g. T4) and the model ships in bf16. "
        "Use an L4 or A100 runtime."
    )""")

code(f"""# Install dependencies. MiniMax Music 3 merged into diffusers on 2026-08-13
# (huggingface/diffusers#14456) but hasn't shipped in a PyPI release yet, so we pin the
# merge commit. Once a release newer than v0.39.0 is out, plain `diffusers` works too.
# Takes a few minutes.
import sys

if "diffusers" in sys.modules or "transformers" in sys.modules:
    print("diffusers/transformers already imported in this kernel — after the install finishes, "
          "Runtime -> Restart session, then rerun from the top.")

%pip install -q -U "git+https://github.com/huggingface/diffusers@{DIFFUSERS_COMMIT}" transformers accelerate soundfile openai gradio

from importlib.metadata import version

print(" | ".join(f"{{p}} {{version(p)}}" for p in ("diffusers", "transformers", "accelerate", "gradio")))""")

code("""# Optional: Hugging Face token from Colab secrets (key icon in the left sidebar,
# add a secret named HF_TOKEN). The weights are public, so this is only needed for the
# "compose with an LLM" cell further down, and to dodge anonymous-download rate limits.
import os

try:
    from google.colab import userdata
    os.environ["HF_TOKEN"] = userdata.get("HF_TOKEN")
    print("HF_TOKEN set.")
except Exception:
    print("No HF_TOKEN secret — fine for generation, the composer cell won't work.")""")

code("""#@title Storage { display-mode: "form" }
#@markdown With Drive on, every song saves to `My Drive/<drive_folder>` as a WAV plus a
#@markdown `.json` of the exact inputs and seed that made it, and normally survives
#@markdown runtime recycling. (Mounting pops an authorization prompt; if you decline it,
#@markdown set `save_to_drive` off instead.) WAVs are ~10 MB per minute of audio.
#@markdown With Drive off, songs go to `/content/songs` and vanish with the runtime.
save_to_drive = True  #@param {type:"boolean"}
drive_folder = "MiniMax-Music3"  #@param {type:"string"}

import json
import os

import soundfile as sf

if save_to_drive:
    from google.colab import drive

    drive.mount("/content/drive")
    _root = os.path.realpath("/content/drive/MyDrive")
    SONGS_DIR = os.path.realpath(os.path.join(_root, drive_folder))
    assert SONGS_DIR == _root or SONGS_DIR.startswith(_root + os.sep), \
        "drive_folder must stay inside My Drive"
else:
    SONGS_DIR = "/content/songs"
os.makedirs(SONGS_DIR, exist_ok=True)


def save_song(audio, seed, meta):
    # Microsecond stamp rules out same-second name collisions; writing to .part and
    # renaming keeps a dying runtime or a full Drive from leaving truncated files
    # under the final names.
    from datetime import datetime

    base = f"{SONGS_DIR}/song_{datetime.now().strftime('%Y%m%d-%H%M%S-%f')}_seed{seed}"
    sf.write(base + ".wav.part", audio.T, pipe.sampling_rate, format="WAV", subtype="PCM_16")
    with open(base + ".json.part", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    os.replace(base + ".wav.part", base + ".wav")
    os.replace(base + ".json.part", base + ".json")
    return base + ".wav"


print("Saving songs to", SONGS_DIR)""")

code("""# Load the pipeline. Strategy depends on VRAM:
#   >=28 GB (A100): everything on the GPU
#   22-28 GB (L4):  automatic CPU offload, ~22 GB peak per the model card
#   <22 GB:         also stream the 8B language model layer by layer (slow)
#
# If a long song dies with CUDA out-of-memory: Runtime -> Restart session, set
# FORCE_LM_STREAMING = True, and rerun from the top. (Re-running this cell without a
# restart would load a second copy of the weights.)
FORCE_LM_STREAMING = False
LM_STREAMING_VRAM_GB = 22

import torch
from diffusers import ModularPipeline

vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3

if vram_gb >= 28 and not FORCE_LM_STREAMING:
    pipe = ModularPipeline.from_pretrained("MiniMaxAI/MiniMax-Music3")
    pipe.load_components(dtype=torch.bfloat16)
    pipe.to("cuda")
    print("Loaded fully on GPU.")
else:
    from diffusers import ComponentsManager

    manager = ComponentsManager()
    manager.enable_auto_cpu_offload(device="cuda")
    pipe = ModularPipeline.from_pretrained("MiniMaxAI/MiniMax-Music3", components_manager=manager)
    pipe.load_components(dtype=torch.bfloat16)
    if vram_gb < LM_STREAMING_VRAM_GB or FORCE_LM_STREAMING:
        from diffusers.hooks import apply_group_offloading

        apply_group_offloading(
            pipe.language_model,
            onload_device=torch.device("cuda"),
            offload_type="leaf_level",
            use_stream=True,
        )
        print("Loaded with CPU offload + layer streaming for the language model.")
    else:
        print("Loaded with automatic CPU offload.")

PIPE_READY = True
print(f"Sampling rate: {pipe.sampling_rate} Hz, frame rate: {pipe.frame_rate:.0f} frames/s")""")

code("""# Recommended one-minute smoke test: exercises the whole stack (language model ->
# flow matching -> vocoder) on a 5-second, 4-step request before you commit GPU time to a
# full song. The audio will sound rough at 4 steps — that's expected.
assert globals().get("PIPE_READY"), (
    "The pipeline isn't loaded — run the cells above first (install -> load). "
    "A fresh or restarted runtime starts empty, even if outputs are still visible."
)

import numpy as np
import torch
from diffusers.guiders import ClassifierFreeGuidance

pipe.update_components(guider=ClassifierFreeGuidance(guidance_scale=1.7))
torch.cuda.reset_peak_memory_stats()
test_audio = pipe(
    prompt="Basic Attributes: bpm is 100. key is C, and scale is major. Acoustic pop.",
    lyrics="[verse]\\nMorning light begins to rise",
    audio_duration=5.0,
    num_inference_steps=4,
    generator=torch.Generator("cuda").manual_seed(0),
    output_type="np",
    output="audios",
)[0]
assert test_audio.ndim == 2 and test_audio.shape[0] == 2 and np.isfinite(test_audio).all()
print(f"OK — {test_audio.shape[-1] / pipe.sampling_rate:.1f}s of audio, "
      f"peak VRAM {torch.cuda.max_memory_allocated() / 1024**3:.1f} GB")""")

md("""## Writing the song

Two inputs drive the model:

**Lyrics** use section tags, each alone on its own line:
`[intro] [verse] [pre-chorus] [chorus] [post-chorus] [bridge] [instrumental] [solo] [outro]`.
About 12–16 sung words per 10 seconds. For an instrumental, use `[instrumental]` sections
with no words.

**The structured caption** is three paragraphs the model was trained on, joined together:

1. *Global metadata* — `Basic Attributes: bpm is …. key is …, and scale is ….  <genre>.`
   followed by the emotional progression, listening scenarios, and production profile.
2. *Vocal details* — singer gender and timbre, delivery per section, harmonies, vocal FX
   (or `Instrumental, no vocals.` plus what carries the lead melody).
3. *Arrangement* — which instruments enter, exit, or intensify in each section; how the
   groove develops; textures and transitions.

Describe an energy arc, not an equipment list. The defaults below are the official Space's
demo song — run them as-is once to check everything works, then make it yours. If you'd
rather not write all this by hand, the composer cell near the bottom drafts it from a
one-line description.""")

code(f'''# Song inputs — edit freely (defaults are the official Space's demo song)

lyrics = """{DEFAULT_LYRICS}"""

global_metadata = """{DEFAULT_GLOBAL}"""

vocal_details = """{DEFAULT_VOCALS}"""

arrangement = """{DEFAULT_ARRANGEMENT}"""''')

code("""#@title Generate { display-mode: "form" }
#@markdown Settings match the Space's knobs. The duration is an upper bound — the model may end the song earlier.
duration_seconds = 60  #@param {type:"slider", min:5, max:300, step:5}
num_inference_steps = 30  #@param {type:"slider", min:4, max:60, step:1}
guidance_scale = 1.7  #@param {type:"slider", min:1.0, max:4.0, step:0.1}
seed = 0  #@param {type:"integer"}
randomize_seed = True  #@param {type:"boolean"}

_required = ("save_song", "lyrics", "global_metadata", "vocal_details", "arrangement")
_missing = [n for n in _required if n not in globals()]
assert globals().get("PIPE_READY") and not _missing, (
    f"Missing setup ({', '.join(_missing) or 'pipeline load incomplete'}) — "
    "run the cells above first (install -> storage -> load -> song inputs). "
    "A fresh or restarted runtime starts empty, even if outputs are still visible."
)

import random
import time

import torch
from diffusers.guiders import ClassifierFreeGuidance
from IPython.display import Audio, display

if randomize_seed:
    seed = random.randint(0, 2**32 - 1)

if duration_seconds > 120 and torch.cuda.get_device_properties(0).total_memory / 1024**3 < 28:
    print("Heads-up: songs over ~2 minutes are untested on 24 GB GPUs. If this dies with CUDA "
          "out-of-memory, restart the runtime and set FORCE_LM_STREAMING = True in the load cell.")

pipe.update_components(guider=ClassifierFreeGuidance(guidance_scale=float(guidance_scale)))

caption = "\\n".join(s.strip() for s in (global_metadata, vocal_details, arrangement) if s.strip())

start = time.time()
audio = pipe(
    prompt=caption,
    lyrics=lyrics,
    audio_duration=float(duration_seconds),
    num_inference_steps=int(num_inference_steps),
    generator=torch.Generator("cuda").manual_seed(seed),
    output_type="np",
    output="audios",
)[0]
if hasattr(audio, "cpu"):
    audio = audio.float().cpu().numpy()

out_path = save_song(audio, seed, {
    "seed": seed,
    "duration_requested": duration_seconds,
    "num_inference_steps": num_inference_steps,
    "guidance_scale": guidance_scale,
    "audio_seconds": round(audio.shape[-1] / pipe.sampling_rate, 1),
    "generation_seconds": round(time.time() - start),
    "global_metadata": global_metadata,
    "vocal_details": vocal_details,
    "arrangement": arrangement,
    "lyrics": lyrics,
})
print(f"seed {seed} — {audio.shape[-1] / pipe.sampling_rate:.1f}s of audio in {time.time() - start:.0f}s -> {out_path}")

# The inline player embeds base64 PCM in the notebook (~10 MB per minute), so long songs
# get a 60-second preview here; the WAV on disk is always the full song.
preview = audio if audio.shape[-1] <= 90 * pipe.sampling_rate else audio[:, : 60 * pipe.sampling_rate]
display(Audio(preview, rate=pipe.sampling_rate))
if preview.shape[-1] < audio.shape[-1]:
    print(f"(player shows the first 60s — download the full song from {out_path})")""")

md("""Every song lands in the storage folder from the Storage cell — your Drive folder if
you mounted it — as a WAV plus a `.json` recording the seed, settings, caption, and
lyrics that made it. When a seed comes out great, that sidecar is how you reproduce it.

## Seed sweep

The same inputs give meaningfully different songs on different seeds, so the usual
workflow is: generate a batch, listen, keep the winners. The cell below runs the current
song inputs across several seeds. Each song saves as soon as it finishes, so stopping
the cell (or losing the runtime) keeps everything generated up to that point. Budget GPU
time accordingly: a 10-song sweep at 60 s each is on the order of an hour on an A100.""")

code("""#@title Seed sweep — same song, many seeds { display-mode: "form" }
num_songs = 5  #@param {type:"slider", min:2, max:20, step:1}
sweep_duration_seconds = 60  #@param {type:"slider", min:5, max:300, step:5}
sweep_steps = 30  #@param {type:"slider", min:4, max:60, step:1}
sweep_guidance = 1.7  #@param {type:"slider", min:1.0, max:4.0, step:0.1}
seed_mode = "random"  #@param ["random", "sequential from base_seed"]
base_seed = 0  #@param {type:"integer"}
#@markdown `preview_seconds` controls the inline player per song (0 = no players, just files).
preview_seconds = 30  #@param {type:"slider", min:0, max:60, step:5}

_required = ("save_song", "lyrics", "global_metadata", "vocal_details", "arrangement")
_missing = [n for n in _required if n not in globals()]
assert globals().get("PIPE_READY") and not _missing, (
    f"Missing setup ({', '.join(_missing) or 'pipeline load incomplete'}) — "
    "run the cells above first (install -> storage -> load -> song inputs). "
    "A fresh or restarted runtime starts empty, even if outputs are still visible."
)

import random
import time

import torch
from diffusers.guiders import ClassifierFreeGuidance
from IPython.display import Audio, display

if sweep_duration_seconds > 120 and torch.cuda.get_device_properties(0).total_memory / 1024**3 < 28:
    print("Heads-up: songs over ~2 minutes are untested on 24 GB GPUs. If this dies with CUDA "
          "out-of-memory, restart the runtime and set FORCE_LM_STREAMING = True in the load cell.")

pipe.update_components(guider=ClassifierFreeGuidance(guidance_scale=float(sweep_guidance)))
caption = "\\n".join(s.strip() for s in (global_metadata, vocal_details, arrangement) if s.strip())

if seed_mode == "random":
    seeds = random.sample(range(2**32), k=int(num_songs))
else:
    seeds = [(int(base_seed) + i) % 2**32 for i in range(num_songs)]

results = []
for i, seed in enumerate(seeds, 1):
    print(f"[{i}/{num_songs}] seed {seed}")
    start = time.time()
    audio = pipe(
        prompt=caption,
        lyrics=lyrics,
        audio_duration=float(sweep_duration_seconds),
        num_inference_steps=int(sweep_steps),
        generator=torch.Generator("cuda").manual_seed(int(seed)),
        output_type="np",
        output="audios",
    )[0]
    if hasattr(audio, "cpu"):
        audio = audio.float().cpu().numpy()
    path = save_song(audio, seed, {
        "seed": seed,
        "duration_requested": sweep_duration_seconds,
        "num_inference_steps": sweep_steps,
        "guidance_scale": sweep_guidance,
        "audio_seconds": round(audio.shape[-1] / pipe.sampling_rate, 1),
        "generation_seconds": round(time.time() - start),
        "global_metadata": global_metadata,
        "vocal_details": vocal_details,
        "arrangement": arrangement,
        "lyrics": lyrics,
    })
    results.append((seed, audio.shape[-1] / pipe.sampling_rate, path))
    print(f"    {results[-1][1]:.1f}s of audio in {time.time() - start:.0f}s -> {path}")
    if preview_seconds:
        display(Audio(audio[:, : int(preview_seconds * pipe.sampling_rate)], rate=pipe.sampling_rate))
    del audio

print(f"\\nSweep done — {len(results)} songs:")
for seed, secs, path in results:
    print(f"  seed {seed:>10} — {secs:5.1f}s — {path}")""")

composer_system = (
    "You write inputs for MiniMax Music 3, a lyrics+description music generation model.\n"
    "Given a song description and a target duration, produce:\n"
    "1. " + LYRICS_RULES + "\n"
    "2-4. global_metadata, vocal_details, arrangement — a structured caption. " + CAPTION_CONTRACT + "\n"
    "Answer with ONLY a JSON object with keys: lyrics, global_metadata, vocal_details, arrangement."
)

code('''#@title Optional: compose lyrics + caption from a one-line description { display-mode: "form" }
#@markdown Uses an LLM via the Hugging Face inference router (needs the HF_TOKEN secret).
#@markdown Overwrites `lyrics`, `global_metadata`, `vocal_details`, `arrangement` — re-run the Generate cell afterwards.
song_description = "a slow-burning desert blues about driving all night, gravelly male vocals"  #@param {type:"string"}
target_duration_seconds = 60  #@param {type:"slider", min:5, max:300, step:5}
composer_model = "deepseek-ai/DeepSeek-V4-Flash-0731"  #@param {type:"string"}
#@markdown `composer_model` is a base router model id — provider suffixes are tried automatically.

import json
import os
import time

from openai import OpenAI

assert os.environ.get("HF_TOKEN"), "This cell needs the HF_TOKEN Colab secret (see the token cell above)."

_COMPOSER_SYSTEM = """''' + composer_system + '''"""

# Bare model ids can route to a provider that 403s; the official Space works around this
# with distinct provider suffixes, bounded timeouts, and a second pass for transient 429s.
client = OpenAI(base_url="https://router.huggingface.co/v1", api_key=os.environ["HF_TOKEN"], max_retries=0)
user_message = f"Song description: {song_description}\\nTarget duration: {int(target_duration_seconds)} seconds."
required = ("lyrics", "global_metadata", "vocal_details", "arrangement")

data = last_error = None
for attempt in range(2):
    for provider, timeout in (("baseten", 45), ("deepinfra", 75), ("novita", 100)):
        try:
            reply = client.with_options(timeout=timeout).chat.completions.create(
                model=f"{composer_model}:{provider}",
                messages=[
                    {"role": "system", "content": _COMPOSER_SYSTEM},
                    {"role": "user", "content": user_message},
                ],
            ).choices[0].message.content or ""
            start, end = reply.find("{"), reply.rfind("}")
            if start == -1 or end <= start:
                raise ValueError("no JSON object in the reply")
            candidate = json.loads(reply[start : end + 1])
            missing = [k for k in required if k not in candidate]
            if missing:
                raise ValueError(f"reply missing keys: {missing}")
            data = candidate
            break
        except Exception as e:
            print(f"{composer_model}:{provider}: {type(e).__name__}: {e}")
            last_error = e
    if data:
        break
    if attempt == 0:
        time.sleep(3)

if data is None:
    raise RuntimeError("All composer providers failed — try again in a minute, or write the song inputs by hand.") from last_error

lyrics = data["lyrics"]
global_metadata = data["global_metadata"]
vocal_details = data["vocal_details"]
arrangement = data["arrangement"]
for part in (lyrics, global_metadata, vocal_details, arrangement):
    print(part)
    print()
print(f"Now set the Generate cell's duration slider to ~{int(target_duration_seconds)}s and run it — "
      "the lyrics are sized for that length.")''')

md("""## Optional: web UI

A small Gradio app with the same inputs, served through a public share link — handy for
using the notebook from a phone while the runtime keeps working. Caveats:

- The share URL is public and unauthenticated — anyone with the link can queue jobs on
  your GPU. Links expire when the runtime recycles.
- Generation is non-streaming: the player fills in when the song is done, with no
  progress in between (watch the notebook cell output for the step counter).
- Songs save to the storage folder from the Storage cell, same as the notebook cells.""")

code("""_required = ("save_song", "lyrics", "global_metadata", "vocal_details", "arrangement")
_missing = [n for n in _required if n not in globals()]
assert globals().get("PIPE_READY") and not _missing, (
    f"Missing setup ({', '.join(_missing) or 'pipeline load incomplete'}) — "
    "run the cells above first (install -> storage -> load -> song inputs). "
    "A fresh or restarted runtime starts empty, even if outputs are still visible."
)

import random
import time

import gradio as gr
import numpy as np
import torch
from diffusers.guiders import ClassifierFreeGuidance


def ui_generate(gm, vd, arr, lyr, dur, steps, guidance, seed, randomize):
    if randomize:
        seed = random.randint(0, 2**32 - 1)
    seed = int(seed)
    pipe.update_components(guider=ClassifierFreeGuidance(guidance_scale=float(guidance)))
    caption = "\\n".join(s.strip() for s in (gm, vd, arr) if s.strip())
    start = time.time()
    audio = pipe(
        prompt=caption,
        lyrics=lyr,
        audio_duration=float(dur),
        num_inference_steps=int(steps),
        generator=torch.Generator("cuda").manual_seed(seed),
        output_type="np",
        output="audios",
    )[0]
    if hasattr(audio, "cpu"):
        audio = audio.float().cpu().numpy()
    path = save_song(audio, seed, {
        "seed": seed,
        "duration_requested": dur,
        "num_inference_steps": steps,
        "guidance_scale": guidance,
        "audio_seconds": round(audio.shape[-1] / pipe.sampling_rate, 1),
        "generation_seconds": round(time.time() - start),
        "global_metadata": gm,
        "vocal_details": vd,
        "arrangement": arr,
        "lyrics": lyr,
    })
    pcm = (audio.T * 32767.0).astype(np.int16)
    return (pipe.sampling_rate, pcm), seed, path


with gr.Blocks(title="MiniMax Music 3") as demo:
    gr.Markdown("# MiniMax Music 3")
    with gr.Row():
        with gr.Column():
            gm_box = gr.Textbox(label="Global metadata", value=global_metadata, lines=5)
            vd_box = gr.Textbox(label="Vocal details", value=vocal_details, lines=5)
            arr_box = gr.Textbox(label="Arrangement", value=arrangement, lines=5)
            lyr_box = gr.Textbox(label="Lyrics", value=lyrics, lines=10)
        with gr.Column():
            dur_sl = gr.Slider(5, 300, value=60, step=5, label="Max duration (s)")
            steps_sl = gr.Slider(4, 60, value=30, step=1, label="Flow-matching steps")
            guid_sl = gr.Slider(1.0, 4.0, value=1.7, step=0.1, label="Guidance scale")
            seed_box = gr.Number(value=0, label="Seed", precision=0)
            rand_ck = gr.Checkbox(value=True, label="Randomize seed")
            go = gr.Button("Generate", variant="primary")
            out_audio = gr.Audio(label="Song", type="numpy")
            used_seed = gr.Number(label="Seed used", precision=0)
            saved_to = gr.Textbox(label="Saved to", interactive=False)

    go.click(
        ui_generate,
        [gm_box, vd_box, arr_box, lyr_box, dur_sl, steps_sl, guid_sl, seed_box, rand_ck],
        [out_audio, used_seed, saved_to],
    )

demo.queue().launch(share=True)""")

md("""## Notes

- The first generation includes one-time CUDA initialization; later runs are faster.
- On 24 GB GPUs, long songs may run out of memory — the ~22 GB offloaded footprint is the
  official number for typical requests, not a five-minute guarantee. If it happens:
  Runtime → Restart session, set `FORCE_LM_STREAMING = True` in the load cell, rerun.
- Interrupting a generation mid-flight keeps everything already saved, but the pipeline's
  internal state after an interrupt isn't guaranteed clean — if the next run errors or
  GPU memory stays high, restart the session.
- Drive syncing is asynchronous: before deliberately killing a runtime right after a
  sweep, give it a minute, or run `from google.colab import drive; drive.flush_and_unmount()`
  in a scratch cell.
- Weights cache in `/root/.cache/huggingface` and are gone when the runtime recycles, so
  each fresh session re-downloads ~22 GB.
- The text prompt is capped at 5,000 tokens and audio at 9,000 frames (six minutes).
- Section tags and captions steer the model rather than guarantee an exact structure —
  tempo, key, and arrangement can drift from what you asked for. Different seeds give
  meaningfully different songs; when one comes out close, keep its seed.
- Weights are released under a Creative Commons license — check the
  [model card](https://huggingface.co/MiniMaxAI/MiniMax-Music3) before commercial use.""")

# --- write ------------------------------------------------------------------

notebook = {
    "nbformat": 4,
    "nbformat_minor": 0,
    "metadata": {
        "colab": {"provenance": [], "gpuType": "A100", "machine_shape": "hm"},
        "kernelspec": {"name": "python3", "display_name": "Python 3"},
        "language_info": {"name": "python"},
        "accelerator": "GPU",
    },
    "cells": [
        {**c, "source": c["source"].splitlines(keepends=True)} for c in cells
    ],
}

out = Path(__file__).resolve().parent.parent / "MiniMax_Music3_Colab.ipynb"
out.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n")
print(f"wrote {out} ({len(cells)} cells)")
