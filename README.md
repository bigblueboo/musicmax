# musicmax

A Colab port of the [MiniMax Music 3 Studio Space](https://huggingface.co/spaces/MiniMaxAI/MiniMax-Music3): full-song generation (lyrics + structured caption → 44.1 kHz stereo, up to 5 minutes) from the open [MiniMaxAI/MiniMax-Music3](https://huggingface.co/MiniMaxAI/MiniMax-Music3) weights via the diffusers `ModularPipeline`. The Space's ZeroGPU/AoTI machinery is dropped; the notebook picks a loading strategy from available VRAM instead: A100 runs fully on GPU, L4 uses automatic CPU offload, and smaller Ampere-or-newer GPUs stream the 8B language model layer by layer. T4 is refused — it lacks native bfloat16 and its standard Colab shape lacks the host RAM the load needs.

## Run

Upload `MiniMax_Music3_Colab.ipynb` at [colab.research.google.com](https://colab.research.google.com) (File → Upload notebook), switch the runtime to a GPU (A100 recommended; L4 is expected to fit per the model card's offload recipe, though this repo hasn't GPU-validated it; T4 is refused), and run the cells top to bottom. A one-minute smoke-test cell validates the whole stack before you spend GPU time on a full song. Once this repo is pushed to GitHub it also opens directly:

    https://colab.research.google.com/github/bigblueboo/musicmax/blob/main/MiniMax_Music3_Colab.ipynb

Optional Colab secret `HF_TOKEN` enables the "compose lyrics + caption from a description" cell (Hugging Face inference router) and avoids anonymous download throttling.

## Quality gates

```
python3 tools/build_notebook.py        # regenerates the notebook (source of truth)
python3 tools/test_notebook.py         # dry-run: executes all cells against a mocked runtime
```

The test suite needs only Python 3 + numpy. It verifies the notebook regenerates from the
builder, statically checks cross-cell name resolution and the pinned install line, executes
every cell in notebook order against strict fakes (torch, diffusers, google.colab, gradio,
openai, soundfile) across a matrix of GPU/RAM scenarios and form-param variants — asserting
offload-branch selection (including that the manager reaches `from_pretrained`), saved
WAV+JSON pairs, sidecar contents, per-call guidance, composer failover, and UI wiring —
and re-runs each inference cell in empty and partially-populated namespaces to assert the
notebook's own guard message fires instead of a `NameError`. It is a control-flow dry run:
real installs, CUDA behavior, Drive durability, and Gradio compatibility still need an
actual Colab run.

## Layout

- `tools/build_notebook.py` — generates the notebook; edit this, not the `.ipynb`.
- `tools/test_notebook.py` — mocked dry-run test suite for the generated notebook.
- `space/` — reference copy of the official Space (`app.py`, model card, pinned diffusers `minimax_music3` module), fetched 2026-08-13. Not shipped anywhere.

## Notes

- diffusers support merged 2026-08-13 in [huggingface/diffusers#14456](https://github.com/huggingface/diffusers/pull/14456); the notebook pins the merge commit (`2da7040b`) until a PyPI release includes it. `transformers`/`accelerate`/`gradio` are unpinned — after the first validated Colab run, record the printed versions and pin them here.
- Pipeline knobs mirror the Space: duration 5–300 s, flow-matching steps default 30, guidance 1.7 via the `ClassifierFreeGuidance` guider, seed control.
- Songs auto-save to a Google Drive folder (optional mount; falls back to `/content/songs`), each WAV paired with a `.json` sidecar of the seed and inputs. A seed-sweep cell batch-generates the same song across random or sequential seeds, saving each as it finishes.
- Long songs on 24 GB GPUs are unverified; the notebook warns past 2 minutes and has a `FORCE_LM_STREAMING` escape hatch in the load cell.
- Reviewed twice by GPT-5.6 Pro (expert CLI, 2026-08-13/14); all P0/P1 findings addressed. Still static-only — nothing has executed on a GPU yet.
