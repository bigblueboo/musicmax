# musicmax

A Colab port of the [MiniMax Music 3 Studio Space](https://huggingface.co/spaces/MiniMaxAI/MiniMax-Music3): full-song generation (lyrics + structured caption → 44.1 kHz stereo, up to 5 minutes) from the open [MiniMaxAI/MiniMax-Music3](https://huggingface.co/MiniMaxAI/MiniMax-Music3) weights via the diffusers `ModularPipeline`. The Space's ZeroGPU/AoTI machinery is dropped; the notebook picks a loading strategy from available VRAM instead (A100 fully on-GPU, L4 with auto CPU offload, T4 with layer-streamed LM).

## Run

Upload `MiniMax_Music3_Colab.ipynb` at [colab.research.google.com](https://colab.research.google.com) (File → Upload notebook), switch the runtime to a GPU (A100 recommended, L4 fine; T4 is refused — pre-Ampere, no native bfloat16, and its standard shape lacks the ~30 GB host RAM the load needs), and run the cells top to bottom. A one-minute smoke-test cell validates the whole stack before you spend GPU time on a full song. Once this repo is pushed to GitHub it also opens directly:

    https://colab.research.google.com/github/bigblueboo/musicmax/blob/main/MiniMax_Music3_Colab.ipynb

Optional Colab secret `HF_TOKEN` enables the "compose lyrics + caption from a description" cell (Hugging Face inference router) and avoids anonymous download throttling.

## Quality gates

```
python3 tools/build_notebook.py        # regenerates the notebook (source of truth)
python3 tools/test_notebook.py         # dry-run: executes all cells against a mocked runtime
```

The test suite needs only Python 3 + numpy. It statically checks cross-cell name
resolution, executes every cell in notebook order against strict fakes (torch, diffusers,
google.colab, gradio, openai, soundfile) across five GPU/RAM scenarios asserting saved
files, branch selection, and UI wiring, and re-runs each inference cell in an empty
namespace to assert it fails with the notebook's own guard message instead of a
`NameError` — the Colab fresh-kernel case.

## Layout

- `tools/build_notebook.py` — generates the notebook; edit this, not the `.ipynb`.
- `tools/test_notebook.py` — mocked dry-run test suite for the generated notebook.
- `space/` — reference copy of the official Space (`app.py`, model card, pinned diffusers `minimax_music3` module), fetched 2026-08-13. Not shipped anywhere.

## Notes

- diffusers support merged 2026-08-13 in [huggingface/diffusers#14456](https://github.com/huggingface/diffusers/pull/14456); the notebook pins the merge commit (`2da7040b`) until a PyPI release includes it. `transformers`/`accelerate`/`gradio` are unpinned — after the first validated Colab run, record the printed versions and pin them here.
- Pipeline knobs mirror the Space: duration 5–300 s, flow-matching steps default 30, guidance 1.7 via the `ClassifierFreeGuidance` guider, seed control.
- Songs auto-save to a Google Drive folder (optional mount; falls back to `/content/songs`), each WAV paired with a `.json` sidecar of the seed and inputs. A seed-sweep cell batch-generates the same song across random or sequential seeds, saving each as it finishes.
- Long songs on 24 GB GPUs are unverified; the notebook warns past 2 minutes and has a `FORCE_LM_STREAMING` escape hatch in the load cell.
- Reviewed by GPT-5.6 Pro (expert CLI) on 2026-08-13; all P0/P1 findings addressed. Still static-only — nothing has executed on a GPU yet.
