---
title: MiniMax Music 3 Studio
emoji: 🎵
colorFrom: pink
colorTo: purple
sdk: gradio
sdk_version: 6.24.0
app_file: app.py
pinned: true
suggested_hardware: zero-a10g
---

# MiniMax Music 3 Studio — diffusers demo

Streams full songs from lyrics + a structured caption using the `MiniMaxMusic3Pipeline` diffusers port.
The input surface is a single Suno-inspired custom `gr.HTML` composer (Simple ↔ Studio modes, section-tag
chips, structured-caption fields per the official prompting guide) that drives Gradio events via
`trigger()`/`props.value`; styling uses only theme CSS vars so it follows the Citrus theme natively.

- Weights: `MiniMaxAI/MiniMax-Music3`
- AoTI kernels: `diffusers-internal-dev/MiniMax-Music3-aoti` (compiled on RTX Pro 6000, matching ZeroGPU hardware)
- Generation streams chunk by chunk with a configurable playback headroom. The 8B language-model stage runs eager
  on ZeroGPU (its JIT StaticCache ladder needs a persistent process); AoTI-exporting the LM decode step per cache
  bucket is the follow-up that brings the extra ~1.9x.