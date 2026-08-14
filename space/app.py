# Gradio demo for the MiniMax Music 3 diffusers port. Inputs follow the official prompt guide:
# a Structured Caption (Global Metadata / Vocal Details / Arrangement) + tagged lyrics.
import json
import os
import random
import time

import gradio as gr
import numpy as np
import spaces
import torch
from huggingface_hub import snapshot_download

from diffusers import ModularPipeline
from diffusers.models.modeling_outputs import Transformer2DModelOutput

PIPE = ModularPipeline.from_pretrained("MiniMaxAI/MiniMax-Music3")
PIPE.load_components(dtype=torch.bfloat16)
PIPE.to("cuda")


def _encode_prompt(caption, lyrics, device):
    # the modular TextEncoderStep's logic, needed here because the app drives the AR stage manually
    import diffusers.modular_pipelines.minimax_music3.encoders as P

    text = (
        f"{P._IM_START}{P._CAPTION_START}{P._clean_caption(caption)}{P._CAPTION_END}"
        f"{P._LYRICS_START}{P._normalize_lyrics(lyrics)}{P._LYRICS_END}{P._IM_END}{P._AUDIO_START}"
    )
    input_ids = PIPE.tokenizer(text, return_tensors="pt")["input_ids"]
    if input_ids.shape[1] > P._MAX_PROMPT_TOKENS:
        raise gr.Error(f"The assembled prompt has {input_ids.shape[1]} tokens; the maximum is {P._MAX_PROMPT_TOKENS}.")
    unconditional_ids = input_ids.clone()
    unconditional_ids[:, 1:-2] = P._AUDIO_CFG_TOKEN_ID
    return torch.cat((input_ids, unconditional_ids), dim=0).to(device)

# AoTI-compiled kernels (RTX Pro 6000 variant). The transformer artifact is static over full 689-latent
# chunks; the once-per-song final short chunk falls back to eager.
_AOTI_DIR = snapshot_download("diffusers-internal-dev/MiniMax-Music3-aoti")
_eager_transformer_forward = PIPE.transformer.forward
spaces.aoti_load_from_package_dir(PIPE.transformer, f"{_AOTI_DIR}/transformer")
_aoti_transformer_forward = PIPE.transformer.forward


def _guarded_transformer_forward(hidden_states, timestep, encoder_hidden_states, return_dict=True):
    if hidden_states.shape[-1] == 689:
        out = _aoti_transformer_forward(hidden_states, timestep, encoder_hidden_states)
        if not isinstance(out, Transformer2DModelOutput):
            out = Transformer2DModelOutput(sample=out[0] if isinstance(out, (tuple, list)) else out)
        return out
    return _eager_transformer_forward(hidden_states, timestep, encoder_hidden_states, return_dict=return_dict)


PIPE.transformer.forward = _guarded_transformer_forward
spaces.aoti_load_from_package_dir(PIPE.vocoder, f"{_AOTI_DIR}/vocoder")

# AoTI LM decode step, one artifact per StaticCache bucket; eager per-frame glue. Eager full-sequence
# prefill writes directly into each artifact's cache buffers (aliased StaticCache), matching eager exactly.
import copy as _copy

import torch.nn as _nn
from transformers import StaticCache
from transformers.integrations.executorch import TorchExportableModuleForDecoderOnlyLM

_LM = PIPE.language_model
_BUCKETS = [1024, 2048, 4096, 8192]
_STOP_CHECK_INTERVAL = 25

_lm_headless = _copy.copy(_LM)
_lm_headless._modules = dict(_LM._modules)  # nn.Module shallow copies share _modules
_lm_headless.lm_head = _nn.Identity()
_lm_headless.generation_config = _copy.deepcopy(_LM.generation_config)
_lm_headless.generation_config.cache_implementation = "static"

_LM_STEPS = {}
for _bucket in _BUCKETS:
    _exportable = TorchExportableModuleForDecoderOnlyLM(
        _lm_headless, batch_size=2, max_cache_len=_bucket, device="cuda"
    )
    for _m in _exportable.modules():
        _m._non_persistent_buffers_set.clear()
    spaces.aoti_load_from_package_dir(_exportable.model, f"{_AOTI_DIR}/lm_step_{_bucket}")
    _LM_STEPS[_bucket] = _exportable.model


def _aliased_cache(step_module, bucket):
    cache = StaticCache(max_cache_len=bucket, config=_LM.config.get_text_config())
    cache.early_initialization(
        2, _LM.config.num_key_value_heads, _LM.config.head_dim, _LM.dtype, torch.device("cuda")
    )
    for i, layer in enumerate(cache.layers):
        layer.keys = step_module.get_buffer(f"key_cache_{i}")
        layer.values = step_module.get_buffer(f"value_cache_{i}")
        layer.cumulative_length = step_module.get_buffer(f"cumulative_length_{i}")
        layer.keys.zero_()
        layer.values.zero_()
        layer.cumulative_length.zero_()
    return cache


def _hop_lm_cache(src_bucket, dst_bucket, used):
    src, dst = _LM_STEPS[src_bucket], _LM_STEPS[dst_bucket]
    for i in range(_LM.config.num_hidden_layers):
        dst.get_buffer(f"key_cache_{i}")[:, :, :used] = src.get_buffer(f"key_cache_{i}")[:, :, :used]
        dst.get_buffer(f"value_cache_{i}")[:, :, :used] = src.get_buffer(f"value_cache_{i}")[:, :, :used]
        dst.get_buffer(f"cumulative_length_{i}").copy_(src.get_buffer(f"cumulative_length_{i}"))


def _iter_frames_aoti(text_ids, max_frames, generator=None):
    import diffusers.modular_pipelines.minimax_music3.encoders as P

    prompt_len = text_ids.shape[1]
    bucket = _BUCKETS[0]
    while bucket < prompt_len + 16:
        bucket *= 2
    step = _LM_STEPS[bucket]
    cache = _aliased_cache(step, bucket)
    prompt_embeds = _LM.model.embed_tokens(text_ids)
    output = _LM.model(
        inputs_embeds=prompt_embeds,
        past_key_values=cache,
        cache_position=torch.arange(prompt_len, device="cuda"),
        use_cache=True,
    )
    last_hidden = output.last_hidden_state[:, -1]

    vocab_mask = torch.ones(_LM.config.vocab_size, dtype=torch.bool, device="cuda")
    vocab_mask[P._AUDIO_CODE_OFFSET : P._AUDIO_CODE_OFFSET + P._SEMANTIC_VOCAB_SIZE] = False
    vocab_mask[P._AUDIO_END_TOKEN_ID] = False

    emitted = 0
    position = prompt_len
    pending = []
    for frame_index in range(max_frames + 1):
        if position + 2 >= bucket:
            new_bucket = bucket * 2
            _hop_lm_cache(bucket, new_bucket, position)
            bucket = new_bucket
            step = _LM_STEPS[bucket]
        logits = _LM.lm_head(last_hidden).float()
        logits = logits.masked_fill(vocab_mask, -float("inf"))
        conditional, unconditional = logits[0:1], logits[1:2]
        guided = unconditional + (conditional - unconditional) * P._AR_CFG_SCALE
        threshold = torch.topk(conditional, P._AR_CFG_TOP_K, dim=-1).values[..., -1, None]
        guided = guided.masked_fill(conditional < threshold, -float("inf"))
        guided = guided.masked_fill(vocab_mask.unsqueeze(0), -float("inf"))
        sampled = P._sample_top_k(guided, generator)
        semantic_code = (sampled - P._AUDIO_CODE_OFFSET).clamp_min(0).repeat(2)
        frame_codes, depth_hidden = P._generate_depth_codes(PIPE, last_hidden, semantic_code, generator)
        frame_hidden = torch.cat((last_hidden[:1].clone(), depth_hidden), dim=-1) if frame_index > 0 else None
        pending.append((sampled, frame_hidden))
        if len(pending) >= _STOP_CHECK_INTERVAL or frame_index == max_frames:
            stop_flags = torch.cat([s == P._AUDIO_END_TOKEN_ID for s, _ in pending]).tolist()
            for flag, (_, fh) in zip(stop_flags, pending):
                if flag:
                    return
                if fh is not None:
                    emitted += 1
                    yield fh
                    if emitted >= max_frames:
                        return
            pending = []
        feedback = P._embed_audio_frame(PIPE, frame_codes)
        last_hidden = step(inputs_embeds=feedback, cache_position=torch.tensor([position], device="cuda"))[:, -1]
        position += 1
    for _, fh in pending:
        if fh is not None:
            yield fh


PIPE._iter_frames = _iter_frames_aoti


def _iter_frames_eager(text_ids, max_frames, generator=None):
    # Yields one hidden state [1, 32768] per generated frame (eager LM path).
    import diffusers.modular_pipelines.minimax_music3.encoders as P

    lm = PIPE.language_model
    embeds = lm.model.embed_tokens(text_ids)
    output = lm.model(inputs_embeds=embeds, use_cache=True)
    past_key_values = output.past_key_values
    last_hidden = output.last_hidden_state[:, -1]

    vocab_mask = torch.ones(lm.config.vocab_size, dtype=torch.bool, device=text_ids.device)
    vocab_mask[P._AUDIO_CODE_OFFSET : P._AUDIO_CODE_OFFSET + P._SEMANTIC_VOCAB_SIZE] = False
    vocab_mask[P._AUDIO_END_TOKEN_ID] = False

    emitted = 0
    for frame_index in range(max_frames + 1):
        logits = lm.lm_head(last_hidden).float().masked_fill(vocab_mask, -float("inf"))
        conditional, unconditional = logits[0:1], logits[1:2]
        guided = unconditional + (conditional - unconditional) * P._AR_CFG_SCALE
        threshold = torch.topk(conditional, P._AR_CFG_TOP_K, dim=-1).values[..., -1, None]
        guided = guided.masked_fill(conditional < threshold, -float("inf"))
        guided = guided.masked_fill(vocab_mask.unsqueeze(0), -float("inf"))
        sampled = P._sample_top_k(guided, generator)
        if int(sampled.item()) == P._AUDIO_END_TOKEN_ID:
            break
        semantic_code = (sampled - P._AUDIO_CODE_OFFSET).repeat(2)
        frame_codes, depth_hidden = P._generate_depth_codes(PIPE, last_hidden, semantic_code, generator)
        if frame_index > 0:
            emitted += 1
            yield torch.cat((last_hidden[:1].clone(), depth_hidden), dim=-1)
            if emitted >= max_frames:
                break
        feedback = P._embed_audio_frame(PIPE, frame_codes)
        output = lm.model(inputs_embeds=feedback, past_key_values=past_key_values, use_cache=True)
        past_key_values = output.past_key_values
        last_hidden = output.last_hidden_state[:, -1]


# eager fallback available as _iter_frames_eager

# LM_COMPILE=1 (default): compile the 8B backbone's decode step with a StaticCache — measured 1.9x on the
# autoregressive stage, which dominates song time. The DIT stays eager: SDPA auto-dispatch already runs
# FlashAttention-2 there and torch.compile measured slower end-to-end. First generation per cache bucket
# pays ~1 min of compilation.
if os.environ.get("LM_COMPILE", "0") == "1":
    from transformers import StaticCache

    _lm = PIPE.language_model
    _depth = PIPE.rvq_depth_decoder

    def _lm_decode_step(inputs_embeds, cache_position, cache):
        output = _lm.model(
            inputs_embeds=inputs_embeds, past_key_values=cache, cache_position=cache_position, use_cache=True
        )
        return output.last_hidden_state[:, -1]

    _compiled_lm_step = torch.compile(_lm_decode_step, fullgraph=True)

    def _new_cache(length):
        return StaticCache(config=_lm.config, max_batch_size=2, max_cache_len=length, device="cuda", dtype=_lm.dtype)

    def _grow_cache(old, new_len):
        # Migrate K/V into the next bucket: allocated stays within 2x of used, and every bucket size hits its
        # pre-compiled specialization (attention cost scales with the ALLOCATED static length).
        new = _new_cache(new_len)
        for old_layer, new_layer in zip(old.layers, new.layers):
            used = int(old_layer.cumulative_length.item())
            new_layer.lazy_initialization(old_layer.keys[:, :, :1], old_layer.values[:, :, :1])
            new_layer.keys[:, :, :used] = old_layer.keys[:, :, :used]
            new_layer.values[:, :, :used] = old_layer.values[:, :, :used]
            new_layer.cumulative_length.copy_(old_layer.cumulative_length)
        return new

    def _iter_frames_compiled(text_ids, max_frames, generator=None):
        # Yields one hidden state [1, 32768] per generated frame, so windows can be decoded mid-generation.
        import diffusers.modular_pipelines.minimax_music3.encoders as P

        prompt_len = text_ids.shape[1]
        bucket = 1024
        while bucket < prompt_len + 16:
            bucket *= 2
        cache = _new_cache(bucket)
        embeds = _lm.model.embed_tokens(text_ids)
        output = _lm.model(
            inputs_embeds=embeds,
            past_key_values=cache,
            cache_position=torch.arange(prompt_len, device="cuda"),
            use_cache=True,
        )
        last_hidden = output.last_hidden_state[:, -1]

        vocab_mask = torch.ones(_lm.config.vocab_size, dtype=torch.bool, device="cuda")
        vocab_mask[P._AUDIO_CODE_OFFSET : P._AUDIO_CODE_OFFSET + P._SEMANTIC_VOCAB_SIZE] = False
        vocab_mask[P._AUDIO_END_TOKEN_ID] = False

        emitted = 0
        cache_position = torch.tensor([prompt_len], device="cuda")
        for frame_index in range(max_frames + 1):
            if int(cache_position.item()) + 2 >= bucket:
                bucket *= 2
                cache = _grow_cache(cache, bucket)
            logits = _lm.lm_head(last_hidden).float()
            logits = logits.masked_fill(vocab_mask, -float("inf"))
            conditional, unconditional = logits[0:1], logits[1:2]
            guided = unconditional + (conditional - unconditional) * P._AR_CFG_SCALE
            threshold = torch.topk(conditional, P._AR_CFG_TOP_K, dim=-1).values[..., -1, None]
            guided = guided.masked_fill(conditional < threshold, -float("inf"))
            guided = guided.masked_fill(vocab_mask.unsqueeze(0), -float("inf"))
            sampled = P._sample_top_k(guided, generator)
            if int(sampled.item()) == P._AUDIO_END_TOKEN_ID:
                break
            semantic_code = (sampled - P._AUDIO_CODE_OFFSET).repeat(2)
            frame_codes, depth_hidden = P._generate_depth_codes(PIPE, last_hidden, semantic_code, generator)
            if frame_index > 0:
                emitted += 1
                yield torch.cat((last_hidden[:1].clone(), depth_hidden), dim=-1)
                if emitted >= max_frames:
                    break
            feedback = P._embed_audio_frame(PIPE, frame_codes)
            last_hidden = _compiled_lm_step(feedback, cache_position, cache).clone()
            cache_position = cache_position + 1

    def _generate_frames_compiled(text_ids, max_frames, generator=None):
        frame_hiddens = list(_iter_frames_compiled(text_ids, max_frames, generator))
        if not frame_hiddens:
            raise gr.Error("The model generated zero audio frames — try different lyrics or a longer duration.")
        return torch.stack(frame_hiddens, dim=1)

    PIPE.generate_frames = _generate_frames_compiled
    PIPE._iter_frames = _iter_frames_compiled

    # Each distinct bucket size compiles once per process; keep every specialization cached.
    torch._dynamo.config.cache_size_limit = 16

    # Pre-warm the common cache buckets at startup so users never hit a compile pause (each bucket size is one
    # dynamo specialization). The default covers songs up to ~80s; longer buckets compile on first use.
    @torch.inference_mode()
    def _warm_bucket(bucket):
        print(f"[warmup] compiling decode step for cache bucket {bucket}...", flush=True)
        cache = StaticCache(config=_lm.config, max_batch_size=2, max_cache_len=bucket, device="cuda", dtype=_lm.dtype)
        embeds = torch.zeros(2, 8, _lm.config.hidden_size, device="cuda", dtype=_lm.dtype)
        _lm.model(inputs_embeds=embeds, past_key_values=cache, cache_position=torch.arange(8, device="cuda"), use_cache=True)
        _compiled_lm_step(embeds[:, :1], torch.tensor([8], device="cuda"), cache)

    # The full ladder covers every slider duration (300s -> 7574 slots -> bucket 8192).
    for bucket in [int(b) for b in os.environ.get("WARM_BUCKETS", "1024,2048,4096,8192").split(",") if b]:
        _warm_bucket(bucket)
    # One short end-to-end generation covers the remaining one-time CUDA/cuDNN/SDPA initialization in the
    # flow-matching and vocoder stages.
    print("[warmup] end-to-end pass...", flush=True)
    PIPE(
        prompt="a short warm-up jingle",
        lyrics="[instrumental]",
        audio_duration=4.0,
        num_inference_steps=30,
        generator=torch.Generator("cuda").manual_seed(0),
    )
    print("[warmup] done", flush=True)


_CHUNK, _HOP, _HOP_SAMPLES = 200, 100, 86 * 512
_CROP_RIGHT_SAMPLES = (344 - 86) * 512


@torch.inference_mode()
def _decode_window(hidden_window, previous, generator, steps, guidance):
    previous_latent, previous_condition = previous
    condition = PIPE.condition_encoder(hidden_window)
    condition = condition.to(PIPE.transformer.dtype)
    latents = randn_like_seeded = torch.randn(
        (1, PIPE.transformer.config.in_channels, condition.shape[1]),
        generator=generator, device="cuda", dtype=condition.dtype,
    )
    overlap, noise_prompt = 0, None
    if previous_latent is not None:
        overlap = min(previous_latent.shape[-1], latents.shape[-1])
        noise_prompt = latents[..., :overlap].clone()
        condition[:, :overlap] = previous_condition[:, :overlap]
    condition_input = torch.cat((condition, torch.zeros_like(condition)), dim=0)
    PIPE.scheduler.set_timesteps(sigmas=np.linspace(1.0, 1.0 / steps, steps), device="cuda")
    for timestep in PIPE.scheduler.timesteps:
        if overlap > 0:
            t = timestep.to(latents.dtype)
            latents[..., :overlap] = (1.0 - (1.0 - 1e-6) * t) * noise_prompt + t * previous_latent[..., :overlap]
        velocity = PIPE.transformer(
            latents.expand(2, -1, -1).contiguous(), timestep.expand(2).to(latents.dtype), condition_input
        ).sample
        velocity = velocity[1:2] + guidance * (velocity[0:1] - velocity[1:2])
        latents = PIPE.scheduler.step(velocity, timestep, latents).prev_sample
    if overlap > 0:
        latents[..., :overlap] = previous_latent[..., :overlap]
    overlap_start = max(0, latents.shape[-1] - 2 * 172)
    overlap_end = max(overlap_start, latents.shape[-1] - 172)
    carry = (latents[..., overlap_start:overlap_end], condition[:, overlap_start:overlap_end])
    waveform = PIPE.vocoder(latents.to(PIPE.vocoder.dtype)).float().clamp(-1.0, 1.0)[0]
    return waveform, carry


def _to_int16(waveform):
    return (waveform.cpu().numpy().T * 32767.0).astype(np.int16)


def _pcm_msg(wave_int16, sr, seq, gen, off):
    # One streamed-player message: base64 of interleaved int16 stereo PCM with the chunk's absolute
    # sample offset. The custom gr.HTML player replaces the streaming gr.Audio (its HLS path never
    # re-attaches after the first stream and can't autoplay reliably), plays these gaplessly via
    # Web Audio, and stays lossless. Gradio's frontend coalesces rapid per-component updates (only
    # the newest survives a flush), so a chunk can be dropped: offsets keep the timeline correct,
    # and the final "done" message carries the finished wav's URL so the player re-fetches the
    # complete file whenever anything is missing.
    import base64

    return {"cmd": "chunk", "sr": int(sr), "ch": 2, "seq": int(seq), "gen": gen, "off": int(off),
            "pcm": base64.b64encode(np.ascontiguousarray(wave_int16).tobytes()).decode()}


_SONGS_DIR = "/tmp/mm3_songs"
os.makedirs(_SONGS_DIR, exist_ok=True)
os.environ.setdefault("GRADIO_ALLOWED_PATHS", f"{_SONGS_DIR},{os.path.abspath('examples')}")


def _file_url(path):
    return "/gradio_api/file=" + os.path.abspath(path)


@torch.inference_mode()
def _stream_windows(text_ids, max_frames, ar_generator, dit_generator, steps, guidance):
    frames = []
    windows_done = 0
    carry = (None, None)
    for hidden in PIPE._iter_frames(text_ids, max_frames, ar_generator):
        frames.append(hidden)
        window_start = windows_done * _HOP
        if len(frames) > window_start + _CHUNK:
            window = torch.stack(frames[window_start : window_start + _CHUNK], dim=1)
            waveform, carry = _decode_window(window, carry, dit_generator, steps, guidance)
            left = 0 if windows_done == 0 else _HOP_SAMPLES
            windows_done += 1
            yield waveform[:, left : waveform.shape[-1] - _CROP_RIGHT_SAMPLES]
    if not frames:
        raise gr.Error("The model generated zero audio frames — try different lyrics or a longer duration.")
    total = len(frames)
    window_starts = [0] if total <= _CHUNK else list(range(0, total - _HOP, _HOP))
    for w in range(windows_done, len(window_starts)):
        window_start = window_starts[w]
        window = torch.stack(frames[window_start : min(window_start + _CHUNK, total)], dim=1)
        waveform, carry = _decode_window(window, carry, dit_generator, steps, guidance)
        left = 0 if w == 0 else _HOP_SAMPLES
        right = _CROP_RIGHT_SAMPLES if w < len(window_starts) - 1 else 0
        yield waveform[:, left : waveform.shape[-1] - right]


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

_CAPTION_CONTRACT = """The three caption fields follow the exact labeled style the model was trained on. Be concrete and musical; describe an energy arc and instrument lifecycles, never a static equipment list or decorative adjectives. Never contradict an explicit user constraint: instrumental stays instrumental; never reverse a required vocal gender, tempo limit, required instrument, or exclusion. Do not quote or paraphrase lyric lines inside the caption. Total caption length roughly 250-400 words.

global_metadata: one paragraph, in order: "Basic Attributes: bpm is <number>. key is <letter>, and scale is <major|minor>. <Genre / Subgenre>." then "Global Emotional Progression: <how the emotion evolves from the opening through the final section>." then "Application Scenarios & Imagery: <two or three vivid listening scenarios>." then "Sonics & Production Profile: <soundstage, frequency balance, dynamics, production character>."

vocal_details: one paragraph: "Vocal Gender & Timbre: Singer A (<Male|Female>), <timbre and register>." then "Vocal Style: <delivery, and how it shifts per section>." then "Harmony/Backing Vocals: <where harmonies or doubles appear and their character>." then "Vocal FX: <restrained treatment: reverb, delay, light compression>." For instrumental pieces write "Instrumental, no vocals." and name the instrument or texture carrying the lead melodic role.

arrangement: one paragraph: "Instrument Lifecycle Description (Primary/Secondary Layering): Primary: <core instruments present start to finish and their role>. Secondary: <instruments that enter, exit or intensify, and in which sections>." then "Groove & Foundation Progression: <how drums, bass and groove develop across sections>." then "Embellishments, Textures & Spatial FX: <fills, textures, transitional gestures, stereo and space treatment where relevant>." State what enters, exits, changes or intensifies for every section of the song, aligned with the lyric section tags."""

_LYRICS_RULES = """lyrics: singable lyrics using ONLY these section tags, each ALWAYS ALONE on its own line: [intro] [verse] [pre-chorus] [chorus] [post-chorus] [bridge] [instrumental] [solo] [outro]. Never put words on the same line as a tag. Size the structure to the duration: <=30s: one verse + one chorus; ~60s: verse/pre-chorus/chorus/verse/chorus; >=120s: full structure with bridge and outro. Roughly 12-16 sung words per 10 seconds. Musical instructions (tempo, instruments, dynamics) never belong in the lyrics. If the song is instrumental, use [instrumental] sections with no words."""

_COMPOSER_SYSTEM = f"""You write inputs for MiniMax Music 3, a lyrics+description music generation model.
Given a song description and a target duration, produce:
1. {_LYRICS_RULES}
2-4. global_metadata, vocal_details, arrangement — a structured caption. {_CAPTION_CONTRACT}
Answer with ONLY a JSON object with keys: lyrics, global_metadata, vocal_details, arrangement."""

_LYRICS_SYSTEM = f"""You write lyrics for MiniMax Music 3, a lyrics+description music generation model.
Given a lyrics instruction, the current structured prompt (global metadata, vocal details, arrangement) and a target duration, write lyrics coherent with that structured prompt.
{_LYRICS_RULES}
Answer with ONLY a JSON object with key: lyrics."""

_PROMPT_SYSTEM = f"""You write the structured caption for MiniMax Music 3, a lyrics+description music generation model.
Given a sound instruction and/or lyrics, produce global_metadata, vocal_details and arrangement. Build the arrangement timeline around the lyric section tags when lyrics are provided. {_CAPTION_CONTRACT}
Answer with ONLY a JSON object with keys: global_metadata, vocal_details, arrangement."""


def _llm_json(system, user, required=()):
    import json as _json

    from openai import OpenAI

    # Bounded timeout: a hung provider must fail over, not freeze the UI at the composing step.
    client = OpenAI(base_url="https://router.huggingface.co/v1", api_key=os.environ["HF_TOKEN"], max_retries=0)
    last_error = None
    # Three DISTINCT providers, all verified enabled for this account (bare/":fastest" can route to
    # together, which 403s here and killed the fallbacks). Timeouts sized to measured composer latency.
    # Two passes over the chain: under load every provider can 429 transiently, and a second pass a few
    # seconds later usually lands.
    for attempt in range(2):
        for model, timeout in (
            ("deepseek-ai/DeepSeek-V4-Flash-0731:baseten", 45),
            ("deepseek-ai/DeepSeek-V4-Flash-0731:deepinfra", 75),
            ("deepseek-ai/DeepSeek-V4-Flash-0731:novita", 100),
        ):
            try:
                completion = client.with_options(timeout=timeout).chat.completions.create(
                    model=model,
                    messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                )
                text = completion.choices[0].message.content or ""
                # Tolerate fences/preambles and reject truncated replies: parse the outermost {...} span.
                start, end = text.find("{"), text.rfind("}")
                if start == -1 or end <= start:
                    raise ValueError(f"no JSON object in composer reply (finish_reason={completion.choices[0].finish_reason})")
                data = _json.loads(text[start : end + 1])
                # Valid JSON with the wrong shape must retry too, not KeyError later.
                missing = [key for key in required if key not in data]
                if missing:
                    raise ValueError(f"composer reply missing keys: {missing}")
                return data
            except Exception as e:
                print(f"composer attempt failed ({model}, pass {attempt + 1}): {type(e).__name__}: {e}", flush=True)
                last_error = e
        if attempt == 0:
            time.sleep(3)
    raise gr.Error(
        "The composer model is overloaded right now — try again in a moment, "
        "or write the lyrics and structured prompt directly in the Studio tab."
    ) from last_error


def compose_song(description, duration):
    if not description.strip():
        raise gr.Error("Describe the song you want first.")
    data = _llm_json(
        _COMPOSER_SYSTEM,
        f"Song description: {description}\nTarget duration: {int(duration)} seconds.",
        required=("lyrics", "global_metadata", "vocal_details", "arrangement"),
    )
    return data["lyrics"], data["global_metadata"], data["vocal_details"], data["arrangement"]


# ---------------------------------------------------------------------------
# Custom streaming player (gr.HTML). Replaces the streaming gr.Audio: in Gradio 6 the
# streaming Audio output rides HLS and AudioPlayer.svelte's load_stream() never re-attaches
# after the first stream (`stream_active` is only cleared on the non-stream path), so a 2nd
# generation glitches; autoplay also fires outside a user gesture so browsers block it.
# This player receives base64 int16 PCM messages ({cmd: reset|chunk|done}) as generator
# yields, schedules them gaplessly with Web Audio, and is armed for autoplay from the
# Generate click (a real gesture) via window.__mmArmAudio.
# ---------------------------------------------------------------------------

_PLAYER_HTML = """
<div class="pl-wrap">
  <div class="pl-head">
    <span class="pl-label">&#9835; Your song</span>
    <span class="pl-headright">
      <span class="pl-live" data-role="live" hidden><span class="pl-dot"></span>streaming</span>
      <button type="button" class="pl-stop" data-role="stopgen" hidden title="Stop generating — keeps what was already streamed">
        <svg viewBox="0 0 24 24"><rect x="7" y="7" width="10" height="10" rx="1.5"/></svg>Stop
      </button>
    </span>
  </div>
  <div class="pl-loader" data-role="loader" hidden>
    <div class="pl-loader-row"><span class="pl-dot"></span><span data-role="loader-text">Starting&#8230;</span></div>
    <div class="pl-bar"><div class="pl-bar-fill"></div></div>
  </div>
  <div class="pl-empty" data-role="empty">
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><path d="M9 18V5l12-2v13"/><circle cx="6" cy="18" r="3"/><circle cx="18" cy="16" r="3"/></svg>
  </div>
  <div class="pl-body" data-role="body" hidden>
    <button type="button" class="pl-btn" data-role="play" aria-label="Play / pause">
      <svg viewBox="0 0 24 24" data-role="ic-play"><path d="M8 5v14l11-7z"/></svg>
      <svg viewBox="0 0 24 24" data-role="ic-pause" style="display:none"><path d="M6 5h4v14H6zM14 5h4v14h-4z"/></svg>
    </button>
    <span class="pl-time" data-role="time">0:00</span>
    <canvas class="pl-wave" data-role="wave"></canvas>
    <span class="pl-time" data-role="dur">0:00</span>
    <button type="button" class="pl-btn pl-sm" data-role="mute" aria-label="Mute / unmute">
      <svg viewBox="0 0 24 24" data-role="ic-vol"><path d="M3 9v6h4l5 5V4L7 9H3zm13.5 3a4.5 4.5 0 0 0-2.5-4v8a4.5 4.5 0 0 0 2.5-4zM14 3.2v2.1a7 7 0 0 1 0 13.4v2.1a9 9 0 0 0 0-17.6z"/></svg>
      <svg viewBox="0 0 24 24" data-role="ic-mute" style="display:none"><path d="M3 9v6h4l5 5V4L7 9H3zm13.6 3 2.7-2.7-1.4-1.4-2.7 2.7-2.7-2.7-1.4 1.4 2.7 2.7-2.7 2.7 1.4 1.4 2.7-2.7 2.7 2.7 1.4-1.4-2.7-2.7z"/></svg>
    </button>
    <a class="pl-btn pl-sm" data-role="dl" download="minimax-music3.wav" aria-label="Download wav" hidden>
      <svg viewBox="0 0 24 24"><path d="M12 3v10.6l-3.3-3.3-1.4 1.4L12 17.4l4.7-4.7-1.4-1.4-3.3 3.3V3h-2zM5 19h14v2H5z"/></svg>
    </a>
  </div>
</div>
"""

_PLAYER_CSS = """
.pl-wrap { background: var(--block-background-fill); border: var(--block-border-width, 1px) solid var(--block-border-color, var(--border-color-primary)); border-radius: var(--block-radius, 12px); box-shadow: var(--block-shadow, none); padding: 10px 14px; display: flex; flex-direction: column; gap: 7px; }
.pl-head { display: flex; align-items: center; justify-content: space-between; }
.pl-label { color: var(--block-title-text-color, var(--body-text-color)); font-size: var(--block-title-text-size, 13px); font-weight: var(--block-title-text-weight, 600); }
.pl-headright { display: inline-flex; align-items: center; gap: 10px; }
.pl-stop { display: inline-flex; align-items: center; gap: 5px; background: transparent; color: var(--body-text-color-subdued); border: 1px solid var(--border-color-primary); border-radius: 999px; padding: 3px 11px; font-family: inherit; font-size: 11.5px; font-weight: 600; cursor: pointer; box-shadow: none; transition: color .15s, border-color .15s; }
.pl-stop svg { width: 11px; height: 11px; fill: currentColor; }
.pl-stop:hover { color: var(--error-text-color, #d64545); border-color: var(--error-border-color, #d64545); }
.pl-stop[hidden] { display: none; }
.pl-live { display: inline-flex; align-items: center; gap: 6px; color: var(--color-accent); font-size: 11.5px; font-weight: 600; }
.pl-live[hidden] { display: none; }
.pl-dot { width: 8px; height: 8px; border-radius: 50%; background: var(--color-accent); animation: pl-pulse 1.1s ease-in-out infinite; }
@keyframes pl-pulse { 0%, 100% { opacity: .25; transform: scale(.8); } 50% { opacity: 1; transform: scale(1.1); } }
.pl-loader { display: flex; flex-direction: column; gap: 7px; padding: 6px 0 4px; }
.pl-loader[hidden] { display: none; }
.pl-loader-row { display: flex; align-items: center; gap: 8px; color: var(--body-text-color); font-size: 12.5px; }
.pl-bar { height: 3px; border-radius: 999px; background: var(--background-fill-secondary); overflow: hidden; }
.pl-bar-fill { width: 35%; height: 100%; border-radius: 999px; background: var(--button-primary-background-fill, var(--color-accent)); animation: pl-slide 1.3s cubic-bezier(.45, .1, .55, .9) infinite; }
@keyframes pl-slide { 0% { transform: translateX(-110%); } 100% { transform: translateX(400%); } }
.pl-empty { display: flex; align-items: center; justify-content: center; padding: 18px 0; color: var(--body-text-color-subdued); }
.pl-empty svg { width: 42px; height: 42px; opacity: .45; }
.pl-empty[hidden] { display: none; }
.pl-body { display: flex; align-items: center; gap: 9px; }
.pl-body[hidden] { display: none; }
.pl-btn { width: 34px; height: 34px; flex: none; border-radius: 50%; border: 1px solid var(--border-color-primary); background: var(--background-fill-secondary); color: var(--body-text-color); display: flex; align-items: center; justify-content: center; cursor: pointer; padding: 0; box-shadow: none; transition: color .15s, border-color .15s; }
.pl-btn svg { width: 16px; height: 16px; fill: currentColor; }
.pl-btn:hover { border-color: var(--color-accent); color: var(--color-accent); }
.pl-sm { width: 28px; height: 28px; }
.pl-sm svg { width: 13px; height: 13px; }
.pl-time { font-family: var(--font-mono, ui-monospace, monospace); font-size: 11.5px; color: var(--body-text-color-subdued); flex: none; min-width: 36px; text-align: center; }
.pl-wave { flex: 1; height: 52px; min-width: 60px; cursor: pointer; }
"""

_PLAYER_JS = """
const $ = function(s) { return element.querySelector(s); };
const playBtn = $('[data-role="play"]'), icPlay = $('[data-role="ic-play"]'), icPause = $('[data-role="ic-pause"]');
const muteBtn = $('[data-role="mute"]'), icVol = $('[data-role="ic-vol"]'), icMute = $('[data-role="ic-mute"]');
const dlLink = $('[data-role="dl"]'), liveEl = $('[data-role="live"]'), emptyEl = $('[data-role="empty"]');
const bodyEl = $('[data-role="body"]'), timeEl = $('[data-role="time"]'), durEl = $('[data-role="dur"]');
const canvas = $('[data-role="wave"]');
const loaderEl = $('[data-role="loader"]'), loaderText = $('[data-role="loader-text"]');
const stopBtn = $('[data-role="stopgen"]');
function setLoader(text) {
  if (text) { loaderText.textContent = text; loaderEl.hidden = false; emptyEl.hidden = true; }
  else {
    loaderEl.hidden = true;
    if (totalFrames === 0) { emptyEl.hidden = false; bodyEl.hidden = true; }
  }
}
const cx2d = canvas.getContext('2d');

function show(el, on) { el.style.display = on ? '' : 'none'; }
let ctx = null, gain = null, autoplayPending = false, userStopped = false;
let sr = 44100, chunks = [], totalFrames = 0, lastSeq = 0, curGen = null;
let sources = [], baseTime = 0, pausedAt = 0;
let playing = false, muted = false, streamingNow = false, doneFlag = false;
let peaks = [], peakFrames = 0, PEAK_STEP = 5512;

function ensureCtx() {
  if (!ctx) {
    ctx = new (window.AudioContext || window.webkitAudioContext)();
    gain = ctx.createGain();
    gain.connect(ctx.destination);
  }
  if (ctx.state === 'suspended') ctx.resume();
}
window.__mmArmAudio = function() { try { ensureCtx(); } catch (e) {} };

function fmt(t) { t = Math.max(0, t); const m = Math.floor(t / 60), s = Math.floor(t % 60); return m + ':' + (s < 10 ? '0' : '') + s; }
function bufferedDur() { return totalFrames / sr; }
function pos() {
  if (!playing || !ctx) return pausedAt;
  return Math.min(ctx.currentTime - baseTime, bufferedDur());
}
function stopSources() { sources.forEach(function(s) { try { s.stop(); } catch (e) {} }); sources = []; }

function makeBuffer(c) {
  const b = ctx.createBuffer(2, c.frames, sr);
  b.getChannelData(0).set(c.l);
  b.getChannelData(1).set(c.r);
  return b;
}
function scheduleChunk(c) {
  const t0 = baseTime + c.start / sr, now = ctx.currentTime;
  const src = ctx.createBufferSource();
  src.buffer = makeBuffer(c);
  src.connect(gain);
  if (t0 >= now) src.start(t0);
  else if (now - t0 < c.frames / sr) src.start(now, now - t0);
  else return;
  sources.push(src);
}
function playFrom(t) {
  document.querySelectorAll('video').forEach(function(v) { try { v.pause(); } catch (e) {} });
  ensureCtx();
  stopSources();
  t = Math.max(0, Math.min(t, bufferedDur()));
  baseTime = ctx.currentTime - t;
  chunks.forEach(scheduleChunk);
  playing = true; autoplayPending = false;
  show(icPlay, false); show(icPause, true);
}
function pause() {
  pausedAt = pos();
  stopSources();
  playing = false;
  show(icPlay, true); show(icPause, false);
}

function addPeaks(c) {
  const mono = c.l, n = c.frames;
  let i = peakFrames % PEAK_STEP === 0 ? 0 : PEAK_STEP - (peakFrames % PEAK_STEP);
  for (; i < n; i += PEAK_STEP) {
    let m = 0;
    const end = Math.min(i + PEAK_STEP, n);
    for (let j = i; j < end; j += 16) { const a = Math.abs(mono[j]); if (a > m) m = a; }
    peaks.push(m);
  }
  peakFrames = totalFrames;
}

function addPcm(i16, srIn, ch, off) {
  sr = srIn || sr;
  ch = ch || 2;
  const frames = Math.floor(i16.length / ch);
  if (frames < 1) return;
  const l = new Float32Array(frames), r = new Float32Array(frames);
  for (let f = 0; f < frames; f++) {
    l[f] = i16[f * ch] / 32768;
    r[f] = i16[f * ch + (ch > 1 ? 1 : 0)] / 32768;
  }
  const startFrame = typeof off === 'number' ? off : totalFrames;
  const c = { l: l, r: r, i16: i16, frames: frames, start: startFrame };
  chunks.push(c);
  totalFrames = Math.max(totalFrames, startFrame + frames);
  addPeaks(c);
  setLoader(null);
  if (streamingNow) liveEl.hidden = false;
  emptyEl.hidden = true; bodyEl.hidden = false;
  if (playing) {
    if (ctx.currentTime - baseTime > c.start / sr + 0.05) playFrom(c.start / sr); // underrun at live edge: rebase
    else scheduleChunk(c);
  } else if (autoplayPending && ctx && ctx.state === 'running' && !doneFlag) {
    playFrom(0); // armed by the Generate gesture -> reliable autoplay
  }
}

function addChunk(msg) {
  if (msg.seq && msg.seq <= lastSeq) return;
  lastSeq = msg.seq || lastSeq + 1;
  const bytes = Uint8Array.from(atob(msg.pcm), function(c) { return c.charCodeAt(0); });
  addPcm(new Int16Array(bytes.buffer), msg.sr, msg.ch || 2, msg.off);
}

let loadToken = 0;
async function streamWav(url, srHint, chHint) {
  // Progressive PCM streaming of a cached wav: walk the RIFF chunks to the data section,
  // then feed interleaved int16 frames into the player as they arrive off the network.
  const myToken = loadToken;
  const resp = await fetch(url);
  if (!resp.ok) throw new Error('fetch ' + resp.status);
  const reader = resp.body.getReader();
  let pending = new Uint8Array(0), headerParsed = false;
  let wsr = srHint || 44100, wch = chHint || 2, dataRemaining = Infinity;
  const concat = function(a, b) { const o = new Uint8Array(a.length + b.length); o.set(a); o.set(b, a.length); return o; };
  while (true) {
    const step = await reader.read();
    if (loadToken !== myToken) { try { reader.cancel(); } catch (e) {} return false; }
    if (step.value && step.value.length) pending = concat(pending, step.value);
    if (!headerParsed && pending.length >= 12) {
      const dv = new DataView(pending.buffer, pending.byteOffset, pending.byteLength);
      let pos = 12, found = false;
      while (pos + 8 <= pending.length) {
        const id = String.fromCharCode(pending[pos], pending[pos + 1], pending[pos + 2], pending[pos + 3]);
        const size = dv.getUint32(pos + 4, true);
        if (id === 'fmt ' && pos + 16 <= pending.length) { wch = dv.getUint16(pos + 10, true) || wch; wsr = dv.getUint32(pos + 12, true) || wsr; }
        if (id === 'data') { dataRemaining = size; pos += 8; found = true; break; }
        pos += 8 + size + (size % 2);
      }
      if (found) { pending = pending.slice(pos); headerParsed = true; }
    }
    if (headerParsed) {
      const frameBytes = wch * 2;
      const threshold = totalFrames === 0 ? Math.floor(wsr / 8) * frameBytes : Math.floor(wsr / 2) * frameBytes;
      let usable = Math.min(pending.length, dataRemaining);
      usable -= usable % frameBytes;
      if (usable > 0 && (usable >= threshold || step.done)) {
        const bytes = pending.slice(0, usable);
        addPcm(new Int16Array(bytes.buffer), wsr, wch);
        pending = pending.slice(usable);
        dataRemaining -= usable;
      }
    }
    if (step.done) break;
  }
  return loadToken === myToken;
}

async function repair(url, frames) {
  // A coalesced flush can swallow a chunk message; the done message carries the finished wav's
  // URL, so whenever anything is missing the full lossless file is fetched and swapped in.
  try {
    const resp = await fetch(url);
    if (!resp.ok) throw new Error('fetch ' + resp.status);
    const ab = await resp.arrayBuffer();
    if (!ctx) {
      ctx = new (window.AudioContext || window.webkitAudioContext)();
      gain = ctx.createGain(); gain.connect(ctx.destination);
      if (muted) gain.gain.value = 0;
    }
    const buf = await ctx.decodeAudioData(ab);
    const wasPos = pos(), wasPlaying = playing;
    stopSources();
    sr = buf.sampleRate;
    const L = buf.getChannelData(0), R = buf.numberOfChannels > 1 ? buf.getChannelData(1) : L;
    const n = buf.length;
    const i16 = new Int16Array(n * 2);
    for (let f = 0; f < n; f++) {
      i16[2 * f] = Math.max(-32768, Math.min(32767, Math.round(L[f] * 32767)));
      i16[2 * f + 1] = Math.max(-32768, Math.min(32767, Math.round(R[f] * 32767)));
    }
    chunks = [{ l: Float32Array.from(L), r: Float32Array.from(R), i16: i16, frames: n, start: 0 }];
    totalFrames = n;
    peaks = []; peakFrames = 0; addPeaks(chunks[0]);
    emptyEl.hidden = true; bodyEl.hidden = false;
    if (wasPlaying) playFrom(Math.min(wasPos, n / sr));
    else if (autoplayPending && ctx.state === 'running') playFrom(0);
  } catch (e) { console.error('player repair failed:', e); }
}

function reset() {
  loadToken++;
  stopSources();
  playing = false; doneFlag = false; pausedAt = 0; lastSeq = 0; curGen = null; autoplayPending = true;
  chunks = []; totalFrames = 0; peaks = []; peakFrames = 0;
  show(icPlay, true); show(icPause, false);
  dlLink.hidden = true;
  if (dlLink.href) { try { URL.revokeObjectURL(dlLink.href); } catch (e) {} dlLink.removeAttribute('href'); }
  streamingNow = true; userStopped = false;
  liveEl.hidden = true;
  stopBtn.hidden = false;
  bodyEl.hidden = true; emptyEl.hidden = false;
}

function makeWavBlob() {
  const dataLen = totalFrames * 4;
  const buf = new ArrayBuffer(44 + dataLen);
  const v = new DataView(buf);
  function ws(o, s) { for (let i = 0; i < s.length; i++) v.setUint8(o + i, s.charCodeAt(i)); }
  ws(0, 'RIFF'); v.setUint32(4, 36 + dataLen, true); ws(8, 'WAVE'); ws(12, 'fmt ');
  v.setUint32(16, 16, true); v.setUint16(20, 1, true); v.setUint16(22, 2, true);
  v.setUint32(24, sr, true); v.setUint32(28, sr * 4, true); v.setUint16(32, 4, true); v.setUint16(34, 16, true);
  ws(36, 'data'); v.setUint32(40, dataLen, true);
  let o = 44;
  chunks.forEach(function(c) {
    for (let f = 0; f < c.frames; f++) {
      v.setInt16(o, c.i16[f * 2], true); v.setInt16(o + 2, c.i16[f * 2 + 1], true); o += 4;
    }
  });
  return new Blob([buf], { type: 'audio/wav' });
}

async function finish(v) {
  streamingNow = false; doneFlag = true;
  liveEl.hidden = true;
  stopBtn.hidden = true;
  setLoader(null);
  const missing = v && v.url && (!totalFrames || (v.frames && totalFrames < v.frames) || chunks.length !== lastSeq);
  if (missing) await repair(v.url, v.frames);
  if (totalFrames > 0) { dlLink.href = URL.createObjectURL(makeWavBlob()); dlLink.hidden = false; }
}

function draw() {
  const w = canvas.clientWidth, h = canvas.clientHeight;
  if (w > 0 && (canvas.width !== w * devicePixelRatio || canvas.height !== h * devicePixelRatio)) {
    canvas.width = w * devicePixelRatio; canvas.height = h * devicePixelRatio;
  }
  cx2d.setTransform(devicePixelRatio, 0, 0, devicePixelRatio, 0, 0);
  cx2d.clearRect(0, 0, w, h);
  const style = getComputedStyle(element);
  const accent = style.getPropertyValue('--color-accent').trim() || 'darkorange';
  const dim = style.getPropertyValue('--border-color-primary').trim() || '#666';
  const bars = Math.max(1, Math.floor(w / 3));
  const frac = bufferedDur() > 0 ? pos() / bufferedDur() : 0;
  for (let b = 0; b < bars; b++) {
    const p0 = Math.floor(b * peaks.length / bars), p1 = Math.max(p0 + 1, Math.floor((b + 1) * peaks.length / bars));
    let m = 0;
    for (let p = p0; p < p1 && p < peaks.length; p++) if (peaks[p] > m) m = peaks[p];
    const bh = Math.max(2, m * (h - 6));
    cx2d.fillStyle = (b / bars) <= frac ? accent : dim;
    cx2d.fillRect(b * 3, (h - bh) / 2, 2, bh);
  }
  if (playing) {
    timeEl.textContent = fmt(pos());
    if (playing && pos() >= bufferedDur() && doneFlag) pause();
  } else {
    timeEl.textContent = fmt(pausedAt);
  }
  durEl.textContent = fmt(bufferedDur());
  requestAnimationFrame(draw);
}
requestAnimationFrame(draw);

stopBtn.addEventListener('click', function() {
  userStopped = true;
  trigger('stop');
  streamingNow = false; doneFlag = true;
  liveEl.hidden = true; stopBtn.hidden = true;
  setLoader(null);
  if (totalFrames > 0) { dlLink.href = URL.createObjectURL(makeWavBlob()); dlLink.hidden = false; }
});
playBtn.addEventListener('click', function() {
  if (playing) pause();
  else {
    if (doneFlag && pausedAt >= bufferedDur() - 0.05) pausedAt = 0;
    playFrom(pausedAt);
  }
});
muteBtn.addEventListener('click', function() {
  muted = !muted;
  if (gain) gain.gain.value = muted ? 0 : 1;
  show(icVol, !muted); show(icMute, muted);
});
canvas.addEventListener('click', function(e) {
  if (!totalFrames) return;
  const rect = canvas.getBoundingClientRect();
  const t = ((e.clientX - rect.left) / rect.width) * bufferedDur();
  if (playing) playFrom(t); else { pausedAt = t; }
});

// The watch effect coalesces rapid value updates (only the newest survives a flush), so
// messages are processed through an ordered async queue and any missed seq range is pulled
// back from the server buffer (server.fetch_chunks). Inline PCM is the fast path.
async function handleMsg(v) {
  if (v.cmd === 'reset') {
    reset();
    setLoader(v.loader ? (v.status || 'Starting...') : null);
    if (!v.loader) stopBtn.hidden = true;
    return;
  }
  if (userStopped && v.cmd !== 'done') return;
  if (v.cmd === 'status') { if (!doneFlag || streamingNow) setLoader(v.text); return; }
  if (v.cmd === 'load') {
    streamingNow = false;
    liveEl.hidden = true; stopBtn.hidden = true;
    curGen = v.gen || curGen;
    // fire-and-forget: awaiting here would block the queue, and a later reset (which cancels
    // this stream via loadToken) could never run
    streamWav(v.url, v.sr, v.ch).then(function(complete) {
      if (complete) {
        doneFlag = true;
        if (totalFrames > 0) { dlLink.href = URL.createObjectURL(makeWavBlob()); dlLink.hidden = false; }
      }
    }).catch(function(e) { console.error('player load failed:', e); });
    return;
  }
  if (v.cmd === 'chunk') {
    if (curGen && v.gen && v.gen !== curGen) reset();
    curGen = v.gen || curGen;
    addChunk(v);
  } else if (v.cmd === 'done') {
    if (v.gen && curGen && v.gen !== curGen) reset();
    await finish(v);
  }
}
document.addEventListener('play', function(e) {
  if (e.target && e.target.tagName === 'VIDEO' && playing) pause();
}, true);
new MutationObserver(function(muts) {
  for (const m of muts) for (const n of m.addedNodes) {
    if (n.nodeType !== 1) continue;
    if ((n.matches && n.matches('.toast-body.error')) ||
        (n.querySelector && n.querySelector('.toast-body.error'))) setLoader(null);
  }
}).observe(document.body, { childList: true, subtree: true });
let msgQueue = Promise.resolve();
watch('value', function() {
  const v = props.value;
  if (!v || !v.cmd) return;
  msgQueue = msgQueue.then(function() { return handleMsg(v); }).catch(function(e) { console.error('player msg error:', e); });
});
"""


def render_video(wav_path, title):
    # Social share visualizer: warm citrus bars on a dark gradient, rendered via numpy -> ffmpeg pipe (CPU).
    if not wav_path:
        return gr.skip()
    import subprocess

    import scipy.io.wavfile

    title = " ".join((title or "").split())[:96] or "MiniMax Music 3"
    try:
        sr, wave = scipy.io.wavfile.read(wav_path)
    except (FileNotFoundError, OSError):
        return gr.skip()  # the visitor left and gradio cleaned the cached wav
    mono = wave.astype(np.float32).mean(axis=1) / 32768.0
    fps, size, bars = 24, 720, 56
    total_frames = int(len(mono) / sr * fps)
    window = int(sr / fps * 2)
    bar_w = size // (bars + 6)
    x0 = (size - bars * bar_w) // 2

    from PIL import Image, ImageDraw, ImageFont

    def _font(px):
        for path in ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
                     "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf"):
            try:
                return ImageFont.truetype(path, px)
            except OSError:
                continue
        return ImageFont.load_default(size=px)

    def _fit_title(draw, text, max_w):
        # Adaptive title sizing: shrink to fit, then wrap to two lines at the space nearest the middle.
        for px in range(30, 15, -2):
            f = _font(px)
            if draw.textlength(text, font=f) <= max_w:
                return [(text, f, 56)]
        spaces = [i for i, ch in enumerate(text) if ch == " "]
        split = min(spaces, key=lambda i: abs(i - len(text) // 2)) if spaces else len(text) // 2
        lines = [text[:split].strip(), text[split:].strip()]
        for px in range(24, 11, -2):
            f = _font(px)
            if all(draw.textlength(line, font=f) <= max_w for line in lines):
                break
        return [(lines[0], f, 40), (lines[1], f, 72)]

    # warm dark gradient with a soft vignette
    grad_y = np.linspace(0.0, 1.0, size)[:, None, None]
    bg = np.array([10.0, 10.0, 13.0]) * (1 - grad_y) + np.array([27.0, 18.0, 10.0]) * grad_y
    gx, gy = np.meshgrid(np.linspace(-1, 1, size), np.linspace(-1, 1, size))
    vignette = 1.0 - 0.38 * np.clip(np.sqrt(gx * gx + gy * gy) - 0.35, 0.0, 1.0) ** 1.5
    bg = (np.repeat(bg, size, axis=1) * vignette[..., None]).astype(np.uint8)

    overlay = Image.fromarray(bg)
    draw = ImageDraw.Draw(overlay)
    if title:
        for line, f, y in _fit_title(draw, title[:96], size - 48):
            draw.text((size // 2, y), line, fill=(240, 238, 232), anchor="mm", font=f)
    draw.text((size // 2, size - 52), "MiniMax Music 3", fill=(245, 158, 11), anchor="mm", font=_font(30))
    draw.text((size // 2, size - 24), "made with diffusers", fill=(150, 140, 124), anchor="mm", font=_font(16))
    base = np.asarray(overlay, dtype=np.uint8)

    # citrus palette across the bars: yellow -> orange -> ember
    _yellow, _orange, _ember = np.array([250.0, 204.0, 86.0]), np.array([245.0, 140.0, 32.0]), np.array([196.0, 74.0, 22.0])
    palette = []
    for b in range(bars):
        t = b / max(bars - 1, 1)
        col = _yellow + (_orange - _yellow) * (t * 2) if t < 0.5 else _orange + (_ember - _orange) * ((t - 0.5) * 2)
        palette.append(col)

    out_path = wav_path.replace(".wav", "_viz.mp4")
    ffmpeg = subprocess.Popen(
        ["ffmpeg", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{size}x{size}", "-r", str(fps),
         "-i", "pipe:", "-i", wav_path, "-c:v", "libx264", "-preset", "veryfast", "-pix_fmt", "yuv420p",
         "-c:a", "aac", "-shortest", out_path],
        stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    freqs = np.fft.rfftfreq(window, 1 / sr)
    band_edges = np.geomspace(40, 12000, bars + 1)
    smooth = np.zeros(bars)
    mid = size // 2 - 30
    prog_y, prog_xa, prog_xb = size - 92, int(size * 0.1), int(size * 0.9)
    for i in range(total_frames):
        start = int(i * sr / fps)
        chunk = mono[start : start + window]
        if len(chunk) < window:
            chunk = np.pad(chunk, (0, window - len(chunk)))
        spectrum = np.abs(np.fft.rfft(chunk * np.hanning(window)))
        levels = np.array([
            spectrum[m].mean() if (m := (freqs >= band_edges[b]) & (freqs < band_edges[b + 1])).any() else 0.0
            for b in range(bars)
        ])
        levels = np.log1p(12 * np.nan_to_num(levels))
        smooth = np.maximum(levels, smooth * 0.85)
        frame = base.copy()
        for b in range(bars):
            rel = min(smooth[b] / 4.5, 1.0)
            h = max(3, int(rel * (size * 0.26)))
            x = x0 + b * bar_w
            col = palette[b] * (0.45 + 0.55 * rel)
            glow = (col * 0.30).astype(np.uint8)
            region = frame[mid - h - 5 : mid + h + 5, x : x + bar_w - 2]
            np.maximum(region, glow, out=region)
            frame[mid - h : mid + h, x + 2 : x + bar_w - 4] = col.astype(np.uint8)
        frame[prog_y : prog_y + 3, prog_xa : prog_xb] = (52, 40, 26)
        px = prog_xa + int((prog_xb - prog_xa) * (i / max(total_frames - 1, 1)))
        frame[prog_y : prog_y + 3, prog_xa : px] = (245, 158, 11)
        ffmpeg.stdin.write(frame.tobytes())
    ffmpeg.stdin.close()
    ffmpeg.wait()
    return out_path


# GPU wall time fitted from on-Space measurements (see project notes); steps scale the flow-matching share.
def get_duration(description, lyrics, global_meta, vocal_details, arrangement, duration, seed, randomize_seed, headroom, steps, guidance):
    return min(int(float(duration) * (_DUR_A + _DUR_B * float(steps) / 30.0) + _DUR_C), 600)


# Fitted on-Space (xlarge): wall = 0.71*dur + 0.15*dur*(steps/30) + ~1s; margin for cold-worker init.
_DUR_A, _DUR_B, _DUR_C = 0.75, 0.20, 15


@spaces.GPU(duration=get_duration, size="xlarge")
@torch.inference_mode()
def generate(description, lyrics, global_meta, vocal_details, arrangement, duration, seed, randomize_seed, headroom, steps, guidance):
    caption = "\n".join(s.strip() for s in (global_meta, vocal_details, arrangement) if s.strip())
    if not caption:
        raise gr.Error("Fill in the structured prompt (or use Prompt your song) first.")
    if not lyrics.strip():
        raise gr.Error("Lyrics are required (section tags like [verse] must be on their own line).")
    if randomize_seed:
        seed = random.randint(0, MAX_SEED)
    seed = int(seed)
    # This yield leaves the @spaces.GPU worker only once the GPU is allocated and the body runs —
    # it is the exact "ZeroGPU acquired" signal for the player's stage loader.
    yield {"cmd": "status", "text": "ZeroGPU acquired — the band is warming up..."}, "ZeroGPU acquired — warming up...", gr.skip(), seed
    steps, guidance, sr = int(steps), float(guidance), PIPE.sampling_rate
    text_ids = _encode_prompt(caption, lyrics, "cuda")
    max_frames = min(int(float(duration) * PIPE.frame_rate), 9000)
    generator = torch.Generator("cuda").manual_seed(int(seed))
    ar_generator = torch.Generator("cuda").manual_seed(generator.initial_seed())
    dit_generator = torch.Generator("cuda").manual_seed(generator.initial_seed() + 1)

    import uuid

    gen_id = uuid.uuid4().hex
    start = time.time()
    streamed = 0.0
    off_samples = 0
    pending = []
    started = False
    chunks = []
    seq = 0
    for chunk in _stream_windows(text_ids, max_frames, ar_generator, dit_generator, steps, guidance):
        chunks.append(chunk)
        streamed += chunk.shape[-1] / sr
        if started:
            seq += 1
            arr = _to_int16(chunk)
            msg = _pcm_msg(arr, sr, seq, gen_id, off_samples)
            off_samples += arr.shape[0]
            yield msg, f"streaming... {streamed:.1f}s of audio at {time.time() - start:.0f}s", gr.skip(), seed
        else:
            pending.append(chunk)
            if streamed >= float(headroom):
                started = True
                seq += 1
                arr = _to_int16(torch.cat(pending, dim=-1))
                msg = _pcm_msg(arr, sr, seq, gen_id, off_samples)
                off_samples += arr.shape[0]
                yield msg, f"streaming... {streamed:.1f}s", gr.skip(), seed
                pending = []
            else:
                yield {"cmd": "status", "text": f"buffering {streamed:.1f}/{headroom:.0f}s of headroom..."}, f"buffering {streamed:.1f}/{headroom:.0f}s of headroom...", gr.skip(), seed
    if pending:
        seq += 1
        arr = _to_int16(torch.cat(pending, dim=-1))
        msg = _pcm_msg(arr, sr, seq, gen_id, off_samples)
        off_samples += arr.shape[0]
        yield msg, gr.skip(), gr.skip(), seed
    import tempfile

    import scipy.io.wavfile

    full = _to_int16(torch.cat(chunks, dim=-1))
    wav_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False, dir=_SONGS_DIR)
    scipy.io.wavfile.write(wav_file.name, sr, full)
    yield {"cmd": "done", "gen": gen_id, "frames": int(full.shape[0]), "url": _file_url(wav_file.name)}, f"done: {streamed:.1f}s of audio in {time.time() - start:.0f}s — rendering share video...", wav_file.name, seed


MAX_SEED = np.iinfo(np.int32).max

# ---------------------------------------------------------------------------
# Suno-inspired composer: one custom gr.HTML drives the whole input surface.
# The component's JSON value is the single source of truth ({mode, description,
# instrumental, title, lyrics, global_meta, vocals, arrangement}); JS gathers the
# fields into props.value right before firing an event, and watch('value')
# applies server-side updates (composed lyrics/prompt) back into the DOM.
# Events: 'submit' = simple generate (compose + sing), 'edit' = compose & review,
# 'click' = studio generate. Styling uses only Gradio theme CSS vars so it
# follows Citrus (and dark mode) natively.
# ---------------------------------------------------------------------------

_COMPOSER_DEFAULTS = {
    "mode": "simple",
    "description": "",
    "instrumental": False,
    "title": "",
    "lyrics": DEFAULT_LYRICS,
    "global_meta": DEFAULT_GLOBAL,
    "vocals": DEFAULT_VOCALS,
    "arrangement": DEFAULT_ARRANGEMENT,
}


def _normalize_state(state):
    merged = dict(_COMPOSER_DEFAULTS)
    if isinstance(state, dict):
        merged.update({k: state[k] for k in merged if k in state and state[k] is not None})
    return merged


def _composed_description(state):
    description = state["description"].strip()
    if state["instrumental"]:
        description = (description + "\n" if description else "") + "Instrumental, no vocals."
    return description


_IDEA_CHIPS = [
    "a smoky late-night soul ballad about old flames, warm female voice",
    "a defiant punk anthem about staying up too late",
    "a cozy lo-fi hip hop beat for studying, no vocals",
]

_PRESETS = [
    # Official examples from the MiniMax Music 3 project page (https://minimax-ai.github.io/music3-demo/):
    # caption and lyrics verbatim; the cached audio is the officially showcased generation.
    {"name": "It's In My Head",
     "lyrics": "(Hook)\nDon’t waste your time on me youre already\nThe only one who keeps me rock steady\nAnd the voices in my head\nAll assure me that that’s what you said\nYou miss me.\nAnd if fate fell short this time,\nThen your fading smile keeps me whole for a while\nThe feeling of your hand in mine,\nIs something that I never wanna,\nForget about that summer\n\n[verse 1]\nI was in the 9th grade when I fell in love for the first time\nAnd her name? Well it never ended up as hers-mine\nI loved her for 4 years of her time and when she spurned mine\nI felt like Hamilton shot in the side after burrs lie\nAnd I’m not saying I regret it, in fact I’m indebted without you how would I know what a true friend is, but\nAfter a couple of beer flasks and years past a new true love did appear so a sincere task\nWould be infatuation of the strongest and the strangest and i know it might sound lame but her name was my whole playlist and,\nI wouldn’t change it for the world\nThe feeling of bliss when ya kiss curled up with your girl,\nAnd then she went and broke my heart,\nAnd I’m not saying that it’s hard but it’s hard to see each other apart,\nNow I guess I finally understand,\nWhat they meant when they said I should’ve ran\n\n(Hook)\nDon’t waste your time on me youre already\nThe only one who keeps me rock steady\nAnd the voices in my head\nAll assure me that that’s what you said\nYou miss me.\nAnd if fate fell short this time,\nThen your fading smile keeps me whole for a while\nThe feeling of your hand in mine,\nIs something that I never wanna,\nForget about that summer\n\n[verse 2]\nNow I’m not saying that there’s any affection that’s headed in your direction this is just a reflection\nOn the fact that I hated you, but lately I’ve been thinking maybe I was afraid of you,\nBut the fickle predicament of imprisonment was at the interlude we introduced a listing of differences,\nYa maybe we both could have changed,\nOr it was just my fault for insinuating you were deranged\nA few months apart and everyday is a present,\nHesitant of heartfelt cause the harpy harkened unpleasant,\nBut the truth is in the face of the fact that I’m laughing\nI’m actually happy now I never thought that could happen\nBut lovin is free, and a few words could change a person,\nAnd every single human on earth feels a range of hurtin,\nThe world is a clock and no one can stop it\nDon’t waste time on a Love that’s proven toxic\n\n(Hook)\nDon’t waste your time on me youre already\nThe only one who keeps me rock steady\nAnd the voices in my head\nAll assure me that that’s what you said\nYou miss me.\nAnd if fate fell short this time,\nThen your fading smile keeps me whole for a while\nThe feeling of your hand in mine,\nIs something that I never wanna,\nForget about that summer\n\n[guitar solo]\n\n[hook, Accapella]",
     "global_meta": "2000s pop punk",
     "vocals": "", "arrangement": ""},
    {"name": "Behind The Glass",
     "lyrics": "(Verse 1)\nSylvia asked from behind the glass\nIs there no way out of the mind\nThe books were stacked, the shelves were full\nBut the door she could not find\n(Verse 2)\nThe wolf climbed up the tower stairs\nOne more page, one more light\nThe library gleamed, the stockings shone\nBut no one slept at night\n(Chorus)\nThe way out of the mind\nIs not another thought\nIt’s the floor beneath your feet\nThe strings your fingers caught\nIt’s the needle and the breath\nThe toes you finally feel\nThe way out of the mind, my love\nIs everything that’s real\n(Verse 3)\nPhoebe sat down on the wood\nNo shoes, no suit, no name\nShe closed her eyes, she found a chord\nAnd nothing was the same\n(Bridge)\nFerme la porte de la tour\nDescends pieds nus ce soir\nLe loup n’a plus besoin de lire\nIl a besoin de voir\n(Last Chorus)\nThe way out of the mind\nIs not another word\nIt’s the song you finally sing\nAfter all the ones you’ve heard\nIt’s one last dance with her\nBefore you close your eyes\nThe way out of the mind, my love\nIs where the body lies\n(Outro)\nGo sing, Daniel.\nFeel your toes.\nOne last dance.",
     "global_meta": "Bossa nova with piano and acoustix guitar",
     "vocals": "", "arrangement": ""},
    {"name": "Everything",
     "lyrics": "[Verse 1]\nWoke up this morning, breath in my chest\nDidn’t earn it, still I’m blessed\nClock keeps ticking, can’t rewind\nEvery second drawing a line\n\nChoices echo, seeds we sow\nIn the light or down below\nCan’t keep drifting, can’t pretend\nThis life ain’t just about the end\n\n[Pre-Chorus]\nThere’s a fire calling deep inside\nMore than money, more than pride\n\n[Chorus]\nLive like your eternal life depends on it\nEvery word, every step, every moment\nDon’t just talk it, don’t just sing\nLet your whole life mean everything\nLive like your eternal life depends on it\nNo more halfway, no more counterfeit\nStand on truth, don’t compromise\nLive forever in these borrowed lives\n\n[Verse 2]\nLove your neighbor, lift the weak\nFind the lost, be who they seek\nGrace ain’t cheap, it cost too much\nStill He gave that healing touch\n\nWhen it’s hard and nights are long\nStill choose right over wrong\nYou can fall but don’t you stay\nGet back up and find your way\n\n[Pre-Chorus]\nThere’s a kingdom you can’t see\nBut it’s closer than your heartbeat\n\n[Chorus]\nLive like your eternal life depends on it\nEvery breath is heaven-sent, don’t waste it\nWalk in faith, not by sight\nShine in darkness, be the light\nLive like your eternal life depends on it\nNot tomorrow—right now, commit\nHeart on fire, spirit alive\nLive like forever’s on the line\n\n[Bridge]\nThis ain’t a game, this ain’t pretend\nWhere you start ain’t where you end\nMercy’s wide but truth is real\nWhat you sow is what you’ll feel\n\nSo give Him all, don’t hold back\nStay the course, stay on track\nWhen the final day arrives\nYou’ll know you truly lived your life\n\n[Breakdown]\nOhhh… don’t just survive\nYou were made for more than time\n\n[Final Chorus]\nLive like your eternal life depends on it\nEvery heartbeat got purpose in it\nLift your hands, walk in grace\nRun your race, keep the pace\nLive like your eternal life depends on it\nLet your soul and your life be honest\nWhen it’s over, you’ll testify—\nYou didn’t just live… you lived for life.",
     "global_meta": "Violin intro , funk, male vocals, beat, ethereal, Neo soul, urban funk",
     "vocals": "", "arrangement": ""}
]

_COMPOSER_HTML = """
<div class="mm-card">
  <div class="mm-head">
    <div class="mm-seg" role="tablist">
      <button type="button" class="mm-seg-btn" data-mode="simple" aria-selected="true">Simple</button>
      <button type="button" class="mm-seg-btn" data-mode="studio" aria-selected="false">Studio</button>
    </div>
    <span class="mm-headhint" data-role="headhint">a full song from a one-line idea</span>
  </div>

  <div class="mm-view" data-view="simple">
    <textarea class="mm-desc" data-role="description" rows="4"
      placeholder="Describe your song&#8230;  e.g. a smoky late-night soul ballad about old flames, warm female voice"></textarea>
    <div class="mm-chips" data-role="idea-chips"></div>
    <div class="mm-foot">
      <label class="mm-toggle"><input type="checkbox" data-role="instrumental">Instrumental</label>
      <span class="mm-spacer"></span>
      <button type="button" class="mm-ghost" data-role="compose"
        title="Write the lyrics &amp; structured prompt now and review them in Studio before generating audio">&#9998; Write lyrics &amp; review</button>
      <button type="button" class="mm-primary" data-role="generate-simple">&#9834;&#160;&#160;Generate</button>
    </div>
  </div>

  <div class="mm-view" data-view="studio" hidden>
    <div class="mm-panel">
      <div class="mm-panehead">
        <span class="mm-label">Lyrics</span>
        <span class="mm-tags" data-role="tag-chips"></span>
      </div>
      <div class="mm-assistbar">
        <span class="mm-spark">&#10024;</span>
        <input type="text" data-role="lyrics-assist-prompt"
          placeholder="Describe lyrics to write for you&#8230;  e.g. nostalgic road-trip song, punchy one-line chorus">
        <button type="button" class="mm-assistgo" data-role="lyrics-assist">Write</button>
      </div>
      <textarea data-role="lyrics" class="mm-lyrics" rows="11" spellcheck="false"></textarea>
      <div class="mm-hint">Section tags sit <b>alone on their own line</b> &mdash; words on a tag line are dropped.
        Musical directions (tempo, instruments, dynamics) belong in Arrangement, never in the lyrics.</div>
    </div>
    <div class="mm-panel">
      <div class="mm-panehead">
        <span class="mm-label">Structured prompt</span>
      </div>
      <div class="mm-assistbar">
        <span class="mm-spark">&#10024;</span>
        <input type="text" data-role="prompt-assist-prompt"
          placeholder="Describe the sound to write for you&#8230;  e.g. dreamy shoegaze, slow build, whispered vocals">
        <button type="button" class="mm-assistgo" data-role="prompt-assist">Write</button>
      </div>
      <div class="mm-field">
        <div class="mm-sublabel">Global metadata <span class="mm-opt">genre &middot; BPM &middot; key &amp; scale &middot; mood arc &middot; scenario &middot; production</span></div>
        <textarea data-role="global" rows="3"></textarea>
      </div>
      <div class="mm-field">
        <div class="mm-sublabel">Vocal details <span class="mm-opt">gender &middot; timbre &middot; style per section &middot; harmonies &middot; effects</span></div>
        <textarea data-role="vocals" rows="2"></textarea>
      </div>
      <div class="mm-field">
        <div class="mm-sublabel">Arrangement <span class="mm-opt">instruments per section &middot; groove &middot; bass &middot; textures &middot; spatial fx</span></div>
        <textarea data-role="arrangement" rows="3"></textarea>
      </div>
      <div class="mm-fieldrow">
        <span class="mm-sublabel">Title</span>
        <input type="text" data-role="title" placeholder="Untitled &mdash; shown on the share video">
      </div>
    </div>
    <div class="mm-presets"><span class="mm-preset-label">Presets</span><span class="mm-chips" data-role="preset-chips"></span></div>
    <div class="mm-foot">
      <span class="mm-spacer"></span>
      <button type="button" class="mm-primary" data-role="generate-studio">&#9834;&#160;&#160;Generate</button>
    </div>
  </div>

  <div class="mm-status" data-role="compose-status" hidden>
    <span class="mm-pulse"></span><span data-role="compose-status-text"></span>
  </div>
</div>
"""

_COMPOSER_CSS = """
.mm-card { background: var(--block-background-fill); border: var(--block-border-width, 1px) solid var(--block-border-color, var(--border-color-primary)); border-radius: var(--block-radius, 12px); box-shadow: var(--block-shadow, none); padding: 16px; display: flex; flex-direction: column; gap: 12px; }
.mm-head { display: flex; align-items: center; justify-content: space-between; gap: 10px; }
.mm-seg { display: inline-flex; background: var(--background-fill-secondary); border: 1px solid var(--border-color-primary); border-radius: 999px; padding: 3px; gap: 2px; }
.mm-seg-btn { border: none; background: transparent; color: var(--body-text-color-subdued); padding: 5px 16px; border-radius: 999px; font-family: inherit; font-size: 14px; font-weight: 600; cursor: pointer; transition: background .15s, color .15s; }
.mm-seg-btn[aria-selected="true"] { background: var(--button-primary-background-fill); color: var(--button-primary-text-color); }
.mm-headhint { color: var(--body-text-color-subdued); font-size: 12.5px; text-align: right; }
.mm-view { display: flex; flex-direction: column; gap: 12px; }
.mm-view[hidden] { display: none; }
.mm-card textarea, .mm-card input[type="text"] { width: 100%; box-sizing: border-box; background: var(--input-background-fill); border: var(--input-border-width, 1px) solid var(--input-border-color, var(--border-color-primary)); border-radius: var(--input-radius, 8px); padding: 10px 12px; color: var(--body-text-color); font-family: inherit; font-size: var(--input-text-size, 14px); line-height: 1.5; resize: vertical; transition: border-color .15s, box-shadow .15s; }
.mm-card textarea::placeholder, .mm-card input::placeholder { color: var(--input-placeholder-color, var(--body-text-color-subdued)); }
.mm-card textarea:focus, .mm-card input[type="text"]:focus { outline: none; border-color: var(--input-border-color-focus, var(--color-accent)); box-shadow: var(--input-shadow-focus, none); }
.mm-desc { font-size: 16px; min-height: 118px; }
.mm-lyrics { font-family: var(--font-mono, ui-monospace, SFMono-Regular, Menlo, monospace); font-size: 13px; }
.mm-chips { display: flex; flex-wrap: wrap; gap: 6px; }
.mm-chip { background: var(--button-secondary-background-fill); color: var(--button-secondary-text-color); border: 1px solid var(--button-secondary-border-color, var(--border-color-primary)); border-radius: 999px; padding: 4px 12px; font-family: inherit; font-size: 12.5px; cursor: pointer; transition: border-color .15s, background .15s; }
.mm-chip:hover { background: var(--button-secondary-background-fill-hover, var(--button-secondary-background-fill)); border-color: var(--color-accent); }
.mm-tags { display: flex; flex-wrap: wrap; gap: 4px; }
.mm-panehead .mm-tags { flex: 1; justify-content: flex-end; }
.mm-tag { background: transparent; color: var(--body-text-color-subdued); border: 1px dashed var(--border-color-primary); border-radius: 6px; padding: 2px 8px; font-family: var(--font-mono, ui-monospace, monospace); font-size: 11.5px; cursor: pointer; transition: color .15s, border-color .15s; }
.mm-tag:hover { color: var(--color-accent); border-color: var(--color-accent); }
.mm-foot { display: flex; align-items: center; gap: 10px; }
.mm-spacer { flex: 1; }
.mm-toggle { display: inline-flex; align-items: center; gap: 7px; color: var(--body-text-color); font-size: 14px; cursor: pointer; user-select: none; }
.mm-toggle input { width: 16px; height: 16px; accent-color: var(--color-accent); cursor: pointer; }
.mm-primary { background: var(--button-primary-background-fill); color: var(--button-primary-text-color); border: var(--button-border-width, 1px) solid var(--button-primary-border-color, transparent); border-radius: var(--button-large-radius, var(--radius-lg, 8px)); padding: 10px 24px; font-family: inherit; font-size: var(--button-large-text-size, 16px); font-weight: var(--button-large-text-weight, 600); cursor: pointer; box-shadow: var(--button-primary-shadow, none); transition: background .15s, box-shadow .15s, transform .05s; }
.mm-primary:hover { background: var(--button-primary-background-fill-hover, var(--button-primary-background-fill)); box-shadow: var(--button-primary-shadow-hover, var(--button-primary-shadow, none)); }
.mm-primary:active { transform: translateY(1px); box-shadow: var(--button-primary-shadow-active, none); }
.mm-ghost { background: transparent; color: var(--body-text-color-subdued); border: none; border-radius: var(--radius-lg, 8px); padding: 8px 10px; font-family: inherit; font-size: 13.5px; cursor: pointer; box-shadow: none; transition: color .15s; }
.mm-ghost:hover { color: var(--color-accent); }
.mm-panel { background: var(--background-fill-secondary); border: 1px solid var(--border-color-primary); border-radius: var(--radius-lg, 10px); padding: 12px; display: flex; flex-direction: column; gap: 9px; }
.mm-panehead { display: flex; align-items: baseline; justify-content: space-between; gap: 10px; flex-wrap: wrap; }
.mm-assistbar { display: flex; align-items: center; gap: 7px; background: var(--block-background-fill); border: 1px dashed var(--border-color-primary); border-radius: 999px; padding: 3px 5px 3px 12px; transition: border-color .15s; }
.mm-assistbar:focus-within { border-style: solid; border-color: var(--input-border-color-focus, var(--color-accent)); }
.mm-assistbar .mm-spark { font-size: 13px; opacity: .8; }
.mm-card .mm-assistbar input[type="text"] { flex: 1; background: transparent; border: none; border-radius: 0; padding: 6px 0; font-size: 13px; }
.mm-card .mm-assistbar input[type="text"]:focus { box-shadow: none; border: none; }
.mm-assistgo { background: transparent; color: var(--color-accent); border: 1px solid var(--color-accent); border-radius: 999px; padding: 4px 14px; font-family: inherit; font-size: 12.5px; font-weight: 600; cursor: pointer; white-space: nowrap; transition: background .15s, color .15s; }
.mm-assistgo:hover { background: var(--button-primary-background-fill); border-color: var(--button-primary-border-color, transparent); color: var(--button-primary-text-color); }
.mm-label { color: var(--block-title-text-color, var(--body-text-color)); font-size: var(--block-title-text-size, 13px); font-weight: var(--block-title-text-weight, 600); }
.mm-sublabel { color: var(--block-title-text-color, var(--body-text-color)); font-size: 12.5px; font-weight: 600; margin-bottom: 4px; white-space: nowrap; }
.mm-opt { color: var(--body-text-color-subdued); font-weight: 400; font-size: 11px; white-space: normal; }
.mm-field { display: flex; flex-direction: column; }
.mm-fieldrow { display: flex; align-items: center; gap: 10px; }
.mm-fieldrow .mm-sublabel { margin-bottom: 0; }
.mm-fieldrow input { flex: 1; }
.mm-hint { color: var(--body-text-color-subdued); font-size: 11.5px; line-height: 1.45; margin-top: 4px; }
.mm-presets { display: flex; align-items: center; gap: 8px; }
.mm-preset-label { color: var(--body-text-color-subdued); font-size: 12px; font-weight: 600; }
.mm-status { display: flex; align-items: center; gap: 8px; color: var(--body-text-color-subdued); font-size: 13px; }
.mm-status[hidden] { display: none; }
.mm-pulse { width: 9px; height: 9px; border-radius: 50%; background: var(--color-accent); animation: mm-pulse 1.1s ease-in-out infinite; }
@keyframes mm-pulse { 0%, 100% { opacity: .25; transform: scale(.8); } 50% { opacity: 1; transform: scale(1.1); } }
"""

_COMPOSER_JS = """
const $ = function(s) { return element.querySelector(s); };
const $$ = function(s) { return Array.from(element.querySelectorAll(s)); };
const F = {
  description: $('[data-role="description"]'),
  instrumental: $('[data-role="instrumental"]'),
  title: $('[data-role="title"]'),
  lyrics: $('[data-role="lyrics"]'),
  global_meta: $('[data-role="global"]'),
  vocals: $('[data-role="vocals"]'),
  arrangement: $('[data-role="arrangement"]'),
};
const IDEAS = __IDEAS__;
const PRESETS = __PRESETS__;
const TAGS = ['[intro]', '[verse]', '[pre-chorus]', '[chorus]', '[post-chorus]', '[bridge]', '[instrumental]', '[solo]', '[outro]'];
const HINTS = { simple: 'a full song from a one-line idea', studio: 'lyrics + structured caption, full control' };
let mode = 'simple';
let lastPushed = '';

function setMode(m) {
  mode = m;
  $('[data-view="simple"]').hidden = (m !== 'simple');
  $('[data-view="studio"]').hidden = (m !== 'studio');
  $$('.mm-seg-btn').forEach(function(b) { b.setAttribute('aria-selected', String(b.dataset.mode === m)); });
  $('[data-role="headhint"]').textContent = HINTS[m] || '';
}
function setVal(el, v) { if (el.value !== v) el.value = v; }
function readState() {
  return {
    mode: mode,
    description: F.description.value,
    instrumental: F.instrumental.checked,
    title: F.title.value,
    lyrics: F.lyrics.value,
    global_meta: F.global_meta.value,
    vocals: F.vocals.value,
    arrangement: F.arrangement.value,
  };
}
function gather(extra) { const s = Object.assign(readState(), extra || {}); lastPushed = JSON.stringify(s); props.value = s; }
function applyState(v) {
  if (!v) return;
  setVal(F.description, v.description || '');
  if (F.instrumental.checked !== !!v.instrumental) F.instrumental.checked = !!v.instrumental;
  setVal(F.title, v.title || '');
  setVal(F.lyrics, v.lyrics || '');
  setVal(F.global_meta, v.global_meta || '');
  setVal(F.vocals, v.vocals || '');
  setVal(F.arrangement, v.arrangement || '');
  if (v.mode) setMode(v.mode);
}
function status(msg) {
  const el = $('[data-role="compose-status"]');
  el.hidden = !msg;
  if (msg) $('[data-role="compose-status-text"]').textContent = msg;
}
function insertTag(tag) {
  const ta = F.lyrics;
  const v = ta.value;
  const s = ta.selectionStart == null ? v.length : ta.selectionStart;
  const before = v.slice(0, s), after = v.slice(s);
  let ins = tag;
  if (before.length && !before.endsWith('\\n')) ins = '\\n' + ins;
  if (!after.startsWith('\\n')) ins = ins + '\\n';
  ta.value = before + ins + after;
  const pos = (before + ins).length;
  ta.focus();
  ta.setSelectionRange(pos, pos);
}
IDEAS.forEach(function(t, i) {
  const b = document.createElement('button');
  b.type = 'button'; b.className = 'mm-chip'; b.textContent = t;
  b.addEventListener('click', function() {
    F.description.value = t;
    armAudio();
    gather({example_key: 'idea_' + i}); trigger('apply');
  });
  $('[data-role="idea-chips"]').appendChild(b);
});
PRESETS.forEach(function(p, i) {
  const b = document.createElement('button');
  b.type = 'button'; b.className = 'mm-chip'; b.textContent = p.name;
  b.addEventListener('click', function() {
    setVal(F.lyrics, p.lyrics); setVal(F.global_meta, p.global_meta);
    setVal(F.vocals, p.vocals); setVal(F.arrangement, p.arrangement);
    armAudio();
    gather({example_key: 'preset_' + i}); trigger('apply');
  });
  $('[data-role="preset-chips"]').appendChild(b);
});
TAGS.forEach(function(t) {
  const b = document.createElement('button');
  b.type = 'button'; b.className = 'mm-tag'; b.textContent = t;
  b.addEventListener('click', function() { insertTag(t); });
  $('[data-role="tag-chips"]').appendChild(b);
});
$$('.mm-seg-btn').forEach(function(b) { b.addEventListener('click', function() { setMode(b.dataset.mode); }); });
function armAudio() { if (window.__mmArmAudio) window.__mmArmAudio(); }
$('[data-role="generate-simple"]').addEventListener('click', function() { status(''); armAudio(); gather(); trigger('submit'); });
$('[data-role="generate-studio"]').addEventListener('click', function() { status(''); armAudio(); gather(); trigger('click'); });
$('[data-role="compose"]').addEventListener('click', function() { gather({assist: 'all'}); status('Writing lyrics and structured prompt with MiniMax-M3...'); trigger('edit'); });
function runAssist(target, msg) {
  const inp = $('[data-role="' + target + '-assist-prompt"]');
  gather({assist: target, assist_prompt: inp.value.trim()});
  status(msg);
  trigger('edit');
}
$('[data-role="lyrics-assist"]').addEventListener('click', function() { runAssist('lyrics', 'Writing lyrics with MiniMax-M3...'); });
$('[data-role="prompt-assist"]').addEventListener('click', function() { runAssist('prompt', 'Writing the structured prompt with MiniMax-M3...'); });
$('[data-role="lyrics-assist-prompt"]').addEventListener('keydown', function(e) { if (e.key === 'Enter') { e.preventDefault(); runAssist('lyrics', 'Writing lyrics with MiniMax-M3...'); } });
$('[data-role="prompt-assist-prompt"]').addEventListener('keydown', function(e) { if (e.key === 'Enter') { e.preventDefault(); runAssist('prompt', 'Writing the structured prompt with MiniMax-M3...'); } });
watch('value', function() {
  const v = props.value;
  if (JSON.stringify(v) === lastPushed) return;
  status('');
  applyState(v);
});
applyState(props.value);
""".replace("__IDEAS__", json.dumps(_IDEA_CHIPS)).replace("__PRESETS__", json.dumps(_PRESETS))


# Cached example renders (built once via the API, committed under examples/). Keys: idea_N / preset_N.
_EXAMPLES = {}
if os.path.exists("examples/manifest.json"):
    with open("examples/manifest.json") as f:
        _EXAMPLES = json.load(f)


def load_example(raw_state):
    # Generator on purpose: the streaming gr.Audio only accepts values arriving through a
    # generator event's stream, exactly like generate()'s chunks.
    key = raw_state.get("example_key", "") if isinstance(raw_state, dict) else ""
    state = _normalize_state(raw_state)
    example = _EXAMPLES.get(key)
    if not example:
        yield state, gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip()
        return
    for k in ("description", "instrumental", "title", "lyrics", "global_meta", "vocals", "arrangement"):
        if k in example:
            state[k] = example[k]
    wav = example.get("wav") if example.get("wav") and os.path.exists(example.get("wav", "")) else None
    video = example.get("video") if example.get("video") and os.path.exists(example.get("video", "")) else None
    if example.get("official"):
        stats_md = "official example from the [MiniMax Music 3 project page](https://minimax-ai.github.io/music3-demo/)"
    else:
        stats_md = f"cached example — seed {example.get('seed')}"
    title = example.get("title") or state["description"]
    yield state, {"cmd": "reset", "nonce": random.random()}, stats_md, video or gr.skip(), wav or gr.skip(), title
    if wav:
        import uuid
        import wave as _wave

        with _wave.open(wav) as w:
            frames, wav_sr, wav_ch = w.getnframes(), w.getframerate(), w.getnchannels()
        # "load" streams the cached wav progressively in the player (raw PCM over fetch):
        # playback starts within the first fraction of a second instead of after the full download.
        load = {"cmd": "load", "gen": uuid.uuid4().hex, "frames": frames, "sr": wav_sr, "ch": wav_ch,
                "url": _file_url(wav)}
        yield gr.skip(), load, gr.skip(), gr.skip(), gr.skip(), gr.skip()


def compose_assist(raw_state, duration):
    # One LLM event, three targets: 'all' (simple CTA -> review in Studio), 'lyrics', 'prompt' (per-pane
    # assist bars). Each pane assist has its own typed instruction (assist_prompt); the other pane's current
    # content rides along as context so both halves stay coherent.
    target = raw_state.get("assist", "all") if isinstance(raw_state, dict) else "all"
    instruction = (raw_state.get("assist_prompt", "") if isinstance(raw_state, dict) else "").strip()
    state = _normalize_state(raw_state)
    description = _composed_description(state)
    if target == "lyrics":
        data = _llm_json(
            _LYRICS_SYSTEM,
            f"Lyrics instruction: {instruction or description or '(none — write lyrics that fit the structured prompt)'}\n"
            f"Current structured prompt, keep the lyrics coherent with it:\n"
            f"Global metadata: {state['global_meta']}\nVocal details: {state['vocals']}\n"
            f"Arrangement: {state['arrangement']}\nTarget duration: {int(duration)} seconds.",
            required=("lyrics",),
        )
        state["lyrics"] = data["lyrics"]
        return state, "Lyrics written — tweak them, or press Generate."
    if target == "prompt":
        data = _llm_json(
            _PROMPT_SYSTEM,
            f"Sound instruction: {instruction or description or '(none — describe a sound that fits the lyrics)'}\n"
            f"Current lyrics, keep the structured prompt coherent with them:\n{state['lyrics']}",
            required=("global_metadata", "vocal_details", "arrangement"),
        )
        state.update(global_meta=data["global_metadata"], vocals=data["vocal_details"], arrangement=data["arrangement"])
        return state, "Structured prompt written — tweak it, or press Generate."
    lyr, gm, vd, arr = compose_song(description, duration)
    state.update(mode="studio", lyrics=lyr, global_meta=gm, vocals=vd, arrangement=arr)
    return state, "Lyrics & structured prompt ready — review and tweak them, then press Generate."


def simple_generate(state, duration, seed, randomize_seed, headroom, steps, guidance):
    # One event so the output components engage (spinner) from the first click, through composing and singing.
    state = _normalize_state(state)
    description = _composed_description(state)
    title = state["title"] or state["description"]
    yield gr.skip(), "writing lyrics & structured prompt...", gr.skip(), gr.skip(), gr.skip(), title
    lyr, gm, vd, arr = compose_song(description, duration)
    state.update(lyrics=lyr, global_meta=gm, vocals=vd, arrangement=arr)
    yield {"cmd": "status", "text": "Lyrics ready — acquiring ZeroGPU..."}, "lyrics ready — acquiring ZeroGPU...", gr.skip(), gr.skip(), state, gr.skip()
    for audio, status, wav, used_seed in generate(
        description, lyr, gm, vd, arr, duration, seed, randomize_seed, headroom, steps, guidance
    ):
        yield audio, status, wav, used_seed, gr.skip(), gr.skip()


def studio_generate(state, duration, seed, randomize_seed, headroom, steps, guidance):
    state = _normalize_state(state)
    yield {"cmd": "status", "text": "Acquiring ZeroGPU..."}, gr.skip(), gr.skip(), gr.skip(), state["title"] or state["description"]
    for audio, status, wav, used_seed in generate(
        state["description"], state["lyrics"], state["global_meta"], state["vocals"], state["arrangement"],
        duration, seed, randomize_seed, headroom, steps, guidance,
    ):
        yield audio, status, wav, used_seed, gr.skip()


CSS = """
#col-container { max-width: 1300px; margin: 0 auto; }
.dark .gradio-container { color: var(--body-text-color); }
.html-container{padding: 0}
.mm3-logo { display: block; margin: 8px auto 0; width: 500px; max-width: 100%; }
.mm3-logo-dark { display: none; }
.dark .mm3-logo-light { display: none; }
.dark .mm3-logo-dark { display: block; }
"""

import base64

_LOGO_LIGHT_B64 = base64.b64encode(open("logo_light.png", "rb").read()).decode()
_LOGO_DARK_B64 = base64.b64encode(open("logo_dark.png", "rb").read()).decode()

with gr.Blocks(theme=gr.themes.Citrus(), css=CSS) as demo:
    with gr.Column(elem_id="col-container"):
        gr.HTML(
            '<img src="data:image/png;base64,' + _LOGO_LIGHT_B64 + '" class="mm3-logo mm3-logo-light" alt="MiniMax Music 3">'
            '<img src="data:image/png;base64,' + _LOGO_DARK_B64 + '" class="mm3-logo mm3-logo-dark" alt="MiniMax Music 3">',
            container=False,
            padding=False,
        )
        gr.Markdown(
            "MiniMax Music 3 is a music generation model designed to support the creation of full-length songs "
            "[[model]](https://huggingface.co/MiniMaxAI/MiniMax-Music3) | "
            "[[project]](https://minimax-ai.github.io/music3-demo/) | "
            "[[run locally with diffusers]](https://github.com/huggingface/diffusers/blob/82319140e0456fd58beff0a251c38825bfc310de/docs/source/en/api/pipelines/minimax_music3.md) | "
            "[[prompting guide and skill]](https://github.com/MiniMax-AI/MiniMax-Music3/)"
        )
        with gr.Row():
            with gr.Column():
                composer = gr.HTML(
                    value=dict(_COMPOSER_DEFAULTS),
                    html_template=_COMPOSER_HTML,
                    css_template=_COMPOSER_CSS,
                    js_on_load=_COMPOSER_JS,
                    container=False,
                    padding=False,
                )
                duration = gr.Slider(5, 300, value=60, step=5, label="Maximum song duration (seconds)")
                with gr.Accordion("Advanced", open=False):
                    seed = gr.Slider(label="Seed", minimum=0, maximum=MAX_SEED, step=1, value=0)
                    randomize_seed = gr.Checkbox(label="Randomize seed", value=True)
                    headroom = gr.Slider(0, 60, value=0, step=1, label="Playback headroom (s)")
                    steps = gr.Slider(4, 60, value=30, step=1, label="Flow-matching steps per chunk")
                    guidance = gr.Slider(1.0, 4.0, value=1.7, step=0.1, label="Guidance scale")
            with gr.Column():
                player = gr.HTML(
                    value=None, html_template=_PLAYER_HTML, css_template=_PLAYER_CSS,
                    js_on_load=_PLAYER_JS, container=False, padding=False,
                )
                stats = gr.Markdown()
                video_out = gr.Video(label="Share video", autoplay=False)
        # Hidden File (not gr.State) so the full-song wav is exposed on the API — used to build cached examples.
        wav_state = gr.File(visible=False)
        video_title = gr.State("")

    _knobs = [duration, seed, randomize_seed, headroom, steps, guidance]

    ev_simple = composer.submit(lambda: ({"cmd": "reset", "loader": True, "status": "Writing lyrics & structured prompt with MiniMax-M3...", "nonce": random.random()}, None), None, [player, video_out]).then(
        simple_generate, [composer] + _knobs,
        [player, stats, wav_state, seed, composer, video_title],
        concurrency_limit=None, show_progress="minimal",
    )
    ev_simple.then(render_video, [wav_state, video_title], video_out, concurrency_limit=16).then(
        lambda s: s.replace(" — rendering share video...", " — share video ready."), stats, stats
    )

    ev_studio = composer.click(lambda: ({"cmd": "reset", "loader": True, "status": "Acquiring ZeroGPU...", "nonce": random.random()}, "", None), None, [player, stats, video_out]).then(
        studio_generate, [composer] + _knobs,
        [player, stats, wav_state, seed, video_title],
        concurrency_limit=None, show_progress="minimal",
    )
    ev_studio.then(render_video, [wav_state, video_title], video_out, concurrency_limit=16).then(
        lambda s: s.replace(" — rendering share video...", " — share video ready."), stats, stats
    )

    player.stop(
        lambda: "stopped — kept the part that was already streamed.", None, stats,
        cancels=[ev_simple, ev_studio], show_progress="hidden",
    )

    composer.edit(compose_assist, [composer, duration], [composer, stats], concurrency_limit=None, show_progress="minimal")

    composer.apply(load_example, [composer], [composer, player, stats, video_out, wav_state, video_title], concurrency_limit=None, show_progress="minimal")

if __name__ == "__main__":
    demo.launch()
