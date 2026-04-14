---
name: Generate Video
description: Prepare the built-in video model, then generate a video from a text prompt (text-to-video) or from a text prompt plus a workspace-local image (text-image-to-video) using the host-native local model service powered by LTX-2.3 (22B distilled, 4-bit quantized) via the MLX-native `mlx-video-with-audio` runner on Apple Silicon.
---

# Purpose

Use this skill when you need to prepare the built-in local video model and then generate a video artifact from a text prompt, optionally conditioned on a workspace-local image.

# When To Use

Use when:
- the user wants a new video generated from a text prompt
- the user wants image-conditioned video generation from a text prompt plus a workspace-local image
- the task should stay inside the built-in local video capability path
- the resulting video should be saved into the runtime workspace for later reuse

Skip when:
- the task is video analysis rather than generation
- the user explicitly wants a different backend or service not provided by this skill
- the task is better handled by a remote API video service rather than a local open-source model

# Supported Surface

This skill supports both modes through a single unified LTX-2.3 audio-video pipeline:

- `text_to_video`
  - one text prompt
  - one generated output video (video-only; audio track is deliberately suppressed)
- `text_image_to_video`
  - one text prompt
  - one workspace-local input image used to condition the generation (by default the first frame)
  - one generated output video (video-only)

It does **not** currently support:

- multi-image or multi-video conditioning
- synchronized audio + video output (the underlying runner supports it, but this skill currently passes `--no-audio` for simpler outputs)
- LoRA loading
- prompt expansion

The mode is selected automatically by the `generate_video.py` script: if `--image-path` is provided, the skill runs `text_image_to_video`; otherwise it runs `text_to_video`.

# Skill Mode

- `script_mode: multi`
- Preferred for video generation because the agent should reason between preparation and inference.
- Default phase scripts:
  - `scripts/prepare_model.py`: prepare/download the built-in video model before generation
  - `scripts/generate_video.py`: run the actual video inference and save the output artifact
- Core Agent should call this skill directly instead of passing provider/model config through Helix CLI.
- Default handler path: `skills/builtin_skills/generate-video/scripts/prepare_model.py`

# Procedure

1. Gather context:
   - confirm the exact prompt and desired scene
   - determine whether generation is text-only or image-conditioned
   - if image-conditioned, confirm the input image is already inside the workspace
2. Prepare the model first:
   - run `scripts/prepare_model.py`
   - downloads the LTX-2.3 22B distilled q4 MLX weights (~23GB — this variant was quantized by the same maintainer as the runner's default LTX-2 weights, so it's config-aligned with `mlx-video-with-audio`)
   - wait for `status=ok` before moving on to inference
   - this step is idempotent; it is safe even if the model is already cached
   - `--timeout` controls both the script HTTP wait and the local model service request budget for this call
   - if the job may run long, set exec `timeout_seconds` larger than `--timeout`
3. Plan output:
   - choose one workspace-local target using `--output-path` or `--output-dir`
   - keep generated artifacts in deterministic, reusable locations
4. Infer:
   - run `scripts/generate_video.py` with `--prompt` and one output target option
   - pass `--image-path` only when image-conditioned video is actually required
   - pass generation knobs only when they materially improve the requested output
   - let the skill choose its built-in backend/model; do not add provider/model args
   - first inference will also lazy-download the Gemma text encoder (~24GB on first run); subsequent runs are fast
5. Verify:
   - inspect runtime stdout for `status`, `task_type`, `output_path`, `fps`, `num_frames`, and any service/runtime error details
   - confirm the generated video file exists in the workspace
6. Report:
   - return the resulting artifact path and any important generation constraints or failures

# Runtime Contract

All scripts in this skill must:
1. print one final JSON object to stdout
2. use stderr only for unexpected runtime failures
3. keep stdout concise but informative so workflow history remains readable

The next reasoning step should inspect runtime stdout/stderr before deciding the next phase.

# Action Input Templates

## Phase 1: Prepare Model

```json
{
  "code_type": "python",
  "timeout_seconds": 3600,
  "script_path": "skills/builtin_skills/generate-video/scripts/prepare_model.py",
  "script_args": [
    "--timeout", "3000"
  ]
}
```

## Phase 2: Generate Video

- required:
  - `--prompt`
  - one of:
    - `--output-dir`
    - `--output-path`

Recommended argument rules:
- use workspace-local paths only
- pass `--image-path` only when you want text-image-to-video rather than text-to-video
- pass only one `--image-path`; this skill does not support multiple conditions
- if `--output-dir` is used, let the script choose a deterministic file name under that directory
- keep `--size`, `--num-frames`, and `--num-inference-steps` close to defaults unless the user requests a specific output shape
- prefer a known safe preset over inventing unusual video dimensions
- `--timeout` controls both the script HTTP wait and the local model service request budget for this call
- when generation may run long, set exec `timeout_seconds` larger than `--timeout`
- do not pass provider, model, or API settings; the skill owns its backend and model choice

### Prompt Guidance

- For text-to-video:
  - describe the subject, motion, camera movement, lighting, and mood in 1–4 focused sentences
  - avoid cramming multiple unrelated scene changes into one clip
- For text-image-to-video:
  - treat the input image as the conditioning frame (by default the first frame)
  - describe how that scene should animate rather than describing a completely different scene
- LTX-2.3 responds well to cinematic, descriptive language about motion and composition.

### Parameter Reference

- `--prompt`
  - required
  - best practice: 1–4 sentences focused on subject, motion, camera, lighting, and mood
- `--image-path`
  - optional
  - only use when the user wants image-conditioned video
  - must point to a single workspace-local image
- `--image-strength`
  - default: `1.0`
  - recommended working range: `0.5–1.0`
  - conditioning strength; `1.0` = full denoise starting from the image, `0.0` = keep original (no generation effect)
  - only applied when `--image-path` is set
- `--image-frame-idx`
  - default: `0`
  - recommended value: `0` (use the image as the first frame)
  - which frame index the conditioning image should occupy; only applied when `--image-path` is set
- `--size`
  - default: `832x480`
  - recommended presets:
    - landscape: `832x480`, `1024x576`
    - square: `576x576`
  - best practice: keep both dimensions divisible by `32`
- `--num-frames`
  - default: `65`
  - recommended working range: `49–121`
  - LTX-2.3 decodes at 24 fps, so `65` frames ≈ `2.7s`, `121` frames ≈ `5.0s`
- `--num-inference-steps`
  - default: `30`
  - recommended working range: `20–40`
  - this is the distilled checkpoint — 30 steps is the sweet spot; going higher rarely helps
- `--guidance-scale`
  - default: `3.0`
  - recommended working range: `2.0–5.0`
  - distilled LTX-2.3 runs best at modest CFG values; avoid high guidance which tends to over-constrain motion
- `--seed`
  - default: `42`
  - keep fixed for reproducibility; change only when you want a different take

### Best-Practice Defaults

- Quick preview (T2V)
  - `--size 832x480 --num-frames 49 --num-inference-steps 20 --guidance-scale 3.0 --seed 42`
- Standard final clip (T2V)
  - `--size 832x480 --num-frames 65 --num-inference-steps 30 --guidance-scale 3.0 --seed 42`
- Image-conditioned clip (I2V)
  - `--image-path <workspace image> --image-strength 1.0 --image-frame-idx 0 --size 832x480 --num-frames 65 --num-inference-steps 30 --guidance-scale 3.0 --seed 42`

Example text-to-video:

```json
{
  "code_type": "python",
  "timeout_seconds": 3600,
  "script_path": "skills/builtin_skills/generate-video/scripts/generate_video.py",
  "script_args": [
    "--prompt", "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage.",
    "--size", "832x480",
    "--num-frames", "65",
    "--num-inference-steps", "30",
    "--guidance-scale", "3.0",
    "--seed", "42",
    "--output-dir", "generated_videos/cats-boxing",
    "--timeout", "3000"
  ]
}
```

Example text-image-to-video:

```json
{
  "code_type": "python",
  "timeout_seconds": 3600,
  "script_path": "skills/builtin_skills/generate-video/scripts/generate_video.py",
  "script_args": [
    "--prompt", "A calm cinematic aerial reveal of this house at sunset, camera slowly orbiting.",
    "--image-path", "sessions/test_1/project/reference-house.png",
    "--image-strength", "1.0",
    "--image-frame-idx", "0",
    "--size", "832x480",
    "--num-frames", "65",
    "--num-inference-steps", "30",
    "--guidance-scale", "3.0",
    "--seed", "42",
    "--output-path", "sessions/test_1/project/generated-house-reveal.mp4",
    "--timeout", "3000"
  ]
}
```

# Output JSON Shape

## Prepare Phase

```json
{
  "executed_skill": "generate-video",
  "phase": "prepare",
  "status": "ok|error",
  "model_used": "...",
  "error_code": "...",
  "message": "..."
}
```

## Generate Phase

```json
{
  "executed_skill": "generate-video",
  "phase": "generate",
  "status": "ok|error",
  "task_type": "text_to_video|text_image_to_video",
  "prompt": "...",
  "image_path": "...",
  "output_path": "...",
  "fps": 24,
  "num_frames": 65,
  "model_used": "...",
  "error_code": "...",
  "message": "..."
}
```

# Error Handling Rule

1. If the local model service variables are missing, stop internal retries and return control to requester or runtime with the configuration failure.
2. If model preparation fails, do not continue to inference until preparation succeeds.
3. If output path validation fails, do not retry with the same invalid path; choose a new workspace-local path first.
4. If image-conditioned generation is requested, do not proceed without a valid workspace-local `--image-path`.
5. If generation fails due to service/model runtime issues, surface the exact `error_code` and `message` rather than masking them.

# Skill Dependencies

- (none)

This skill is self-contained for local text-to-video and text-image-to-video generation and does not require another skill for normal execution.

# Notes

- This skill calls the runtime-managed local inference host through `HELIX_LOCAL_MODEL_SERVICE_URL`.
- The skill ships a `model_spec.json` next to `SKILL.md` and sends that full spec payload to the host.
- `scripts/prepare_model.py` calls `/models/prepare`; `scripts/generate_video.py` calls `/infer`.
- The built-in backend is `mlx` — the same venv that powers `generate-image`.
- The built-in model repo is `notapalindrome/ltx23-mlx-av-q4`, a community MLX-native 4-bit quantization of Lightricks' LTX-2.3 22B distilled checkpoint.
- Inference is dispatched to the `mlx-video-with-audio` pip package (`python -m mlx_video.generate_av`), which runs natively on Metal via MLX — no PyTorch-MPS workarounds needed.
- The runner's unified audio-video pipeline handles both T2V and I2V through a single model; the adapter switches modes by passing `--image` (plus `--image-strength` and `--image-frame-idx`) to the subprocess when the caller supplies `--image-path`.
- The runner's `--model-repo` is pointed at the locally-downloaded weights directory, so the weights are not re-downloaded through `snapshot_download` at inference time.
- The Gemma text encoder (`mlx-community/gemma-3-12b-it-bf16`, ~24GB) is downloaded lazily by the runner on first use into the same local HF cache; subsequent inferences skip the download.
- LTX-2.3 produces video at a fixed output rate of 24 fps; `fps` in the output JSON reflects this.
- Audio generation is supported by the underlying runner but deliberately disabled here (`--no-audio`) — a future follow-up may expose it as a knob.
