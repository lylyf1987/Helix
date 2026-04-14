"""Host adapter for the generate-video skill (LTX-2.3 MLX via mlx-video-with-audio)."""

from __future__ import annotations

import os
import subprocess
from typing import Any

from helix.runtime.local_model_service.adapters import _BaseBackend
from helix.runtime.local_model_service.helpers import (
    _ensure_worker_dependencies,
    _parse_float,
    _parse_int,
    _parse_size,
    _request_inputs,
    _resolve_service_workspace_root,
    _resolve_workspace_path,
)

_DEPENDENCIES = (
    "mlx-video-with-audio",
)


class _MLXVideoBackend(_BaseBackend):
    """Runs LTX-2.3 MLX q4 via the mlx-video-with-audio `generate_av` CLI."""

    _OUTPUT_FPS = 24  # mlx-video-with-audio default; LTX-2.3 native rate.

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._ready = False

    def _load(self) -> None:
        assert self.python_bin is not None
        _ensure_worker_dependencies(self.python_bin, _DEPENDENCIES)
        self._ready = True

    def handle(self, payload: dict[str, Any]) -> dict[str, Any]:
        inputs = _request_inputs(payload)
        prompt = str(inputs.get("prompt", "")).strip()
        if not prompt:
            return self._error(error_code="video_prompt_missing", message="prompt is required")
        workspace_root = _resolve_service_workspace_root(payload)
        output_path = _resolve_workspace_path(
            workspace_root, str(inputs.get("output_path", "")).strip(), expect_exists=False,
        )
        image_path_text = str(inputs.get("image_path", "")).strip()
        image_path = (
            _resolve_workspace_path(workspace_root, image_path_text, expect_exists=True)
            if image_path_text
            else None
        )

        width, height = _parse_size(str(inputs.get("size", "")).strip() or "832x480")
        num_frames = _parse_int(inputs.get("num_frames"), default=65, minimum=1)
        num_inference_steps = _parse_int(inputs.get("num_inference_steps"), default=30, minimum=1)
        cfg_scale = _parse_float(inputs.get("guidance_scale"), default=3.0, minimum=0.0)
        seed = _parse_int(inputs.get("seed"), default=42, minimum=0)
        image_strength = _parse_float(inputs.get("image_strength"), default=1.0, minimum=0.0)
        image_frame_idx = _parse_int(inputs.get("image_frame_idx"), default=0, minimum=0)

        try:
            if not self._ready:
                self._load()
            assert self.model_root is not None
            assert self.python_bin is not None
            cmd = [
                str(self.python_bin),
                "-m", "mlx_video.generate_av",
                "--prompt", prompt,
                "--height", str(height),
                "--width", str(width),
                "--num-frames", str(num_frames),
                "--num-inference-steps", str(num_inference_steps),
                "--cfg-scale", str(cfg_scale),
                "--seed", str(seed),
                "--fps", str(self._OUTPUT_FPS),
                "--model-repo", str(self.model_root),
                "--output-path", str(output_path),
                "--no-audio",
            ]
            if image_path is not None:
                cmd.extend([
                    "--image", str(image_path),
                    "--image-strength", str(image_strength),
                    "--image-frame-idx", str(image_frame_idx),
                ])
            env = os.environ.copy()
            # The runner defaults to mlx-community/gemma-3-12b-it-bf16 as the text encoder and
            # downloads it lazily on first use via huggingface_hub. Point HF_HOME at the service
            # models dir so that download lands next to the other cached weights.
            assert self.cache_root is not None
            env.setdefault("HF_HOME", str(self.cache_root))
            env.setdefault("HF_HUB_CACHE", str(self.cache_root))
            completed = subprocess.run(
                cmd, env=env,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                text=True, check=False,
            )
            if completed.returncode != 0:
                detail = (completed.stderr or completed.stdout or "").strip()
                raise RuntimeError(detail or "mlx_video.generate_av exited non-zero")
            if not output_path.exists():
                raise RuntimeError(f"mlx_video.generate_av did not produce {output_path}")
        except Exception as exc:
            return self._error(error_code="video_runtime_error", message=str(exc))

        rel = str(output_path.relative_to(workspace_root))
        return self._ok(
            outputs={"output_path": rel, "fps": self._OUTPUT_FPS, "num_frames": num_frames},
            message=f"generated video at {rel}",
        )


def create_adapter(**kwargs):
    return _MLXVideoBackend(**kwargs)
