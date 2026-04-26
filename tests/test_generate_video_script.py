"""Tests for the generate-video inference handler."""

from __future__ import annotations

import importlib.util
import json
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
SCRIPT_PATH = (
    ROOT
    / "helix"
    / "builtin_skills"
    / "generate-video"
    / "scripts"
    / "generate_video.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location("generate_video_script", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


script = _load_module()


class _FakeResponse:
    def __init__(self, body: bytes, *, status: int = 200) -> None:
        self._body = body
        self.status = status

    def read(self, _size: int = -1) -> bytes:
        return self._body

    def __enter__(self) -> "_FakeResponse":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None


def test_generate_video_text_only_success(monkeypatch):
    with tempfile.TemporaryDirectory() as td:
        workspace = Path(td)
        monkeypatch.chdir(workspace)
        monkeypatch.setenv("HELIX_LOCAL_MODEL_SERVICE_URL", "http://local-model.example")
        monkeypatch.setenv("HELIX_LOCAL_MODEL_SERVICE_TOKEN", "secret-token")

        def fake_urlopen(req, timeout=0):
            assert timeout == 1200
            payload = json.loads(req.data.decode("utf-8"))
            assert payload["task_type"] == "text_to_video"
            assert payload["request_timeout_seconds"] == 1200
            assert payload["inputs"]["prompt"] == "A cinematic beach scene"
            assert payload["inputs"]["size"] == "704x512"
            assert payload["inputs"]["output_path"] == "generated/video.mp4"
            assert payload["inputs"]["num_frames"] == 161
            assert payload["inputs"]["num_inference_steps"] == 50
            assert payload["inputs"]["guidance_scale"] == 3.0
            assert payload["inputs"]["seed"] == 42
            assert "image_path" not in payload["inputs"]
            return _FakeResponse(
                json.dumps(
                    {
                        "status": "ok",
                        "task_type": "text_to_video",
                        "backend": "mlx",
                        "model_id": "builtin.generate-video.ltx23-mlx",
                        "outputs": {
                            "output_path": "generated/video.mp4",
                            "fps": 25,
                            "num_frames": 161,
                        },
                        "error_code": "",
                        "message": "generated video at generated/video.mp4",
                    }
                ).encode("utf-8")
            )

        monkeypatch.setattr(script, "urlopen", fake_urlopen)

        out, code = script.run(
            type(
                "Args",
                (),
                {
                    "prompt": "A cinematic beach scene",
                    "image_path": "",
                    "image_strength": 1.0,
                    "image_frame_idx": 0,
                    "size": "704x512",
                    "num_frames": 161,
                    "num_inference_steps": 50,
                    "guidance_scale": 3.0,
                    "seed": 42,
                    "output_path": "generated/video.mp4",
                    "output_dir": "",
                    "timeout": 1200,
                },
            )()
        )

        assert code == 0
        assert out["status"] == "ok"
        assert out["phase"] == "generate"
        assert out["task_type"] == "text_to_video"
        assert out["output_path"] == "generated/video.mp4"
        assert out["fps"] == 25
        assert out["num_frames"] == 161
        assert out["model_used"] == "notapalindrome/ltx23-mlx-av-q4"


def test_generate_video_image_conditioned_success(monkeypatch):
    with tempfile.TemporaryDirectory() as td:
        workspace = Path(td)
        image_path = workspace / "assets" / "frame.png"
        image_path.parent.mkdir(parents=True, exist_ok=True)
        image_path.write_bytes(b"png")
        monkeypatch.chdir(workspace)
        monkeypatch.setenv("HELIX_LOCAL_MODEL_SERVICE_URL", "http://local-model.example")
        monkeypatch.setenv("HELIX_LOCAL_MODEL_SERVICE_TOKEN", "secret-token")

        def fake_urlopen(req, timeout=0):
            payload = json.loads(req.data.decode("utf-8"))
            assert payload["task_type"] == "text_image_to_video"
            assert payload["request_timeout_seconds"] == 1200
            assert payload["inputs"]["image_path"] == "assets/frame.png"
            assert payload["inputs"]["image_strength"] == 0.85
            assert payload["inputs"]["image_frame_idx"] == 0
            return _FakeResponse(
                json.dumps(
                    {
                        "status": "ok",
                        "task_type": "text_image_to_video",
                        "backend": "mlx",
                        "model_id": "builtin.generate-video.ltx23-mlx",
                        "outputs": {
                            "output_path": "generated/conditioned.mp4",
                            "fps": 25,
                            "num_frames": 121,
                        },
                        "error_code": "",
                        "message": "generated video at generated/conditioned.mp4",
                    }
                ).encode("utf-8")
            )

        monkeypatch.setattr(script, "urlopen", fake_urlopen)

        out, code = script.run(
            type(
                "Args",
                (),
                {
                    "prompt": "Animate this frame gently.",
                    "image_path": "assets/frame.png",
                    "image_strength": 0.85,
                    "image_frame_idx": 0,
                    "size": "704x512",
                    "num_frames": 121,
                    "num_inference_steps": 50,
                    "guidance_scale": 3.0,
                    "seed": 42,
                    "output_path": "generated/conditioned.mp4",
                    "output_dir": "",
                    "timeout": 1200,
                },
            )()
        )

        assert code == 0
        assert out["status"] == "ok"
        assert out["task_type"] == "text_image_to_video"
        assert out["image_path"] == "assets/frame.png"
        assert out["output_path"] == "generated/conditioned.mp4"


