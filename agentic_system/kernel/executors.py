from __future__ import annotations

import shlex
import subprocess
import sys
from pathlib import Path


def execute(
    *,
    action_input: dict[str, object],
    workspace: str | Path,
    timeout_seconds: int | None = None,
) -> dict[str, str]:
    cwd = Path(workspace).expanduser().resolve()
    cwd.mkdir(parents=True, exist_ok=True)

    if not isinstance(action_input, dict):
        raise ValueError("exec action requires object action_input")

    code_type = str(action_input.get("code_type", "bash")).strip().lower()
    script_path = str(action_input.get("script_path", "")).strip()
    script = str(action_input.get("script", "")).strip()
    raw_script_args = action_input.get("script_args", [])
    if isinstance(raw_script_args, (list, tuple)):
        script_args = [str(arg) for arg in raw_script_args if str(arg).strip()]
    elif isinstance(raw_script_args, str):
        raw_args_text = raw_script_args.strip()
        if raw_args_text:
            try:
                script_args = [arg for arg in shlex.split(raw_args_text) if arg.strip()]
            except ValueError:
                script_args = [raw_args_text]
        else:
            script_args = []
    else:
        script_args = []

    normalized_code_type = str(code_type).strip().lower()
    path_value = str(script_path or "").strip()
    script_value = str(script or "").strip()
    args_value = [str(arg) for arg in (script_args or []) if str(arg).strip()]

    has_path = bool(path_value)
    has_script = bool(script_value)
    if has_path == has_script:
        raise ValueError("Exactly one of script_path or script must be provided")
    if has_script and args_value:
        raise ValueError("script_args is only supported when script_path is provided")

    timeout_value: int | None = None
    if timeout_seconds is not None:
        timeout_value = max(1, int(timeout_seconds))

    if normalized_code_type == "python":
        if has_path:
            result = subprocess.run(
                [sys.executable, path_value, *args_value],
                cwd=str(cwd),
                capture_output=True,
                text=True,
                timeout=timeout_value,
            )
        else:
            result = subprocess.run(
                [sys.executable, "-c", script_value],
                cwd=str(cwd),
                capture_output=True,
                text=True,
                timeout=timeout_value,
            )
    elif normalized_code_type == "bash":
        if has_path:
            result = subprocess.run(
                ["bash", path_value, *args_value],
                cwd=str(cwd),
                capture_output=True,
                text=True,
                timeout=timeout_value,
            )
        else:
            result = subprocess.run(
                script_value,
                cwd=str(cwd),
                shell=True,
                capture_output=True,
                text=True,
                timeout=timeout_value,
            )
    else:
        raise ValueError(f"Unsupported code_type: {code_type}")

    return {
        "stdout": result.stdout,
        "stderr": result.stderr,
    }
