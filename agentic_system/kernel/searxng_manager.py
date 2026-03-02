"""Workspace-local SearXNG lifecycle management for runtime."""

from __future__ import annotations

import json
import os
import secrets
import shlex
import signal
import subprocess
import sys
import time

from pathlib import Path
from typing import Any
from urllib.parse import urlencode, urlparse
from urllib.request import Request, urlopen


class SearxngManager:
    """Start/stop local SearXNG and keep runtime startup path simple."""

    _DEFAULT_BASE_URL = "http://127.0.0.1:8888"
    _DEFAULT_START_CMD = "python -m searx.webapp"
    _START_TIMEOUT_SECONDS = 15.0
    _MANAGED_DIRNAME = "searxng"
    _MANAGED_VENV_DIRNAME = ".venv"

    def __init__(self, workspace: Path) -> None:
        self.workspace = Path(workspace).expanduser().resolve()
        self._process: subprocess.Popen[Any] | None = None
        self._start_command: str = ""
        self._log_paths: tuple[Path, Path] | None = None

    @staticmethod
    def _note(message: str) -> None:
        print(f"runtime> {message}")

    @classmethod
    def _resolve_base_url(cls) -> str:
        raw = str(os.getenv("SEARXNG_BASE_URL", cls._DEFAULT_BASE_URL)).strip()
        return raw or cls._DEFAULT_BASE_URL

    @staticmethod
    def _is_local_loopback_url(url: str) -> bool:
        parsed = urlparse(str(url).strip())
        host = (parsed.hostname or "").strip().lower()
        return host in {"127.0.0.1", "localhost"}

    @staticmethod
    def _is_healthy(base_url: str, timeout_seconds: float = 2.0) -> bool:
        base = str(base_url or "").strip().rstrip("/")
        if not base:
            return False
        query = urlencode({"q": "healthcheck", "format": "json", "language": "en-US"})
        endpoint = f"{base}/search?{query}"
        req = Request(
            endpoint,
            headers={
                "User-Agent": "AgenticSystemRuntime/1.0",
                "Accept": "application/json",
                "Accept-Language": "en-US,en;q=0.9",
            },
        )
        try:
            with urlopen(req, timeout=max(0.1, float(timeout_seconds))) as resp:
                status = int(getattr(resp, "status", 200) or 200)
                if status >= 400:
                    return False
                body = resp.read(256_000).decode("utf-8", errors="replace")
        except Exception:
            return False
        try:
            parsed = json.loads(body)
        except Exception:
            return False
        return isinstance(parsed, dict) and isinstance(parsed.get("results", []), list)

    @classmethod
    def _resolve_start_command(cls) -> list[str]:
        raw = str(os.getenv("SEARXNG_START_CMD", cls._DEFAULT_START_CMD)).strip()
        if not raw:
            raw = cls._DEFAULT_START_CMD
        try:
            parsed = [part for part in shlex.split(raw) if part.strip()]
        except ValueError:
            parsed = [raw]
        return parsed if parsed else [cls._DEFAULT_START_CMD]

    def _managed_root(self) -> Path:
        return self.workspace / ".runtime" / self._MANAGED_DIRNAME

    def _managed_settings_path(self) -> Path:
        return self._managed_root() / "local_settings.yml"

    def _managed_python_path(self) -> Path:
        venv_dir = self._managed_root() / self._MANAGED_VENV_DIRNAME
        if os.name == "nt":
            return venv_dir / "Scripts" / "python.exe"
        return venv_dir / "bin" / "python"

    @staticmethod
    def _render_settings_yaml(secret_key: str) -> str:
        return "\n".join(
            [
                "use_default_settings: true",
                "",
                "general:",
                '  instance_name: "Agentic System Local SearXNG"',
                "",
                "search:",
                "  formats:",
                "    - html",
                "    - json",
                "",
                "server:",
                f'  secret_key: "{secret_key}"',
                "  limiter: false",
                "  image_proxy: false",
                "",
            ]
        )

    def _ensure_managed_settings(self) -> Path:
        settings_path = self._managed_settings_path()
        settings_path.parent.mkdir(parents=True, exist_ok=True)
        if settings_path.exists():
            return settings_path
        secret_key = secrets.token_urlsafe(32)
        settings_path.write_text(self._render_settings_yaml(secret_key), encoding="utf-8")
        return settings_path

    def _run_checked_subprocess(
        self,
        command: list[str],
        *,
        env: dict[str, str] | None = None,
    ) -> None:
        completed = subprocess.run(
            command,
            cwd=str(self.workspace),
            env=dict(env if env is not None else os.environ),
            capture_output=True,
            text=True,
        )
        if completed.returncode == 0:
            return
        stderr_tail = (completed.stderr or "").strip()[-1200:]
        stdout_tail = (completed.stdout or "").strip()[-1200:]
        details: list[str] = [f"exit_code={completed.returncode}"]
        if stdout_tail:
            details.append(f"stdout_tail={stdout_tail}")
        if stderr_tail:
            details.append(f"stderr_tail={stderr_tail}")
        command_text = " ".join(shlex.quote(part) for part in command)
        raise RuntimeError(f"command failed: {command_text}; {'; '.join(details)}")

    def _python_has_searx_module(self, python_path: Path, *, env: dict[str, str] | None = None) -> bool:
        if not python_path.exists():
            return False
        command = [
            str(python_path),
            "-c",
            "import importlib.util,sys;sys.exit(0 if importlib.util.find_spec('searx') else 1)",
        ]
        completed = subprocess.run(
            command,
            cwd=str(self.workspace),
            env=dict(env if env is not None else os.environ),
            capture_output=True,
            text=True,
        )
        return completed.returncode == 0

    def _install_managed_searxng(self, python_path: Path, *, env: dict[str, str]) -> None:
        self._note("installing workspace-local searxng (no approval required)...")
        install_commands = [
            [
                str(python_path),
                "-m",
                "pip",
                "install",
                "-U",
                "pip",
                "setuptools",
                "wheel",
                "pyyaml",
                "msgspec",
            ],
            [
                str(python_path),
                "-m",
                "pip",
                "install",
                "--use-pep517",
                "--no-build-isolation",
                "git+https://github.com/searxng/searxng.git",
            ],
        ]
        for command in install_commands:
            self._run_checked_subprocess(command, env=env)

    def _ensure_managed_command(self) -> tuple[list[str], dict[str, str]] | None:
        settings_path = self._ensure_managed_settings()
        launch_env = dict(os.environ)
        launch_env["SEARXNG_SETTINGS_PATH"] = str(settings_path)

        python_path = self._managed_python_path()
        if not python_path.exists():
            self._note("creating workspace-local searxng runtime environment...")
            venv_dir = python_path.parent.parent
            self._run_checked_subprocess([sys.executable, "-m", "venv", str(venv_dir)], env=launch_env)

        if not self._python_has_searx_module(python_path, env=launch_env):
            self._install_managed_searxng(python_path, env=launch_env)

        if not self._python_has_searx_module(python_path, env=launch_env):
            self._note("workspace-local searxng install completed but module is still unavailable.")
            return None

        return [str(python_path), "-m", "searx.webapp"], launch_env

    def _resolve_launch_plan(self) -> tuple[list[str], dict[str, str]] | None:
        raw_start_command = str(os.getenv("SEARXNG_START_CMD", "")).strip()
        if raw_start_command:
            return self._resolve_start_command(), dict(os.environ)

        managed = self._ensure_managed_command()
        if managed is not None:
            return managed

        return self._resolve_start_command(), dict(os.environ)

    @staticmethod
    def _terminate_process(process: subprocess.Popen[Any]) -> None:
        if process.poll() is not None:
            return
        try:
            if os.name == "nt":
                process.terminate()
            else:
                os.killpg(process.pid, signal.SIGTERM)
            process.wait(timeout=2.0)
            return
        except Exception:
            pass
        try:
            if os.name == "nt":
                process.kill()
            else:
                os.killpg(process.pid, signal.SIGKILL)
        except Exception:
            pass

    def _wait_ready(
        self,
        *,
        base_url: str,
        process: subprocess.Popen[Any],
        timeout_seconds: float,
    ) -> bool:
        deadline = time.time() + max(0.5, float(timeout_seconds))
        while time.time() < deadline:
            if self._is_healthy(base_url, timeout_seconds=1.0):
                return True
            if process.poll() is not None:
                return False
            time.sleep(0.35)
        return self._is_healthy(base_url, timeout_seconds=1.5)

    def _start_native(self, base_url: str, command: list[str], launch_env: dict[str, str]) -> bool:
        logs_dir = self.workspace / ".runtime" / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)
        stdout_path = logs_dir / "searxng_stdout.log"
        stderr_path = logs_dir / "searxng_stderr.log"
        self._log_paths = (stdout_path, stderr_path)
        try:
            with stdout_path.open("a", encoding="utf-8") as stdout_handle, stderr_path.open(
                "a", encoding="utf-8"
            ) as stderr_handle:
                process = subprocess.Popen(
                    command,
                    cwd=str(self.workspace),
                    stdout=stdout_handle,
                    stderr=stderr_handle,
                    start_new_session=True,
                    env=dict(launch_env),
                )
        except OSError:
            self._log_paths = None
            return False

        if self._wait_ready(base_url=base_url, process=process, timeout_seconds=self._START_TIMEOUT_SECONDS):
            self._process = process
            self._start_command = " ".join(shlex.quote(part) for part in command)
            return True
        self._terminate_process(process)
        return False

    def ensure_backend(self) -> None:
        base_url = self._resolve_base_url()
        if self._is_healthy(base_url):
            return
        if not self._is_local_loopback_url(base_url):
            self._note(
                "searxng endpoint unreachable and not local loopback; "
                f"set SEARXNG_BASE_URL to a reachable endpoint (current: {base_url})"
            )
            return
        launch_plan = self._resolve_launch_plan()
        if launch_plan is None:
            self._note("unable to resolve a runnable searxng launch plan.")
            return
        command, launch_env = launch_plan
        if self._start_native(base_url, command, launch_env):
            self._note(
                "started native searxng in background "
                f"at {base_url} using command: {self._start_command}"
            )
            return
        command_preview = " ".join(command)
        self._note(
            "unable to auto-start native searxng. "
            f"tried command: {command_preview}. "
            "Set SEARXNG_START_CMD to a valid command if you want to override runtime auto-bootstrap."
        )

    def shutdown(self) -> None:
        process = self._process
        self._process = None
        self._start_command = ""
        if process is not None:
            self._terminate_process(process)

        log_paths = self._log_paths
        self._log_paths = None
        if not log_paths:
            return
        for path in log_paths:
            try:
                path.unlink(missing_ok=True)
            except OSError:
                pass
        log_dir = log_paths[0].parent
        try:
            if log_dir.exists() and not any(log_dir.iterdir()):
                log_dir.rmdir()
        except OSError:
            pass
