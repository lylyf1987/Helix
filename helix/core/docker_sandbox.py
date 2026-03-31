"""Docker-backed sandbox executor for exec actions."""

from __future__ import annotations

import hashlib
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlparse, urlunparse

from helix.core.sandbox import _format_output_block, _normalize_exec_input
from helix.core.state import Turn


_DOCKER_BUILD_ROOT = Path(__file__).resolve().parent.parent / "runtime" / "docker"
_SANDBOX_DOCKERFILE = _DOCKER_BUILD_ROOT / "exec-sandbox.Dockerfile"
_SANDBOX_ENTRYPOINT = _DOCKER_BUILD_ROOT / "helix_exec.sh"
_SEARXNG_IMAGE = "docker.io/searxng/searxng:latest"
_DOCKER_INFO_TIMEOUT = 5
_DOCKER_BUILD_TIMEOUT = int(os.environ.get("AGENTIC_DOCKER_BUILD_TIMEOUT", "1800"))
_DOCKER_TIMEOUT = int(os.environ.get("AGENTIC_DOCKER_SANDBOX_TIMEOUT", "600"))
_DOCKER_MEMORY = os.environ.get("AGENTIC_DOCKER_SANDBOX_MEMORY", "2g")
_DOCKER_CPUS = os.environ.get("AGENTIC_DOCKER_SANDBOX_CPUS", "2.0")
_DOCKER_PIDS = os.environ.get("AGENTIC_DOCKER_SANDBOX_PIDS", "256")
_SEARXNG_READY_TIMEOUT = int(os.environ.get("AGENTIC_DOCKER_SEARXNG_READY_TIMEOUT", "30"))
_SEARXNG_READY_POLL = float(os.environ.get("AGENTIC_DOCKER_SEARXNG_READY_POLL", "1.0"))
_PASS_ENV_PREFIXES = (
    "IMAGE_ANALYSIS_",
    "IMAGE_GENERATION_",
    "SEARXNG_",
    "OLLAMA_",
    "DEEPSEEK_",
    "LMSTUDIO_",
    "LM_API_",
    "ZAI_",
    "OPENAI_COMPAT_",
    "OPENAI_API_KEY",
)


def docker_is_available() -> tuple[bool, str]:
    """Return whether Docker is usable from this runtime."""
    try:
        completed = subprocess.run(
            ["docker", "info"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
            timeout=_DOCKER_INFO_TIMEOUT,
            check=False,
        )
    except FileNotFoundError:
        return False, "docker CLI not found"
    except subprocess.TimeoutExpired:
        return False, "docker info timed out"

    if completed.returncode == 0:
        return True, ""
    detail = (completed.stderr or "").strip() or "docker info failed"
    return False, detail


def _hash_directory(root: Path) -> str:
    """Hash all files under a directory to derive a content-addressed image tag."""
    digest = hashlib.sha256()
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        relative = path.relative_to(root).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()[:12]


def _workspace_slug(workspace: Path) -> str:
    return hashlib.sha256(str(workspace).encode("utf-8")).hexdigest()[:10]


def _dockerize_loopback_url(url: str) -> str:
    """Translate host-loopback URLs into a Docker-reachable hostname."""
    candidate = str(url).strip()
    if not candidate:
        return candidate

    parsed = urlparse(candidate)
    hostname = (parsed.hostname or "").strip().lower()
    if hostname not in {"127.0.0.1", "0.0.0.0", "localhost", "::1"}:
        return candidate

    netloc = "host.docker.internal"
    if parsed.port is not None:
        netloc = f"{netloc}:{parsed.port}"
    return urlunparse(parsed._replace(netloc=netloc))


def _write_searxng_settings(config_dir: Path) -> None:
    config_dir.mkdir(parents=True, exist_ok=True)
    settings = "\n".join([
        "use_default_settings: true",
        "",
        "server:",
        '  secret_key: "helix-docker-sandbox"',
        "  limiter: false",
        "",
        "search:",
        "  safe_search: 0",
        "  formats:",
        "    - html",
        "    - json",
        "",
    ])
    (config_dir / "settings.yml").write_text(settings, encoding="utf-8")


def _collect_logged_result(
    *,
    process: subprocess.Popen[Any],
    stdout_path: Path,
    stderr_path: Path,
    extra_stderr: str = "",
) -> dict[str, Any]:
    """Collect stdout/stderr from temp log files and remove them."""
    if process.poll() is None:
        process.wait()

    stdout = ""
    stderr = ""
    if stdout_path.exists():
        stdout = stdout_path.read_text(encoding="utf-8", errors="replace")
        stdout_path.unlink(missing_ok=True)
    if stderr_path.exists():
        stderr = stderr_path.read_text(encoding="utf-8", errors="replace")
        stderr_path.unlink(missing_ok=True)

    if extra_stderr:
        if stderr and not stderr.endswith("\n"):
            stderr += "\n"
        stderr += extra_stderr.strip() + "\n"

    return {
        "stdout": stdout,
        "stderr": stderr,
        "return_code": int(process.returncode or 0),
    }


class DockerSandboxExecutor:
    """Callable Docker-backed executor matching ``SandboxExecutor``."""

    backend_name = "docker"

    def __init__(self, workspace: Path, *, searxng_base_url: str | None = None) -> None:
        self.workspace = Path(workspace).expanduser().resolve()
        self.slug = _workspace_slug(self.workspace)
        self.image_tag = f"helix-sandbox:{_hash_directory(_DOCKER_BUILD_ROOT)}"
        self.network_name = f"helix-sandbox-net-{self.slug}"
        self.cache_volume = f"helix-sandbox-cache-{self.slug}"
        self.searxng_name = f"helix-searxng-{self.slug}"
        self.searxng_config_dir = self.workspace / ".runtime" / "docker" / "searxng" / "config"
        self.searxng_data_dir = self.workspace / ".runtime" / "docker" / "searxng" / "data"
        requested = str(searxng_base_url or "").strip()
        self._managed_searxng = not requested
        self._requested_searxng_base_url = requested
        self._effective_searxng_base_url = (
            f"http://{self.searxng_name}:8080"
            if self._managed_searxng
            else _dockerize_loopback_url(requested)
        )
        self.approval_profile = f"docker-online-rw-workspace-v1:{self.image_tag}"

    def status_fields(self) -> dict[str, str]:
        return {
            "sandbox_backend": self.backend_name,
            "sandbox_profile": self.approval_profile,
            "docker_image": self.image_tag,
            "docker_network": self.network_name,
            "docker_searxng": self._effective_searxng_base_url,
        }

    def tool_environment(self) -> dict[str, str]:
        return {"SEARXNG_BASE_URL": self._effective_searxng_base_url}

    def _run_docker(
        self,
        args: list[str],
        *,
        check: bool = True,
        timeout: int = _DOCKER_BUILD_TIMEOUT,
    ) -> subprocess.CompletedProcess[str]:
        completed = subprocess.run(
            ["docker", *args],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout,
            check=False,
        )
        if check and completed.returncode != 0:
            detail = (completed.stderr or completed.stdout or "").strip()
            raise RuntimeError(detail or f"docker {' '.join(args)} failed")
        return completed

    def _ensure_image(self) -> None:
        inspect = self._run_docker(["image", "inspect", self.image_tag], check=False)
        if inspect.returncode == 0:
            return

        self._run_docker(
            [
                "build",
                "-t",
                self.image_tag,
                "-f",
                str(_SANDBOX_DOCKERFILE),
                str(_DOCKER_BUILD_ROOT),
            ],
            timeout=_DOCKER_BUILD_TIMEOUT,
        )

    def _ensure_network(self) -> None:
        inspect = self._run_docker(["network", "inspect", self.network_name], check=False)
        if inspect.returncode == 0:
            return
        self._run_docker(["network", "create", self.network_name])

    def _ensure_cache_volume(self) -> None:
        inspect = self._run_docker(["volume", "inspect", self.cache_volume], check=False)
        if inspect.returncode != 0:
            self._run_docker(["volume", "create", self.cache_volume])

        uid = str(os.getuid())
        gid = str(os.getgid())
        self._run_docker(
            [
                "run",
                "--rm",
                "-v",
                f"{self.cache_volume}:/helix-cache",
                self.image_tag,
                "bash",
                "-lc",
                (
                    "mkdir -p "
                    "/helix-cache/home "
                    "/helix-cache/pip "
                    "/helix-cache/npm "
                    "/helix-cache/npm-global "
                    "/helix-cache/venv "
                    "&& chown -R "
                    f"{uid}:{gid} /helix-cache"
                ),
            ],
        )

    def _ensure_searxng_service(self) -> None:
        if not self._managed_searxng:
            return

        self.searxng_config_dir.mkdir(parents=True, exist_ok=True)
        self.searxng_data_dir.mkdir(parents=True, exist_ok=True)
        _write_searxng_settings(self.searxng_config_dir)

        inspect = self._run_docker(
            ["inspect", "-f", "{{.State.Running}}", self.searxng_name],
            check=False,
        )
        if inspect.returncode == 0 and inspect.stdout.strip() == "true":
            self._wait_for_searxng_ready()
            return
        if inspect.returncode == 0:
            self._run_docker(["rm", "-f", self.searxng_name], check=False)

        self._run_docker(
            [
                "run",
                "-d",
                "--name",
                self.searxng_name,
                "--restart",
                "unless-stopped",
                "--network",
                self.network_name,
                "-v",
                f"{self.searxng_config_dir}:/etc/searxng",
                "-v",
                f"{self.searxng_data_dir}:/var/cache/searxng",
                _SEARXNG_IMAGE,
            ],
            timeout=_DOCKER_BUILD_TIMEOUT,
        )
        self._wait_for_searxng_ready()

    def _wait_for_searxng_ready(self) -> None:
        deadline = time.time() + max(1, _SEARXNG_READY_TIMEOUT)
        probe = (
            "from urllib.request import urlopen\n"
            f"urlopen('{self._effective_searxng_base_url.rstrip('/')}/search?q=test&format=json', timeout=5).read(64)\n"
            "print('ready')\n"
        )
        last_error = "searxng readiness probe did not return success"
        while time.time() < deadline:
            completed = self._run_docker(
                [
                    "run",
                    "--rm",
                    "--network",
                    self.network_name,
                    self.image_tag,
                    "python",
                    "-c",
                    probe,
                ],
                check=False,
                timeout=15,
            )
            if completed.returncode == 0:
                return
            detail = (completed.stderr or completed.stdout or "").strip()
            if detail:
                last_error = detail
            time.sleep(max(0.1, _SEARXNG_READY_POLL))
        raise RuntimeError(f"SearXNG did not become ready: {last_error}")

    @staticmethod
    def _payload_uses_searxng(payload: dict[str, Any]) -> bool:
        script_path = str(payload.get("script_path", "") or "")
        script = str(payload.get("script", "") or "")
        script_args = str(payload.get("script_args", "") or "")
        joined = "\n".join([script_path, script, script_args]).lower()
        return (
            "search-online-context" in joined
            or "search_searxng.py" in joined
            or "search_and_fetch.py" in joined
            or "fetch_pages.py" in joined
            or "searxng" in joined
        )

    def _ensure_support_services(self, payload: dict[str, Any]) -> None:
        self._ensure_image()
        self._ensure_network()
        self._ensure_cache_volume()
        if self._payload_uses_searxng(payload):
            self._ensure_searxng_service()

    def _build_container_environment(self, workspace_root: Path) -> dict[str, str]:
        tmpdir = workspace_root / ".runtime" / "tmp"
        tmpdir.mkdir(parents=True, exist_ok=True)
        env = {
            "HELIX_CACHE_ROOT": "/helix-cache",
            "HOME": "/helix-cache/home",
            "PIP_CACHE_DIR": "/helix-cache/pip",
            "NPM_CONFIG_CACHE": "/helix-cache/npm",
            "NPM_CONFIG_PREFIX": "/helix-cache/npm-global",
            "PIP_DISABLE_PIP_VERSION_CHECK": "1",
            "TMPDIR": str(tmpdir),
            "TEMP": str(tmpdir),
            "TMP": str(tmpdir),
            "CHROME_BIN": "/usr/bin/chromium",
            "CHROMEDRIVER": "/usr/bin/chromedriver",
            "SEARXNG_BASE_URL": self._effective_searxng_base_url,
        }
        for key, value in os.environ.items():
            if not value:
                continue
            if not key.startswith(_PASS_ENV_PREFIXES):
                continue
            if key.endswith("_BASE_URL") or key == "SEARXNG_BASE_URL":
                env[key] = _dockerize_loopback_url(value)
            else:
                env[key] = value
        env["SEARXNG_BASE_URL"] = self._effective_searxng_base_url
        return env

    @staticmethod
    def _build_container_command(
        code_type: str,
        has_path: bool,
        path_value: str,
        script_value: str,
        args_value: list[str],
    ) -> list[str]:
        if code_type == "python":
            if has_path:
                return ["python", path_value, *args_value]
            return ["python", "-c", script_value]
        if code_type == "bash":
            if has_path:
                return ["bash", path_value, *args_value]
            # Avoid login-shell PATH resets so the entrypoint-managed venv
            # remains the active Python toolchain for all execs.
            return ["bash", "-c", script_value]
        raise ValueError(f"Unsupported code_type: {code_type}")

    def _remove_container(self, name: str) -> None:
        self._run_docker(["rm", "-f", name], check=False, timeout=30)

    def __call__(self, payload: dict, workspace: Path) -> Turn:
        workspace_root = Path(workspace).expanduser().resolve()
        timeout = payload.get("timeout_seconds", _DOCKER_TIMEOUT)
        try:
            timeout_seconds = int(timeout)
        except (TypeError, ValueError):
            timeout_seconds = _DOCKER_TIMEOUT
        job_name = str(payload.get("job_name", "unnamed_job")).strip() or "unnamed_job"

        try:
            self._ensure_support_services(payload)
            code_type, has_path, path_value, script_value, args_value = _normalize_exec_input(payload)
            container_command = self._build_container_command(
                code_type,
                has_path,
                path_value,
                script_value,
                args_value,
            )
        except Exception as exc:
            return Turn(
                role="runtime",
                content=f"Job '{job_name}' failed to start: {exc}",
            )

        runtime_logs = workspace_root / ".runtime" / "logs"
        runtime_logs.mkdir(parents=True, exist_ok=True)
        stdout_fd, stdout_name = tempfile.mkstemp(
            prefix=f"{job_name}_stdout_",
            suffix=".log",
            dir=str(runtime_logs),
        )
        stderr_fd, stderr_name = tempfile.mkstemp(
            prefix=f"{job_name}_stderr_",
            suffix=".log",
            dir=str(runtime_logs),
        )
        stdout_path = Path(stdout_name)
        stderr_path = Path(stderr_name)

        container_name = f"helix-exec-{self.slug}-{int(time.time() * 1000)}"
        env = self._build_container_environment(workspace_root)
        uid_gid = f"{os.getuid()}:{os.getgid()}"
        docker_args = [
            "docker",
            "run",
            "--name",
            container_name,
            "--rm",
            "--init",
            "--network",
            self.network_name,
            "--read-only",
            "--tmpfs",
            "/tmp:exec,mode=1777",
            "--tmpfs",
            "/run:mode=755",
            "--shm-size",
            "512m",
            "--cap-drop",
            "ALL",
            "--security-opt",
            "no-new-privileges",
            "--memory",
            _DOCKER_MEMORY,
            "--cpus",
            _DOCKER_CPUS,
            "--pids-limit",
            _DOCKER_PIDS,
            "--user",
            uid_gid,
            "--workdir",
            str(workspace_root),
            "--mount",
            f"type=bind,src={workspace_root},dst={workspace_root}",
            "--mount",
            f"type=volume,src={self.cache_volume},dst=/helix-cache",
        ]
        if sys.platform.startswith("linux"):
            docker_args.extend(["--add-host", "host.docker.internal:host-gateway"])
        for key, value in sorted(env.items()):
            docker_args.extend(["-e", f"{key}={value}"])
        docker_args.append(self.image_tag)
        docker_args.extend(container_command)

        stdout_file = os.fdopen(stdout_fd, "w", encoding="utf-8")
        stderr_file = os.fdopen(stderr_fd, "w", encoding="utf-8")
        try:
            process = subprocess.Popen(
                docker_args,
                cwd=str(workspace_root),
                stdout=stdout_file,
                stderr=stderr_file,
                start_new_session=True,
            )
        finally:
            stdout_file.close()
            stderr_file.close()

        try:
            process.wait(timeout=timeout_seconds)
            result = _collect_logged_result(
                process=process,
                stdout_path=stdout_path,
                stderr_path=stderr_path,
            )
        except subprocess.TimeoutExpired:
            self._remove_container(container_name)
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
            result = _collect_logged_result(
                process=process,
                stdout_path=stdout_path,
                stderr_path=stderr_path,
                extra_stderr=f"\nruntime> exec terminated after {timeout_seconds}s timeout",
            )
        except KeyboardInterrupt:
            self._remove_container(container_name)
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
            result = _collect_logged_result(
                process=process,
                stdout_path=stdout_path,
                stderr_path=stderr_path,
                extra_stderr="\nruntime> exec terminated by user (KeyboardInterrupt)",
            )

        stdout = result["stdout"]
        stderr = result["stderr"]
        rc = result["return_code"]

        status = "succeeded" if rc == 0 else "failed"
        content = f"Job '{job_name}' {status}. (Exit code: {rc})"
        if stdout:
            content += _format_output_block("stdout", stdout)
        if stderr:
            content += _format_output_block("stderr", stderr)
        return Turn(role="runtime", content=content)
