"""Interactive runtime host for the agent loop and session state."""

from __future__ import annotations

import os
import shutil

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from prompt_toolkit import PromptSession
from prompt_toolkit.key_binding import KeyBindings

from .kernel import (
    FlowEngine,
    ModelRouter,
    PromptEngine,
    SearxngManager,
    StorageEngine,
    prompt_auto_write_override,
    prompt_exec_approval,
)


@dataclass(frozen=True)
class RuntimeConfig:
    """Normalized runtime options from CLI inputs."""

    provider: str
    mode: str
    model_name: str | None
    image_analysis_provider: str
    image_analysis_model: str
    image_generation_provider: str
    image_generation_model: str

    @classmethod
    def from_inputs(
        cls,
        *,
        provider: str,
        mode: str,
        model_name: str | None,
        image_analysis_provider: str | None,
        image_analysis_model: str | None,
        image_generation_provider: str | None,
        image_generation_model: str | None,
    ) -> RuntimeConfig:
        def _norm(value: str | None) -> str:
            return str(value).strip() if value is not None else ""

        return cls(
            provider=_norm(provider).lower() or "ollama",
            mode=_norm(mode).lower() or "controlled",
            model_name=model_name,
            image_analysis_provider=_norm(image_analysis_provider) or "none",
            image_analysis_model=_norm(image_analysis_model) or "none",
            image_generation_provider=_norm(image_generation_provider) or "none",
            image_generation_model=_norm(image_generation_model) or "none",
        )


class AgentRuntime:
    """Own runtime lifecycle: bootstrap assets, read user input, and drive loop."""

    _CMD_EXIT = "__EXIT__"
    _CMD_REFRESH = "__REFRESH__"
    _DEFAULT_SEARXNG_BASE_URL = "http://127.0.0.1:8888"
    _PROMPT_CONTINUATION = "... "
    _TOKEN_WINDOW_LIMIT = 70000
    _HELP_TEXT = "\n".join(
        [
            "Commands:",
            "  /help            Show help.",
            "  /status          Show runtime status overview.",
            "  /status workflow_summary   Show workflow_summary.",
            "  /status workflow_hist      Show workflow_hist lines.",
            "  /status full_proc_hist     Show full_proc_hist lines.",
            "  /status action_hist        Show LLM selected action history.",
            "  /status core_agent_prompt  Show the last full prompt sent to core_agent.",
            "  /refresh         Start a new session in current workspace.",
            "  /exit            Quit.",
        ]
    )

    def __init__(
        self,
        workspace: str | Path,
        provider: str = "ollama",
        mode: str = "controlled",
        session_id: str | None = None,
        model_name: str | None = None,
        image_analysis_provider: str = "none",
        image_analysis_model: str = "none",
        image_generation_provider: str = "none",
        image_generation_model: str = "none",
    ) -> None:
        self.workspace = Path(workspace).expanduser().resolve()
        self.workspace.mkdir(parents=True, exist_ok=True)
        self.packaged_prompts_root = Path(__file__).resolve().parent / "prompts"
        self.packaged_skills_root = Path(__file__).resolve().parent.parent / "skills"

        self.config = RuntimeConfig.from_inputs(
            provider=provider,
            mode=mode,
            model_name=model_name,
            image_analysis_provider=image_analysis_provider,
            image_analysis_model=image_analysis_model,
            image_generation_provider=image_generation_provider,
            image_generation_model=image_generation_model,
        )
        # Backward-compatible direct attributes used by status output and tests.
        self.provider = self.config.provider
        self.mode = self.config.mode
        self.image_analysis_provider = self.config.image_analysis_provider
        self.image_analysis_model = self.config.image_analysis_model
        self.image_generation_provider = self.config.image_generation_provider
        self.image_generation_model = self.config.image_generation_model

        self._prompt_session = self._build_prompt_session()
        self._command_handlers: dict[str, Callable[[str], str]] = {
            "/help": self._cmd_help,
            "/status": self._cmd_status,
            "/refresh": self._cmd_refresh,
            "/exit": self._cmd_exit,
        }

        self._configure_skill_provider_environment()
        self._searxng = SearxngManager(workspace=self.workspace)
        self._initialize_kernel(session_id=session_id)
        self._persist()

    def _build_prompt_session(self) -> PromptSession:
        """Create multiline prompt-toolkit session (Ctrl+D submits)."""
        bindings = KeyBindings()

        @bindings.add("c-d")
        def _submit(event: Any) -> None:
            event.app.exit(result=event.app.current_buffer.text)

        return PromptSession(key_bindings=bindings)

    def _initialize_kernel(self, *, session_id: str | None) -> None:
        """Initialize storage, model routing, prompting, and orchestration engines."""
        self.state = StorageEngine(workspace=self.workspace, session_id=session_id)
        if session_id is not None:
            self.state.load_state()
        self.model_router = ModelRouter(provider=self.config.provider, model_name=self.config.model_name)
        self.prompt_engine = PromptEngine(
            workspace=self.workspace,
            token_window_limit=self._TOKEN_WINDOW_LIMIT,
            compact_keep_last_k=10,
        )
        self.engine = FlowEngine(
            workspace=self.workspace,
            mode=self.config.mode,
            model_router=self.model_router,
            prompt_engine=self.prompt_engine,
            approval_handler=prompt_exec_approval,
            write_policy_handler=prompt_auto_write_override,
        )

    def _configure_skill_provider_environment(self) -> None:
        """Expose image skill provider/model choices via environment variables."""
        os.environ["IMAGE_ANALYSIS_PROVIDER"] = self.config.image_analysis_provider
        os.environ["IMAGE_ANALYSIS_MODEL"] = self.config.image_analysis_model
        os.environ["IMAGE_GENERATION_PROVIDER"] = self.config.image_generation_provider
        os.environ["IMAGE_GENERATION_MODEL"] = self.config.image_generation_model
        os.environ.setdefault("SEARXNG_BASE_URL", self._DEFAULT_SEARXNG_BASE_URL)

    def _bootstrap_runtime_assets(self) -> None:
        """Sync packaged prompts/skills into the runtime workspace."""
        runtime_prompts_root = self.workspace / "prompts"
        runtime_skills_root = self.workspace / "skills"
        runtime_prompts_root.mkdir(parents=True, exist_ok=True)
        runtime_skills_root.mkdir(parents=True, exist_ok=True)
        self._copy_packaged_prompts(runtime_prompts_root)
        self._copy_packaged_skills(runtime_skills_root)

    def _copy_packaged_prompts(self, runtime_prompts_root: Path) -> None:
        """Copy runtime prompt templates from package into workspace."""
        for file_name in ("agent_system_prompt.json", "agent_role_description.json"):
            source = self.packaged_prompts_root / file_name
            target = runtime_prompts_root / file_name
            if source.exists():
                shutil.copy2(source, target)

    def _copy_packaged_skills(self, runtime_skills_root: Path) -> None:
        """Replace runtime skill folders with packaged built-ins on each start."""
        for scope in ("core-agent", "all-agents"):
            source_scope = self.packaged_skills_root / scope
            target_scope = runtime_skills_root / scope
            target_scope.mkdir(parents=True, exist_ok=True)
            if not source_scope.exists():
                continue
            for skill_dir in sorted(path for path in source_scope.iterdir() if path.is_dir()):
                target_dir = target_scope / skill_dir.name
                if target_dir.exists():
                    if target_dir.is_dir():
                        shutil.rmtree(target_dir)
                    else:
                        target_dir.unlink()
                shutil.copytree(skill_dir, target_dir)

    @staticmethod
    def _render_status_value(value: Any) -> str:
        if isinstance(value, list):
            if not value:
                return "(empty)"
            return "\n".join(str(line) for line in value)
        if isinstance(value, str):
            return value if value.strip() else "(empty)"
        return "(empty)"

    def _status_overview_text(self) -> str:
        """Build compact runtime/session overview for `/status`."""
        return "\n".join(
            [
                f"session_id={self.state.session_id}",
                f"provider={self.provider}",
                f"image_analysis_provider={self.image_analysis_provider}",
                f"image_analysis_model={self.image_analysis_model}",
                f"image_generation_provider={self.image_generation_provider}",
                f"image_generation_model={self.image_generation_model}",
                f"mode={self.mode}",
                f"full_proc_hist_lines={len(self.state.full_proc_hist)}",
                f"workflow_hist_lines={len(self.state.workflow_hist)}",
                f"action_hist_lines={len(getattr(self.state, 'action_hist', []))}",
                f"exec_approval_exact={len(getattr(self.state, 'exec_approval_exact', []))}",
                f"exec_approval_pattern={len(getattr(self.state, 'exec_approval_pattern', []))}",
                f"exec_approval_path={len(getattr(self.state, 'exec_approval_path', []))}",
                f"exec_auto_write_allowlist={len(getattr(self.state, 'exec_auto_write_allowlist', []))}",
            ]
        )

    def _status_text(self, target: str) -> str:
        """Render `/status` output for a specific target payload."""
        normalized_target = str(target).strip().lower()
        if not normalized_target:
            return self._status_overview_text()
        status_values: dict[str, Any] = {
            "workflow_summary": getattr(self.state, "workflow_summary", ""),
            "workflow_hist": getattr(self.state, "workflow_hist", []),
            "full_proc_hist": getattr(self.state, "full_proc_hist", []),
            "action_hist": getattr(self.state, "action_hist", []),
            "core_agent_prompt": getattr(self.engine, "last_core_agent_prompt", ""),
        }
        if normalized_target not in status_values:
            return (
                "Unknown /status target. Use: "
                "workflow_summary | workflow_hist | full_proc_hist | action_hist | core_agent_prompt"
            )
        return self._render_status_value(status_values[normalized_target])

    def _cmd_help(self, _arg: str) -> str:
        return self._HELP_TEXT

    def _cmd_status(self, arg: str) -> str:
        return self._status_text(arg)

    def _cmd_refresh(self, _arg: str) -> str:
        return self._CMD_REFRESH

    def _cmd_exit(self, _arg: str) -> str:
        return self._CMD_EXIT

    def _handle_command(self, command_line: str) -> str:
        """Process built-in slash commands and return printable output token/text."""
        parts = command_line.split(maxsplit=1)
        cmd = parts[0].lower()
        arg = parts[1] if len(parts) > 1 else ""
        handler = self._command_handlers.get(cmd)
        if handler is None:
            return f"Unknown command: {cmd}. Use /help."
        return handler(arg)

    def _refresh_session(self) -> None:
        """Start a fresh session while keeping workspace/runtime config."""
        self._persist()
        self.state = StorageEngine(workspace=self.workspace, session_id=None)
        print(f"Session refreshed. New session_id={self.state.session_id}")

    def start(self, show_banner: bool = True) -> int:
        """Run interactive REPL until requester exits or EOF occurs."""
        self._bootstrap_runtime_assets()
        self._searxng.ensure_backend()
        if show_banner:
            print(f"Session {self.state.session_id} started in provider={self.provider}, mode={self.mode}")
            print("Type /help for commands. Type /exit to quit.")
            print("Multiline input enabled: Enter adds new lines, Ctrl+D submits, Ctrl+C cancels current input.")

        try:
            first_user_prompt = True
            while True:
                if not first_user_prompt:
                    print()
                try:
                    line = str(
                        self._prompt_session.prompt(
                            "user> ",
                            multiline=True,
                            prompt_continuation=lambda _w, _n, _s: self._PROMPT_CONTINUATION,
                        )
                    )
                except EOFError:
                    print()
                    break
                except KeyboardInterrupt:
                    print("\nInterrupted. Use /exit to quit.")
                    continue
                first_user_prompt = False

                stripped = line.strip()
                if not stripped:
                    print("No input provided.")
                    continue

                if stripped.startswith("/"):
                    command_out = self._handle_command(stripped)
                    if command_out == self._CMD_EXIT:
                        break
                    if command_out == self._CMD_REFRESH:
                        self._refresh_session()
                        continue
                    if command_out:
                        print(command_out)
                    continue

                self.engine.process_user_message(
                    state=self.state,
                    user_text=stripped,
                )
            return 0
        finally:
            self.shutdown()

    def shutdown(self) -> None:
        """Persist runtime state before process exit."""
        self._searxng.shutdown()
        self._persist()

    def _persist(self) -> None:
        self.state.save_state()
