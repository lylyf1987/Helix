from __future__ import annotations

import os
import shutil

from pathlib import Path
from typing import Any
from prompt_toolkit import PromptSession
from prompt_toolkit.key_binding import KeyBindings

from .kernel import (
    FlowEngine,
    PromptEngine,
    StorageEngine,
)
from .kernel.model_router import ModelRouter


class AgentRuntime:
    _CMD_EXIT = "__EXIT__"
    _CMD_REFRESH = "__REFRESH__"
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

        self.provider = str(provider).strip().lower() or "ollama"
        self.mode = str(mode).strip().lower() or "controlled"
        self.image_analysis_provider = self._normalized_optional_text(image_analysis_provider) or "none"
        self.image_analysis_model = self._normalized_optional_text(image_analysis_model) or "none"
        self.image_generation_provider = self._normalized_optional_text(image_generation_provider) or "none"
        self.image_generation_model = self._normalized_optional_text(image_generation_model) or "none"

        self._multiline_input_enabled = False
        self._prompt_session: Any | None = None
        self._initialize_input_mode()
        self._configure_skill_provider_environment()
        self._initialize_kernel(session_id=session_id, model_name=model_name)
        self._persist()

    @staticmethod
    def _normalized_optional_text(value: str | None) -> str:
        return str(value).strip() if value is not None else ""

    def _initialize_kernel(self, *, session_id: str | None, model_name: str | None) -> None:
        self.state = StorageEngine(workspace=self.workspace, session_id=session_id)
        if session_id is not None:
            self.state.load_state()
        self.model_router = ModelRouter(provider=self.provider, model_name=model_name)
        self.prompt_engine = PromptEngine(
            workspace=self.workspace,
            token_window_limit=70000,
            compact_keep_last_k=10,
        )
        self.engine = FlowEngine(
            workspace=self.workspace,
            mode=self.mode,
            model_router=self.model_router,
            prompt_engine=self.prompt_engine,
            approval_handler=self._default_approval_prompt,
            write_policy_handler=self._auto_write_override_prompt,
        )

    def _configure_skill_provider_environment(self) -> None:
        os.environ["IMAGE_ANALYSIS_PROVIDER"] = self.image_analysis_provider
        os.environ["IMAGE_ANALYSIS_MODEL"] = self.image_analysis_model
        os.environ["IMAGE_GENERATION_PROVIDER"] = self.image_generation_provider
        os.environ["IMAGE_GENERATION_MODEL"] = self.image_generation_model

    def _initialize_input_mode(self) -> None:
        try:
            bindings = KeyBindings()

            @bindings.add("c-d")
            def _submit(event: Any) -> None:
                event.app.exit(result=event.app.current_buffer.text)

            self._prompt_session = PromptSession(key_bindings=bindings)
            self._multiline_input_enabled = True
        except Exception:
            self._multiline_input_enabled = False
            self._prompt_session = None

    @staticmethod
    def _continuation_prompt(_width: int, _line_number: int, _is_soft_wrap: bool) -> str:
        return "... "

    def _read_user_input(self, prompt: str) -> str:
        if not self._multiline_input_enabled or self._prompt_session is None:
            return input(prompt)
        return str(
            self._prompt_session.prompt(
                prompt,
                multiline=True,
                prompt_continuation=self._continuation_prompt,
            )
        )

    def _bootstrap_runtime_assets(self) -> None:
        runtime_prompts_root = self.workspace / "prompts"
        runtime_skills_root = self.workspace / "skills"
        runtime_prompts_root.mkdir(parents=True, exist_ok=True)
        runtime_skills_root.mkdir(parents=True, exist_ok=True)
        self._copy_packaged_prompts(runtime_prompts_root)
        self._copy_packaged_skills(runtime_skills_root)

    def _copy_packaged_prompts(self, runtime_prompts_root: Path) -> None:
        for file_name in ("agent_system_prompt.json", "agent_role_description.json"):
            source = self.packaged_prompts_root / file_name
            target = runtime_prompts_root / file_name
            if source.exists():
                shutil.copy2(source, target)

    def _copy_packaged_skills(self, runtime_skills_root: Path) -> None:
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
    def _default_approval_prompt(signature: str) -> tuple[bool, str]:
        print()
        print("Runtime confirmation required for exec action.")
        print(signature)
        print("Approve this execution? [y/N/s/p/k]")
        print("  y: allow once")
        print("  s: allow same exact exec for this session")
        print("  p: allow same script/pattern for this session")
        print("  k: allow same script_path for this session (ignore args)")
        choice = input("> ").strip().lower()
        if choice in {"y", "yes", "once"}:
            return True, "once"
        if choice in {"s", "session", "exact"}:
            return True, "session"
        if choice in {"p", "pattern"}:
            return True, "pattern"
        if choice in {"k", "path", "skill"}:
            return True, "path"
        return False, "deny"

    @staticmethod
    def _auto_write_override_prompt(note: str, suggested_paths: list[str]) -> str | None:
        print()
        print("Runtime auto-mode write policy blocked external write.")
        if note.strip():
            print(note.strip())
        if suggested_paths:
            print("Suggested external paths (from command context):")
            for idx, item in enumerate(suggested_paths, start=1):
                print(f"  {idx}. {item}")
        print("Allow one external writable path for this session? [y/N]")
        choice = input("> ").strip().lower()
        if choice not in {"y", "yes"}:
            return None
        default_path = suggested_paths[0] if suggested_paths else ""
        if default_path:
            print(f"Enter writable path (blank uses default: {default_path})")
            entered = input("> ").strip()
            return entered or default_path
        print("Enter writable path (absolute or ~/...):")
        entered = input("> ").strip()
        return entered or None

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

    def _handle_command(self, command_line: str) -> str:
        parts = command_line.split(maxsplit=1)
        cmd = parts[0].lower()
        if cmd == "/help":
            return self._HELP_TEXT
        if cmd == "/refresh":
            return self._CMD_REFRESH
        if cmd == "/exit":
            return self._CMD_EXIT
        if cmd == "/status":
            target = parts[1] if len(parts) > 1 else ""
            return self._status_text(target)
        return f"Unknown command: {cmd}. Use /help."

    def start(self, show_banner: bool = True) -> int:
        self._bootstrap_runtime_assets()
        if show_banner:
            print(f"Session {self.state.session_id} started in provider={self.provider}, mode={self.mode}")
            print("Type /help for commands. Type /exit to quit.")
            if self._multiline_input_enabled:
                print("Multiline input enabled: Enter adds new lines, Ctrl+D submits, Ctrl+C cancels current input.")

        try:
            first_user_prompt = True
            while True:
                if not first_user_prompt:
                    print()
                try:
                    line = self._read_user_input("user> ")
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
                        self._persist()
                        self.state = StorageEngine(workspace=self.workspace, session_id=None)
                        print(f"Session refreshed. New session_id={self.state.session_id}")
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
        self._persist()

    def _persist(self) -> None:
        self.state.save_state()
