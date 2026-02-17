from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from .prompts import build_prompt

class StorageEngine:
    def __init__(self, workspace: str | Path, session_id: str | None = None) -> None:
        self.workspace = Path(workspace).expanduser().resolve()
        self.workspace.mkdir(parents=True, exist_ok=True)
        self.sessions_root = self.workspace / "sessions"
        self.prompts_root = self.workspace / "prompts"
        self.knowledge_root = self.workspace / "knowledge"
        self.skills_root = self.workspace / "skills"
        self.sessions_root.mkdir(parents=True, exist_ok=True)
        self.prompts_root.mkdir(parents=True, exist_ok=True)
        (self.knowledge_root / "docs").mkdir(parents=True, exist_ok=True)
        (self.knowledge_root / "index").mkdir(parents=True, exist_ok=True)
        (self.skills_root / "core-agent").mkdir(parents=True, exist_ok=True)
        (self.skills_root / "all-agents").mkdir(parents=True, exist_ok=True)
        self.system_prompts_path = self.prompts_root / "agent_system_prompt.json"
        self.step_prompts_path = self.prompts_root / "agent_step_prompt.json"
        self.agent_role_descriptions_path = self.prompts_root / "agent_role_description.json"
        self.session_id = session_id or f"session_{uuid4().hex[:12]}"
        self.session_dir = self.sessions_root / self.session_id
        self.session_dir.mkdir(parents=True, exist_ok=True)
        self.state_path = self.session_dir / "state.json"
        # Session state fields.
        self.full_proc_hist: list[str] = []
        self.workflow_hist: list[str] = []
        self.workflow_summary: str = ""

    def load_state(self) -> bool:
        if not self.state_path.exists():
            return False
        raw = json.loads(self.state_path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            return False
        self._deserialize_state(raw)
        return True

    def save_state(self) -> None:
        tmp = self.state_path.with_suffix(".tmp")
        payload = self._serialize_state()
        tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        tmp.replace(self.state_path)

    def update_state(
        self,
        *,
        role: str | None = None,
        text: str | None = None,
        to_workflow_hist: bool = True,
        prompt_engine: Any | None = None,
        model_router: Any | None = None,
    ) -> None:
        if role is not None:
            line = self.format_line(role, text or "")
            self.full_proc_hist.append(line)
            if to_workflow_hist:
                self.workflow_hist.append(line)

        if not self.workflow_hist:
            return
        
        if prompt_engine is None or model_router is None:
            return

        step_prompt = prompt_engine.get_step_prompt("workflow_summary")
        if not isinstance(step_prompt, str) or not step_prompt.strip():
            return

        prompt = build_prompt(
            "",
            step_prompt,
            {
                "workflow_summary": self.workflow_summary,
                "workflow_history": self.workflow_hist,
            },
        )
        try:
            out = model_router.generate(prompt=prompt, task_type="thinking")
            if not isinstance(out, dict):
                return
            candidate = out.get("workflow_summary")
            if isinstance(candidate, str) and candidate.strip():
                self.workflow_summary = candidate.strip()
        except Exception:
            return

    @staticmethod
    def utc_now_iso() -> str:
        return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")

    @classmethod
    def format_line(cls, role: str, text: str) -> str:
        return f"[{cls.utc_now_iso()}] {role}> : {text}"

    def _serialize_state(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "full_proc_hist": self.full_proc_hist,
            "workflow_hist": self.workflow_hist,
            "workflow_summary": self.workflow_summary,
        }

    def _deserialize_state(self, raw: dict[str, Any]) -> None:
        loaded_id = str(raw.get("session_id", "")).strip()
        if loaded_id:
            self.session_id = loaded_id
            self.session_dir = self.sessions_root / self.session_id
            self.session_dir.mkdir(parents=True, exist_ok=True)
            self.state_path = self.session_dir / "state.json"
        full_proc_hist = raw.get("full_proc_hist", [])
        workflow_hist = raw.get("workflow_hist", raw.get("llm_hist", []))
        workflow_summary = raw.get("workflow_summary", raw.get("runtime_summary", ""))
        self.full_proc_hist = list(full_proc_hist if isinstance(full_proc_hist, list) else [])
        self.workflow_hist = list(workflow_hist if isinstance(workflow_hist, list) else [])
        self.workflow_summary = str(workflow_summary if isinstance(workflow_summary, str) else "")

    # Compatibility methods kept for current orchestrator contract.
    def ensure_agent_specs(
        self,
        default_system_prompts: dict[str, str],
        default_agent_role_descriptions: dict[str, str],
    ) -> None:
        system_prompts = self.load_system_prompts()
        role_descriptions = self.load_agent_role_descriptions()

        updated_system_prompts = dict(system_prompts)
        for key, value in default_system_prompts.items():
            role = str(key).strip()
            if role and role not in updated_system_prompts:
                updated_system_prompts[role] = str(value)

        updated_role_descriptions = dict(role_descriptions)
        for key, value in default_agent_role_descriptions.items():
            role = str(key).strip()
            if role and role not in updated_role_descriptions:
                updated_role_descriptions[role] = str(value)

        if updated_system_prompts != system_prompts or not self.system_prompts_path.exists():
            self.save_system_prompts(updated_system_prompts)
        if updated_role_descriptions != role_descriptions or not self.agent_role_descriptions_path.exists():
            self.save_agent_role_descriptions(updated_role_descriptions)

        if not self.step_prompts_path.exists():
            self._save_json_map(self.step_prompts_path, {})

    def load_system_prompts(self) -> dict[str, str]:
        return self._load_json_map(self.system_prompts_path)

    def save_system_prompts(self, system_prompts: dict[str, str]) -> None:
        self._save_json_map(self.system_prompts_path, system_prompts)

    def load_agent_role_descriptions(self) -> dict[str, str]:
        return self._load_json_map(self.agent_role_descriptions_path)

    def save_agent_role_descriptions(self, agent_role_descriptions: dict[str, str]) -> None:
        self._save_json_map(self.agent_role_descriptions_path, agent_role_descriptions)

    @staticmethod
    def _normalize_json_map(raw: Any) -> dict[str, str]:
        if not isinstance(raw, dict):
            return {}
        return {
            str(key): str(value)
            for key, value in raw.items()
            if str(key).strip()
        }

    def _load_json_map(self, path: Path) -> dict[str, str]:
        if not path.exists():
            return {}
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return {}
        return self._normalize_json_map(raw)

    def _save_json_map(self, path: Path, payload: dict[str, str]) -> None:
        normalized = self._normalize_json_map(payload)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(normalized, indent=2), encoding="utf-8")
        tmp.replace(path)
