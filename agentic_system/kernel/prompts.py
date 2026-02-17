from __future__ import annotations

import json
from pathlib import Path
import shutil
from typing import Any

_PACKAGED_PROMPTS_ROOT = Path(__file__).resolve().parents[1] / "prompts"
_SYSTEM_PROMPTS_PATH = _PACKAGED_PROMPTS_ROOT / "agent_system_prompt.json"
_STEP_PROMPTS_PATH = _PACKAGED_PROMPTS_ROOT / "agent_step_prompt.json"
_ROLE_DESCRIPTIONS_PATH = _PACKAGED_PROMPTS_ROOT / "agent_role_description.json"
_AGENT_ROLES_PLACEHOLDER = "{{AGENT_ROLES_FROM_JSON}}"
_REQUIRED_STEPS = (
    "context",
    "retrieve_ltm",
    "plan",
    "do_tasks",
    "act",
    "verify",
    "iterate",
    "create_sub_agent",
    "assign_task",
    "document",
    "create_skill",
    "promotion_check",
    "report",
    "invalid_step_repair",
    "stm_compaction",
    "workflow_summary",
)


class PromptEngine:
    def __init__(self, workspace: str | Path, packaged_root: str | Path | None = None) -> None:
        self.workspace = Path(workspace).expanduser().resolve()
        self.runtime_prompts_root = self.workspace / "prompts"
        default_packaged = Path(__file__).resolve().parents[1] / "prompts"
        self.packaged_root = Path(packaged_root).resolve() if packaged_root else default_packaged
        self.system_prompts_path = self.runtime_prompts_root / "agent_system_prompt.json"
        self.legacy_system_prompts_path = self.runtime_prompts_root / "agent_systemp_prompt.json"
        self.step_prompts_path = self.runtime_prompts_root / "agent_step_prompt.json"
        self.agent_role_descriptions_path = self.runtime_prompts_root / "agent_role_description.json"
        self._bootstrap_runtime_prompts()

    def _bootstrap_runtime_prompts(self) -> None:
        self.runtime_prompts_root.mkdir(parents=True, exist_ok=True)

        system_target = self.system_prompts_path
        if not system_target.exists():
            source = self.packaged_root / "agent_system_prompt.json"
            legacy_source = self.packaged_root / "agent_systemp_prompt.json"
            if source.exists():
                shutil.copy2(source, system_target)
            elif legacy_source.exists():
                shutil.copy2(legacy_source, system_target)
            else:
                system_target.write_text("{}", encoding="utf-8")

        for file_name in ("agent_step_prompt.json", "agent_role_description.json"):
            source_file = self.packaged_root / file_name
            target_file = self.runtime_prompts_root / file_name
            if target_file.exists():
                continue
            if source_file.exists():
                shutil.copy2(source_file, target_file)
                continue
            target_file.write_text("{}", encoding="utf-8")

    def get_system_prompt(self, agent_role: str, fallback_role: str = "core_agent") -> str:
        prompts = self._load_system_prompts()
        role = str(agent_role).strip()
        selected = ""
        if role and role in prompts and prompts[role].strip():
            selected = prompts[role]
        else:
            fallback = str(fallback_role).strip()
            if fallback and fallback in prompts and prompts[fallback].strip():
                selected = prompts[fallback]
        if not selected:
            return ""

        descriptions = self._load_json_map(self.agent_role_descriptions_path)
        lines = ["Agent Roles (Loaded from agent_role_description.json):"]
        if descriptions:
            for item_role in sorted(descriptions.keys()):
                desc = str(descriptions.get(item_role, "")).strip()
                lines.append(f"- {item_role}: {desc if desc else '(no description)'}")
        else:
            lines.append("- (no role descriptions found)")
        roles_section = "\n".join(lines)

        text = selected.strip()
        if _AGENT_ROLES_PLACEHOLDER in text:
            return text.replace(_AGENT_ROLES_PLACEHOLDER, roles_section)
        return text + "\n\n" + roles_section

    def list_agent_roles_with_descriptions(self) -> dict[str, str]:
        prompts = self._load_system_prompts()
        descriptions = self._load_json_map(self.agent_role_descriptions_path)
        all_roles = sorted(set(prompts.keys()) | set(descriptions.keys()))
        return {role: descriptions.get(role, "") for role in all_roles}

    def get_step_prompt(self, step_name: str) -> str:
        step = str(step_name).strip()
        if not step:
            return ""
        prompts = self._load_json_map(self.step_prompts_path)
        prompt = prompts.get(step, "")
        if isinstance(prompt, str) and prompt.strip():
            return prompt
        return ""

    @staticmethod
    def build_prompt(system_prompt: str | None, step_prompt: str, input_payload: dict[str, Any]) -> str:
        sections: list[str] = []
        if isinstance(system_prompt, str) and system_prompt.strip():
            sections.append(str(system_prompt).strip())
        sections.append(str(step_prompt).strip())

        workflow_summary = input_payload.get("workflow_summary")
        workflow_history = input_payload.get("workflow_history")
        if workflow_summary is not None or workflow_history is not None:
            text_blocks: list[str] = []
            if workflow_summary is not None:
                summary_text = str(workflow_summary).strip()
                text_blocks.append("Workflow Summary:")
                text_blocks.append(summary_text if summary_text else "(empty)")
            if workflow_history is not None:
                if isinstance(workflow_history, list):
                    history_text = "\n".join(str(line) for line in workflow_history)
                else:
                    history_text = str(workflow_history)
                history_text = history_text.strip()
                text_blocks.append("Workflow History:")
                text_blocks.append(history_text if history_text else "(empty)")
            sections.append("\n".join(text_blocks))

        return "\n\n".join(sections)

    def _load_system_prompts(self) -> dict[str, str]:
        if not self.system_prompts_path.exists() and self.legacy_system_prompts_path.exists():
            return self._load_json_map(self.legacy_system_prompts_path)
        return self._load_json_map(self.system_prompts_path)

    def _load_json_map(self, path: Path) -> dict[str, str]:
        if not path.exists():
            return {}
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return {}
        return self._normalize_map(raw)

    @staticmethod
    def _normalize_map(raw: Any) -> dict[str, str]:
        if not isinstance(raw, dict):
            return {}
        normalized: dict[str, str] = {}
        for key, value in raw.items():
            role = str(key).strip()
            if not role:
                continue
            if isinstance(value, list):
                normalized[role] = "\n".join(str(item) for item in value)
            else:
                normalized[role] = str(value)
        return normalized


def _normalize_map(raw: Any) -> dict[str, str]:
    if not isinstance(raw, dict):
        return {}
    out: dict[str, str] = {}
    for key, value in raw.items():
        role = str(key).strip()
        if not role:
            continue
        if isinstance(value, list):
            out[role] = "\n".join(str(item) for item in value)
        else:
            out[role] = str(value)
    return out


def _load_json_map(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return _normalize_map(raw)


SYSTEM_PROMPTS_BY_ROLE_DEFAULT = _load_json_map(_SYSTEM_PROMPTS_PATH)
if "core_agent" not in SYSTEM_PROMPTS_BY_ROLE_DEFAULT:
    SYSTEM_PROMPTS_BY_ROLE_DEFAULT["core_agent"] = ""
if "sub_agent" not in SYSTEM_PROMPTS_BY_ROLE_DEFAULT:
    SYSTEM_PROMPTS_BY_ROLE_DEFAULT["sub_agent"] = ""

AGENT_ROLE_DESCRIPTIONS_DEFAULT = _load_json_map(_ROLE_DESCRIPTIONS_PATH)
if "core_agent" not in AGENT_ROLE_DESCRIPTIONS_DEFAULT:
    AGENT_ROLE_DESCRIPTIONS_DEFAULT["core_agent"] = ""

SYSTEM_PROMPT_SUB = SYSTEM_PROMPTS_BY_ROLE_DEFAULT.get("sub_agent", "")

_RAW_STEP_PROMPTS = _load_json_map(_STEP_PROMPTS_PATH)
STEP_PROMPTS = {
    name: (
        _RAW_STEP_PROMPTS[name]
        if isinstance(_RAW_STEP_PROMPTS.get(name), str) and _RAW_STEP_PROMPTS[name].strip()
        else ""
    )
    for name in _REQUIRED_STEPS
}


def build_prompt(system_prompt: str | None, step_prompt: str, input_payload: dict[str, Any]) -> str:
    return PromptEngine.build_prompt(system_prompt, step_prompt, input_payload)
