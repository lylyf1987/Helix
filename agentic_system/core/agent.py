"""Agent — the LLM brain.

A pure function conceptually: (system_prompt, state) → Action.
The agent itself is stateless; all state lives in the environment.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Optional

from .action import Action, parse_action, ActionParseError, ALLOWED_CORE_ACTIONS, ALLOWED_SUB_ACTIONS
from .state import State, format_turn
from ..providers import ModelProvider


# --------------------------------------------------------------------------- #
# Prompt Building & Context Loading
# --------------------------------------------------------------------------- #

_SKILLS_META = "{{SKILLS_META_FROM_JSON}}"
_KNOWLEDGE_META = "{{KNOWLEDGE_META_FROM_JSON}}"
_BUILTIN_LOADERS = "{{BUILTIN_REFERENCE_LOADERS}}"
_WORKSPACE_ROOT = "{{WORKSPACE_ROOT}}"
_SESSION_ID = "{{SESSION_ID}}"
_SESSION_ROOT = "{{SESSION_ROOT}}"
_PROJECT_ROOT = "{{PROJECT_ROOT}}"
_DOCS_ROOT = "{{DOCS_ROOT}}"
_STATE_ROOT = "{{STATE_ROOT}}"
_RUNTIME_WORKSPACE = "{{RUNTIME_WORKSPACE}}"
_SUB_AGENT_ROLE = "{{SUB_AGENT_ROLE}}"

_BUILTIN_REFERENCE_LOADERS = [
    {
        "loader": "load-skill",
        "purpose": "Load full SKILL.md and scripts list for a target skill into workflow_history.",
        "script_path": "skills/all-agents/load-skill/scripts/load_skill.py",
        "code_type": "python",
        "required_args": ["--skill-id", "<skill_id>", "--scope", "all-agents|core-agent"],
    },
]

# Package-level prompts directory
_PACKAGE_PROMPTS = Path(__file__).resolve().parent.parent / "prompts"


def _load_sys_prompt(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}
    if not isinstance(raw, dict):
        return {}
    result: dict[str, str] = {}
    for key, value in raw.items():
        role = str(key).strip()
        if not role:
            continue
        if isinstance(value, list):
            result[role] = "\n".join(str(item) for item in value)
        else:
            result[role] = str(value)
    return result


def _parse_frontmatter(text: str) -> dict[str, str]:
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        return {}
    end = -1
    for idx in range(1, len(lines)):
        if lines[idx].strip() == "---":
            end = idx
            break
    if end == -1:
        return {}
    result: dict[str, str] = {}
    for raw in lines[1:end]:
        line = raw.strip()
        if not line or ":" not in line:
            continue
        key, value = line.split(":", 1)
        result[key.strip()] = value.strip()
    return result


def _parse_csv(value: str) -> list[str]:
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def _load_skills(skills_root: Path, *, allowed_scopes: Optional[set[str]] = None) -> list[dict[str, Any]]:
    skills_root = Path(skills_root)
    if not skills_root.exists():
        return []

    builtin_ids = {"load-skill", "load-knowledge-docs"}
    rows: list[dict[str, Any]] = []
    for scope_dir in sorted(skills_root.iterdir()):
        if not scope_dir.is_dir():
            continue
        scope = scope_dir.name
        if allowed_scopes is not None and scope not in allowed_scopes:
            continue
        for skill_dir in sorted(scope_dir.iterdir()):
            if not skill_dir.is_dir():
                continue
            skill_id = skill_dir.name
            if skill_id in builtin_ids:
                continue
            skill_md = skill_dir / "SKILL.md"
            if not skill_md.exists():
                continue
            try:
                text = skill_md.read_text(encoding="utf-8")
            except OSError:
                continue
            fm = _parse_frontmatter(text)
            name = fm.get("name", skill_id).strip() or skill_id
            handler = fm.get("handler", "").strip()
            description = fm.get("description", "").strip()
            path = f"skills/{scope}/{skill_id}"
            handler_path = f"{path}/{handler}" if handler else ""

            rows.append({
                "skill_id": skill_id,
                "scope": scope,
                "path": path,
                "handler": handler_path,
                "name": name,
                "description": description,
                "required_tools": _parse_csv(fm.get("required_tools", "")),
                "recommended_tools": _parse_csv(fm.get("recommended_tools", "")),
                "forbidden_tools": _parse_csv(fm.get("forbidden_tools", "")),
            })

    rows.sort(key=lambda r: (r["scope"], r["skill_id"]))
    return rows


def _normalize_tags(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    return []


def _title_from_path(path_text: str) -> str:
    path = Path(str(path_text).strip())
    stem = path.stem.strip()
    return stem or "untitled"


def _load_knowledge_catalog(knowledge_root: Path, *, limit: int = 80) -> list[dict[str, Any]]:
    catalog_path = Path(knowledge_root) / "index" / "catalog.json"
    if not catalog_path.exists():
        return []
    try:
        raw = json.loads(catalog_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return []
    if not isinstance(raw, list):
        return []

    rows: list[dict[str, Any]] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        path = str(item.get("path", "")).strip()
        legacy_doc_id = str(item.get("doc_id", "")).strip()
        if not path and legacy_doc_id:
            path = f"knowledge/docs/{legacy_doc_id}.md"
        if not path:
            continue
        title = str(item.get("title", "")).strip() or _title_from_path(path)
        rows.append({
            "title": title,
            "summary": str(item.get("summary", "")).strip(),
            "tags": _normalize_tags(item.get("tags", [])),
            "path": path,
        })

    rows.sort(key=lambda r: (r["title"].lower(), r["path"]))
    return rows[:max(1, limit)]


def _build_system_prompt(
    workspace_path: Path,
    role: str = "core_agent",
    *,
    session_id: str = "none",
    session_root: Optional[Path] = None,
    project_root: Optional[Path] = None,
    docs_root: Optional[Path] = None,
    state_root: Optional[Path] = None,
    sub_agent_role: str = "",
) -> str:
    """Build the complete system prompt from templates + runtime metadata.

    Works for both core-agents (role="core_agent") and sub-agents
    (role="sub_agent").  The *role* selects which template to load
    from ``agent_system_prompt.json``.  Sub-agent-specific placeholders
    (``{{SUB_AGENT_ROLE}}``, ``{{SUB_AGENT_CONTEXT}}``) are filled when
    the corresponding arguments are provided.
    """
    workspace = Path(workspace_path).expanduser().resolve()
    session_label = str(session_id).strip() or "none"
    session_dir = Path(session_root).expanduser().resolve() if session_root is not None else workspace / "sessions" / session_label
    project_dir = Path(project_root).expanduser().resolve() if project_root is not None else session_dir / "project"
    docs_dir = Path(docs_root).expanduser().resolve() if docs_root is not None else session_dir / "docs"
    state_dir = Path(state_root).expanduser().resolve() if state_root is not None else session_dir / ".state"
    
    # 1. Load template
    templates = _load_sys_prompt(_PACKAGE_PROMPTS / "agent_system_prompt.json")
    template = templates.get(role, "")
    if not template:
        return ""

    # 2. Load skills (sub-agents only see "all-agents" scope)
    skill_scopes = {"all-agents"} if role == "sub_agent" else None
    skills = _load_skills(workspace / "skills", allowed_scopes=skill_scopes)
    skills_text = "- (no skills found)" if not skills else "\n".join(
        "- " + json.dumps(row, ensure_ascii=True) for row in skills
    )

    # 3. Load knowledge
    catalog = _load_knowledge_catalog(workspace / "knowledge")
    knowledge_text = "- (no knowledge docs found)" if not catalog else "\n".join(
        "- " + json.dumps(row, ensure_ascii=True) for row in catalog
    )

    # 4. Built-in loaders
    loaders_text = "\n".join(
        "- " + json.dumps(row, ensure_ascii=True) for row in _BUILTIN_REFERENCE_LOADERS
    )

    # 5. Replace placeholders
    prompt = template
    if _SKILLS_META in prompt:
        prompt = prompt.replace(_SKILLS_META, skills_text)
    if _KNOWLEDGE_META in prompt:
        prompt = prompt.replace(_KNOWLEDGE_META, knowledge_text)
    if _BUILTIN_LOADERS in prompt:
        prompt = prompt.replace(_BUILTIN_LOADERS, loaders_text)
    if _WORKSPACE_ROOT in prompt:
        prompt = prompt.replace(_WORKSPACE_ROOT, str(workspace))
    if _SESSION_ID in prompt:
        prompt = prompt.replace(_SESSION_ID, session_label)
    if _SESSION_ROOT in prompt:
        prompt = prompt.replace(_SESSION_ROOT, str(session_dir))
    if _PROJECT_ROOT in prompt:
        prompt = prompt.replace(_PROJECT_ROOT, str(project_dir))
    if _DOCS_ROOT in prompt:
        prompt = prompt.replace(_DOCS_ROOT, str(docs_dir))
    if _STATE_ROOT in prompt:
        prompt = prompt.replace(_STATE_ROOT, str(state_dir))
    if _RUNTIME_WORKSPACE in prompt:
        prompt = prompt.replace(_RUNTIME_WORKSPACE, str(workspace))

    # Sub-agent-specific placeholders
    if _SUB_AGENT_ROLE in prompt:
        prompt = prompt.replace(_SUB_AGENT_ROLE, sub_agent_role)

    return prompt


# --------------------------------------------------------------------------- #
# Agent
# --------------------------------------------------------------------------- #


class Agent:
    """LLM-based agent that produces Actions from State.

    The agent is stateless — it reads the current State (built by the
    Environment) and produces an Action.  All persistent state lives
    in the Environment's history.

    Prompt building is owned by the agent:

    - **Core agents** pass ``workspace=`` — a ``PromptBuilder`` assembles the
      system prompt from the workspace's skills, knowledge, and prompt
      templates at construction time.
    - **Sub-agents** pass ``system_prompt=`` directly (simple role prompt,
      no workspace).

    Exactly one of ``workspace`` or ``system_prompt`` must be provided.

    Args:
        model: Any object satisfying ``ModelProvider`` protocol.
        workspace: Workspace directory (builds system prompt via PromptBuilder).
        system_prompt: Pre-built system prompt string (for sub-agents).
        allowed_actions: Which action types this agent can emit.
        on_token: Callback fired with each streamed token (for UI).
    """

    def __init__(
        self,
        model: ModelProvider,
        *,
        name: str = "core-agent",
        workspace: Optional[Path] = None,
        role: str = "core_agent",
        session_id: str = "none",
        session_root: Optional[Path] = None,
        project_root: Optional[Path] = None,
        docs_root: Optional[Path] = None,
        state_root: Optional[Path] = None,
        sub_agent_role: str = "",
        system_prompt: Optional[str] = None,
        allowed_actions: frozenset[str] = ALLOWED_CORE_ACTIONS,
    ) -> None:
        if workspace is not None and system_prompt is not None:
            raise ValueError("Provide workspace or system_prompt, not both.")
        if workspace is None and system_prompt is None:
            raise ValueError("Provide either workspace or system_prompt.")

        self.model = model
        self.name = name
        self.allowed_actions = allowed_actions
        self.last_prompt = ""

        if workspace is not None:
            self.system_prompt = _build_system_prompt(
                workspace,
                role,
                session_id=session_id,
                session_root=session_root,
                project_root=project_root,
                docs_root=docs_root,
                state_root=state_root,
                sub_agent_role=sub_agent_role,
            )
        else:
            self.system_prompt = system_prompt  # type: ignore[assignment]

    def act(
        self,
        state: State,
        *,
        stream: bool = True,
        chunk_callback: Optional[Callable[[str], None]] = None,
    ) -> Action:
        """Given current state, produce the next Action.

        Raises:
            ActionParseError: If the model output fails parsing/validation.
        """
        prompt = self._build_prompt(state)
        self.last_prompt = prompt
        raw_output = self.model.generate(
            prompt,
            stream=stream,
            chunk_callback=chunk_callback,
        )
        return parse_action(raw_output, allowed_actions=self.allowed_actions)

    # ----- prompt construction --------------------------------------------- #

    def _build_prompt(self, state: State) -> str:
        """Compose the final prompt from system prompt + state context.

        Layout (optimized for LLM attention):
            1. system_prompt     — identity, skills, constraints (beginning)
            2. workflow_summary  — compacted long-term memory
            3. workflow_history  — chronological turns (excluding latest)
            4. latest_context    — the immediate turn to respond to (end)
        """
        sections: list[str] = [self.system_prompt]

        if state.workflow_summary:
            sections.append(
                f"<workflow_summary>\n{state.workflow_summary}\n</workflow_summary>"
            )
        else:
            sections.append(
                "<workflow_summary>\n<empty>\n</workflow_summary>"
            )

        if state.observation:
            # Split: history (all but last) + latest (last turn)
            history_turns = state.observation[:-1]
            latest_turn = state.observation[-1]

            if history_turns:
                history_text = "\n".join(
                    format_turn(t) for t in history_turns
                )
                sections.append(
                    f"<workflow_history>\n{history_text}\n</workflow_history>"
                )
            else:
                sections.append(
                    "<workflow_history>\n<empty>\n</workflow_history>"
                )

            latest_text = format_turn(latest_turn)
            sections.append(
                f"<latest_context>\n{latest_text}\n</latest_context>"
            )

        return "\n\n".join(sections)
