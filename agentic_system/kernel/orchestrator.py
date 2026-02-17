from __future__ import annotations

import json
from collections import deque
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable

from .constants import DEFAULT_LIMITS, STEP_ORDER, TERMINAL_TOKENS
from .executors import compact_observation, execute
from .knowledge import KnowledgeEngine
from .model_router import ModelRouter
from .policy import PolicyEngine
from .prompts import (
    PromptEngine,
    AGENT_ROLE_DESCRIPTIONS_DEFAULT,
    SYSTEM_PROMPTS_BY_ROLE_DEFAULT,
    build_prompt,
)
from .skills import SkillEngine
from .storage import StorageEngine
from .validators import (
    validate_assignment,
    validate_llm_step_output,
    validate_memory_patch,
    validate_plan_schema,
    validate_promotion_proposal,
    validate_skill_proposal,
    validate_subagent_spec,
    validate_verify_schema,
)


class FlowEngine:
    def __init__(
        self,
        workspace: str | Path,
        model_router: ModelRouter,
        mode: str,
        prompt_engine: PromptEngine,
        skill_engine: SkillEngine,
        knowledge: KnowledgeEngine,
        policy: PolicyEngine,
        approval_handler: Callable[[str], tuple[bool, str]] | None = None,
        limits: dict[str, int] | None = None,
    ) -> None:
        self.workspace = Path(workspace).expanduser().resolve()
        self.model_router = model_router
        self.mode = mode
        self.prompt_engine = prompt_engine
        self.skill_engine = skill_engine
        self.knowledge = knowledge
        self.policy = policy
        self.approval_handler = approval_handler
        self.limits = deepcopy(DEFAULT_LIMITS)
        if limits:
            self.limits.update(limits)
        self.action_info = None

    def _ensure_runtime_fields(self, state: StorageEngine) -> None:
        if not isinstance(getattr(state, "full_proc_hist", None), list):
            state.full_proc_hist = []
        if not isinstance(getattr(state, "workflow_hist", None), list):
            state.workflow_hist = []
        if not isinstance(getattr(state, "workflow_summary", None), str):
            state.workflow_summary = ""
        state.ensure_agent_specs(
            default_system_prompts=SYSTEM_PROMPTS_BY_ROLE_DEFAULT,
            default_agent_role_descriptions=AGENT_ROLE_DESCRIPTIONS_DEFAULT,
        )

    def validate_action_or_repair(
        self,
        state: StorageEngine,
        proposed_action: Any,
        agent_role: str,
        objective: str,
    ) -> str | None:
        nxt = self.normalize_action(str(proposed_action).strip() if proposed_action is not None else None)
        if nxt in self.allowed_steps(agent_role):
            return nxt

        role = str(agent_role).strip() or "core_agent"
        fallback_role = "core_agent" if self._agent_kind_for_role(role) == "core" else "sub_agent"
        system_prompt = self.prompt_engine.get_system_prompt(role, fallback_role=fallback_role)
        caps = self.load_capability_snapshot(agent_role)
        env = self.build_envelope_for_step("context", state, caps, agent_role, objective)
        repair_prompt = build_prompt(system_prompt, self.prompt_engine.get_step_prompt("invalid_step_repair"), env)
        repair_out = self.model_router.generate(
            prompt=repair_prompt,
            task_type="thinking",
        )
        try:
            validate_llm_step_output("invalid_step_repair", repair_out)
        except ValueError:
            return "report"

        repaired = self.normalize_action(repair_out.get("action"))
        if repaired in self.allowed_steps(agent_role):
            state.update_state(
                role="runtime",
                text=repair_out.get("raw_response", "repaired action"),
                prompt_engine=self.prompt_engine,
                model_router=self.model_router,
            )
            return repaired
        return "report"

    @staticmethod
    def _task_type_for_step(step: str) -> str:
        if step in {"act", "create_skill"}:
            return "coding"
        return "thinking"

    def call_step_llm(
        self,
        state: StorageEngine,
        current_step: str,
        caps: dict[str, Any],
        agent_role: str,
        objective: str,
    ) -> dict[str, Any]:
        envelope = self.build_envelope_for_step(current_step, state, caps, agent_role, objective)
        role = str(agent_role).strip() or "core_agent"
        fallback_role = "core_agent" if self._agent_kind_for_role(role) == "core" else "sub_agent"
        system_prompt = self.prompt_engine.get_system_prompt(role, fallback_role=fallback_role)
        prompt = build_prompt(system_prompt, self.prompt_engine.get_step_prompt(current_step), envelope)
        try:
            out = self.model_router.generate(
                prompt=prompt,
                task_type=self._task_type_for_step(current_step),
            )
            validate_llm_step_output(current_step, out)
            return out
        except Exception as exc:
            return {
                "action": "report",
                "raw_response": f"[Model error] {exc}",
                "action_input": {},
            }

    def handle_context(self, state: StorageEngine, structured: dict[str, Any]) -> None:
        selected = structured.get("selected_doc_ids", [])
        reasons = structured.get("selected_reasons", {})
        if not isinstance(selected, list):
            selected = []
        if not isinstance(reasons, dict):
            reasons = {}
        selected_ids = [str(item) for item in selected if str(item).strip()]
        if len(selected_ids) != len(set(selected_ids)):
            raise ValueError("duplicate doc ids")
        if set(selected_ids) != set(str(key) for key in reasons.keys()):
            raise ValueError("selected_reasons mismatch")
        state.ltm_context = self.knowledge.load_knowledge(selected_ids)

    def handle_retrieve_ltm(self, state: StorageEngine, structured: dict[str, Any]) -> None:
        requested_ids: list[str] = []
        raw_items = structured.get("ltms", [])
        if not isinstance(raw_items, list):
            raw_items = []
        for meta in raw_items:
            if not isinstance(meta, dict):
                continue
            doc_id = str(meta.get("doc_id", "")).strip()
            if doc_id:
                requested_ids.append(doc_id)
        ltms = self.knowledge.load_knowledge(requested_ids)
        for doc in ltms:
            state.update_state(
                role="retrieved_memory",
                text=str(doc.get("title", "doc")),
                prompt_engine=self.prompt_engine,
                model_router=self.model_router,
            )
        state.ltm_context.extend(ltms)

    def handle_plan(self, state: StorageEngine, structured: dict[str, Any], caps: dict[str, Any]) -> None:
        plan = validate_plan_schema(structured, caps)
        state.plan = plan
        state.task_queue = deque(plan.get("tasks", []))

    def handle_do_tasks(self, state: StorageEngine, structured: dict[str, Any], agent_role: str) -> dict[str, str]:
        if not state.task_queue:
            return {"force_next_step": "verify"}

        requested_task_id = str(structured.get("task_id", "")).strip()
        requested_route_raw = str(structured.get("route", "")).strip().lower()
        requested_route = requested_route_raw if requested_route_raw else ""

        if requested_route == "done":
            return {"force_next_step": "verify"}

        task: dict[str, Any] | None = None
        if requested_task_id:
            for item in list(state.task_queue):
                if str(item.get("task_id")) == requested_task_id:
                    task = item
                    break
        if task is None:
            task = state.task_queue[0]

        state.active_task = task
        state.active_action = None

        route = requested_route or str(task.get("route", "act")).strip().lower()
        if route == "done":
            return {"force_next_step": "verify"}
        if route == "assign_task" and self._agent_kind_for_role(agent_role) == "core":
            return {"force_next_step": "assign_task"}
        return {"force_next_step": "act"}

    def resolve_action_from_skills(self, task: dict[str, Any]) -> dict[str, Any]:
        task_type = str(task.get("type", "bash")).lower()
        params = task.get("params", {})
        if not isinstance(params, dict):
            params = {}

        if task_type == "pythonexec":
            code = str(params.get("code") or params.get("script") or "print('No code provided')")
            return {"executor": "PythonExec", "code": code}

        command = str(params.get("command") or params.get("cmd") or f"echo {task.get('purpose', 'task')}")
        return {"executor": "Bash", "command": command}

    def _run_working_loop(self, state: StorageEngine) -> None:
        self._ensure_runtime_fields(state)

        state.terminated = False
        state.final_report = None
        max_turns = int(self.limits.get("max_inner_turns", 60))
        turns = 0

        while turns < max_turns and not state.terminated:
            turns += 1
            current = self.action_info if isinstance(self.action_info, dict) else {}
            action_raw = current.get("action")
            action = str(action_raw).strip().lower() if action_raw is not None else "none"
            if action in TERMINAL_TOKENS:
                state.terminated = True
                break

            if action != "call_llm":
                state.update_state(
                    role="runtime",
                    text=f"unsupported_action> : {action}",
                    prompt_engine=self.prompt_engine,
                    model_router=self.model_router,
                )
                state.final_report = f"Unsupported action: {action}"
                state.terminated = True
                break

            action_input = current.get("action_input", {})
            if not isinstance(action_input, dict):
                action_input = {}

            agent_role_raw = action_input.get("agent_role")
            if not isinstance(agent_role_raw, str) or not agent_role_raw.strip():
                state.update_state(
                    role="runtime",
                    text="invalid_action_input> : call_llm requires action_input.agent_role",
                    prompt_engine=self.prompt_engine,
                    model_router=self.model_router,
                )
                state.final_report = "Invalid action_input for call_llm: missing agent_role."
                state.terminated = True
                break
            agent_role = agent_role_raw.strip()
            fallback_role = "core_agent"
            system_prompt = self.prompt_engine.get_system_prompt(agent_role, fallback_role=fallback_role)
            step_name = str(action_input.get("step_name", "")).strip()
            step_prompt = self.prompt_engine.get_step_prompt(step_name) if step_name else ""
            task_type = str(action_input.get("task_type", "thinking")).strip() or "thinking"
            prompt_input = {
                "workflow_summary": action_input.get("workflow_summary", state.workflow_summary),
                "workflow_history": action_input.get("workflow_history", state.workflow_hist),
            }
            prompt = build_prompt(system_prompt, step_prompt, prompt_input)
            try:
                llm_out = self.model_router.generate(prompt=prompt, task_type=task_type)
            except Exception as exc:
                llm_out = {
                    "raw_response": f"[Model error] {exc}",
                    "action": "none",
                    "action_input": {},
                }
            if not isinstance(llm_out, dict):
                llm_out = {"raw_response": "", "action": "none", "action_input": {}}

            raw_response = llm_out.get("raw_response")
            raw_response_text = raw_response if isinstance(raw_response, str) else ""
            state.update_state(
                role=agent_role,
                text=raw_response_text,
                prompt_engine=self.prompt_engine,
                model_router=self.model_router,
            )

            next_action_raw = llm_out.get("action")
            next_action = str(next_action_raw).strip().lower() if next_action_raw is not None else "none"
            next_action_input = llm_out.get("action_input", {})
            if not isinstance(next_action_input, dict):
                next_action_input = {}
            next_action_input.setdefault("agent_role", agent_role)
            next_action_input.setdefault("workflow_summary", state.workflow_summary)
            next_action_input.setdefault("workflow_history", state.workflow_hist)
            self.action_info = {
                "action": next_action,
                "action_input": next_action_input,
            }

            if next_action in TERMINAL_TOKENS:
                state.final_report = raw_response_text
                state.terminated = True
                state.update_state(
                    role="runtime",
                    text=f"loop_end> : {state.final_report}",
                    prompt_engine=self.prompt_engine,
                    model_router=self.model_router,
                )
                break

        if not state.terminated and turns >= max_turns:
            state.terminated = True
            state.final_report = state.final_report or "Loop ended by runtime limits."
            state.update_state(
                role="runtime",
                text=f"loop_end> : {state.final_report}",
                prompt_engine=self.prompt_engine,
                model_router=self.model_router,
            )

        if state.final_report is None:
            state.final_report = ""
        state.save_state()

    def run_core_session(
        self,
        state: StorageEngine,
        command_handler: Callable[[str], str] | None = None,
    ) -> None:
        self._ensure_runtime_fields(state)
        while True:
            try:
                line = input("user> ")
            except EOFError:
                print()
                break
            except KeyboardInterrupt:
                print("\nInterrupted. Use /exit to quit.")
                continue

            stripped = line.strip()
            if not stripped:
                print("No input provided.")
                continue

            if command_handler is not None and stripped.startswith("/"):
                command_out = command_handler(stripped)
                if command_out == "__EXIT__":
                    break
                if command_out:
                    print(command_out)
                continue

            state.update_state(
                role="user",
                text=stripped,
                prompt_engine=self.prompt_engine,
                model_router=self.model_router,
            )
            state.save_state()
            self.action_info = {
                "action": "call_llm",
                "action_input": {
                    "agent_role": "core_agent"
                }
            }
            self._run_working_loop(state=state)
