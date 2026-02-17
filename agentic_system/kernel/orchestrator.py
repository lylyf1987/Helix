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
        self._ensure_working_fields(state)

        action = self.env_info.get("action")
        turns = 0
        while action != "none":
            turns += 1
            if action == "call_llm":
                action_input = self.env_info.get("action_input", {})
                agent_role = str(action_input.get("agent_role", "core_agent")).strip()
                fallback_role = "core_agent" if self._agent_kind_for_role(agent_role) == "core" else "sub_agent"
                system_prompt = self.prompt_engine.get_system_prompt(agent_role, fallback_role=fallback_role)
                action_input["agent_role"] = agent_role
                action_input["system_prompt"] = system_prompt
                self.env_info["action_input"] = action_input
            caps = self.load_capability_snapshot(agent_role)
            self.compact_history_if_needed(
                state=state,
                caps=caps,
                step_prompt=self.prompt_engine.get_step_prompt(current_step),
                agent_role=agent_role,
                objective=objective,
            )
            llm_out = self.call_step_llm(
                state=state,
                current_step=current_step,
                caps=caps,
                agent_role=agent_role,
                objective=objective,
            )
            state.update_state(
                role=agent_role,
                text=llm_out.get("raw_response", ""),
                prompt_engine=self.prompt_engine,
                model_router=self.model_router,
            )

            structured = llm_out.get("action_input", {})
            if not isinstance(structured, dict):
                structured = {}
            forced: str | None = None

            try:
                if current_step == "context":
                    self.handle_context(state, structured)
                elif current_step == "retrieve_ltm":
                    self.handle_retrieve_ltm(state, structured)
                elif current_step == "plan":
                    self.handle_plan(state, structured, caps)
                elif current_step == "do_tasks":
                    forced = self.handle_do_tasks(state, structured, agent_role).get("force_next_step")
                elif current_step == "act":
                    out = self.handle_act(state, structured)
                    forced = out.get("force_next_step")
                    if out.get("status") == "success":
                        actions_executed += 1
                elif current_step == "verify":
                    forced = self.handle_verify(state, structured).get("force_next_step")
                elif current_step == "iterate":
                    forced = self.handle_iterate(state, structured).get("force_next_step")
                elif current_step == "create_sub_agent":
                    forced = self.handle_create_sub_agent(state, structured, agent_role).get("force_next_step")
                elif current_step == "assign_task":
                    forced = self.handle_assign_task(state, structured, depth, agent_role).get("force_next_step")
                elif current_step == "document":
                    forced = self.handle_document(state, structured).get("force_next_step")
                elif current_step == "create_skill":
                    forced = self.handle_create_skill(state, structured).get("force_next_step")
                elif current_step == "promotion_check":
                    forced = self.handle_promotion_check(state, structured).get("force_next_step")
                elif current_step == "report":
                    forced = self.handle_report(state, llm_out.get("raw_response", "")).get("force_next_step")
            except Exception as exc:
                state.update_state(
                    role="runtime",
                    text=f"handler_error> : {current_step}: {exc}",
                    prompt_engine=self.prompt_engine,
                    model_router=self.model_router,
                )
                forced = "report"

            if actions_executed >= int(limits["max_exec_actions"]):
                current_step = "report"
                continue

            proposed_next = forced if forced is not None else llm_out.get("action")
            next_action = self.validate_action_or_repair(state, proposed_next, agent_role, objective)

            if next_action in TERMINAL_TOKENS:
                state.terminated = True
                if not state.final_report:
                    state.final_report = llm_out.get("raw_response", "Loop complete.")
                state.update_state(
                    role="runtime",
                    text=f"loop_end> : {state.final_report}",
                    prompt_engine=self.prompt_engine,
                    model_router=self.model_router,
                )
                break

            current_step = str(next_action)

        if not state.final_report:
            state.final_report = "Loop ended by runtime limits."
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
                    "agent_role": "core_agent",
                    "workflow_summary": state.workflow_summary,
                    "workflow_history": state.workflow_hist,
                }
            }
            self._run_working_loop(state=state)
