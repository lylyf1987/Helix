"""Universal agent loop — the heart of the framework.

This single loop is used by both core-agents and sub-agents.
The entire orchestration reduces to: state → agent → action → environment → observation.
"""

from __future__ import annotations

import json
import random
import sys
import time
from pathlib import Path
from typing import Any, TextIO, Callable

from ..core.action import Action, ActionParseError, ALLOWED_SUB_ACTIONS
from ..core.agent import Agent
from ..core.environment import Environment, CompactionError, ExecutionInterrupted
from ..core.state import State, Turn
from ..providers.openai_compat import LLMTransientError
from .display import iter_exec_payload_items, write_runtime


DEFAULT_MAX_TURNS = 9999999
DEFAULT_MAX_RETRIES = 10
DEFAULT_LLM_RETRIES = 30
_LLM_RETRY_BASE_DELAY = 2.0
_LLM_RETRY_MAX_DELAY = 60.0


def run_loop(
    agent: Agent,
    env: Environment,
    *,
    model: Any = None,
    max_turns: int = DEFAULT_MAX_TURNS,
    max_retries: int = DEFAULT_MAX_RETRIES,
    output: TextIO = sys.stdout,
    on_turn_start: Callable[[str], None] | None = None,
    on_turn_end: Callable[[], None] | None = None,
    on_turn_error: Callable[[], None] | None = None,
    on_token_chunk: Callable[[str], None] | None = None,
) -> str:
    """Universal agent loop.

    Used identically by core-agents and sub-agents.
    Runs until the agent emits a "chat" action (returning control to the
    caller) or the turn limit is reached.

    Args:
        agent: The LLM-based agent.
        env: The sandbox environment.
        max_turns: Maximum loop iterations before forced stop.
        max_retries: Maximum consecutive parse failures before forced stop.
        output: Stream for runtime status messages.
        on_turn_start: Optional callback fired before the agent acts.
        on_turn_end: Optional callback fired after a valid agent output is finalized.
        on_turn_error: Optional callback fired after a parse-failed attempt.
        on_token_chunk: Optional callback for streaming agent responses.

    Returns:
        The agent's final response text.
    """
    consecutive_failures = 0

    for _ in range(max_turns):
        # 1. Build state from environment (compacts if needed)
        try:
            state = env.build_state()
        except CompactionError as exc:
            msg = (
                f"Session paused: context window is full and compaction failed ({exc}). "
                f"Please start a new session or reduce context."
            )
            _print(output, f"runtime> {msg}\n")
            return msg

        # 2. Agent decides
        if on_turn_start:
            on_turn_start(agent.role)
        try:
            action = _act_with_retry(
                agent, state,
                chunk_callback=on_token_chunk,
                on_turn_error=on_turn_error,
                output=output,
            )
            consecutive_failures = 0
        except ActionParseError as exc:
            if on_turn_error:
                on_turn_error()
            consecutive_failures += 1
            _print(output, f"runtime> Output parse error (attempt {consecutive_failures}/{max_retries}): {exc}\n")
            # Record only the first parse error to avoid flooding the observation window.
            if consecutive_failures == 1:
                env.record(Turn(
                    role="runtime",
                    content=f"Output parse error: {exc}. Please respond with valid <output>...</output> JSON.",
                ))
            if consecutive_failures >= max_retries:
                msg = "Loop ended: too many consecutive parse failures."
                _print(output, f"runtime> {msg}\n")
                return msg
            continue

        # 3. Finalize streaming display (adds trailing newline)
        if on_turn_end:
            on_turn_end()

        # 4. Record agent turn with full action details
        record_content = _format_agent_record(action)
        env.record(Turn(
            role=agent.role,
            content=record_content,
        ))

        # 5. Execute action
        if action.type == "chat":
            # Done — return to caller
            return action.response

        if action.type == "think":
            # Loop continues — response already recorded
            pass

        elif action.type == "exec":
            _print(output, f"runtime> Executing: {action.payload.get('job_name', 'unnamed')}...\n")
            try:
                observation = env.execute(action)
            except ExecutionInterrupted as exc:
                env.record(exc.observation)
                _print(output, f"runtime> {exc.observation.content}\n")
                return exc.observation.content
            env.record(observation)

        elif action.type == "delegate":
            _print(output, f"runtime> Delegating to sub-agent: {action.payload.get('role', 'unknown')}...\n")
            result = _delegate(action, env, model)
            env.record(Turn(
                role="sub_agent",
                content=result,
            ))

    # Turn limit reached
    msg = "Loop ended: maximum turns reached."
    _print(output, f"runtime> {msg}\n")
    return msg


# --------------------------------------------------------------------------- #
# Sub-agent delegation
# --------------------------------------------------------------------------- #


def _delegate(action: Action, env: Environment, model: Any) -> str:
    """Spawn a sub-agent to handle a delegated task.

    Creates an isolated Environment sharing the parent's workspace, executor,
    compactor, and approval hook, then runs a recursive loop.

    If state_root is set on env, sub-agent state is persisted and restored
    across delegations to the same role.
    """
    task = action.payload

    if model is None:
        return "Delegation failed: no model reference. Pass model= to run_loop()."

    role = task.get("role", "assistant")
    objective = task.get("objective", "")
    context = task.get("context", "")
    role_description = task.get("role_description", "")

    # Build sub-environment sharing parent's infrastructure
    sub_env = Environment(
        workspace=env.workspace,
        mode=env.mode,
        token_limit=env.token_limit,
        keep_last_k=env.keep_last_k,
        executor=env._executor,
        compactor=env._compactor,
        state_root=env.state_root,
    )
    sub_env._on_before_execute = env._on_before_execute

    # Restore previous sub-agent state if available
    state_root = env.state_root
    if state_root is not None:
        sub_state_path = state_root / "sub_agents" / f"{role}.json"
        sub_env.load_session(sub_state_path)

    # Append new objective as a core_agent turn (like a new user message)
    seed_content = objective
    if context:
        seed_content += f"\n\nContext:\n{context}"
    sub_env.record(Turn(role="core_agent", content=seed_content))

    # Build sub-agent (cannot delegate further)
    sub_agent = Agent(
        model,
        workspace=env.workspace,
        role="sub_agent",
        sub_agent_role=role,
        sub_agent_description=role_description,
        allowed_actions=ALLOWED_SUB_ACTIONS,
    )

    result = run_loop(sub_agent, sub_env)

    # Persist sub-agent state and update meta registry
    if state_root is not None:
        sub_state_path = state_root / "sub_agents" / f"{role}.json"
        sub_env.save_session(sub_state_path)
        _update_sub_agents_meta(state_root, role, role_description)

    return result


# --------------------------------------------------------------------------- #
# Sub-agent meta registry
# --------------------------------------------------------------------------- #


def _load_sub_agents_meta(state_root: Path) -> list[dict]:
    """Load the sub-agents meta registry."""
    meta_path = state_root / "sub_agents_meta.json"
    if not meta_path.exists():
        return []
    try:
        data = json.loads(meta_path.read_text(encoding="utf-8"))
        return data if isinstance(data, list) else []
    except (json.JSONDecodeError, OSError):
        return []


def _save_sub_agents_meta(state_root: Path, meta: list[dict]) -> None:
    """Save the sub-agents meta registry."""
    meta_path = state_root / "sub_agents_meta.json"
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = meta_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    tmp.replace(meta_path)


def _update_sub_agents_meta(state_root: Path, role: str, description: str) -> None:
    """Add or update a sub-agent entry in the meta registry."""
    meta = _load_sub_agents_meta(state_root)
    for entry in meta:
        if entry.get("role") == role:
            if description:
                entry["description"] = description
            return _save_sub_agents_meta(state_root, meta)
    meta.append({
        "role": role,
        "description": description or f"Sub-agent: {role}",
    })
    _save_sub_agents_meta(state_root, meta)


def _act_with_retry(
    agent: Agent,
    state: State,
    *,
    chunk_callback: Callable[[str], None] | None,
    on_turn_error: Callable[[], None] | None,
    output: TextIO,
    max_retries: int = DEFAULT_LLM_RETRIES,
) -> Action:
    """Call agent.act() with retry on transient LLM errors.

    Retries with exponential backoff + jitter. Does NOT record anything
    to the environment — the agent never produced output.
    """
    last_exc: LLMTransientError | None = None
    for attempt in range(1, max_retries + 1):
        try:
            return agent.act(state, chunk_callback=chunk_callback)
        except LLMTransientError as exc:
            last_exc = exc
            if on_turn_error:
                on_turn_error()
            if attempt >= max_retries:
                raise
            delay = min(_LLM_RETRY_BASE_DELAY * (2 ** (attempt - 1)), _LLM_RETRY_MAX_DELAY) + random.uniform(0, 1)
            if exc.retry_after is not None:
                delay = max(delay, exc.retry_after)
            _print(output, f"runtime> LLM provider error (attempt {attempt}/{max_retries}, retrying in {delay:.1f}s): {exc}\n")
            time.sleep(delay)
    raise last_exc  # all retries exhausted


def _print(output: TextIO, text: str) -> None:
    """Print runtime-styled output."""
    if text:
        write_runtime(text, output)

# --------------------------------------------------------------------------- #
# Agent record formatting
# --------------------------------------------------------------------------- #


def _format_agent_record(action: Action) -> str:
    """Format agent response + action details into a readable record.

    This is what the LLM will see in workflow_history, so it must be clear
    enough for the LLM to trace its own decisions.

    Examples:
        I'll search the project structure.
        [next_action] exec
        [action_input]
          job_name: list-project-files
          code_type: bash
          script: find . -type f

        Let me think about the best approach.
        [next_action] think

        Here are the results you asked for.
        [next_action] chat

        I'll delegate the research to a sub-agent.
        [next_action] delegate
        [action_input]
          role: researcher
          objective: Find papers on RLHF
    """
    parts = [action.response, f"[next_action] {action.type}"]

    if action.type == "exec" and action.payload:
        lines = ["[action_input]"]
        for key, value in iter_exec_payload_items(action.payload):
            text = str(value)
            if "\n" in text:
                lines.append(f"  {key}:")
                lines.extend(f"    {row}" for row in text.split("\n"))
            else:
                lines.append(f"  {key}: {text}")
        if len(lines) > 1:
            parts.append("\n".join(lines))

    elif action.type == "delegate" and action.payload:
        lines = ["[action_input]"]
        for key in ("role", "role_description", "objective"):
            value = action.payload.get(key)
            if value:
                lines.append(f"  {key}: {value}")
        context = action.payload.get("context")
        if context:
            text = str(context)
            if "\n" in text:
                lines.append(f"  context:")
                lines.extend(f"    {row}" for row in text.split("\n"))
            else:
                lines.append(f"  context: {text}")
        if len(lines) > 1:
            parts.append("\n".join(lines))

    return "\n".join(parts)
