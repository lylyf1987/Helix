"""Phase 1 verification tests for the core RL-inspired abstractions."""

import re
import sys
import tempfile
from http.client import RemoteDisconnected
from io import StringIO
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from helix.core.state import State, Turn
from helix.core.action import (
    Action,
    parse_action,
    ActionParseError,
    ALLOWED_CORE_ACTIONS,
    ALLOWED_SUB_ACTIONS,
)
from helix.core.agent import Agent
from helix.core.compactor import Compactor
from helix.core.environment import Environment
from helix.runtime.loop import run_loop
from helix.runtime.approval import ApprovalPolicy


def test_turn_creation():
    t = Turn(role="user", content="hello")
    assert t.role == "user"
    assert t.content == "hello"
    assert t.timestamp  # auto-generated
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}", t.timestamp)
    print("  Turn creation OK")


def test_state_creation():
    t = Turn(role="user", content="hello")
    s = State(observation=[t], workflow_summary="")
    assert s.observation == [t]
    print("  State creation OK")


def test_parse_chat():
    raw = '<output>{"response": "Hi!", "next_action": "chat", "action_input": {}}</output>'
    a = parse_action(raw)
    assert a.type == "chat"
    assert a.response == "Hi!"
    assert a.payload == {}
    print("  Parse chat OK")


def test_parse_think():
    raw = '<output>{"response": "Thinking...", "next_action": "think", "action_input": {}}</output>'
    a = parse_action(raw)
    assert a.type == "think"
    print("  Parse think OK")


def test_parse_exec():
    raw = '<output>{"response": "Running", "next_action": "exec", "action_input": {"job_name": "test", "code_type": "bash", "script": "echo hello"}}</output>'
    a = parse_action(raw)
    assert a.type == "exec"
    assert a.payload["code_type"] == "bash"
    assert a.payload["script"] == "echo hello"
    print("  Parse exec OK")


def test_parse_exec_script_args_array():
    raw = '<output>{"response": "Running", "next_action": "exec", "action_input": {"job_name": "test", "code_type": "python", "script_path": "skills/x.py", "script_args": ["--query", "hello", "--limit", "5"]}}</output>'
    a = parse_action(raw)
    assert a.type == "exec"
    assert a.payload["script_args"] == ["--query", "hello", "--limit", "5"]
    print("  Parse exec script_args array OK")


def test_parse_exec_script_args_string():
    raw = '<output>{"response": "Running", "next_action": "exec", "action_input": {"job_name": "test", "code_type": "python", "script_path": "skills/x.py", "script_args": "--query \\"hello\\" --limit 5"}}</output>'
    a = parse_action(raw)
    assert a.type == "exec"
    assert a.payload["script_args"] == '--query "hello" --limit 5'
    print("  Parse exec script_args string OK")


def test_parse_delegate():
    raw = '<output>{"response": "Delegating", "next_action": "delegate", "action_input": {"role": "researcher", "objective": "Find info"}}</output>'
    a = parse_action(raw)
    assert a.type == "delegate"
    assert a.payload["role"] == "researcher"
    print("  Parse delegate OK")



def test_parse_error_missing_tags():
    try:
        parse_action("no tags here")
        assert False, "Should have raised"
    except ActionParseError:
        print("  Parse error (missing tags) OK")


def test_parse_error_invalid_json():
    try:
        parse_action("<output>not json</output>")
        assert False, "Should have raised"
    except ActionParseError:
        print("  Parse error (invalid JSON) OK")


def test_parse_error_invalid_action():
    raw = '<output>{"response": "Hi!", "next_action": "fly", "action_input": {}}</output>'
    try:
        parse_action(raw)
        assert False, "Should have raised"
    except ActionParseError:
        print("  Parse error (invalid action) OK")


def test_parse_error_exec_missing_script():
    raw = '<output>{"response": "R", "next_action": "exec", "action_input": {"code_type": "bash"}}</output>'
    try:
        parse_action(raw)
        assert False, "Should have raised"
    except ActionParseError:
        print("  Parse error (exec missing script) OK")


def test_parse_error_exec_both_script_and_path():
    raw = '<output>{"response": "R", "next_action": "exec", "action_input": {"code_type": "bash", "script": "echo x", "script_path": "/foo"}}</output>'
    try:
        parse_action(raw)
        assert False, "Should have raised"
    except ActionParseError:
        print("  Parse error (exec both script+path) OK")


def test_parse_error_exec_invalid_script_args_type():
    raw = '<output>{"response": "R", "next_action": "exec", "action_input": {"code_type": "python", "script_path": "skills/x.py", "script_args": {"bad": true}}}</output>'
    try:
        parse_action(raw)
        assert False, "Should have raised"
    except ActionParseError:
        print("  Parse error (exec invalid script_args type) OK")


def test_parse_error_exec_script_args_with_script():
    raw = '<output>{"response": "R", "next_action": "exec", "action_input": {"code_type": "bash", "script": "echo x", "script_args": ["--bad"]}}</output>'
    try:
        parse_action(raw)
        assert False, "Should have raised"
    except ActionParseError:
        print("  Parse error (exec script_args with script) OK")


def test_parse_error_delegate_missing_role():
    raw = '<output>{"response": "D", "next_action": "delegate", "action_input": {"objective": "o"}}</output>'
    try:
        parse_action(raw)
        assert False, "Should have raised"
    except ActionParseError:
        print("  Parse error (delegate missing role) OK")


def test_sub_agent_cannot_delegate():
    raw = '<output>{"response": "D", "next_action": "delegate", "action_input": {"role": "r", "objective": "o"}}</output>'
    try:
        parse_action(raw, allowed_actions=ALLOWED_SUB_ACTIONS)
        assert False, "Should have raised"
    except ActionParseError:
        print("  Sub-agent delegate denied OK")



def test_environment_dual_history():
    with tempfile.TemporaryDirectory() as td:
        env = Environment(workspace=Path(td))
        env.record(Turn(role="user", content="hello"))
        env.record(Turn(role="agent", content="hi back"))
        assert len(env.full_history) == 2
        assert len(env.observation) == 2
        print("  Dual history OK")


def test_environment_build_state():
    with tempfile.TemporaryDirectory() as td:
        env = Environment(workspace=Path(td))
        env.record(Turn(role="user", content="hello"))
        state = env.build_state()
        assert len(state.observation) == 1
        print("  build_state OK")


def test_environment_compaction():
    class CompactorModel:
        def generate(self, messages, *, chunk_callback=None, **_kwargs):
            return "## Session Goal\nTest session\n## Current Status\nCompacted."

    with tempfile.TemporaryDirectory() as td:
        env = Environment(workspace=Path(td), token_limit=200, keep_last_k=2,
                          compactor=Compactor(CompactorModel()))
        # Add many turns to exceed budget
        for i in range(20):
            env.record(Turn(role="agent", content=f"Turn {i} " + "x" * 100))
        assert len(env.full_history) == 20  # full_history never compacted
        state = env.build_state()
        # After compaction: only the 2 recent turns remain in observation.
        assert len(state.observation) == 2
        assert env.workflow_summary  # summary was generated
        assert len(env.full_history) == 20  # full_history still intact
        print("  Compaction OK")


def test_environment_will_compact():
    """will_compact() predicts whether the next build_state() will invoke the compactor."""
    with tempfile.TemporaryDirectory() as td:
        # Case 1: observation fits within token budget → no compaction.
        env = Environment(workspace=Path(td), token_limit=200, keep_last_k=2)
        env.record(Turn(role="user", content="small"))
        assert env.will_compact() is False

        # Case 2: observation exceeds budget AND len(obs) > keep_last_k → compaction.
        env = Environment(workspace=Path(td), token_limit=200, keep_last_k=2)
        for i in range(20):
            env.record(Turn(role="agent", content=f"Turn {i} " + "x" * 100))
        assert env.will_compact() is True

        # Case 3: observation exceeds budget but len(obs) <= keep_last_k → nothing to compact.
        env = Environment(workspace=Path(td), token_limit=50, keep_last_k=2)
        env.record(Turn(role="user", content="x" * 400))
        assert env.will_compact() is False
        print("  will_compact OK")


def test_agent_messages_structured_context():
    state = State(
        observation=[
            Turn(role="runtime", content="Job succeeded."),
            Turn(role="user", content="what now?"),
        ],
        workflow_summary="## Summary\nCompacted",
    )
    with tempfile.TemporaryDirectory() as td:
        agent = Agent(type("MockModel", (), {"generate": lambda self, *args, **kwargs: ""})(), workspace=Path(td))

        messages = agent._build_messages(state)

        # System prompt is clean (no summary mixed in)
        assert messages[0]["role"] == "system"
        assert "## Summary\nCompacted" not in messages[0]["content"]
        # Structured context in single user message
        assert len(messages) == 2
        assert messages[1]["role"] == "user"
        ctx = messages[1]["content"]
        assert "<workflow_summary>" in ctx
        assert "## Summary\nCompacted" in ctx
        assert "<workflow_history>" in ctx
        assert "runtime> Job succeeded." in ctx
        assert "<latest_context>" in ctx
        assert "user> what now?" in ctx
        print("  Structured context (summary + history + latest) OK")


def test_environment_compaction_error():
    """Compaction raises CompactionError when no compactor is available."""
    from helix.core.compactor import CompactionError

    with tempfile.TemporaryDirectory() as td:
        env = Environment(workspace=Path(td), token_limit=200, keep_last_k=2)
        # No compactor set — compaction should fail
        for i in range(20):
            env.record(Turn(role="agent", content=f"Turn {i} " + "x" * 100))
        try:
            env.build_state()
            assert False, "Should have raised CompactionError"
        except CompactionError:
            pass
        print("  Compaction error (no model) OK")


def test_environment_persistence():
    with tempfile.TemporaryDirectory() as td:
        env = Environment(workspace=Path(td))
        env.record(Turn(role="user", content="hello"))
        env.record(Turn(role="agent", content="world"))
        env.workflow_summary = "test summary"

        sp = Path(td) / "session.json"
        env.save_session(sp)

        env2 = Environment(workspace=Path(td))
        loaded = env2.load_session(sp)
        assert loaded
        assert len(env2.full_history) == 2
        assert len(env2.observation) == 2
        assert env2.workflow_summary == "test summary"
        assert env2.full_history[0].content == "hello"
        print("  Persistence OK")


def test_run_loop_compaction_failure_records_runtime_turn():
    """Compaction failures are runtime-handled: retry, then record and return.

    The unified error handling contract treats ``CompactionError`` like
    ``LLMTransientError`` — runtime retries with backoff, then on exhaustion
    records a runtime Turn and returns to the requester (user or parent
    core-agent). The agent cannot influence infrastructure issues so there
    is no point feeding the error back into the agent's prompt.
    """

    class FailingCompactionModel:
        def __init__(self) -> None:
            self.calls = 0

        def generate(self, messages, *, chunk_callback=None, **_kwargs):
            self.calls += 1
            raise RemoteDisconnected("compactor closed connection")

    class UnusedAgentModel:
        def generate(self, messages, *, chunk_callback=None, **_kwargs):
            assert False, "Agent model should not be called when compaction fails"

    # Collapse the outer compaction retry to a single attempt so the test
    # runs fast; the Compactor class still performs its own 3-attempt inner
    # retry (see helix/core/compactor.py), so the model is called 3 times.
    from helix.runtime import loop as loop_module
    original_retries = loop_module.DEFAULT_COMPACTION_RETRIES
    loop_module.DEFAULT_COMPACTION_RETRIES = 1
    try:
        with tempfile.TemporaryDirectory() as td:
            compactor_model = FailingCompactionModel()
            env = Environment(workspace=Path(td), token_limit=10, keep_last_k=1,
                              compactor=Compactor(compactor_model))
            env.record(Turn(role="user", content="x" * 120))
            env.record(Turn(role="runtime", content="y" * 120))

            agent = Agent(UnusedAgentModel(), workspace=Path(td))
            captured = StringIO()
            result = run_loop(agent, env, output=captured)

            assert "compaction failed" in result.lower()
            assert compactor_model.calls == 3
            assert "compaction failed" in captured.getvalue().lower()
            # Runtime Turn is appended so the requester sees what happened.
            assert len(env.full_history) == 3
            assert env.full_history[-1].role == "runtime"
            assert "compaction failed" in env.full_history[-1].content.lower()
            print("  run_loop (compaction failure recorded) OK")
    finally:
        loop_module.DEFAULT_COMPACTION_RETRIES = original_retries


def test_run_loop_notifies_on_compaction_start():
    """When compaction is about to run, the user sees a 'Context window full' notice."""

    class SuccessfulCompactorModel:
        def generate(self, messages, *, chunk_callback=None, **_kwargs):
            return "## Session Goal\nTest\n## Current Status\nCompacted."

    class ChatAgentModel:
        def generate(self, messages, *, chunk_callback=None, **_kwargs):
            return '<output>{"response": "done", "next_action": "chat", "action_input": {}}</output>'

    with tempfile.TemporaryDirectory() as td:
        env = Environment(workspace=Path(td), token_limit=200, keep_last_k=2,
                          compactor=Compactor(SuccessfulCompactorModel()))
        for i in range(20):
            env.record(Turn(role="agent", content=f"Turn {i} " + "x" * 100))

        agent = Agent(ChatAgentModel(), workspace=Path(td))
        captured = StringIO()
        run_loop(agent, env, output=captured)

        output_text = captured.getvalue().lower()
        assert "context window full" in output_text
        assert "compacting older chat history" in output_text
        print("  run_loop (compaction-start notification) OK")


def test_run_loop_compaction_notification_fires_once_per_turn():
    """The 'Context window full' notice appears once per turn, not once per retry."""

    class FailingCompactorModel:
        def generate(self, messages, *, chunk_callback=None, **_kwargs):
            raise RemoteDisconnected("compactor closed connection")

    class UnusedAgentModel:
        def generate(self, messages, *, chunk_callback=None, **_kwargs):
            assert False, "Agent model should not be called when compaction fails"

    from helix.runtime import loop as loop_module
    original_retries = loop_module.DEFAULT_COMPACTION_RETRIES
    original_base_delay = loop_module._RETRY_BASE_DELAY
    loop_module.DEFAULT_COMPACTION_RETRIES = 3
    loop_module._RETRY_BASE_DELAY = 0.0  # make the test fast
    try:
        with tempfile.TemporaryDirectory() as td:
            env = Environment(workspace=Path(td), token_limit=10, keep_last_k=1,
                              compactor=Compactor(FailingCompactorModel()))
            env.record(Turn(role="user", content="x" * 120))
            env.record(Turn(role="runtime", content="y" * 120))

            agent = Agent(UnusedAgentModel(), workspace=Path(td))
            captured = StringIO()
            run_loop(agent, env, output=captured)

            output_text = captured.getvalue().lower()
            assert output_text.count("context window full") == 1, (
                f"expected exactly one compaction-start notice, got "
                f"{output_text.count('context window full')}:\n{output_text}"
            )
            # Retry/failure messages should still be present (shows we did retry).
            assert "compaction error" in output_text
            print("  run_loop (notification fires once per turn) OK")
    finally:
        loop_module.DEFAULT_COMPACTION_RETRIES = original_retries
        loop_module._RETRY_BASE_DELAY = original_base_delay


def test_run_loop_chat():
    """Test that run_loop correctly handles a mock agent returning chat."""

    class MockModel:
        def generate(self, messages, *, chunk_callback=None, **_kwargs):
            return '<output>{"response": "Hello user!", "next_action": "chat", "action_input": {}}</output>'

    with tempfile.TemporaryDirectory() as td:
        env = Environment(workspace=Path(td))
        env.record(Turn(role="user", content="hi"))
        agent = Agent(MockModel(), workspace=Path(td))
        result = run_loop(agent, env, output=sys.stderr)
        assert result == "Hello user!"
        assert len(env.full_history) == 2  # user + agent
        print("  run_loop (chat) OK")


def test_run_loop_reasoning_tokens_never_enter_history():
    """Reasoning-stream tokens go to the callback only — history stays clean."""

    class ReasoningModel:
        def generate(self, messages, *, chunk_callback=None, reasoning_callback=None, **_kwargs):
            if reasoning_callback is not None:
                reasoning_callback("INTERNAL_THOUGHTS_SHOULD_NEVER_LEAK ")
                reasoning_callback("AND_NEITHER_SHOULD_THIS")
            final = '<output>{"response": "Clean answer", "next_action": "chat", "action_input": {}}</output>'
            if chunk_callback is not None:
                chunk_callback(final)
            return final

    reasoning_seen: list[str] = []

    with tempfile.TemporaryDirectory() as td:
        env = Environment(workspace=Path(td))
        env.record(Turn(role="user", content="hi"))
        agent = Agent(ReasoningModel(), workspace=Path(td))
        result = run_loop(
            agent, env,
            output=sys.stderr,
            on_reasoning_chunk=reasoning_seen.append,
        )

    # Reasoning reached the callback
    assert reasoning_seen == [
        "INTERNAL_THOUGHTS_SHOULD_NEVER_LEAK ",
        "AND_NEITHER_SHOULD_THIS",
    ]
    # But nothing reasoning-related is in the returned answer or any history turn
    assert result == "Clean answer"
    for turn in env.full_history:
        assert "INTERNAL_THOUGHTS" not in turn.content
        assert "NEITHER_SHOULD_THIS" not in turn.content
    print("  run_loop reasoning stays out of history OK")


def test_run_loop_think_then_chat():
    """Test that run_loop handles think -> chat sequence."""
    call_count = [0]

    class MockModel:
        def generate(self, messages, *, chunk_callback=None, **_kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                return '<output>{"response": "Let me think...", "next_action": "think", "action_input": {}}</output>'
            return '<output>{"response": "Done thinking!", "next_action": "chat", "action_input": {}}</output>'

    with tempfile.TemporaryDirectory() as td:
        env = Environment(workspace=Path(td))
        env.record(Turn(role="user", content="think first"))
        agent = Agent(MockModel(), workspace=Path(td))
        result = run_loop(agent, env, output=sys.stderr)
        assert result == "Done thinking!"
        assert call_count[0] == 2
        # user + think_response + chat_response = 3
        assert len(env.full_history) == 3
        print("  run_loop (think→chat) OK")


def test_run_loop_parse_retry():
    """Test that run_loop retries on parse failure then succeeds."""
    call_count = [0]

    class MockModel:
        def generate(self, messages, *, chunk_callback=None, **_kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                return "bad output no tags"
            return '<output>{"response": "Fixed!", "next_action": "chat", "action_input": {}}</output>'

    with tempfile.TemporaryDirectory() as td:
        env = Environment(workspace=Path(td))
        env.record(Turn(role="user", content="test"))
        agent = Agent(MockModel(), workspace=Path(td))
        result = run_loop(agent, env, output=sys.stderr)
        assert result == "Fixed!"
        assert call_count[0] == 2
        print("  run_loop (parse retry) OK")


def test_run_loop_parse_retry_observation_reflects_latest_error():
    """Exactly one parse-error runtime Turn lives in observation across a
    multi-retry cycle; its content is overwritten in place so it reflects
    the *latest* failure, not the first."""
    call_count = [0]

    class MockModel:
        def generate(self, messages, *, chunk_callback=None, **_kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                return "bad output no tags"                          # missing <output> tags
            if call_count[0] == 2:
                return "<output>not-valid-json</output>"              # invalid JSON
            if call_count[0] == 3:
                return (
                    '<output>{"response": "x", '
                    '"next_action": "banana", "action_input": {}}</output>'
                )                                                     # invalid next_action
            return (
                '<output>{"response": "Finally fixed.", '
                '"next_action": "chat", "action_input": {}}</output>'
            )

    with tempfile.TemporaryDirectory() as td:
        env = Environment(workspace=Path(td))
        env.record(Turn(role="user", content="test"))
        agent = Agent(MockModel(), workspace=Path(td))
        result = run_loop(agent, env, output=sys.stderr)

        assert result == "Finally fixed."
        assert call_count[0] == 4

        # Exactly one parse-error runtime Turn, in both history views.
        full_parse_turns = [
            t for t in env.full_history
            if t.role == "runtime" and "Output parse error:" in t.content
        ]
        obs_parse_turns = [
            t for t in env.observation
            if t.role == "runtime" and "Output parse error:" in t.content
        ]
        assert len(full_parse_turns) == 1, (
            f"expected exactly one recorded parse-error turn, got {len(full_parse_turns)}"
        )
        assert len(obs_parse_turns) == 1

        # Content reflects the last failure (invalid next_action 'banana'),
        # not the first (missing tags) or the second (invalid JSON).
        final_turn = full_parse_turns[0]
        assert "banana" in final_turn.content
        assert "Missing <output>" not in final_turn.content
        assert "Invalid JSON" not in final_turn.content
        print("  run_loop (parse retry records latest error) OK")


def test_run_loop_max_retries():
    """Test that run_loop stops after DEFAULT_PARSE_RETRIES consecutive failures.

    The retry cap is a module-level constant rather than a parameter, so we
    patch it to a small value for the duration of the test.
    """

    class MockModel:
        def generate(self, messages, *, chunk_callback=None, **_kwargs):
            return "always bad"

    from helix.runtime import loop as loop_module
    original_retries = loop_module.DEFAULT_PARSE_RETRIES
    loop_module.DEFAULT_PARSE_RETRIES = 2
    try:
        with tempfile.TemporaryDirectory() as td:
            env = Environment(workspace=Path(td))
            env.record(Turn(role="user", content="test"))
            agent = Agent(MockModel(), workspace=Path(td))
            result = run_loop(agent, env, output=sys.stderr)
            assert "parse failures" in result.lower()
            print("  run_loop (max retries) OK")
    finally:
        loop_module.DEFAULT_PARSE_RETRIES = original_retries


def test_run_loop_exec_denied_returns_control():
    """Approval denial is a user abort: UserInterrupted unwinds run_loop back
    to the REPL caller without the agent taking another turn. The denial Turn
    still lands in full_history/observation so the next user prompt has it
    as context."""
    from helix.core.environment import UserInterrupted

    call_count = [0]

    class MockModel:
        def generate(self, messages, *, chunk_callback=None, **_kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                return (
                    '<output>'
                    '{"response": "Checking status.", '
                    '"next_action": "exec", '
                    '"action_input": {"job_name": "check-status", "code_type": "bash", "script": "echo x"}}'
                    '</output>'
                )
            raise AssertionError("run_loop should have unwound before a second agent turn")

    with tempfile.TemporaryDirectory() as td:
        env = Environment(workspace=Path(td), mode="controlled")
        env.on_before_execute(ApprovalPolicy(mode="controlled", prompt=lambda _prompt: "n"))
        env.record(Turn(role="user", content="check status"))
        agent = Agent(MockModel(), workspace=Path(td))

        raised = False
        try:
            run_loop(agent, env, output=sys.stderr)
        except UserInterrupted as exc:
            raised = True
            assert "denied by user" in exc.observation.content.lower()
            assert "core_agent" in exc.observation.content
            assert exc.observation.role == "runtime"

        assert raised, "run_loop must raise UserInterrupted on approval deny"
        assert call_count[0] == 1
        # Denial lands in full_history (and observation) so the next user
        # prompt sees it as context.
        runtime_turns = [t for t in env.full_history if t.role == "runtime"]
        assert any(
            "core_agent" in t.content and "denied by user" in t.content.lower()
            for t in runtime_turns
        )
        observation_turns = [t for t in env.observation if t.role == "runtime"]
        assert any(
            "core_agent" in t.content and "denied by user" in t.content.lower()
            for t in observation_turns
        )
        print("  run_loop (deny → UserInterrupted → returns to user) OK")


def test_run_loop_exec_cancelled_returns_control():
    """Ctrl+C at the approval prompt is a user abort: UserInterrupted unwinds
    run_loop back to the REPL caller without the agent taking another turn."""
    from helix.core.environment import UserInterrupted

    call_count = [0]

    class MockModel:
        def generate(self, messages, *, chunk_callback=None, **_kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                return (
                    '<output>'
                    '{"response": "Checking status.", '
                    '"next_action": "exec", '
                    '"action_input": {"job_name": "check-status", "code_type": "bash", "script": "echo x"}}'
                    '</output>'
                )
            # Unreachable under the new semantics; if the test sees call 2
            # something went wrong with the unwind.
            raise AssertionError("run_loop should have unwound before a second agent turn")

    with tempfile.TemporaryDirectory() as td:
        env = Environment(workspace=Path(td), mode="controlled")
        env.on_before_execute(ApprovalPolicy(
            mode="controlled",
            prompt=lambda _prompt: (_ for _ in ()).throw(KeyboardInterrupt()),
        ))
        env.record(Turn(role="user", content="check status"))
        agent = Agent(MockModel(), workspace=Path(td))

        raised = False
        try:
            run_loop(agent, env, output=sys.stderr)
        except UserInterrupted as exc:
            raised = True
            assert "cancelled by user" in exc.observation.content
            # Loop prefixed with the agent's role on its way up.
            assert "core_agent" in exc.observation.content
            assert exc.observation.role == "runtime"

        assert raised, "run_loop must raise UserInterrupted on Ctrl+C at approval prompt"
        assert call_count[0] == 1
        # Core env should have the prefixed interrupt turn recorded.
        runtime_turns = [t for t in env.full_history if t.role == "runtime"]
        assert any(
            "core_agent" in t.content and "cancelled by user" in t.content
            for t in runtime_turns
        )
        print("  run_loop (approval Ctrl+C → UserInterrupted) OK")


def test_run_loop_transient_error_retries_then_succeeds():
    """Transient LLM error should retry without recording a Turn."""
    from helix.providers.openai_compat import LLMTransientError
    from unittest.mock import patch

    call_count = [0]

    class MockModel:
        def generate(self, messages, *, chunk_callback=None, **_kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                raise LLMTransientError("LLM HTTP 503: overloaded", status_code=503)
            return '<output>{"response": "Recovered!", "next_action": "chat", "action_input": {}}</output>'

    with tempfile.TemporaryDirectory() as td:
        env = Environment(workspace=Path(td))
        env.record(Turn(role="user", content="test"))
        agent = Agent(MockModel(), workspace=Path(td))
        captured = StringIO()
        with patch("helix.runtime.loop.time.sleep"):
            result = run_loop(agent, env, output=captured)

        assert result == "Recovered!"
        assert call_count[0] == 2
        # No transient error recorded as Turn — only user + agent
        assert len(env.full_history) == 2
        assert all(t.role != "runtime" for t in env.full_history)
        assert "retrying" in captured.getvalue().lower()
        print("  run_loop (transient retry → success) OK")


def test_run_loop_transient_error_exhausts_retries():
    """After DEFAULT_LLM_RETRIES attempts, transient error is recorded and returned.

    Under the unified error handling contract, ``LLMTransientError`` is
    runtime-handled: ``_act_with_retry`` retries up to ``DEFAULT_LLM_RETRIES``
    times, then on exhaustion ``run_loop`` catches the exception, records a
    runtime Turn, and returns a summary string to the requester. The error
    must NOT propagate out of ``run_loop``.
    """
    from helix.providers.openai_compat import LLMTransientError
    from helix.runtime.loop import DEFAULT_LLM_RETRIES
    from unittest.mock import patch

    call_count = [0]

    class MockModel:
        def generate(self, messages, *, chunk_callback=None, **_kwargs):
            call_count[0] += 1
            raise LLMTransientError("LLM HTTP 429: rate limited", status_code=429)

    with tempfile.TemporaryDirectory() as td:
        env = Environment(workspace=Path(td))
        env.record(Turn(role="user", content="test"))
        agent = Agent(MockModel(), workspace=Path(td))
        with patch("helix.runtime.loop.time.sleep"):
            result = run_loop(agent, env, output=StringIO())

        assert "llm provider error" in result.lower()
        assert call_count[0] == DEFAULT_LLM_RETRIES
        # Runtime Turn is appended so the requester sees what happened.
        assert len(env.full_history) == 2  # user + runtime
        assert env.full_history[-1].role == "runtime"
        assert "llm provider error" in env.full_history[-1].content.lower()
        print("  run_loop (transient retry exhausted) OK")


def test_run_loop_permanent_error_no_retry():
    """Permanent LLM error should propagate immediately without retry."""
    from helix.providers.openai_compat import LLMPermanentError

    call_count = [0]

    class MockModel:
        def generate(self, messages, *, chunk_callback=None, **_kwargs):
            call_count[0] += 1
            raise LLMPermanentError("LLM HTTP 401: unauthorized", status_code=401)

    with tempfile.TemporaryDirectory() as td:
        env = Environment(workspace=Path(td))
        env.record(Turn(role="user", content="test"))
        agent = Agent(MockModel(), workspace=Path(td))
        try:
            run_loop(agent, env, output=StringIO())
            assert False, "Expected LLMPermanentError to propagate"
        except LLMPermanentError:
            pass

        assert call_count[0] == 1  # no retry
        print("  run_loop (permanent error no retry) OK")


if __name__ == "__main__":
    print("=== State & Turn ===")
    test_turn_creation()
    test_state_creation()

    print("\n=== Action Parsing ===")
    test_parse_chat()
    test_parse_think()
    test_parse_exec()
    test_parse_exec_script_args_array()
    test_parse_exec_script_args_string()
    test_parse_delegate()

    print("\n=== Parse Errors ===")
    test_parse_error_missing_tags()
    test_parse_error_invalid_json()
    test_parse_error_invalid_action()
    test_parse_error_exec_missing_script()
    test_parse_error_exec_both_script_and_path()
    test_parse_error_exec_invalid_script_args_type()
    test_parse_error_exec_script_args_with_script()
    test_parse_error_delegate_missing_role()
    test_sub_agent_cannot_delegate()

    print("\n=== Environment ===")
    test_environment_dual_history()
    test_environment_build_state()
    test_environment_compaction()
    test_environment_will_compact()
    test_agent_messages_structured_context()
    test_environment_compaction_error()
    test_environment_persistence()

    print("\n=== Universal Loop ===")
    test_run_loop_chat()
    test_run_loop_think_then_chat()
    test_run_loop_parse_retry()
    test_run_loop_parse_retry_observation_reflects_latest_error()
    test_run_loop_max_retries()
    test_run_loop_exec_denied_returns_control()
    test_run_loop_exec_cancelled_returns_control()
    test_run_loop_compaction_failure_records_runtime_turn()
    test_run_loop_notifies_on_compaction_start()
    test_run_loop_compaction_notification_fires_once_per_turn()

    print("\n=== LLM Provider Retry ===")
    test_run_loop_transient_error_retries_then_succeeds()
    test_run_loop_transient_error_exhausts_retries()
    test_run_loop_permanent_error_no_retry()

    print("\n✅ All Phase 1 tests passed!")
