"""Action dataclass and output parser for LLM responses."""

from __future__ import annotations

import re
import textwrap
from dataclasses import dataclass, field


# --------------------------------------------------------------------------- #
# Action dataclass
# --------------------------------------------------------------------------- #

ALLOWED_CORE_ACTIONS = frozenset({"chat", "think", "exec", "delegate"})
ALLOWED_SUB_ACTIONS = frozenset({"chat", "think", "exec"})


@dataclass
class Action:
    """Agent output per turn.

    Attributes:
        response: Natural language reasoning/answer (always present).
        type:     One of "chat", "think", "exec", "delegate".
        payload:  {} for chat/think; exec or delegate details otherwise.
    """

    response: str
    type: str
    payload: dict = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# Parse errors
# --------------------------------------------------------------------------- #


class ActionParseError(Exception):
    """Raised when LLM output cannot be parsed into a valid Action."""

    def __init__(self, message: str, raw_text: str = ""):
        super().__init__(message)
        self.raw_text = raw_text


# --------------------------------------------------------------------------- #
# Tag-extraction helpers
# --------------------------------------------------------------------------- #


_ACTION_RE = re.compile(r"<action>\s*(.*?)\s*</action>", re.DOTALL)
_CHILD_RE = re.compile(r"<(\w+)\s*(?:/>|>(.*?)</\1>)", re.DOTALL)


def _extract_tag(body: str, name: str) -> str | None:
    """Return the body of <name>...</name>, or None if the tag is absent.

    A body wrapped in ``<![CDATA[...]]>`` is the verbatim escape hatch:
    the CDATA wrapper is stripped and the inner content is returned with
    no further whitespace processing. Use this when a script body contains
    the literal closing tag (e.g. ``</script>``) that would otherwise close
    the wrapping tag prematurely. The CDATA pattern is tried first so that
    a literal ``</name>`` inside the CDATA isn't mistaken for the close.

    Otherwise, whitespace policy: strip exactly one leading newline (the
    model writes ``<tag>\\n...`` for readability — that newline is
    formatting, not content), strip trailing whitespace, then
    ``textwrap.dedent`` to remove common leading indentation while
    preserving internal structure.
    """
    escaped = re.escape(name)
    cdata_match = re.search(
        rf"<{escaped}>\s*<!\[CDATA\[(.*?)\]\]>\s*</{escaped}>",
        body,
        re.DOTALL,
    )
    if cdata_match is not None:
        return cdata_match.group(1)

    plain_match = re.search(rf"<{escaped}>(.*?)</{escaped}>", body, re.DOTALL)
    if plain_match is None:
        return None
    raw = plain_match.group(1)

    if raw.startswith("\r\n"):
        raw = raw[2:]
    elif raw.startswith("\n"):
        raw = raw[1:]
    raw = raw.rstrip()
    if not raw:
        return ""
    return textwrap.dedent(raw)


def _extract_arg_list(args_body: str) -> list[str]:
    """Extract <arg>...</arg> children from inside <script_args>."""
    return [
        match.group(1).strip()
        for match in re.finditer(r"<arg>(.*?)</arg>", args_body, re.DOTALL)
    ]


# --------------------------------------------------------------------------- #
# Public parser
# --------------------------------------------------------------------------- #


def parse_action(
    raw_llm_output: str,
    *,
    allowed_actions: frozenset[str] = ALLOWED_CORE_ACTIONS,
) -> Action:
    """Parse raw LLM text into an Action.

    Expected format::

        Plain prose response to the latest context, no escaping required.

        <action>
          <exec>                              (or <chat/> | <think/> | <delegate>...</delegate>)
            <job_name>...</job_name>
            <code_type>bash</code_type>
            <script>raw multi-line code</script>
          </exec>
        </action>

    The action type is the single child tag of ``<action>``. Self-closing
    forms (``<chat/>``, ``<think/>``) are accepted alongside paired forms
    (``<chat></chat>``). Tag bodies are raw text — no JSON escaping, no
    HTML entities. If a ``<script>`` body must contain its own literal
    closing tag, wrap it in ``<![CDATA[...]]>``.

    Raises:
        ActionParseError: on any parsing or validation failure.
    """
    match = _ACTION_RE.search(raw_llm_output)
    if match is None:
        raise ActionParseError(
            "Missing <action>...</action> block in model response.",
            raw_text=raw_llm_output,
        )

    response = raw_llm_output[: match.start()].strip()
    if not response:
        raise ActionParseError(
            "Missing response prose before <action> block.",
            raw_text=raw_llm_output,
        )

    body = match.group(1).strip()
    if not body:
        raise ActionParseError(
            "Empty <action> block. Use one of <chat/>, <think/>, "
            "<exec>...</exec>, <delegate>...</delegate>.",
            raw_text=raw_llm_output,
        )

    children = list(_CHILD_RE.finditer(body))
    if len(children) == 0:
        raise ActionParseError(
            "Missing action child tag inside <action>. Use one of "
            "<chat/>, <think/>, <exec>...</exec>, <delegate>...</delegate>.",
            raw_text=raw_llm_output,
        )
    if len(children) > 1:
        names = [c.group(1) for c in children]
        raise ActionParseError(
            f"<action> must contain exactly one child tag, found {len(children)}: {names}.",
            raw_text=raw_llm_output,
        )

    child = children[0]
    if body[: child.start()].strip() or body[child.end():].strip():
        raise ActionParseError(
            "<action> body must contain only the child tag, no extra text.",
            raw_text=raw_llm_output,
        )

    name = child.group(1).strip().lower()
    content = child.group(2) or ""

    if name not in allowed_actions:
        raise ActionParseError(
            f"Unknown action child tag '<{name}>'. Allowed: {sorted(allowed_actions)}.",
            raw_text=raw_llm_output,
        )

    if name in ("chat", "think"):
        return Action(response=response, type=name, payload={})

    if name == "exec":
        payload = _parse_exec_input(content, raw_llm_output)
        _validate_exec_payload(payload, raw_llm_output)
    else:  # delegate
        payload = _parse_delegate_input(content)
        _validate_delegate_payload(payload, raw_llm_output)

    return Action(response=response, type=name, payload=payload)


# --------------------------------------------------------------------------- #
# action_input parsers
# --------------------------------------------------------------------------- #


def _parse_exec_input(input_body: str, raw_llm_output: str) -> dict:
    payload: dict = {}
    for field_name in ("job_name", "code_type", "script", "script_path"):
        value = _extract_tag(input_body, field_name)
        if value is not None:
            payload[field_name] = value

    timeout = _extract_tag(input_body, "timeout_seconds")
    if timeout:
        try:
            payload["timeout_seconds"] = int(timeout.strip())
        except ValueError:
            raise ActionParseError(
                f"exec action <timeout_seconds> must be an integer, got '{timeout.strip()}'.",
                raw_text=raw_llm_output,
            )

    args_body = _extract_tag(input_body, "script_args")
    if args_body is not None and args_body.strip():
        payload["script_args"] = _extract_arg_list(args_body)

    return payload


def _parse_delegate_input(input_body: str) -> dict:
    payload: dict = {}
    for field_name in ("role", "role_description", "objective", "context"):
        value = _extract_tag(input_body, field_name)
        if value is not None:
            payload[field_name] = value
    return payload


# --------------------------------------------------------------------------- #
# Payload validators
# --------------------------------------------------------------------------- #


def _validate_exec_payload(payload: dict, raw_text: str) -> None:
    """Validate exec action_input has required fields."""
    code_type = str(payload.get("code_type", "")).strip().lower()
    if code_type not in ("bash", "python"):
        raise ActionParseError(
            f"exec action requires code_type 'bash' or 'python', got '{code_type}'.",
            raw_text=raw_text,
        )

    has_script = bool(str(payload.get("script", "")).strip())
    has_path = bool(str(payload.get("script_path", "")).strip())

    if not has_script and not has_path:
        raise ActionParseError(
            "exec action requires either 'script' or 'script_path'.",
            raw_text=raw_text,
        )
    if has_script and has_path:
        raise ActionParseError(
            "exec action must have exactly one of 'script' or 'script_path', not both.",
            raw_text=raw_text,
        )

    args = payload.get("script_args")
    if not args:
        return
    if has_script:
        raise ActionParseError(
            "exec action 'script_args' is only allowed when using 'script_path'.",
            raw_text=raw_text,
        )
    if any(not str(item).strip() for item in args):
        raise ActionParseError(
            "exec action 'script_args' must contain only non-empty argument strings.",
            raw_text=raw_text,
        )


_ROLE_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


def _validate_delegate_payload(payload: dict, raw_text: str) -> None:
    """Validate delegate action_input has required fields."""
    role = str(payload.get("role", "")).strip()
    if not role:
        raise ActionParseError(
            "delegate action requires a non-empty 'role' field.",
            raw_text=raw_text,
        )
    if not _ROLE_RE.fullmatch(role):
        raise ActionParseError(
            f"delegate role must match ^[A-Za-z0-9][A-Za-z0-9._-]*$, got '{role}'.",
            raw_text=raw_text,
        )
    if not str(payload.get("objective", "")).strip():
        raise ActionParseError(
            "delegate action requires a non-empty 'objective' field.",
            raw_text=raw_text,
        )
