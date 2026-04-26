"""Streaming display callbacks and extraction for the agent loop."""

from __future__ import annotations

import re
import sys
from typing import Any, Optional, TextIO

_EXEC_PAYLOAD_ORDER = (
    "job_name",
    "code_type",
    "script_path",
    "script",
    "script_args",
    "timeout_seconds",
)

# --------------------------------------------------------------------------- #
# ANSI color scheme
# --------------------------------------------------------------------------- #

# ANSI styles
_RESET = "\033[0m"
_BOLD = "\033[1m"
_DIM = "\033[2m"

# Prefix shown before the first reasoning-stream token.
_THINKING_PREFIX = "thinking> "

# Role prefix badges: bold + colored background + white text
_BADGE = {
    "user":       f"{_BOLD}\033[48;5;240m\033[38;5;255m",   # gray badge
    "core_agent": f"{_BOLD}\033[48;5;25m\033[38;5;255m",    # blue badge
    "runtime":    f"{_BOLD}\033[48;5;130m\033[38;5;255m",   # amber badge
    "sub_agent":  f"{_BOLD}\033[48;5;28m\033[38;5;255m",    # green badge
    "approval":   f"{_BOLD}\033[48;5;130m\033[38;5;255m",   # amber badge
}


def _write_role_block(role: str, text: str, output: TextIO) -> None:
    """Write a block with a colored role badge prefix and content."""
    if not text:
        return
    badge = _BADGE.get(role, f"{_BOLD}")

    # Split into prefix (role>) and content
    if "> " in text:
        prefix, content = text.split("> ", 1)
        prefix_text = f"{prefix}>"
    else:
        prefix_text = role
        content = text

    # Badge + content
    output.write(f"{badge} {prefix_text} {_RESET} {content}")
    if not content.endswith("\n"):
        output.write("\n")
    output.write("\n")
    output.flush()


def write_agent(text: str, output: Optional[TextIO] = None, *, role: str = "core_agent") -> None:
    """Write agent output with the role's badge prefix.

    The role argument selects the badge color — `core_agent` gets the blue
    badge, `sub_agent` gets the green badge. Any unknown role falls back to
    the default bold.
    """
    stream = output if output is not None else sys.stdout
    _write_role_block(role, text, stream)


def write_runtime(text: str, output: Optional[TextIO] = None) -> None:
    """Write runtime output with amber badge prefix."""
    stream = output if output is not None else sys.stdout
    _write_role_block("runtime", text, stream)


def write_approval(text: str, output: Optional[TextIO] = None) -> None:
    """Write approval prompt with amber badge prefix."""
    stream = output if output is not None else sys.stdout
    _write_role_block("approval", text, stream)


# --------------------------------------------------------------------------- #
# Exec payload formatting
# --------------------------------------------------------------------------- #


def _has_display_value(value: Any) -> bool:
    return value not in (None, "", [], {})


def iter_exec_payload_items(payload: dict[str, Any]) -> list[tuple[str, Any]]:
    """Return non-empty exec payload items in stable display order."""
    seen: set[str] = set()
    items: list[tuple[str, Any]] = []

    for key in _EXEC_PAYLOAD_ORDER:
        if key in payload and _has_display_value(payload[key]):
            items.append((key, payload[key]))
            seen.add(key)

    for key in sorted(payload):
        if key in seen or not _has_display_value(payload[key]):
            continue
        items.append((key, payload[key]))

    return items


# --------------------------------------------------------------------------- #
# Streaming response extraction
# --------------------------------------------------------------------------- #


def extract_streaming_response(partial_text: str) -> Optional[str]:
    """Extract the <response> body from a partial XML stream.

    Used to stream tokens to the UI during generation. Returns the response
    text accumulated so far, or None if the opening ``<response>`` tag hasn't
    appeared yet. The closing ``</response>`` tag (if reached) ends the
    extraction; otherwise everything after the opening tag is returned with
    one leading newline trimmed for display readability.

    Tag bodies are raw text — no escape decoding is performed. If the body
    is wrapped in ``<![CDATA[...]]>``, the wrapper is stripped before
    returning.
    """
    open_idx = partial_text.find("<response>")
    if open_idx == -1:
        return None
    body_start = open_idx + len("<response>")
    close_idx = partial_text.find("</response>", body_start)
    body = partial_text[body_start:close_idx] if close_idx != -1 else partial_text[body_start:]

    stripped = body.lstrip()
    if stripped.startswith("<![CDATA["):
        inner = stripped[len("<![CDATA["):]
        end = inner.find("]]>")
        body = inner if end == -1 else inner[:end]
    elif body.startswith("\r\n"):
        body = body[2:]
    elif body.startswith("\n"):
        body = body[1:]

    return body if body else None


# --------------------------------------------------------------------------- #
# Streaming display
# --------------------------------------------------------------------------- #


class StreamingDisplay:
    """Stateful streaming callback that buffers only the parsed response.

    Accumulates raw LLM tokens and uses extract_streaming_response() to
    track the latest response text. The text is only printed if the turn
    later passes parsing/validation. Raw JSON structure remains hidden.
    """

    def __init__(self, output: Optional[TextIO] = None) -> None:
        self._output = output
        self._accumulated = ""
        self._response_text = ""
        self._current_name = "agent"
        self._reasoning_active = False

    @property
    def _stream(self) -> TextIO:
        return self._output if self._output is not None else sys.stdout

    def on_content(self, token: str) -> None:
        """Called per content token during model.generate()."""
        self._close_reasoning()
        self._accumulated += token
        response = extract_streaming_response(self._accumulated)
        if response is None:
            return
        self._response_text = response

    def on_reasoning(self, token: str) -> None:
        """Called per reasoning_content token — stream live in dim/grey.

        Reasoning tokens are written to stdout immediately so the user sees
        the model thinking. They are NOT buffered into the response text and
        therefore never reach the parsed action or the recorded history.
        """
        stream = self._stream
        if not self._reasoning_active:
            stream.write(f"{_DIM}{_THINKING_PREFIX}")
            self._reasoning_active = True
        stream.write(token)
        stream.flush()

    def reset(self, name: str = "agent") -> None:
        """Reset state for a new turn."""
        self._close_reasoning()
        self._accumulated = ""
        self._response_text = ""
        self._current_name = name

    def commit(self) -> None:
        """Print the buffered response as agent output."""
        self._close_reasoning()
        if not self._response_text:
            return
        write_agent(
            f"{self._current_name}> {self._response_text}",
            self._stream,
            role=self._current_name,
        )

    def discard(self) -> None:
        """Drop any buffered response from a failed parse attempt."""
        self._close_reasoning()
        self._accumulated = ""
        self._response_text = ""

    def _close_reasoning(self) -> None:
        """End the dim reasoning zone cleanly so later output isn't styled.

        No-op if no reasoning zone is open — callers don't need to guard.
        Emits a trailing blank line so the following badge block (core_agent
        / sub_agent / runtime) reads as a visually distinct section.
        """
        if not self._reasoning_active:
            return
        stream = self._stream
        stream.write(f"{_RESET}\n\n")
        stream.flush()
        self._reasoning_active = False
