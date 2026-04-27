"""Streaming display callbacks and extraction for the agent loop."""

from __future__ import annotations

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

_RESET = "\033[0m"
_BOLD = "\033[1m"
_DIM = "\033[2m"

_THINKING_PREFIX = "thinking> "

# Role prefix badges: bold + colored background + white text
_BADGE = {
    "user":       f"{_BOLD}\033[48;5;240m\033[38;5;255m",   # gray badge
    "core_agent": f"{_BOLD}\033[48;5;25m\033[38;5;255m",    # blue badge
    "runtime":    f"{_BOLD}\033[48;5;130m\033[38;5;255m",   # amber badge
    "sub_agent":  f"{_BOLD}\033[48;5;28m\033[38;5;255m",    # green badge
    "approval":   f"{_BOLD}\033[48;5;130m\033[38;5;255m",   # amber badge
}

# Boundary marker the streaming display watches for. Any token chars after
# this marker are suppressed from the live UI.
_NEXT_ACTION_TAG = "<next_action>"
_NEXT_ACTION_HOLDBACK = len(_NEXT_ACTION_TAG) - 1  # 12


def _write_role_block(role: str, text: str, output: TextIO) -> None:
    """Write a block with a colored role badge prefix and content."""
    if not text:
        return
    badge = _BADGE.get(role, f"{_BOLD}")

    if "> " in text:
        prefix, content = text.split("> ", 1)
        prefix_text = f"{prefix}>"
    else:
        prefix_text = role
        content = text

    output.write(f"{badge} {prefix_text} {_RESET} {content}")
    if not content.endswith("\n"):
        output.write("\n")
    output.write("\n")
    output.flush()


def write_agent(text: str, output: Optional[TextIO] = None, *, role: str = "core_agent") -> None:
    """Write agent output with the role's badge prefix.

    Used for non-streaming agent output (e.g. delegate result strings).
    The interactive REPL streams agent prose via StreamingDisplay instead.
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
    """Return the printable portion of a partial response buffer.

    With response-outside-`<next_action>`, the response prose is everything
    from the start of the buffer up to (but not including) ``<next_action>``.
    While streaming, we hold back the trailing 12 chars
    (= ``len("<next_action>") - 1``) so a partial ``<next_action>`` arriving
    across token boundaries doesn't leak onto the UI. Once ``<next_action>``
    appears, we return the prose up to it.

    Returns ``None`` only when the buffer is too short to safely flush
    (less than the holdback length and no ``<next_action>`` seen yet).
    """
    idx = partial_text.find(_NEXT_ACTION_TAG)
    if idx != -1:
        text = partial_text[:idx]
        return text if text else None
    if len(partial_text) <= _NEXT_ACTION_HOLDBACK:
        return None
    return partial_text[: len(partial_text) - _NEXT_ACTION_HOLDBACK]


# --------------------------------------------------------------------------- #
# Streaming display
# --------------------------------------------------------------------------- #


class StreamingDisplay:
    """Live-streams agent prose to the terminal as tokens arrive.

    Per turn:
      * ``reset(name)`` writes the role badge header and resets state.
      * ``on_reasoning(token)`` streams reasoning tokens live in dim/grey.
      * ``on_content(token)`` streams response prose live; once
        ``<next_action>`` is seen, all subsequent tokens are suppressed
        (they're the structured action block, not user-visible prose).
      * ``commit()`` closes the turn with a separator newline.
      * ``discard()`` writes a dim retry cue and resets state for a fresh
        attempt under the same role badge — the previously-streamed prose
        stays on screen, the cue makes the duplication legible.
    """

    def __init__(self, output: Optional[TextIO] = None) -> None:
        self._output = output
        self._accumulated = ""
        self._printed_to = 0
        self._in_action = False
        self._current_name = "agent"
        self._reasoning_active = False
        self._badge_printed = False

    @property
    def _stream(self) -> TextIO:
        return self._output if self._output is not None else sys.stdout

    def reset(self, name: str = "agent") -> None:
        """Start a new turn — reset buffers. Badge is written lazily on the
        first content token so reasoning streams cleanly above it."""
        self._close_reasoning()
        self._accumulated = ""
        self._printed_to = 0
        self._in_action = False
        self._current_name = name
        self._badge_printed = False

    def on_content(self, token: str) -> None:
        """Stream a response prose token live, holding back partial `<next_action>`."""
        self._close_reasoning()
        if not self._badge_printed:
            self._write_badge_header()
        self._accumulated += token
        if self._in_action:
            return

        idx = self._accumulated.find(_NEXT_ACTION_TAG, self._printed_to)
        if idx != -1:
            self._flush(idx)
            self._in_action = True
            return

        safe_end = len(self._accumulated) - _NEXT_ACTION_HOLDBACK
        if safe_end > self._printed_to:
            self._flush(safe_end)

    def on_reasoning(self, token: str) -> None:
        """Stream a reasoning token live in dim.

        Reasoning tokens are not buffered into the response and never reach
        the parsed action or recorded history.
        """
        stream = self._stream
        if not self._reasoning_active:
            stream.write(f"{_DIM}{_THINKING_PREFIX}")
            self._reasoning_active = True
        stream.write(token)
        stream.flush()

    def commit(self) -> None:
        """Close the current turn cleanly. Prose is already on screen."""
        self._close_reasoning()
        self._flush_remaining()
        if self._badge_printed:
            self._stream.write("\n\n")
            self._stream.flush()
            self._badge_printed = False

    def discard(self) -> None:
        """Mark a parse-failed attempt with a dim retry cue, prepare for retry.

        The previously-streamed prose stays visible so the user understands
        what the agent tried to say. A dim ``runtime>`` cue separates it
        from the fresh attempt, which gets its own badge header.
        """
        self._close_reasoning()
        self._flush_remaining()
        self._stream.write(f"\n{_DIM}runtime> retrying after parse error{_RESET}\n\n")
        self._stream.flush()
        self._accumulated = ""
        self._printed_to = 0
        self._in_action = False
        self._badge_printed = False
        # Badge is written lazily on the first content token of the retry.

    # ----- internals ------------------------------------------------------- #

    def _flush(self, up_to: int) -> None:
        """Write _accumulated[_printed_to:up_to] to the stream."""
        self._stream.write(self._accumulated[self._printed_to:up_to])
        self._stream.flush()
        self._printed_to = up_to

    def _flush_remaining(self) -> None:
        """Drain any held-back content (the trailing 12-char lookahead).

        Called at end-of-turn when no <next_action> arrived to release the
        holdback on its own. Without this, the last 12 chars of the model's
        reply would be silently dropped from the UI.
        """
        if self._in_action:
            return
        if len(self._accumulated) > self._printed_to:
            self._flush(len(self._accumulated))

    def _write_badge_header(self) -> None:
        badge = _BADGE.get(self._current_name, _BOLD)
        self._stream.write(f"{badge} {self._current_name}> {_RESET} ")
        self._stream.flush()
        self._badge_printed = True

    def _close_reasoning(self) -> None:
        """End the dim reasoning zone cleanly so later output isn't styled."""
        if not self._reasoning_active:
            return
        self._stream.write(f"{_RESET}\n\n")
        self._stream.flush()
        self._reasoning_active = False
