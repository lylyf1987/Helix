"""Streaming display callbacks and extraction for the agent loop."""

from __future__ import annotations

import re
import shutil
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
# Startup banner
# --------------------------------------------------------------------------- #


_BANNER_TITLE_ICON = "✻"
_BANNER_LABEL_WIDTH = 11


def _fit(value: str, width: int, kind: str = "url") -> str:
    """Truncate ``value`` to fit in ``width`` columns using ``…`` as ellipsis.

    ``kind="path"`` truncates at the *start* (``…/agent``) so the trailing
    component (most informative for paths) stays visible. ``kind="url"``
    truncates at the *end* (``https://api.de…``) so the scheme + host stay
    visible. Width must be at least 1.
    """
    if len(value) <= width:
        return value
    if width <= 1:
        return "…"
    if kind == "path":
        return "…" + value[-(width - 1):]
    return value[: width - 1] + "…"


def _default_banner_width() -> int:
    """Pick a banner width based on the terminal columns, with sensible bounds.

    Caps at 100 so very wide terminals don't get a sprawling box; floors at
    64 so very narrow ones still fit the title + status fields.
    """
    try:
        cols = shutil.get_terminal_size((80, 24)).columns
    except OSError:
        cols = 80
    return max(64, min(100, cols - 2))


def _fit_kind(value: str) -> str:
    """Pick the truncation strategy for a banner value.

    URLs (``http://``, ``https://``) end-truncate so the scheme + host stay
    visible. Other values containing ``/`` are treated as paths and
    start-truncate so the trailing component stays visible. Plain values
    end-truncate (rarely matters).
    """
    if value.startswith(("http://", "https://")):
        return "url"
    if "/" in value:
        return "path"
    return "url"


def clear_screen(output: Optional[TextIO] = None) -> None:
    """Clear the visible terminal viewport and move the cursor to home.

    Scrollback is intentionally preserved so the user can scroll up to see
    prior shell context if they need it.
    """
    stream = output if output is not None else sys.stdout
    stream.write("\033[2J\033[H")
    stream.flush()


def write_startup_banner(
    title: str,
    fields: list[tuple[str, str]],
    hint: str,
    output: Optional[TextIO] = None,
    width: Optional[int] = None,
) -> None:
    """Render a Claude-Code-style box-drawn startup banner.

    ``fields`` is an ordered list of ``(label, value)`` pairs — preserves the
    intended display order without depending on dict insertion semantics.
    Long values are middle-truncated to keep the right border straight: any
    value containing a path separator is treated as path-like (truncated at
    the start); other values are truncated at the end.
    """
    stream = output if output is not None else sys.stdout
    if width is None:
        width = _default_banner_width()
    inner = width - 2
    border_h = "─" * inner

    def _line(content: str = "") -> str:
        # Pad the visible content to inner width (excludes ANSI escapes).
        return f"{_DIM}│{_RESET}{content}{' ' * max(0, inner - _visible_len(content))}{_DIM}│{_RESET}\n"

    parts: list[str] = []
    parts.append(f"{_DIM}╭{border_h}╮{_RESET}\n")
    title_text = f" {_BOLD}{_BANNER_TITLE_ICON} {title}{_RESET}"
    parts.append(_line(title_text))
    parts.append(_line())
    for label, value in fields:
        # margins (2 left + 1 right) + label column + space between label and value
        avail = inner - 3 - _BANNER_LABEL_WIDTH - 1
        fitted = _fit(value, avail, kind=_fit_kind(value))
        row = f"  {_BOLD}{label.ljust(_BANNER_LABEL_WIDTH)}{_RESET} {fitted}"
        parts.append(_line(row))
    parts.append(_line())
    parts.append(_line(f"  {hint}"))
    parts.append(f"{_DIM}╰{border_h}╯{_RESET}\n")

    stream.write("".join(parts))
    stream.flush()


_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def _visible_len(text: str) -> int:
    """Return the on-screen width of ``text`` in monospace cells, ignoring
    ANSI escape sequences. Counts each non-escape character as one cell —
    sufficient for the banner's ASCII labels + values + the single sparkle
    icon (which is approximately one cell in modern terminals)."""
    return len(_ANSI_RE.sub("", text))


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
