"""Shared test helpers."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Mapping

from helix.core.state import Turn
from helix.runtime.sandbox import HostSandboxExecutor


def make_xml_output(
    response: str,
    action_type: str,
    action_input: Mapping[str, Any] | None = None,
) -> str:
    """Build an agent reply: prose, blank line, then the `<action>` block.

    Centralizes the format so tests don't drift from the parser's expectations.
    For ``chat`` / ``think`` (no payload), pass ``action_input=None`` and the
    self-closing form ``<action><chat/></action>`` is emitted. For ``exec``
    and ``delegate``, pass an ``action_input`` mapping; lists become `<arg>`
    children, multi-line values get their own line.
    """
    if not action_input:
        return f"{response}\n\n<action><{action_type}/></action>"
    field_lines = "\n".join(_render_field(key, value) for key, value in action_input.items())
    return (
        f"{response}\n\n"
        f"<action>\n<{action_type}>\n{field_lines}\n</{action_type}>\n</action>"
    )


def _render_field(key: str, value: Any) -> str:
    if isinstance(value, (list, tuple)):
        children = "\n".join(f"  <arg>{item}</arg>" for item in value)
        return f"<{key}>\n{children}\n</{key}>"
    text = str(value)
    if "\n" in text:
        return f"<{key}>\n{text}\n</{key}>"
    return f"<{key}>{text}</{key}>"


def sandbox_executor(payload: dict, workspace: Path) -> Turn:
    """Run a single exec payload in a throwaway host-shell sandbox.

    Creates a HostSandboxExecutor, runs one payload, and shuts down.
    For tests and direct Environment usage only.
    """
    requested_searxng = os.environ.get("SEARXNG_BASE_URL", "").strip()
    local_service_env = {
        key: value
        for key, value in os.environ.items()
        if key.startswith("HELIX_LOCAL_MODEL_SERVICE_") and str(value).strip()
    }
    executor = HostSandboxExecutor(
        workspace,
        searxng_base_url=requested_searxng or "https://example.com",
        local_model_service_env=local_service_env,
    )
    try:
        return executor(payload, workspace)
    finally:
        executor.shutdown()
