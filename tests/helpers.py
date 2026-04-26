"""Shared test helpers."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Mapping

from helix.core.state import Turn
from helix.runtime.sandbox import HostSandboxExecutor


def make_xml_output(
    response: str,
    next_action: str,
    action_input: Mapping[str, Any] | None = None,
) -> str:
    """Build an `<output>` block in the agent's XML reply format.

    Centralizes the format so tests don't drift from the parser's expectations.
    Lists in ``action_input`` are emitted as `<arg>` children (e.g. for
    ``script_args``); all other values are stringified into their tag body.
    Pass ``action_input=None`` for ``chat`` / ``think`` (the tag is omitted).
    """
    parts = [
        "<output>",
        f"<response>{response}</response>",
        f"<next_action>{next_action}</next_action>",
    ]
    if action_input:
        parts.append("<action_input>")
        for key, value in action_input.items():
            parts.append(_render_field(key, value))
        parts.append("</action_input>")
    parts.append("</output>")
    return "\n".join(parts)


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
