from __future__ import annotations

import json
import shlex
from typing import Any

from .storage import StorageEngine


def normalize_script_args(raw_script_args: Any) -> list[str]:
    if isinstance(raw_script_args, (list, tuple)):
        return [str(arg).strip() for arg in raw_script_args if str(arg).strip()]
    if isinstance(raw_script_args, str):
        text = raw_script_args.strip()
        if not text:
            return []
        try:
            return [arg for arg in shlex.split(text) if arg.strip()]
        except ValueError:
            return [text]
    return []


def build_exec_exact_signature(action_input: dict[str, Any]) -> str:
    code_type = str(action_input.get("code_type", "bash")).strip().lower() or "bash"
    script_path = str(action_input.get("script_path", "")).strip()
    script = str(action_input.get("script", "")).strip()
    script_args = normalize_script_args(action_input.get("script_args", []))
    normalized = {
        "action": "exec",
        "code_type": code_type,
        "script_path": script_path,
        "script": script,
        "script_args": script_args,
    }
    return json.dumps(normalized, ensure_ascii=True, sort_keys=True)


def build_exec_pattern_signature(action_input: dict[str, Any]) -> str:
    code_type = str(action_input.get("code_type", "bash")).strip().lower() or "bash"
    script_path = str(action_input.get("script_path", "")).strip()
    script = str(action_input.get("script", "")).strip()
    if script_path:
        return f"exec|{code_type}|script_path|{script_path}"
    compact_inline = " ".join(script.split())[:240]
    return f"exec|{code_type}|inline|{compact_inline}"


def build_exec_path_signature(action_input: dict[str, Any]) -> str:
    code_type = str(action_input.get("code_type", "bash")).strip().lower() or "bash"
    script_path = str(action_input.get("script_path", "")).strip()
    if not script_path:
        return ""
    return f"exec|{code_type}|script_path|{script_path}"


def format_exec_value_lines(label: str, value: Any) -> list[str]:
    lines = [f"  - {label}:"]
    raw_text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=True)
    text = str(raw_text)
    if not text:
        return lines
    stripped = text.strip()
    if stripped:
        try:
            parsed = json.loads(stripped)
        except Exception:
            parsed = None
        if isinstance(parsed, (dict, list)):
            text = json.dumps(parsed, ensure_ascii=True, indent=2)
    for row in str(text).splitlines():
        lines.append(f"    {row}")
    return lines


def format_history_block(
    *,
    state: StorageEngine,
    role: str,
    first_line: str,
    continuation_lines: list[str],
) -> str:
    role_name = str(role or "").strip() or "runtime"
    prefix = f"[{state.utc_now_iso()}] {role_name}> "
    lines = [f"{prefix}{first_line}"]
    for row in continuation_lines:
        lines.append(f"{row}")
    return "\n".join(lines)


def format_ui_block(role: str, first_line: str, continuation_lines: list[str]) -> str:
    role_name = str(role or "").strip() or "runtime"
    prefix = f"{role_name}> "
    lines = [f"{prefix}{first_line}"]
    for row in continuation_lines:
        lines.append(f"{row}")
    return "\n".join(lines)


def build_exec_result_lines(exec_result: Any) -> list[str]:
    if not isinstance(exec_result, dict):
        lines: list[str] = []
        lines.append('job "none" with id unknown failed with the stdout and stderr below')
        lines.extend(format_exec_value_lines("stdout", ""))
        lines.extend(format_exec_value_lines("stderr", str(exec_result)))
        return lines
    lines = []
    job_name = str(exec_result.get("job_name", "")).strip() or "none"
    job_id = str(exec_result.get("job_id", "")).strip()
    status = str(exec_result.get("status", "")).strip()
    display_job_id = job_id[4:] if job_id.startswith("job_") else (job_id or "unknown")
    status_text = "finished"
    if status == "completed":
        status_text = "completed successfully"
    elif status == "failed":
        status_text = "failed"
    elif status == "cancelled":
        status_text = "was cancelled"
    elif status:
        status_text = status
    lines.append(
        f'job "{job_name}" with id {display_job_id} {status_text} with the stdout and stderr below'
    )
    lines.extend(format_exec_value_lines("stdout", exec_result.get("stdout", "")))
    lines.extend(format_exec_value_lines("stderr", exec_result.get("stderr", "")))
    return lines


def format_history_record(state: StorageEngine, role: str, text: str) -> str:
    role_name = str(role or "").strip() or "runtime"
    return f"[{state.utc_now_iso()}] {role_name}> {str(text or '')}"


def format_core_agent_record(
    state: StorageEngine,
    raw_response: str,
    action: str,
    action_input: Any,
) -> str:
    action_name = str(action or "").strip().lower() or "unknown"
    payload = dict(action_input) if isinstance(action_input, dict) else {}
    prefix = f"[{state.utc_now_iso()}] core_agent> "
    lines: list[str] = [
        f"{prefix}{str(raw_response)}",
        f"  - next_action: {action_name}",
    ]
    if payload:
        lines.append("  - action_input:")
        for key, value in payload.items():
            if isinstance(value, str) and "\n" in value:
                lines.append(f"    - {key}:")
                for sub_line in value.splitlines():
                    lines.append(f"      {sub_line}")
                continue
            lines.append(f"    - {key}: {json.dumps(value, ensure_ascii=True)}")
    else:
        lines.append("  - action_input: {}")
    return "\n".join(lines)
