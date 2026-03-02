"""Interactive runtime UI prompts for approvals and policy overrides."""

from __future__ import annotations


def prompt_exec_approval(signature: str) -> tuple[bool, str]:
    """Prompt requester for execution approval scope in controlled mode."""
    print()
    print("Runtime confirmation required for exec action.")
    print(signature)
    print("Approve this execution? [y/N/s/p/k]")
    print("  y: allow once")
    print("  s: allow same exact exec for this session")
    print("  p: allow same script/pattern for this session")
    print("  k: allow same script_path for this session (ignore args)")
    choice = input("> ").strip().lower()
    if choice in {"y", "yes", "once"}:
        return True, "once"
    if choice in {"s", "session", "exact"}:
        return True, "session"
    if choice in {"p", "pattern"}:
        return True, "pattern"
    if choice in {"k", "path", "skill"}:
        return True, "path"
    return False, "deny"


def prompt_auto_write_override(note: str, suggested_paths: list[str]) -> str | None:
    """Ask requester for one-off external write override in auto mode."""
    print()
    print("Runtime auto-mode write policy blocked external write.")
    if note.strip():
        print(note.strip())
    if suggested_paths:
        print("Suggested external paths (from command context):")
        for idx, item in enumerate(suggested_paths, start=1):
            print(f"  {idx}. {item}")
    print("Allow one external writable path for this session? [y/N]")
    choice = input("> ").strip().lower()
    if choice not in {"y", "yes"}:
        return None
    default_path = suggested_paths[0] if suggested_paths else ""
    if default_path:
        print(f"Enter writable path (blank uses default: {default_path})")
        entered = input("> ").strip()
        return entered or default_path
    print("Enter writable path (absolute or ~/...):")
    entered = input("> ").strip()
    return entered or None
