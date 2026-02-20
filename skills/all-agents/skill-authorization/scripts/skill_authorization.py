from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

_SKILL_ID_RE = re.compile(r"^[a-z0-9][a-z0-9-]{0,63}$")


def _ok(action: str, skill_id: str, scope: str, summary: str, details: dict[str, Any], next_step: str) -> dict[str, Any]:
    return {
        "status": "ok",
        "action": action,
        "skill_id": skill_id,
        "scope": scope,
        "summary": summary,
        "next_step": next_step,
        "details": details,
    }


def _err(action: str, skill_id: str, scope: str, summary: str, details: dict[str, Any], next_step: str = "review_inputs") -> dict[str, Any]:
    return {
        "status": "error",
        "action": action,
        "skill_id": skill_id,
        "scope": scope,
        "summary": summary,
        "next_step": next_step,
        "details": details,
    }


def _read_skill_frontmatter(path: Path) -> dict[str, str]:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return {}

    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        return {}

    end = -1
    for idx in range(1, len(lines)):
        if lines[idx].strip() == "---":
            end = idx
            break
    if end == -1:
        return {}

    out: dict[str, str] = {}
    for raw in lines[1:end]:
        line = raw.strip()
        if not line or ":" not in line:
            continue
        key, value = line.split(":", 1)
        out[key.strip()] = value.strip()
    return out


def _skill_template(skill_name: str, description: str) -> str:
    return "\n".join(
        [
            "---",
            f"name: {skill_name}",
            "handler:",
            f"description: {description}",
            "required_tools: exec",
            "recommended_tools: exec",
            "forbidden_tools:",
            "---",
            "",
            "# Purpose",
            "",
            "Describe what this skill does.",
            "",
            "# When To Use",
            "",
            "Describe clear trigger conditions.",
            "",
            "# Procedure",
            "",
            "List deterministic steps.",
            "",
            "# Action Input Templates",
            "",
            "Provide concrete action_input examples.",
            "",
            "# Notes",
            "",
            "Add constraints and caveats.",
            "",
        ]
    )


def _normalize_skill_name(skill_id: str) -> str:
    return " ".join(part.capitalize() for part in skill_id.split("-") if part)


def run_inspect(workspace: Path, skill_id: str, scope: str) -> dict[str, Any]:
    skill_dir = workspace / "skills" / scope / skill_id
    skill_md = skill_dir / "SKILL.md"
    scripts_dir = skill_dir / "scripts"

    exists = skill_dir.exists() and skill_dir.is_dir()
    skill_md_exists = skill_md.exists()
    frontmatter = _read_skill_frontmatter(skill_md) if skill_md_exists else {}

    scripts: list[str] = []
    if scripts_dir.exists() and scripts_dir.is_dir():
        for p in sorted(scripts_dir.rglob("*")):
            if p.is_file():
                scripts.append(str(p.relative_to(workspace)))

    summary = "skill exists" if exists else "skill not found"
    next_step = "proceed_with_update" if exists else "scaffold_skill"

    return _ok(
        action="inspect",
        skill_id=skill_id,
        scope=scope,
        summary=summary,
        next_step=next_step,
        details={
            "skill_dir": str(skill_dir.relative_to(workspace)),
            "exists": exists,
            "skill_md_exists": skill_md_exists,
            "frontmatter": frontmatter,
            "scripts": scripts,
        },
    )


def run_scaffold(workspace: Path, skill_id: str, scope: str, description: str, overwrite: bool) -> dict[str, Any]:
    skill_dir = workspace / "skills" / scope / skill_id
    scripts_dir = skill_dir / "scripts"
    skill_md = skill_dir / "SKILL.md"

    created: list[str] = []
    updated: list[str] = []

    if not skill_dir.exists():
        skill_dir.mkdir(parents=True, exist_ok=True)
        created.append(str(skill_dir.relative_to(workspace)))

    if not scripts_dir.exists():
        scripts_dir.mkdir(parents=True, exist_ok=True)
        created.append(str(scripts_dir.relative_to(workspace)))

    skill_name = _normalize_skill_name(skill_id)
    default_description = description.strip() or f"Describe purpose for {skill_id}."
    template = _skill_template(skill_name, default_description)

    if not skill_md.exists():
        skill_md.write_text(template, encoding="utf-8")
        created.append(str(skill_md.relative_to(workspace)))
    elif overwrite:
        skill_md.write_text(template, encoding="utf-8")
        updated.append(str(skill_md.relative_to(workspace)))

    return _ok(
        action="scaffold",
        skill_id=skill_id,
        scope=scope,
        summary="skill scaffold ready",
        next_step="edit_skill_content",
        details={
            "skill_dir": str(skill_dir.relative_to(workspace)),
            "created": created,
            "updated": updated,
        },
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect or scaffold skill packages with structured JSON output.")
    parser.add_argument("--action", required=True, choices=["inspect", "scaffold"])
    parser.add_argument("--skill-id", required=True)
    parser.add_argument("--scope", required=True, choices=["all-agents", "core-agent"])
    parser.add_argument("--description", default="")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--workspace", default=".")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    action = str(args.action)
    skill_id = str(args.skill_id).strip()
    scope = str(args.scope).strip()
    workspace = Path(args.workspace).expanduser().resolve()

    if not _SKILL_ID_RE.match(skill_id):
        out = _err(
            action=action,
            skill_id=skill_id,
            scope=scope,
            summary="invalid skill_id format",
            details={"expected_pattern": "^[a-z0-9][a-z0-9-]{0,63}$"},
        )
        print(json.dumps(out, ensure_ascii=True))
        return 1

    try:
        if action == "inspect":
            out = run_inspect(workspace=workspace, skill_id=skill_id, scope=scope)
        else:
            out = run_scaffold(
                workspace=workspace,
                skill_id=skill_id,
                scope=scope,
                description=str(args.description),
                overwrite=bool(args.overwrite),
            )
        print(json.dumps(out, ensure_ascii=True))
        return 0 if out.get("status") == "ok" else 1
    except Exception as exc:  # unexpected runtime failure
        out = _err(
            action=action,
            skill_id=skill_id,
            scope=scope,
            summary="unexpected failure",
            details={"error_type": exc.__class__.__name__},
            next_step="retry_or_fix_script",
        )
        print(json.dumps(out, ensure_ascii=True))
        print(f"unexpected error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
