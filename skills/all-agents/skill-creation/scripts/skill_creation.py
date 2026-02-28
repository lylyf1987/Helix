from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

_SKILL_ID_RE = re.compile(r"^[a-z0-9][a-z0-9-]{0,63}$")
_EXECUTED_SKILL = "skill-creation"
_SUPPORTED_SCRIPT_MODES = {"none", "single", "multi"}
_REQUIRED_FRONTMATTER_KEYS = [
    "name",
    "handler",
    "description",
    "required_tools",
    "recommended_tools",
    "forbidden_tools",
]
_REQUIRED_SECTIONS = [
    "Purpose",
    "When To Use",
    "Skill Mode",
    "Procedure",
    "Runtime Contract",
    "Action Input Templates",
    "Output JSON Shape",
    "Error Handling Rule",
    "Skill Dependencies",
    "Notes",
]


def _ok(skill_target: str) -> dict[str, Any]:
    return {
        "executed_skill": _EXECUTED_SKILL,
        "status": "ok",
        "skill_created/updated": skill_target,
    }


def _err(skill_target: str = "") -> dict[str, Any]:
    return {
        "executed_skill": _EXECUTED_SKILL,
        "status": "error",
        "skill_created/updated": skill_target,
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


def _extract_h1_sections(skill_text: str) -> set[str]:
    sections: set[str] = set()
    for line in str(skill_text).splitlines():
        raw = line.strip()
        if raw.startswith("# "):
            sections.add(raw[2:].strip().lower())
    return sections


def _normalize_skill_name(skill_id: str) -> str:
    return " ".join(part.capitalize() for part in skill_id.split("-") if part)


def _summarize_list(items: list[str]) -> str:
    if not items:
        return "(none)"
    return ",".join(items)


def _normalize_dependencies(raw_items: list[str]) -> tuple[list[str], list[str]]:
    normalized: list[str] = []
    invalid: list[str] = []
    seen: set[str] = set()

    for raw in raw_items:
        for token in str(raw).split(","):
            skill_id = token.strip()
            if not skill_id:
                continue
            if not _SKILL_ID_RE.match(skill_id):
                invalid.append(skill_id)
                continue
            if skill_id in seen:
                continue
            seen.add(skill_id)
            normalized.append(skill_id)

    return normalized, invalid


def _resolve_script_mode(frontmatter: dict[str, str]) -> tuple[str, bool]:
    raw_mode = str(frontmatter.get("script_mode", "")).strip().lower()
    if raw_mode == "scaffold":
        raw_mode = "single"

    if raw_mode:
        if raw_mode in _SUPPORTED_SCRIPT_MODES:
            return raw_mode, False
        return raw_mode, True

    handler = str(frontmatter.get("handler", "")).strip()
    return ("single" if handler else "none"), False


def _normalize_handler_path(handler: str, script_mode: str, skill_id: str) -> str:
    mode = str(script_mode).strip().lower()
    if mode == "none":
        return ""

    raw = str(handler).strip()
    if mode == "single" and not raw:
        raw = f"scripts/{skill_id.replace('-', '_')}.py"
    if not raw:
        return ""

    candidate = Path(raw)
    if candidate.is_absolute():
        raise ValueError("handler must be a relative path under the skill directory")
    if ".." in candidate.parts:
        raise ValueError("handler cannot escape the skill directory")

    normalized_parts = [part for part in candidate.parts if part not in {"", "."}]
    normalized = "/".join(normalized_parts)
    if not normalized:
        return ""
    return normalized


def _skill_template(
    *,
    skill_id: str,
    scope: str,
    skill_name: str,
    description: str,
    handler_path: str,
    script_mode: str,
    dependencies: list[str],
) -> str:
    mode = str(script_mode).strip().lower()
    dep_lines = [f"- `{dep}`: explain when/why to call this dependency skill." for dep in dependencies]
    if not dep_lines:
        dep_lines = ["- (none)"]

    action_template_lines: list[str]
    if mode == "none":
        action_template_lines = [
            "No dedicated script is required for this skill.",
            "If runtime execution is needed, include explicit ad-hoc exec action_input for that step.",
        ]
    elif mode == "multi":
        action_template_lines = [
            "Provide one action_input example per script-enabled phase.",
            "",
            "```json",
            "{",
            '  "code_type": "python",',
            f'  "script_path": "skills/{scope}/{skill_id}/scripts/<phase_1_script>.py",',
            '  "script_args": ["--example", "value"]',
            "}",
            "```",
            "",
            "```json",
            "{",
            '  "code_type": "python",',
            f'  "script_path": "skills/{scope}/{skill_id}/scripts/<phase_2_script>.py",',
            '  "script_args": ["--example", "value"]',
            "}",
            "```",
        ]
    else:
        action_template_lines = [
            "```json",
            "{",
            '  "code_type": "python",',
            f'  "script_path": "skills/{scope}/{skill_id}/{handler_path}",',
            '  "script_args": ["--example", "value"]',
            "}",
            "```",
        ]

    output_shape_lines = [
        "```json",
        "{",
        f'  "executed_skill": "{skill_id}",',
        '  "status": "ok|error",',
        '  "<result_field>": "..."',
        "}",
        "```",
    ]

    return "\n".join(
        [
            "---",
            f"name: {skill_name}",
            f"handler: {handler_path}",
            f"description: {description}",
            "required_tools: exec",
            "recommended_tools: exec",
            "forbidden_tools:",
            f"script_mode: {mode}",
            "---",
            "",
            "# Purpose",
            "",
            "State the concrete capability this skill provides.",
            "",
            "# When To Use",
            "",
            "List clear trigger conditions and non-trigger conditions.",
            "",
            "# Skill Mode",
            "",
            f"- script_mode: `{mode}`",
            "- allowed: `none`, `single`, `multi`",
            "- `none`: no dedicated script; skill is procedure-first reasoning guidance.",
            "- `single`: one primary script via frontmatter `handler`.",
            "- `multi`: multiple scripts by phase; LLM reasons between phase executions.",
            "",
            "# Procedure",
            "",
            "Use the design workflow order:",
            "1. Gather context.",
            "2. Plan next minimal action.",
            "3. Act (exec only when needed).",
            "4. Observe runtime evidence and verify results.",
            "5. Iterate or report back.",
            "",
            "When exec is used, ensure the next step reasons over runtime stdout/stderr before continuing.",
            "",
            "# Runtime Contract",
            "",
            "Scripts in this skill must produce runtime-friendly output:",
            "1. Print clear, meaningful stdout describing what was attempted and what changed.",
            "2. Use stderr only for failures with actionable messages.",
            "3. End with one final JSON object on stdout containing stable keys for downstream reasoning.",
            "",
            "# Action Input Templates",
            "",
            *action_template_lines,
            "",
            "# Output JSON Shape",
            "",
            *output_shape_lines,
            "",
            "# Error Handling Rule",
            "",
            "Define when to retry internally, and when to stop and return control to requester.",
            "For unrecoverable config/environment issues, instruct core agent to use chat_with_requester.",
            "",
            "# Skill Dependencies",
            "",
            "Reference existing skills by `skill_id` when they already solve sub-problems.",
            "Do not duplicate dependency implementation unless there is a strong reason.",
            *dep_lines,
            "",
            "For each dependency used, include one concrete action_input example and load it with built-in loader before first use.",
            "",
            "# Notes",
            "",
            "Add constraints, caveats, and workspace-boundary requirements.",
            "",
        ]
    )


def _script_template(skill_id: str) -> str:
    module_name = skill_id.replace("-", "_")
    return "\n".join(
        [
            "#!/usr/bin/env python3",
            "from __future__ import annotations",
            "",
            "import argparse",
            "import json",
            "import sys",
            "from typing import Any",
            "",
            f"_EXECUTED_SKILL = {skill_id!r}",
            "",
            "",
            "def _ok(message: str) -> dict[str, Any]:",
            "    return {",
            '        "executed_skill": _EXECUTED_SKILL,',
            '        "status": "ok",',
            '        "message": message,',
            "    }",
            "",
            "",
            "def _err(message: str) -> dict[str, Any]:",
            "    return {",
            '        "executed_skill": _EXECUTED_SKILL,',
            '        "status": "error",',
            '        "message": message,',
            "    }",
            "",
            "",
            "def parse_args() -> argparse.Namespace:",
            f'    parser = argparse.ArgumentParser(description="Runtime handler scaffold for {module_name}")',
            "    parser.add_argument(\"--dry-run\", action=\"store_true\")",
            "    return parser.parse_args()",
            "",
            "",
            "def main() -> int:",
            "    args = parse_args()",
            "    try:",
            "        if args.dry_run:",
            "            out = _ok(\"dry_run completed\")",
            "            print(json.dumps(out, ensure_ascii=True))",
            "            return 0",
            "",
            "        # TODO: implement deterministic skill logic here.",
            "        # Runtime expectation:",
            "        # - stdout: meaningful evidence of what changed",
            "        # - stderr: only errors/failures",
            "        # - final stdout line: one JSON object with stable keys",
            "        out = _ok(\"scaffold script executed; implementation pending\")",
            "        print(json.dumps(out, ensure_ascii=True))",
            "        return 0",
            "    except Exception as exc:  # unexpected runtime failure",
            "        out = _err(f\"unexpected exception: {exc}\")",
            "        print(json.dumps(out, ensure_ascii=True))",
            "        print(\"unexpected error\", file=sys.stderr)",
            "        return 2",
            "",
            "",
            "if __name__ == \"__main__\":",
            "    raise SystemExit(main())",
            "",
        ]
    )


def run_inspect(workspace: Path, skill_id: str, scope: str) -> dict[str, Any]:
    skill_dir = workspace / "skills" / scope / skill_id
    skill_md = skill_dir / "SKILL.md"
    scripts_dir = skill_dir / "scripts"

    exists = skill_dir.exists() and skill_dir.is_dir()
    skill_md_exists = skill_md.exists()
    frontmatter = _read_skill_frontmatter(skill_md) if skill_md_exists else {}

    scripts: list[str] = []
    if scripts_dir.exists() and scripts_dir.is_dir():
        for path in sorted(scripts_dir.rglob("*")):
            if path.is_file():
                scripts.append(str(path.relative_to(workspace)))

    mode, mode_invalid = _resolve_script_mode(frontmatter)

    summary = (
        f"skill_id={skill_id}; action=inspect; scope={scope}; exists={exists}; "
        f"skill_md_exists={skill_md_exists}; scripts_count={len(scripts)}; "
        f"script_mode={mode}; script_mode_invalid={mode_invalid}"
    )
    if frontmatter:
        name = str(frontmatter.get("name", "")).strip()
        handler = str(frontmatter.get("handler", "")).strip()
        description = str(frontmatter.get("description", "")).strip()
        summary = (
            f"{summary}; name={name or '(empty)'}; "
            f"handler={handler or '(empty)'}; description={description or '(empty)'}"
        )
    return _ok(skill_target=summary)


def run_scaffold(
    *,
    workspace: Path,
    skill_id: str,
    scope: str,
    description: str,
    overwrite: bool,
    script_mode: str,
    handler: str,
    dependencies: list[str],
) -> dict[str, Any]:
    mode = str(script_mode).strip().lower()
    if mode not in _SUPPORTED_SCRIPT_MODES:
        raise ValueError(f"unsupported script_mode={mode}")

    handler_rel = _normalize_handler_path(handler=handler, script_mode=mode, skill_id=skill_id)

    skill_dir = workspace / "skills" / scope / skill_id
    scripts_dir = skill_dir / "scripts"
    references_dir = skill_dir / "references"
    assets_dir = skill_dir / "assets"
    skill_md = skill_dir / "SKILL.md"
    handler_file = (skill_dir / handler_rel) if handler_rel else None

    created: list[str] = []
    updated: list[str] = []

    if not skill_dir.exists():
        skill_dir.mkdir(parents=True, exist_ok=True)
        created.append(str(skill_dir.relative_to(workspace)))

    if mode in {"single", "multi"} and not scripts_dir.exists():
        scripts_dir.mkdir(parents=True, exist_ok=True)
        created.append(str(scripts_dir.relative_to(workspace)))

    if not references_dir.exists():
        references_dir.mkdir(parents=True, exist_ok=True)
        created.append(str(references_dir.relative_to(workspace)))
    if not assets_dir.exists():
        assets_dir.mkdir(parents=True, exist_ok=True)
        created.append(str(assets_dir.relative_to(workspace)))

    skill_name = _normalize_skill_name(skill_id)
    default_description = description.strip() or f"Describe purpose for {skill_id}."

    skill_template = _skill_template(
        skill_id=skill_id,
        scope=scope,
        skill_name=skill_name,
        description=default_description,
        handler_path=handler_rel,
        script_mode=mode,
        dependencies=dependencies,
    )
    script_template = _script_template(skill_id)

    if not skill_md.exists():
        skill_md.write_text(skill_template, encoding="utf-8")
        created.append(str(skill_md.relative_to(workspace)))
    elif overwrite:
        skill_md.write_text(skill_template, encoding="utf-8")
        updated.append(str(skill_md.relative_to(workspace)))

    if handler_file is not None:
        handler_file.parent.mkdir(parents=True, exist_ok=True)
        if not handler_file.exists():
            handler_file.write_text(script_template, encoding="utf-8")
            created.append(str(handler_file.relative_to(workspace)))
        elif overwrite:
            handler_file.write_text(script_template, encoding="utf-8")
            updated.append(str(handler_file.relative_to(workspace)))

    if mode == "multi":
        multi_readme = scripts_dir / "README.md"
        multi_content = "\n".join(
            [
                "# Multi-Script Phase Map",
                "",
                "Document phase-to-script mapping here.",
                "Example:",
                "- phase: gather-context -> script: scripts/gather_context.py",
                "- phase: execute-plan -> script: scripts/execute_plan.py",
                "- phase: verify-output -> script: scripts/verify_output.py",
                "",
                "Core agent should reason between phase executions using runtime stdout/stderr evidence.",
                "",
            ]
        )
        if not multi_readme.exists():
            multi_readme.write_text(multi_content, encoding="utf-8")
            created.append(str(multi_readme.relative_to(workspace)))
        elif overwrite:
            multi_readme.write_text(multi_content, encoding="utf-8")
            updated.append(str(multi_readme.relative_to(workspace)))

    references_readme = references_dir / "README.md"
    references_content = "\n".join(
        [
            "# References",
            "",
            "Put large examples and deep technical notes here.",
            "Keep SKILL.md concise and procedural; load reference files only when needed.",
            "",
        ]
    )
    if not references_readme.exists():
        references_readme.write_text(references_content, encoding="utf-8")
        created.append(str(references_readme.relative_to(workspace)))
    elif overwrite:
        references_readme.write_text(references_content, encoding="utf-8")
        updated.append(str(references_readme.relative_to(workspace)))

    summary = (
        f"skill_id={skill_id}; action=scaffold; scope={scope}; script_mode={mode}; "
        f"handler={handler_rel or '(none)'}; dependencies={_summarize_list(dependencies)}; "
        f"created={_summarize_list(created)}; updated={_summarize_list(updated)}"
    )
    return _ok(skill_target=summary)


def run_validate(workspace: Path, skill_id: str, scope: str) -> dict[str, Any]:
    skill_dir = workspace / "skills" / scope / skill_id
    skill_md = skill_dir / "SKILL.md"
    scripts_dir = skill_dir / "scripts"

    errors: list[str] = []
    warnings: list[str] = []

    if not skill_dir.exists() or not skill_dir.is_dir():
        errors.append("skill_dir_missing")
    if not skill_md.exists():
        errors.append("skill_md_missing")

    frontmatter: dict[str, str] = {}
    skill_text = ""
    if skill_md.exists():
        try:
            skill_text = skill_md.read_text(encoding="utf-8")
        except OSError:
            errors.append("skill_md_unreadable")
        frontmatter = _read_skill_frontmatter(skill_md)

    if skill_md.exists() and not frontmatter:
        errors.append("frontmatter_missing_or_invalid")

    for key in _REQUIRED_FRONTMATTER_KEYS:
        if key not in frontmatter:
            errors.append(f"frontmatter_missing:{key}")

    handler = str(frontmatter.get("handler", "")).strip()
    mode, mode_invalid = _resolve_script_mode(frontmatter)
    if mode_invalid:
        errors.append(f"frontmatter_invalid:script_mode={mode}")
    if "script_mode" not in frontmatter:
        warnings.append("frontmatter_missing:script_mode")

    scripts_count = 0
    if scripts_dir.exists() and scripts_dir.is_dir():
        scripts_count = sum(1 for path in scripts_dir.rglob("*") if path.is_file())

    if mode == "none":
        if handler:
            warnings.append("handler_present_for_none_mode")
    elif mode == "single":
        if not handler:
            errors.append("handler_required_for_single_mode")
        else:
            handler_path = skill_dir / handler
            if not handler_path.exists():
                errors.append(f"handler_missing:{handler}")
            elif not handler_path.is_file():
                errors.append(f"handler_not_file:{handler}")
    elif mode == "multi":
        if scripts_count == 0:
            errors.append("multi_mode_requires_scripts")
        if handler:
            handler_path = skill_dir / handler
            if not handler_path.exists():
                errors.append(f"handler_missing:{handler}")
            elif not handler_path.is_file():
                errors.append(f"handler_not_file:{handler}")
        lower_skill_text = skill_text.lower()
        if "phase" not in lower_skill_text:
            warnings.append("multi_mode_missing_phase_mapping")

    lower_skill_text = skill_text.lower()
    if "stdout" not in lower_skill_text:
        warnings.append("skill_md_missing_stdout_guidance")
    if "stderr" not in lower_skill_text:
        warnings.append("skill_md_missing_stderr_guidance")

    sections = _extract_h1_sections(skill_text)
    for section_name in _REQUIRED_SECTIONS:
        if section_name.lower() not in sections:
            errors.append(f"section_missing:{section_name}")

    workflow_terms = ["context", "plan", "act", "verify", "report"]
    missing_terms = [term for term in workflow_terms if term not in lower_skill_text]
    if missing_terms:
        warnings.append(f"procedure_missing_workflow_terms:{','.join(missing_terms)}")

    summary = (
        f"skill_id={skill_id}; action=validate; scope={scope}; script_mode={mode}; "
        f"errors_count={len(errors)}; warnings_count={len(warnings)}; scripts_count={scripts_count}; "
        f"errors={_summarize_list(errors)}; warnings={_summarize_list(warnings)}"
    )
    if errors:
        return _err(skill_target=summary)
    return _ok(skill_target=summary)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect or scaffold skill packages with structured JSON output.")
    parser.add_argument("--action", required=True, choices=["inspect", "scaffold", "validate"])
    parser.add_argument("--skill-id", required=True)
    parser.add_argument("--scope", required=True, choices=["all-agents", "core-agent"])
    parser.add_argument("--description", default="")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--workspace", default=".")
    parser.add_argument("--script-mode", default="single", choices=["none", "single", "multi"])
    parser.add_argument("--handler", default="")
    parser.add_argument("--dependency-skill", action="append", default=[])
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    action = str(args.action)
    skill_id = str(args.skill_id).strip()
    scope = str(args.scope).strip()
    workspace = Path(args.workspace).expanduser().resolve()

    if not _SKILL_ID_RE.match(skill_id):
        out = _err(skill_target=f"skill_creation_error: invalid skill_id={skill_id}")
        print(json.dumps(out, ensure_ascii=True))
        return 1

    dependency_skills, invalid_dependencies = _normalize_dependencies(list(args.dependency_skill or []))
    if invalid_dependencies:
        out = _err(
            skill_target=(
                "skill_creation_error: invalid dependency skill ids="
                f"{_summarize_list(invalid_dependencies)}"
            )
        )
        print(json.dumps(out, ensure_ascii=True))
        return 1

    try:
        if action == "inspect":
            out = run_inspect(workspace=workspace, skill_id=skill_id, scope=scope)
        elif action == "validate":
            out = run_validate(workspace=workspace, skill_id=skill_id, scope=scope)
        else:
            out = run_scaffold(
                workspace=workspace,
                skill_id=skill_id,
                scope=scope,
                description=str(args.description),
                overwrite=bool(args.overwrite),
                script_mode=str(args.script_mode),
                handler=str(args.handler),
                dependencies=dependency_skills,
            )
        print(json.dumps(out, ensure_ascii=True))
        return 0 if out.get("status") == "ok" else 1
    except Exception as exc:  # unexpected runtime failure
        out = _err(skill_target=f"skill_creation_error: unexpected exception: {exc}")
        print(json.dumps(out, ensure_ascii=True))
        print("unexpected error", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
