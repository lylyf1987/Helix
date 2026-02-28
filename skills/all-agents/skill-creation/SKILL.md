---
name: Skill Creation
handler: scripts/skill_creation.py
description: Create and improve skills with script-first scaffolding, validation, workflow alignment, and structured runtime evidence.
required_tools: exec
recommended_tools: exec
forbidden_tools:
---

# Purpose

Use this skill when skill creation/update work is complex or uncertain and you want deterministic, script-first execution.

# Why Script-First

For uncertain skill tasks, run the helper script immediately so runtime history gets clean, structured evidence in `runtime> stdout`.

# Creation Lifecycle (Recommended)

1. `inspect` current skill status and existing structure.
2. `scaffold` baseline package with explicit `script_mode`.
3. Fill `SKILL.md` with concrete workflow-aligned procedure.
4. Implement/adjust script(s) only where needed.
5. `validate` before first use to catch structure/frontmatter/handler/content issues.
6. Iterate from runtime stderr and validation warnings.

# Skill Specification Standard

Generated `SKILL.md` must include these sections:

- `# Purpose`
- `# When To Use`
- `# Skill Mode`
- `# Procedure`
- `# Runtime Contract`
- `# Action Input Templates`
- `# Output JSON Shape`
- `# Error Handling Rule`
- `# Skill Dependencies`
- `# Notes`

Procedure should reflect design workflow:

- gather context -> plan -> act -> verify -> iterate/report

# Script Modes

`--script-mode` supports:

- `none`: no dedicated script is required.
- `single`: one primary script (`handler`) is expected.
- `multi`: multiple phase scripts are expected; LLM reasons between phase executions.

# Dependency By Reference

For new skills, prefer referencing existing skills by `skill_id` in `SKILL.md` instead of duplicating their logic.

Example: planning skill can reference `search-online-context` for research phases.

# Runtime Log Contract

1. `stdout` must contain one final JSON object.
2. Reserve `stderr` for unexpected runtime failures only.
3. Keep JSON concise so `workflow_hist` remains readable.

# Helper Script

- Path: `skills/all-agents/skill-creation/scripts/skill_creation.py`
- Actions:
  1. `inspect`: inspect existing skill package status.
  2. `scaffold`: create/update skill skeleton and section-compliant `SKILL.md`.
  3. `validate`: validate quality gates (frontmatter, script mode, required sections, workflow terms, runtime-log guidance).

# Preferred Action Input Template

Use `code_type=python` with `script_path` and `script_args` array.

Inspect example:

```json
{
  "code_type": "python",
  "script_path": "skills/all-agents/skill-creation/scripts/skill_creation.py",
  "script_args": [
    "--action", "inspect",
    "--skill-id", "search-online-context",
    "--scope", "all-agents"
  ]
}
```

Scaffold single-script example:

```json
{
  "code_type": "python",
  "script_path": "skills/all-agents/skill-creation/scripts/skill_creation.py",
  "script_args": [
    "--action", "scaffold",
    "--skill-id", "new-skill-id",
    "--scope", "all-agents",
    "--script-mode", "single",
    "--description", "One-line purpose of this skill"
  ]
}
```

Scaffold no-script example:

```json
{
  "code_type": "python",
  "script_path": "skills/all-agents/skill-creation/scripts/skill_creation.py",
  "script_args": [
    "--action", "scaffold",
    "--skill-id", "reasoning-only-skill",
    "--scope", "all-agents",
    "--script-mode", "none",
    "--description", "Procedure-first skill without dedicated handler script"
  ]
}
```

Scaffold multi-script with dependencies example:

```json
{
  "code_type": "python",
  "script_path": "skills/all-agents/skill-creation/scripts/skill_creation.py",
  "script_args": [
    "--action", "scaffold",
    "--skill-id", "planning-with-files-lite",
    "--scope", "all-agents",
    "--script-mode", "multi",
    "--dependency-skill", "search-online-context",
    "--dependency-skill", "documentation-distillation",
    "--description", "Planning skill using phase-based scripts and dependency skills"
  ]
}
```

Validate example:

```json
{
  "code_type": "python",
  "script_path": "skills/all-agents/skill-creation/scripts/skill_creation.py",
  "script_args": [
    "--action", "validate",
    "--skill-id", "new-skill-id",
    "--scope", "all-agents"
  ]
}
```

# Output JSON Shape

```json
{
  "executed_skill": "skill-creation",
  "status": "ok|error",
  "skill_created/updated": "summary string with action result and affected paths"
}
```

# Notes

- Use lowercase hyphenated `skill_id`.
- Use this skill before manually drafting large uncertain skill content.
- For generated scripts, enforce runtime-friendly output:
  - `stdout`: clear, meaningful execution evidence and final result.
  - `stderr`: only actionable failures.
  - final stdout line: one stable JSON object.
