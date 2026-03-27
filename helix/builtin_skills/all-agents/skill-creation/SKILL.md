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
2. `scaffold` baseline package with explicit `script_mode` and phase structure.
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
- `single`: one primary script (`handler`) is expected; use only for truly atomic utility tasks.
- `multi`: multiple phase scripts are expected; this is the preferred default for most executable skills so the LLM can reason between phase executions.

Default scaffold behavior:

- If you do not specify `--script-mode`, scaffold defaults to `multi`.
- For executable skills, prefer multiple small scripts over one all-in-one script.
- Use `single` only when the workflow is genuinely one-step and does not benefit from phase-by-phase evidence.

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
   - For `multi`, the scaffold should create several phase scripts, not one monolithic runtime script.
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

Scaffold multi-script example:

```json
{
  "code_type": "python",
  "script_path": "skills/all-agents/skill-creation/scripts/skill_creation.py",
  "script_args": [
    "--action", "scaffold",
    "--skill-id", "new-skill-id",
    "--scope", "all-agents",
    "--script-mode", "multi",
    "--description", "One-line purpose of this skill"
  ]
}
```

This should scaffold phase-oriented files such as:

- `scripts/gather_context.py`
- `scripts/execute_step.py`
- `scripts/verify_result.py`

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

Scaffold single-script exception example:

```json
{
  "code_type": "python",
  "script_path": "skills/all-agents/skill-creation/scripts/skill_creation.py",
  "script_args": [
    "--action", "scaffold",
    "--skill-id", "atomic-utility-skill",
    "--scope", "all-agents",
    "--script-mode", "single",
    "--description", "Tiny one-step utility skill with one deterministic script"
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
- Prefer `multi` for new executable skills so the agent can execute one phase, observe runtime evidence, then choose the next phase.
- For generated scripts, enforce runtime-friendly output:
  - `stdout`: clear, meaningful execution evidence and final result.
  - `stderr`: only actionable failures.
  - final stdout line: one stable JSON object.
