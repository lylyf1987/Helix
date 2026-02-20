---
name: Understand Skill
handler: scripts/understand_skill.py
description: Load full skill details (SKILL.md and script paths) into runtime history before execution.
required_tools: exec
recommended_tools: exec
forbidden_tools:
---

# Purpose

Use this skill to load the full details of another skill into runtime history before using it.

# Runtime Contract

1. `stdout` must contain one final JSON object.
2. Use `stderr` only for unexpected runtime failures.
3. Keep `stdout` factual and structured.

# Script

- Path: `skills/all-agents/understand-skill/scripts/understand_skill.py`
- Executor: `python`

# Preferred Action Input Template

```json
{
  "code_type": "python",
  "script_path": "skills/all-agents/understand-skill/scripts/understand_skill.py",
  "script_args": [
    "--skill-id", "search-online-context",
    "--scope", "all-agents"
  ]
}
```

# Output JSON Shape

```json
{
  "status": "ok|error",
  "action": "understand",
  "skill_id": "...",
  "scope": "all-agents|core-agent",
  "summary": "...",
  "next_step": "...",
  "details": {
    "skill_path": "...",
    "scripts": ["..."],
    "skill_context": "..."
  }
}
```
