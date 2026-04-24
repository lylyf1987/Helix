---
name: Update Skill
description: Update an existing skill's SKILL.md, scripts, or configuration.
---

# Purpose

Use this skill to improve or modify an existing skill — update its procedure, fix its scripts, or refine its description.

# When To Use

- When a skill's procedure needs improvement based on runtime experience.
- When a skill's scripts have bugs or need enhancements.
- When the user asks to update or improve an existing skill.

# Procedure

## Step 1: Read the Current Skill

```json
{
  "job_name": "read-existing-skill",
  "code_type": "bash",
  "script": "cat {workspace}/skills/{path}/SKILL.md"
}
```

## Step 2: List Skill Contents

```json
{
  "job_name": "list-skill-files",
  "code_type": "bash",
  "script": "find {workspace}/skills/{path} -type f"
}
```

## Step 3: Make Changes

Update the SKILL.md or scripts as needed:

```json
{
  "job_name": "update-skill-md",
  "code_type": "python",
  "script": "from pathlib import Path\npath = Path('{workspace}/skills/{path}/SKILL.md')\npath.write_text('''{updated_content}''', encoding='utf-8')\nprint(f'updated {path}')"
}
```

## Step 4: Verify

Read the updated file to confirm the changes are correct:

```json
{
  "job_name": "verify-skill-update",
  "code_type": "bash",
  "script": "cat {workspace}/skills/{path}/SKILL.md"
}
```

# Updating Generative Skills

A generative skill (runs an ML model through the local model service — e.g. `generate-image`, `generate-audio`, `generate-video`) has three moving parts beyond the `SKILL.md`: `model_spec.json`, `host_adapter.py`, and the `scripts/` HTTP clients. Which one you edit decides what has to happen after the edit.

## Which follow-up step does the edit require

| What changed | Re-run `helix model download` | Restart coordinator |
|---|---|---|
| `model_spec.json` (repo_id, backend, include/exclude/required) | yes | yes |
| `host_adapter.py` (deps, `_load`, `handle`) | no | yes |
| `scripts/*.py` (sandbox-side HTTP clients) | no | no |
| `SKILL.md` (procedure, rules, inputs) | no | no |

The "why" is the process boundary for each file:
- **Weights** live under `~/.helix/services/local-model-service/models/`. A spec change doesn't move them — only `helix model download` does.
- **Host adapter** runs inside a long-lived worker subprocess managed by the coordinator. Edits are only picked up on a restart.
- **Sandbox scripts** run in a fresh host-shell process per exec, so edits take effect on the very next call.

Read the authoritative reference in `docs/skills.md` ("Creating a Generative Skill") before changing `model_spec.json` or `host_adapter.py` — the contracts are strict (field names, `_BaseBackend` subclass, `_ok`/`_error` return shape, required `skill_name`/`task_type`/`workspace_root` payload keys).

## Procedure extensions

Apply these **after** the existing Steps 1-4 (read, list, change, verify).

**Step 5a — if `model_spec.json` changed:** re-download to refresh weights and the manifest snapshot.

```json
{
  "job_name": "redownload-gen-skill-model",
  "code_type": "bash",
  "script": "helix model download --skill {skill-name}"
}
```

**Step 5b — if `model_spec.json` or `host_adapter.py` changed:** restart the local model service so the new adapter is loaded.

```json
{
  "job_name": "restart-local-model-service",
  "code_type": "bash",
  "script": "helix stop local-model-service && helix start local-model-service"
}
```

**Step 6 — smoke-test** the affected skill by running `scripts/prepare_model.py` followed by one inference call. Both should return `status: ok`.

# Rules

- Always read the current skill before modifying it.
- Never edit a file under `skills/builtin_skills/` directly — that tree is resynced from the package on every startup, so your edits will be erased. To customize a built-in skill, copy its directory up one level into `skills/{new-name}/` and edit the copy.
- Keep the frontmatter format: only name and description fields.
- Preserve the existing procedure structure unless the update requires restructuring.
- If adding or modifying scripts, follow the script mode guidelines from the create-skill skill.
- Test any script changes by running them after the update.
- For generative skills: after editing `model_spec.json` re-run `helix model download --skill {name}`; after editing `host_adapter.py` or `model_spec.json` restart `local-model-service`; edits to `scripts/*.py` or `SKILL.md` take effect immediately.
