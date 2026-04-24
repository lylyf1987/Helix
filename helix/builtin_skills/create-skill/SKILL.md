---
name: Create Skill
description: Create a new skill with a SKILL.md and optional scripts.
---

# Purpose

Use this skill to create a new reusable skill under the workspace's skills/ directory.

# When To Use

- When a task pattern is worth reusing across sessions.
- When the user explicitly asks to create a skill.
- When you notice a repeated workflow that would benefit from a standardized procedure.

# Skill Structure

A skill is a directory containing at minimum a SKILL.md file:

```
skills/
  my-new-skill/
    SKILL.md                  <- Required: procedure and metadata
    scripts/                  <- Optional: pre-built scripts (if needed)
      my_script.py
```

# SKILL.md Specification

## Frontmatter

Every SKILL.md must begin with exactly these two fields:

```
---
name: Human-Readable Skill Name
description: One-line description of what the skill does.
---
```

## Required Body Sections

- `# Purpose` — what the skill does
- `# When To Use` — when to use it and when to skip it
- `# Procedure` — step-by-step instructions with exec action examples
- `# Rules` — constraints and guidelines

## Optional Body Sections

- `# Action Input Templates` — for skills with scripts
- `# Skill Dependencies` — when referencing other skills

# Script Modes

Choose based on **step complexity**, not step count:

- **No scripts**: Use when every step is simple file I/O or standard commands. The agent follows the SKILL.md procedure directly. Most skills should be this type.
- **Single script**: Use when one step is complex enough that writing the code fresh each time would be error-prone (e.g. API calls with auth/retry, binary format parsing).
- **Multiple scripts**: Use when multiple steps are independently complex (e.g. generate-image with prepare_model.py + generate_image.py).

# Generative Skills

A skill is **generative** when it runs an ML model through the local model service rather than plain code in the exec sandbox. The built-in examples are `generate-image`, `generate-audio`, and `generate-video`. Use this path when the task needs model weights, GPU/MPS access, or a dedicated Python venv for heavy deps (MLX, PyTorch, diffusion pipelines, etc.).

## Extra files

On top of the standard `SKILL.md`, a generative skill adds three files:

```
skills/my-gen-skill/
  SKILL.md                 Procedure the agent follows
  model_spec.json          Which model weights to download (HuggingFace repo + manifest)
  host_adapter.py          Host-side worker that loads the model and serves /infer
  scripts/
    prepare_model.py       Sandbox-side HTTP client → POSTs /models/prepare
    generate_{task}.py     Sandbox-side HTTP client → POSTs /infer
```

## Authoritative reference

Read `docs/skills.md` section **"Creating a Generative Skill"** before writing any of these files. It covers the full contract:
- `model_spec.json` fields (`backend`, `source.repo_id`, `download_manifest.include/exclude/required`)
- `host_adapter.py` — Pattern A (in-process, keep the pipeline warm) vs Pattern B (subprocess per call)
- Handling upstream code that isn't a pip package (install from git, vendor, fetch lazily)
- Sandbox-side script contract (`HELIX_LOCAL_MODEL_SERVICE_URL`/`..._TOKEN` env vars, required `skill_name` / `task_type` / `workspace_root` payload keys)

## Procedure for a generative skill

Follow these steps in order. Deviating from the order usually breaks something (e.g. starting the coordinator before weights exist, or skipping the restart so the new adapter is never loaded).

**1. Read the authoritative reference.**

```json
{
  "job_name": "read-gen-skill-doc",
  "code_type": "bash",
  "script": "cat docs/skills.md | awk '/^## Creating a Generative Skill/,/^## Best Practices/'"
}
```

**2. Review the closest existing generative skill.** `generate-image` is the canonical Pattern A (MLX); `generate-audio` is Pattern A on PyTorch; `generate-video` is Pattern B (subprocess). Pick whichever matches your target most closely and read all its files.

```json
{
  "job_name": "read-example-gen-skill",
  "code_type": "bash",
  "script": "ls skills/builtin_skills/generate-image && cat skills/builtin_skills/generate-image/model_spec.json && echo '---' && cat skills/builtin_skills/generate-image/host_adapter.py"
}
```

**3. Create the skill directory with all four files.** Use the same Python `Path.write_text` pattern shown in Steps 3-4 of the procedural flow below, once per file (`SKILL.md`, `model_spec.json`, `host_adapter.py`, `scripts/prepare_model.py`, `scripts/generate_{task}.py`).

**4. Download the model weights.** This validates `model_spec.json`, provisions the per-backend venv, and populates `~/.helix/services/local-model-service/models/{repo_id}/`.

```json
{
  "job_name": "download-gen-skill-model",
  "code_type": "bash",
  "script": "helix model download --skill {skill-name}"
}
```

If the HF repo is gated, set `HF_TOKEN=hf_xxx` in your shell before the command.

**5. Restart the local model service** so the coordinator picks up the new adapter.

```json
{
  "job_name": "restart-local-model-service",
  "code_type": "bash",
  "script": "helix stop local-model-service && helix start local-model-service"
}
```

**6. Smoke-test the skill.** Run `scripts/prepare_model.py` once (models/prepare should return `status: ok`), then one inference call via `scripts/generate_{task}.py`. Both should print exactly one JSON object to stdout with `status: ok`.

# Procedure

## Step 1: Review Existing Skills as Examples

Before creating a new skill, read an existing skill for reference.

No-script example (simple procedure):
```json
{
  "job_name": "read-example-no-script",
  "code_type": "bash",
  "script": "cat skills/builtin_skills/retrieve-knowledge/SKILL.md"
}
```

Script-based example (with pre-built scripts):
```json
{
  "job_name": "read-example-with-scripts",
  "code_type": "bash",
  "script": "cat skills/builtin_skills/search-online-context/SKILL.md"
}
```

## Step 2: Create the Skill Directory

```json
{
  "job_name": "create-skill-dir",
  "code_type": "bash",
  "script": "mkdir -p skills/{skill-name}"
}
```

User-created skills go directly under `skills/`, not under `skills/builtin_skills/`.

## Step 3: Write the SKILL.md

```json
{
  "job_name": "write-skill-md",
  "code_type": "python",
  "script": "from pathlib import Path\npath = Path('skills/{skill-name}/SKILL.md')\npath.write_text('''{skill_md_content}''', encoding='utf-8')\nprint(f'created {path}')"
}
```

## Step 4: Add Scripts (if needed)

Only if the skill requires pre-built scripts:

```json
{
  "job_name": "write-skill-script",
  "code_type": "python",
  "script": "from pathlib import Path\nscripts_dir = Path('skills/{skill-name}/scripts')\nscripts_dir.mkdir(parents=True, exist_ok=True)\nscript = scripts_dir / '{script_name}.py'\nscript.write_text('''{script_content}''', encoding='utf-8')\nprint(f'created {script}')"
}
```

## Step 5: Verify

```json
{
  "job_name": "verify-skill",
  "code_type": "bash",
  "script": "cat skills/{skill-name}/SKILL.md && echo '---' && find skills/{skill-name} -type f"
}
```

# Rules

- Use lowercase kebab-case for skill directory names.
- User-created skills go under `skills/`, not `skills/builtin_skills/`.
- Never edit a file under `skills/builtin_skills/` directly — that tree is resynced from the package on every startup, so your edits will be erased. To customize a built-in skill, copy its directory up one level into `skills/{new-name}/` and edit the copy.
- Frontmatter must have exactly `name` and `description` — nothing else.
- Default to no-script skills. Only add scripts when step complexity justifies it.
- Reference existing skills by name in the procedure instead of duplicating their logic.
- For scripts: stdout should produce clear execution evidence; stderr only for failures.
- Generative skills must include `model_spec.json`, `host_adapter.py`, and matching `scripts/prepare_model.py` + `scripts/generate_{task}.py`. After creation, run `helix model download --skill {name}` then restart `local-model-service`.
