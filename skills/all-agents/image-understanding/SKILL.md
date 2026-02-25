---
name: Image Understanding
handler: scripts/analyze_image.py
description: Analyze image content against a query context using a vision-capable model.
required_tools: exec
recommended_tools: exec
forbidden_tools:
---

# Purpose

Use this skill when you need objective image-content analysis (not only URL/title inference).

# Required Query Context

`--query` is required.

The Core Agent must prepare the query context before calling this skill:

1. Derive it from requester intent and current workflow context.
2. Make it explicit and testable (for example: style constraints, objects, composition, brand fit).
3. Keep it concise and task-oriented.

Examples:

- "Describe this image for a pet-store hero banner."
- "Check whether this image matches modern minimal blue-white palette."
- "Identify if the image contains two birds on a branch and describe mood."

# Runtime Script

- Script path: `skills/all-agents/image-understanding/scripts/analyze_image.py`
- Executor: `python` via `script_path` + `script_args`

# Runtime Contract

1. `stdout` must contain one final JSON object.
2. `stderr` should be used only for unexpected runtime failures.
3. Keep output concise and structured so runtime history is readable.

# Config and Terminal Error Rule

This skill requires vision model configuration:

- provider (`--provider` or `VISION_PROVIDER`)
- model (`--model` or `VISION_MODEL`)

If missing, script returns:

- `status = "error"`
- `error_code = "vision_config_missing"`

When `error_code` is `vision_config_missing`, Core Agent must stop internal retries and choose `chat_with_requester` to request configuration from the requester.

# Action Input Templates

Remote image URL example:

```json
{
  "code_type": "python",
  "script_path": "skills/all-agents/image-understanding/scripts/analyze_image.py",
  "script_args": [
    "--image-url", "https://example.com/image.jpg",
    "--query", "Describe this image and check if it fits modern blue-white minimal style",
    "--provider", "openai_compatible",
    "--model", "gpt-4o-mini"
  ]
}
```

Local image file example:

```json
{
  "code_type": "python",
  "script_path": "skills/all-agents/image-understanding/scripts/analyze_image.py",
  "script_args": [
    "--image-path", "assets/banner.jpg",
    "--query", "Describe objects, color palette, and mood for marketing banner",
    "--provider", "ollama",
    "--model", "llava:latest"
  ]
}
```

# Output JSON Shape

```json
{
  "executed_skill": "image-understanding",
  "status": "ok|error",
  "image_source": "...",
  "analysis": "...",
  "provider_used": "...",
  "model_used": "...",
  "error_code": "..."
}
```

# Notes

- Supported providers in this script: `openai_compatible`, `openai`, `zai`, `deepseek`, `lmstudio`, `ollama`.
- Provider aliases are normalized internally (for example `openai` -> `openai_compatible`).
- For OpenAI-compatible providers, `VISION_BASE_URL` and `VISION_API_KEY` can be set.
- For Ollama, default base URL is `OLLAMA_BASE_URL` or `http://localhost:11434`.
