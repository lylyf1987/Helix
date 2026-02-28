---
name: Image Understanding
handler: scripts/analyze_image.py
description: Analyze image content against a query context using a configured vision model provider.
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

# Runtime Script

- Script path: `skills/all-agents/image-understanding/scripts/analyze_image.py`
- Executor: `python` via `script_path` + `script_args`

# Runtime Contract

1. `stdout` must contain one final JSON object.
2. `stderr` should be used only for unexpected runtime failures.
3. Keep output concise and structured so runtime history is readable.

# Config and Error Rule

This script is model-provider based (no MCP path).

Required config:

- `--provider` (or runtime env `IMAGE_ANALYSIS_PROVIDER`) in:
  - `openai_compatible|zai|deepseek|lmstudio|ollama`
- `--model` (or runtime env `IMAGE_ANALYSIS_MODEL`)

If missing, script returns:

- `status = "error"`
- `error_code = "vision_config_missing"`

When `error_code` is `vision_config_missing`, Core Agent should choose `chat_with_requester` and ask requester to configure vision provider/model.

# Action Input Templates

Remote URL with Ollama:

```json
{
  "code_type": "python",
  "script_path": "skills/all-agents/image-understanding/scripts/analyze_image.py",
  "script_args": [
    "--image-url", "https://example.com/image.jpg",
    "--query", "Describe this image for a pet-store hero banner",
    "--provider", "ollama",
    "--model", "llava:latest"
  ]
}
```

Local image with OpenAI-compatible:

```json
{
  "code_type": "python",
  "script_path": "skills/all-agents/image-understanding/scripts/analyze_image.py",
  "script_args": [
    "--image-path", "assets/banner.jpg",
    "--query", "Describe objects, color palette, and mood for marketing banner",
    "--provider", "openai_compatible",
    "--model", "gpt-4o-mini",
    "--base-url", "<OPENAI_COMPAT_BASE_URL>"
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

- For `ollama`, URL input is downloaded first and sent as image bytes.
- HTTP 4xx/5xx response bodies from provider calls are surfaced in `analysis` for debugging.
