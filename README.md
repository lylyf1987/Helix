# Agentic System

Agentic System is an RL-inspired agent framework built around one small control law:

```text
state -> agent -> action -> environment -> observation -> state
```

The LLM is the `Agent`. The `Environment` is a real executable sandbox. `State` is structured context, not a vague chat log. The loop stays grounded by runtime evidence.

## Elegant Agentic Loop

![Agentic System working loop](docs/assets/agentic-system-loop.png)

The illustration above captures the design used throughout the repo:

- `State` is built from `workflow_summary`, `workflow_history`, and `latest_context`.
- `Agent` reasons over that state and emits exactly one action.
- `Action` can `chat`, `think`, `exec`, or `delegate`.
- `Sandbox` is the agent's computer: it can run bash and python, and it can use skills as installable software.
- `Runtime Evidence` flows back into state through stdout/stderr observations.
- `User` sits outside the loop.
- `Sub-agent Loop` mirrors the same pattern in an isolated delegated workspace.

## Install

```bash
python -m pip install -e .
```

This installs the `agentic-system` CLI. You can also run:

```bash
python -m agentic_system
```

## Quick Start

You need three things:

1. a workspace directory
2. an LLM provider
3. optional tool backends if you want search or image skills

### Default Local Setup: Ollama

Start Ollama and pull the default models:

```bash
ollama serve
ollama pull llama3.1:8b
ollama pull glm-ocr
ollama pull x/z-image-turbo
```

Then launch the UI:

```bash
agentic-system --workspace .
```

### Other Providers

Use one of these if you do not want Ollama for the core model:

```bash
# Z.AI
export ZAI_API_KEY="your-zai-api-key"
agentic-system --workspace . --provider zai --model glm-5

# DeepSeek
export DEEPSEEK_API_KEY="your-deepseek-api-key"
agentic-system --workspace . --provider deepseek --model deepseek-chat

# LM Studio
export LMSTUDIO_BASE_URL="http://localhost:1234/v1"   # only if not using the default
agentic-system --workspace . --provider lmstudio
```

## Sessions

- `workspace` is the shared project and artifact directory.
- `session_id` is the conversation memory key.
- without `--session-id`, the run is ephemeral
- with `--session-id`, state is saved to `WORKSPACE/.sessions/<session_id>.json`

Examples:

```bash
# fresh ephemeral run
agentic-system --workspace .

# start or resume a named session
agentic-system --workspace . --session-id design-review-01

# start a different session against the same workspace
agentic-system --workspace . --session-id bugfix-02
```

## CLI Essentials

| Argument | Default | Purpose |
|---|---|---|
| `--workspace` | required | workspace root |
| `--session-id` | none | persistent conversation key |
| `--provider` | `ollama` | core model provider |
| `--mode` | `controlled` | `controlled` asks before exec, `auto` does not |
| `--model` | provider default | override the provider's default model |
| `--searxng-base-url` | `http://127.0.0.1:8888` | search backend URL |

Provider model defaults when `--model` is omitted:

- `ollama` -> `llama3.1:8b`
- `deepseek` -> `deepseek-chat`
- `zai` -> `glm-5`
- `lmstudio` -> `local-model`
- `openai_compatible` -> `local-model`

## Runtime Commands

- `/help` — show commands
- `/status` — show runtime configuration and session status
- `/full_history` — open the saved full history view for the current named session
- `/observation` — open the current observation window view
- `/workflow_summary` — open the saved compact summary view
- `/last_prompt` — open the exact last prompt sent to the model
- `/exit` — quit

The inspection commands require `--session-id`.

## Optional Tool Backends

### Image Skills

If you use the built-in image skills, the runtime defaults are:

- image analysis: `ollama` + `glm-ocr`
- image generation: `ollama` + `x/z-image-turbo`

If those models are not available locally, change them with CLI flags or env vars before startup.

### Web Search: SearXNG

The search skill uses an external SearXNG instance. The runtime does not start it for you.

Default URL:

```text
http://127.0.0.1:8888
```

Minimal local setup with Docker:

1. Pull the image:

```bash
docker pull docker.io/searxng/searxng:latest
```

2. Create config and data directories:

```bash
mkdir -p ./searxng/config ./searxng/data
```

3. Create `./searxng/config/settings.yml`:

```yaml
use_default_settings: true

server:
  secret_key: "change-this-to-a-random-string"
  limiter: false

search:
  safe_search: 0
  formats:
    - html
    - json
```

4. Start SearXNG:

```bash
docker run --name searxng -d \
  -p 8888:8080 \
  -v "./searxng/config:/etc/searxng" \
  -v "./searxng/data:/var/cache/searxng" \
  docker.io/searxng/searxng:latest
```

5. Verify it:

```bash
curl http://127.0.0.1:8888
curl 'http://127.0.0.1:8888/search?q=test&format=json'
```

If you already have SearXNG elsewhere:

```bash
agentic-system --workspace . --searxng-base-url http://your-host:port
```

Stop the local container with:

```bash
docker container stop searxng
docker container rm searxng
```

## How To Think About The System

- The loop is small on purpose.
- The LLM does not directly change the world; it acts through the sandbox.
- Memory is layered:
  - `workflow_summary` for durable compact state
  - `workflow_history` for recent evidence
  - workspace files for persistent artifacts
- Skills are reusable software modules synced into `WORKSPACE/skills/`.
- Sub-agents use the same loop in isolated child workspaces.

## Troubleshooting

- `Missing API key for provider 'zai'`
  Set `ZAI_API_KEY` or `OPENAI_COMPAT_API_KEY`.
- `Missing API key for provider 'deepseek'`
  Set `DEEPSEEK_API_KEY` or `OPENAI_COMPAT_API_KEY`.
- Ollama connection failures
  Make sure `ollama serve` is running and the model exists locally.
- Search returns `403 Forbidden`
  Your SearXNG `settings.yml` is likely missing `json` under `search.formats`.
- Search returns connection refused
  SearXNG is not running or the configured URL is wrong.

## Repo Layout

```text
agentic_system/
  core/           -> state, action, agent, environment, loop
  runtime/        -> sandbox, approval, host, cli
  context/        -> prompt building, skill loading, knowledge loading
  providers/      -> ollama and OpenAI-compatible providers
  builtin_skills/ -> shipped skills bootstrapped into each workspace
  prompts/        -> system prompt templates
```
