# Quick Start

## Prerequisites

- Python 3.11+
- An LLM endpoint (local Ollama or cloud API)

## Install

```bash
pip install -e .
```

## Option A: Local LLM with Ollama

### 1. Start Ollama

```bash
ollama serve
ollama pull llama3.1:8b
```

### 2. Start OpenHelix

```bash
helix \
  --endpoint-url http://localhost:11434/v1 \
  --model llama3.1:8b \
  --workspace ~/my-agent \
  --session-id first-session
```

### 3. Try It

```
user> What files are in the current directory?
```

The agent will write a bash script (`ls -la`), execute it in the host-shell sandbox (you'll be asked to approve in controlled mode), read the output, and report back.

The agent always starts in **controlled** mode, which means **every bash or python execution pauses for your approval first**. You'll see a prompt showing the job name, the script, and an `[y/N/s/p/k/a]` menu (`y` = allow once, `s` = allow same exact exec for the session, `p` = allow same script pattern, `k` = allow same script_path ignoring args, `a` = switch the session to auto and approve this exec, `N` = deny and return control to you). If you'd rather let the agent run autonomously without any interruptions, type `/mode auto` at the REPL — `/mode controlled` switches back. Mode is not persisted across restarts.

## Option B: Cloud LLM (DeepSeek)

```bash
helix \
  --endpoint-url https://api.deepseek.com/v1 \
  --api-key $DEEPSEEK_API_KEY \
  --model deepseek-chat \
  --workspace ~/my-agent \
  --session-id first-session
```

Any OpenAI-compatible endpoint works — just provide the URL and model name.

## Thinking Mode: `--think` and `--effort`

Modern reasoning-capable models expose two knobs, both optional:

- **`--think enable|disable`** — turn thinking on or off.
- **`--effort minimal|low|medium|high`** — how hard the model should reason (OpenAI GPT-5/o-series, DeepSeek, Gemini OpenAI-compat).

Omit either flag to fall back to whatever the server does by default. They're independent: you can set one, both, or neither.

```bash
# DeepSeek with thinking on at medium effort
helix \
  --endpoint-url https://api.deepseek.com/v1 \
  --api-key $DEEPSEEK_API_KEY \
  --model deepseek-chat \
  --think enable --effort medium \
  --workspace ~/my-agent --session-id deep-01

# OpenAI GPT-5 (reasoning is always on — just pick the effort)
helix \
  --endpoint-url https://api.openai.com/v1 \
  --api-key $OPENAI_API_KEY \
  --model gpt-5 \
  --effort high \
  --workspace ~/my-agent --session-id design-01

# Local Ollama Qwen3 — disable thinking for faster replies
helix \
  --endpoint-url http://localhost:11434/v1 \
  --model qwen3:8b \
  --think disable \
  --workspace ~/my-agent --session-id quick-chat
```

Under the hood, `--think` injects the three widely-used field conventions (`thinking.type`, `think`, `chat_template_kwargs.enable_thinking`) and `--effort` injects `reasoning_effort`. Servers that don't recognize a field ignore it silently, so the same flag works across OpenAI, DeepSeek, Z.ai/GLM, Ollama, vLLM/SGLang, and Gemini's OpenAI-compat endpoint. Claude extended thinking uses the Anthropic Messages API and is out of scope for OpenAI-compatible endpoints.

## What Happened

When you started OpenHelix:

1. Built-in skills were copied into your workspace under `skills/builtin_skills/`.
2. The REPL started, waiting for your input.

When you sent a message:

1. Your message was recorded as a `[user]` turn.
2. The agent read its system prompt (identity + available skills) and your message.
3. The agent decided on an action (e.g., `exec` with a bash script).
4. In controlled mode, you approved the script via the `[y/N/s/p/k/a]` prompt.
5. The script ran in the host-shell sandbox with `cwd` set to your workspace.
6. The stdout/stderr came back as a `[runtime]` turn.
7. The agent read the result and responded with `chat`.

## Optional: Start Services

### Web Search (SearXNG)

```bash
helix start searxng
```

Now the agent can use the `search-online-context` skill to search the web.

### Local Model Service (for image/audio/video generation)

```bash
# Download model weights from HuggingFace Hub
helix model download --skill generate-image

# Start the service (one shared service across all workspaces)
helix start local-model-service
```

All generative-skill model weights are fetched from **[HuggingFace Hub](https://huggingface.co)** — this is currently the only supported source. If a model is gated or private, set `HF_TOKEN=hf_xxx...` in your shell before running `helix model download`.

Now the agent can use generative skills like `generate-image`.

### Check Service Status

```bash
helix status
```

## Letting the Agent Push to GitHub

Agent scripts run in the host-shell sandbox — i.e. on your shell, as your user, with your environment. That means your normal git credentials (SSH keys in `~/.ssh`, `~/.gitconfig`, any `Host` aliases) are available to the agent with **no extra setup**. A plain `git push` from inside an agent session uses whatever auth your own shell uses.

The approval prompt is the primary safety net:

- In the default `controlled` mode, every `exec` — including `git push` — pauses for your `[y/N/s/p/k/a]` decision first.
- The sandbox's outside-workspace write detector also highlights writes to paths outside `{{WORKSPACE_ROOT}}` in the approval prompt, so unexpected file modifications stand out before you approve.

### Scoping the blast radius

Because the agent inherits your host credentials, it can in principle reach **any** GitHub repo (or SSH target) your keys can reach. Two practical ways to narrow that:

**Fine-grained personal access token, exported into the session's environment.** Set it in your shell before launching `helix`:

```bash
export GH_TOKEN=github_pat_11AB...
helix --endpoint-url ... --model ... --workspace ~/agent --session-id ...
```

Then inside a session, the agent can push over HTTPS with only that token's scope:

```bash
git push https://x-access-token:$GH_TOKEN@github.com/owner/repo.git master
```

Generate the token at https://github.com/settings/tokens?type=beta → "Generate new token" → **fine-grained personal access token**, with **Repository access: Only select repositories** and a short expiration.

**Approval prompts as the interactive gate.** Even without a scoped token, staying in `controlled` mode means every push command surfaces first. Read the command, deny anything unfamiliar.

> **Do not switch to auto mode (`/mode auto`, or `a` at a prompt) unless you've scoped credentials yourself.** Auto mode skips the approval prompt, so there's nothing between the agent and your full git authority.

## Next Steps

- [Introduction](introduction.md) — core concepts and design philosophy
- [Skills](skills.md) — built-in skills and how to create your own
- [Knowledge](knowledge.md) — the hierarchical knowledge system
- [Storage](storage.md) — where OpenHelix puts every file on your machine
