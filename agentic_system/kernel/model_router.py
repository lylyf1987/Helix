from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Any, Callable
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


@dataclass
class ModelResponse:
    provider: str
    model: str
    text: str


@dataclass(frozen=True)
class OpenAICompatProviderConfig:
    base_url_env_keys: tuple[str, ...]
    default_base_url: str
    api_key_env_keys: tuple[str, ...]
    timeout_env_keys: tuple[str, ...]
    core_model_env_key: str
    thinking_model_env_key: str
    default_thinking_model: str
    summarizer_model_env_key: str


def _first_env_value(keys: tuple[str, ...], default: str = "") -> str:
    for key in keys:
        value = os.getenv(key, "").strip()
        if value:
            return value
    return default


def _first_int_env_value(keys: tuple[str, ...], default: int) -> int:
    for key in keys:
        value = os.getenv(key, "").strip()
        if not value:
            continue
        try:
            return int(value)
        except ValueError:
            continue
    return int(default)


def _post_json(url: str, headers: dict[str, str], payload: dict[str, Any], timeout: int = 300) -> dict[str, Any]:
    req = Request(
        url=url,
        method="POST",
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
    )
    try:
        with urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code}: {body}") from exc
    except URLError as exc:
        raise RuntimeError(f"Network error: {exc}") from exc


class OllamaAdapter:
    provider = "ollama"

    def __init__(self) -> None:
        base = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
        self.endpoint = f"{base}/api/generate"
        self.timeout_seconds = int(os.getenv("OLLAMA_TIMEOUT_SECONDS", "300"))
        keep_alive = os.getenv("OLLAMA_KEEP_ALIVE", "").strip()
        self.keep_alive = keep_alive or None

    def generate(
        self,
        model: str,
        prompt: str,
        stream: bool = False,
        chunk_callback: Callable[[str], None] | None = None,
    ) -> ModelResponse:
        payload: dict[str, Any] = {
            "model": model,
            "prompt": prompt,
            "stream": stream,
            "options": {"temperature": 0.2},
        }
        if self.keep_alive:
            payload["keep_alive"] = self.keep_alive
        headers = {"Content-Type": "application/json"}
        if not stream:
            data = _post_json(self.endpoint, headers, payload, timeout=self.timeout_seconds)
            text = str(data.get("response", "") or "")
            if not text and isinstance(data.get("message"), dict):
                text = str(data["message"].get("content", "") or "")
            return ModelResponse(provider=self.provider, model=model, text=text)

        req = Request(
            url=self.endpoint,
            method="POST",
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
        )
        parts: list[str] = []
        try:
            with urlopen(req, timeout=self.timeout_seconds) as resp:
                for line in resp:
                    raw = line.decode("utf-8", errors="replace").strip()
                    if not raw:
                        continue
                    try:
                        item = json.loads(raw)
                    except json.JSONDecodeError:
                        continue
                    piece = str(item.get("response", "") or "")
                    if piece:
                        parts.append(piece)
                        if chunk_callback is not None:
                            chunk_callback(piece)
            return ModelResponse(provider=self.provider, model=model, text="".join(parts))
        except HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"HTTP {exc.code}: {body}") from exc
        except URLError as exc:
            raise RuntimeError(f"Network error: {exc}") from exc


class OpenAICompatibleAdapter:
    def __init__(
        self,
        *,
        provider: str,
        base_url_env_keys: tuple[str, ...],
        default_base_url: str,
        api_key_env_keys: tuple[str, ...],
        timeout_env_keys: tuple[str, ...],
        default_timeout_seconds: int = 300,
    ) -> None:
        self.provider = str(provider).strip().lower() or "openai_compatible"

        raw_base = _first_env_value(base_url_env_keys, default_base_url).strip()
        base = raw_base.rstrip("/")
        if not re.search(r"/v\d+$", base):
            base = f"{base}/v1"
        self.endpoint = f"{base}/chat/completions"
        self.timeout_seconds = _first_int_env_value(timeout_env_keys, default_timeout_seconds)
        self.api_token = _first_env_value(api_key_env_keys, "")

    @staticmethod
    def _content_to_text(content: Any) -> str:
        if isinstance(content, str):
            return content
        if not isinstance(content, list):
            return ""
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                text = item.get("text", "")
                if isinstance(text, str) and text:
                    parts.append(text)
        return "".join(parts)

    @classmethod
    def _extract_response_text(cls, payload: dict[str, Any]) -> str:
        choices = payload.get("choices", [])
        if not isinstance(choices, list) or not choices:
            return ""
        first = choices[0]
        if not isinstance(first, dict):
            return ""
        message = first.get("message")
        if isinstance(message, dict):
            text = cls._content_to_text(message.get("content"))
            if text:
                return text
        text_value = first.get("text")
        if isinstance(text_value, str):
            return text_value
        return ""

    @classmethod
    def _extract_stream_piece(cls, payload: dict[str, Any]) -> str:
        choices = payload.get("choices", [])
        if not isinstance(choices, list) or not choices:
            return ""
        first = choices[0]
        if not isinstance(first, dict):
            return ""
        delta = first.get("delta")
        if isinstance(delta, dict):
            text = cls._content_to_text(delta.get("content"))
            if text:
                return text
        text_value = first.get("text")
        if isinstance(text_value, str):
            return text_value
        return ""

    def generate(
        self,
        model: str,
        prompt: str,
        stream: bool = False,
        chunk_callback: Callable[[str], None] | None = None,
    ) -> ModelResponse:
        payload: dict[str, Any] = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": stream,
            "temperature": 0.2,
        }
        headers = {"Content-Type": "application/json"}
        if self.api_token:
            headers["Authorization"] = f"Bearer {self.api_token}"

        if not stream:
            data = _post_json(self.endpoint, headers, payload, timeout=self.timeout_seconds)
            text = self._extract_response_text(data)
            return ModelResponse(provider=self.provider, model=model, text=text)

        req = Request(
            url=self.endpoint,
            method="POST",
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
        )
        parts: list[str] = []
        try:
            with urlopen(req, timeout=self.timeout_seconds) as resp:
                for line in resp:
                    raw = line.decode("utf-8", errors="replace").strip()
                    if not raw:
                        continue
                    if raw.startswith("data:"):
                        raw = raw[5:].strip()
                    if not raw or raw == "[DONE]":
                        continue
                    try:
                        item = json.loads(raw)
                    except json.JSONDecodeError:
                        continue
                    piece = self._extract_stream_piece(item)
                    if piece:
                        parts.append(piece)
                        if chunk_callback is not None:
                            chunk_callback(piece)
            return ModelResponse(provider=self.provider, model=model, text="".join(parts))
        except HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"HTTP {exc.code}: {body}") from exc
        except URLError as exc:
            raise RuntimeError(f"Network error: {exc}") from exc


class ModelRouter:
    _PROVIDER_ALIASES: dict[str, str] = {
        "openai-compatible": "openai_compatible",
        "openaicompat": "openai_compatible",
        "z.ai": "zai",
        "deepseek-ai": "deepseek",
    }

    _OPENAI_COMPAT_PROVIDER_CONFIGS: dict[str, OpenAICompatProviderConfig] = {
        "lmstudio": OpenAICompatProviderConfig(
            base_url_env_keys=("LMSTUDIO_BASE_URL", "OPENAI_COMPAT_BASE_URL"),
            default_base_url="http://localhost:1234/v1",
            api_key_env_keys=("LMSTUDIO_API_KEY", "LM_API_TOKEN", "OPENAI_COMPAT_API_KEY"),
            timeout_env_keys=("LMSTUDIO_TIMEOUT_SECONDS", "OPENAI_COMPAT_TIMEOUT_SECONDS"),
            core_model_env_key="LMSTUDIO_MODEL_CORE_AGENT",
            thinking_model_env_key="LMSTUDIO_MODEL_THINKING",
            default_thinking_model="local-model",
            summarizer_model_env_key="LMSTUDIO_MODEL_WORKFLOW_SUMMARIZER",
        ),
        "zai": OpenAICompatProviderConfig(
            base_url_env_keys=("ZAI_BASE_URL", "OPENAI_COMPAT_BASE_URL", "LMSTUDIO_BASE_URL"),
            default_base_url="https://api.z.ai/api/paas/v4",
            api_key_env_keys=("ZAI_API_KEY", "OPENAI_COMPAT_API_KEY", "LMSTUDIO_API_KEY", "LM_API_TOKEN"),
            timeout_env_keys=("ZAI_TIMEOUT_SECONDS", "OPENAI_COMPAT_TIMEOUT_SECONDS", "LMSTUDIO_TIMEOUT_SECONDS"),
            core_model_env_key="ZAI_MODEL_CORE_AGENT",
            thinking_model_env_key="ZAI_MODEL_THINKING",
            default_thinking_model="glm-5",
            summarizer_model_env_key="ZAI_MODEL_WORKFLOW_SUMMARIZER",
        ),
        "deepseek": OpenAICompatProviderConfig(
            base_url_env_keys=("DEEPSEEK_BASE_URL", "OPENAI_COMPAT_BASE_URL", "LMSTUDIO_BASE_URL"),
            default_base_url="https://api.deepseek.com",
            api_key_env_keys=("DEEPSEEK_API_KEY", "OPENAI_COMPAT_API_KEY", "LMSTUDIO_API_KEY", "LM_API_TOKEN"),
            timeout_env_keys=("DEEPSEEK_TIMEOUT_SECONDS", "OPENAI_COMPAT_TIMEOUT_SECONDS", "LMSTUDIO_TIMEOUT_SECONDS"),
            core_model_env_key="DEEPSEEK_MODEL_CORE_AGENT",
            thinking_model_env_key="DEEPSEEK_MODEL_THINKING",
            default_thinking_model="deepseek-chat",
            summarizer_model_env_key="DEEPSEEK_MODEL_WORKFLOW_SUMMARIZER",
        ),
        "openai_compatible": OpenAICompatProviderConfig(
            base_url_env_keys=("OPENAI_COMPAT_BASE_URL",),
            default_base_url="http://localhost:1234/v1",
            api_key_env_keys=("OPENAI_COMPAT_API_KEY", "LM_API_TOKEN"),
            timeout_env_keys=("OPENAI_COMPAT_TIMEOUT_SECONDS",),
            core_model_env_key="OPENAI_COMPAT_MODEL_CORE_AGENT",
            thinking_model_env_key="OPENAI_COMPAT_MODEL_THINKING",
            default_thinking_model="local-model",
            summarizer_model_env_key="OPENAI_COMPAT_MODEL_WORKFLOW_SUMMARIZER",
        ),
    }

    @classmethod
    def _normalize_provider_name(cls, provider: str) -> str:
        provider_name = str(provider).strip().lower() or "ollama"
        return cls._PROVIDER_ALIASES.get(provider_name, provider_name)

    @staticmethod
    def _resolve_model_selection(
        *,
        model_override: str | None,
        core_model_env_key: str,
        thinking_model_env_key: str,
        default_thinking_model: str,
        summarizer_model_env_key: str,
    ) -> tuple[str, str]:
        core_model = model_override or os.getenv(
            core_model_env_key,
            os.getenv(thinking_model_env_key, default_thinking_model),
        )
        summarizer_model = os.getenv(summarizer_model_env_key, core_model)
        return core_model, summarizer_model

    def __init__(self, provider: str = "ollama", model_name: str | None = None) -> None:
        provider_name = self._normalize_provider_name(provider)
        self.provider = provider_name

        if provider_name == "ollama":
            self.adapter = OllamaAdapter()
            core_model, summarizer_model = self._resolve_model_selection(
                model_override=model_name,
                core_model_env_key="OLLAMA_MODEL_CORE_AGENT",
                thinking_model_env_key="OLLAMA_MODEL_THINKING",
                default_thinking_model="llama3.1:8b",
                summarizer_model_env_key="OLLAMA_MODEL_WORKFLOW_SUMMARIZER",
            )
        else:
            config = self._OPENAI_COMPAT_PROVIDER_CONFIGS.get(provider_name)
            if config is None:
                raise NotImplementedError(
                    "Provider "
                    f"'{provider_name}' is not implemented yet. "
                    "Use --provider ollama, lmstudio, zai, deepseek, or openai_compatible."
                )
            self.adapter = OpenAICompatibleAdapter(
                provider=provider_name,
                base_url_env_keys=config.base_url_env_keys,
                default_base_url=config.default_base_url,
                api_key_env_keys=config.api_key_env_keys,
                timeout_env_keys=config.timeout_env_keys,
            )
            core_model, summarizer_model = self._resolve_model_selection(
                model_override=model_name,
                core_model_env_key=config.core_model_env_key,
                thinking_model_env_key=config.thinking_model_env_key,
                default_thinking_model=config.default_thinking_model,
                summarizer_model_env_key=config.summarizer_model_env_key,
            )
        self.models: dict[str, str] = {
            "core_agent": core_model,
            "workflow_summarizer": summarizer_model,
            "workflow_history_compactor": summarizer_model,
        }

    def _select_model(self, role: str) -> str:
        role_name = str(role).strip()
        if role_name in self.models:
            return self.models[role_name]
        return self.models["core_agent"]

    @staticmethod
    def _has_non_empty_script_args(action_input: dict[str, Any]) -> bool:
        raw_args = action_input.get("script_args", [])
        if isinstance(raw_args, str):
            return bool(raw_args.strip())
        if isinstance(raw_args, (list, tuple)):
            return any(str(item).strip() for item in raw_args)
        return raw_args not in (None, {})

    @classmethod
    def _validate_core_agent_payload(cls, payload: dict[str, Any]) -> list[str]:
        errors: list[str] = []
        raw_response = payload.get("raw_response", "")
        if not isinstance(raw_response, str) or not raw_response.strip():
            errors.append("raw_response must be a non-empty string")

        action = str(payload.get("action", "")).strip().lower()
        allowed_actions = {"chat_with_requester", "keep_reasoning", "exec"}
        if action not in allowed_actions:
            errors.append("action must be one of chat_with_requester, keep_reasoning, exec")

        action_input = payload.get("action_input")
        if not isinstance(action_input, dict):
            errors.append("action_input must be an object")
            return errors

        if action in {"chat_with_requester", "keep_reasoning"}:
            if action_input:
                errors.append(f"action_input must be {{}} when action is {action}")
            return errors

        if action == "exec":
            code_type = str(action_input.get("code_type", "")).strip().lower()
            if code_type not in {"bash", "python"}:
                errors.append("exec action_input.code_type must be \"bash\" or \"python\"")
            script_path = str(action_input.get("script_path", "")).strip()
            script = str(action_input.get("script", "")).strip()
            has_script_path = bool(script_path)
            has_script = bool(script)
            if has_script_path == has_script:
                errors.append("exec action_input must include exactly one of script_path or script")
            if has_script and cls._has_non_empty_script_args(action_input):
                errors.append("exec action_input.script_args is only allowed when script_path is used")
        return errors

    @staticmethod
    def _validate_workflow_summarizer_payload(payload: dict[str, Any]) -> list[str]:
        candidate = payload.get("workflow_summary")
        if isinstance(candidate, str):
            return []
        return ["workflow_summary must be a string"]

    @staticmethod
    def _validate_workflow_compactor_payload(payload: dict[str, Any]) -> list[str]:
        candidate = payload.get("workflow_hist_compact")
        if isinstance(candidate, str):
            return []
        return ["workflow_hist_compact must be a string"]

    @classmethod
    def _parse_json_payload_with_error(
        cls,
        text: str,
        role: str,
    ) -> tuple[dict[str, Any] | None, str]:
        raw = text.strip()
        if not raw:
            return None, "empty model output"

        match = re.search(r"<output>(.*?)</output>", raw, flags=re.IGNORECASE | re.DOTALL)
        if not match:
            return None, "missing <output>...</output> block"
        block = match.group(1).strip()
        if not block:
            return None, "empty <output> block"
        try:
            parsed = json.loads(block)
        except json.JSONDecodeError as exc:
            return None, f"invalid JSON in <output> ({exc.msg} at line {exc.lineno}, column {exc.colno})"
        if not isinstance(parsed, dict):
            return None, "<output> JSON must be an object"

        role_name = str(role).strip().lower()
        if role_name == "core_agent":
            errors = cls._validate_core_agent_payload(parsed)
            if errors:
                return None, "; ".join(errors)
        elif role_name == "workflow_summarizer":
            errors = cls._validate_workflow_summarizer_payload(parsed)
            if errors:
                return None, "; ".join(errors)
        elif role_name == "workflow_history_compactor":
            errors = cls._validate_workflow_compactor_payload(parsed)
            if errors:
                return None, "; ".join(errors)
        return parsed, ""

    @staticmethod
    def _stream_raw_response_from_chunk_factory(
        callback: Callable[[str], None],
    ) -> Callable[[str], None]:
        key_pattern = re.compile(r'"raw_response"\s*:\s*"')
        search_buffer = ""
        capture = False
        done = False
        escape = False
        unicode_remaining = 0
        unicode_digits = ""

        def emit(token: str) -> None:
            if token:
                callback(token)

        def consume_string_chars(text: str) -> None:
            nonlocal done, escape, unicode_remaining, unicode_digits
            if done:
                return
            for ch in text:
                if done:
                    return
                if unicode_remaining > 0:
                    if ch.lower() in "0123456789abcdef":
                        unicode_digits += ch
                        unicode_remaining -= 1
                        if unicode_remaining == 0:
                            try:
                                emit(chr(int(unicode_digits, 16)))
                            except Exception:
                                emit("\\u" + unicode_digits)
                            unicode_digits = ""
                    else:
                        emit("\\u" + unicode_digits + ch)
                        unicode_remaining = 0
                        unicode_digits = ""
                    continue

                if escape:
                    mapping = {
                        '"': '"',
                        "\\": "\\",
                        "/": "/",
                        "b": "\b",
                        "f": "\f",
                        "n": "\n",
                        "r": "\r",
                        "t": "\t",
                    }
                    if ch == "u":
                        unicode_remaining = 4
                        unicode_digits = ""
                    else:
                        emit(mapping.get(ch, ch))
                    escape = False
                    continue

                if ch == "\\":
                    escape = True
                    continue
                if ch == '"':
                    done = True
                    return
                emit(ch)

        def on_chunk(chunk: str) -> None:
            nonlocal search_buffer, capture
            if done or not chunk:
                return
            if capture:
                consume_string_chars(chunk)
                return

            search_buffer += chunk
            match = key_pattern.search(search_buffer)
            if not match:
                if len(search_buffer) > 256:
                    search_buffer = search_buffer[-256:]
                return

            capture = True
            remainder = search_buffer[match.end() :]
            search_buffer = ""
            if remainder:
                consume_string_chars(remainder)

        return on_chunk

    def generate(
        self,
        role: str = "core_agent",
        final_prompt: str | None = None,
        raw_response_callback: Callable[[str], None] | None = None,
    ) -> dict[str, Any]:
        prompt = str(final_prompt or "").strip()
        if not prompt:
            return {}
        model = self._select_model(role)
        parsed_callback: Callable[[str], None] | None = None
        if raw_response_callback is not None:
            parsed_callback = self._stream_raw_response_from_chunk_factory(raw_response_callback)

        chunk_callback: Callable[[str], None] | None = parsed_callback

        response = self.adapter.generate(
            model=model,
            prompt=prompt,
            stream=True,
            chunk_callback=chunk_callback,
        )
        payload, parse_error = self._parse_json_payload_with_error(
            response.text or "",
            role=role,
        )
        if isinstance(payload, dict):
            payload["_parse_ok"] = True
            payload["_parse_error"] = ""
            return payload
        return {
            "_parse_ok": False,
            "_parse_error": parse_error,
        }
