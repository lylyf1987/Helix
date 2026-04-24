"""Universal LLM provider — OpenAI-compatible chat completions adapter.

Works with any server that speaks the OpenAI chat completions API:
Ollama, vLLM, LM Studio, DeepSeek, Together, OpenRouter, etc.
"""

from __future__ import annotations

import json
import socket
import ssl
from http.client import RemoteDisconnected
from typing import Any, Callable, Optional
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

class LLMProviderError(RuntimeError):
    """Base for all LLM provider errors."""

    def __init__(self, message: str, *, status_code: int | None = None, retry_after: float | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.retry_after = retry_after


class LLMTransientError(LLMProviderError):
    """Transient provider error — caller should retry (429, 5xx, network)."""


class LLMPermanentError(LLMProviderError):
    """Permanent provider error — do not retry (401, 403, 404, other 4xx)."""


_TRANSIENT_HTTP_CODES = {429, 500, 502, 503, 504}


class LLMProvider:
    """Universal LLM provider for OpenAI-compatible chat completions endpoints."""

    def __init__(
        self,
        *,
        endpoint_url: str,
        model: str,
        api_key: str = "",
        timeout: int = 300,
        temperature: float = 0.2,
        think: bool | None = None,
        reasoning_effort: str | None = None,
    ) -> None:
        self.endpoint_url = endpoint_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.timeout = timeout
        self.temperature = temperature
        self.think = think
        self.reasoning_effort = reasoning_effort

    def generate(
        self,
        messages: list[dict[str, str]],
        *,
        chunk_callback: Optional[Callable[[str], None]] = None,
    ) -> str:
        """Generate text via streaming SSE chat completions."""
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": True,
            "temperature": self.temperature,
        }
        if self.think is not None:
            # Inject every widely-used thinking-mode field so a single flag
            # works across OpenAI-compatible servers. Unknown fields are
            # ignored by liberal servers:
            #   - DeepSeek / Z.ai (GLM) / Anthropic-style: thinking.type
            #   - Ollama:                                  think
            #   - vLLM / SGLang (Qwen3-family):            chat_template_kwargs.enable_thinking
            payload["thinking"] = {"type": "enabled" if self.think else "disabled"}
            payload["think"] = self.think
            payload["chat_template_kwargs"] = {"enable_thinking": self.think}
        if self.reasoning_effort is not None:
            # `reasoning_effort` is the de-facto OpenAI-compat standard,
            # accepted by OpenAI (GPT-5/o-series), DeepSeek, and Gemini.
            # Providers that don't recognize it ignore it.
            payload["reasoning_effort"] = self.reasoning_effort
        headers: dict[str, str] = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        req = Request(
            url=self.endpoint_url,
            method="POST",
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
        )
        parts: list[str] = []
        try:
            with urlopen(req, timeout=self.timeout) as resp:
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
            return "".join(parts)
        except HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            msg = f"LLM HTTP {exc.code}: {body}"
            if exc.code in _TRANSIENT_HTTP_CODES:
                retry_after = None
                raw = exc.headers.get("Retry-After") if exc.headers else None
                if raw:
                    try:
                        retry_after = float(raw)
                    except (ValueError, TypeError):
                        pass
                raise LLMTransientError(msg, status_code=exc.code, retry_after=retry_after) from exc
            raise LLMPermanentError(msg, status_code=exc.code) from exc
        except (URLError, TimeoutError, socket.timeout, ConnectionError, RemoteDisconnected, ssl.SSLError) as exc:
            raise LLMTransientError(f"LLM network error: {exc}") from exc

    @staticmethod
    def _extract_stream_piece(data: dict[str, Any]) -> str:
        """Extract text from one streaming SSE chunk."""
        choices = data.get("choices", [])
        if not isinstance(choices, list) or not choices:
            return ""
        first = choices[0]
        if not isinstance(first, dict):
            return ""
        delta = first.get("delta")
        if not isinstance(delta, dict):
            return ""
        content = delta.get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return "".join(
                item.get("text", "")
                for item in content
                if isinstance(item, dict) and isinstance(item.get("text"), str)
            )
        return ""
