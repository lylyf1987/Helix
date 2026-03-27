"""Model provider adapters and protocol.

Exports:
    ModelProvider: Protocol that all providers must satisfy.
    create_provider: Factory to create a provider by name.
"""

from __future__ import annotations

from typing import Any, Callable, Optional, Protocol


# --------------------------------------------------------------------------- #
# ModelProvider protocol
# --------------------------------------------------------------------------- #


class ModelProvider(Protocol):
    """Minimal contract for LLM providers."""

    def generate(
        self,
        prompt: str,
        *,
        stream: bool = False,
        chunk_callback: Optional[Callable[[str], None]] = None,
    ) -> str:
        """Generate text from the given prompt."""
        ...


# --------------------------------------------------------------------------- #
# Provider factory
# --------------------------------------------------------------------------- #


def create_provider(
    provider_name: str,
    *,
    model: Optional[str] = None,
) -> Any:
    """Create the appropriate ModelProvider from a provider name string.

    Returns an object satisfying the ``ModelProvider`` protocol.
    """
    name = provider_name.strip().lower() or "ollama"

    if name == "ollama":
        from .ollama import OllamaProvider
        return OllamaProvider(model=model)

    # All other providers go through OpenAI-compatible adapter
    from .openai_compat import OpenAICompatProvider
    return OpenAICompatProvider(provider=name, model=model)
