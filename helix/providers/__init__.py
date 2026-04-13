"""LLM provider and error hierarchy.

Exports:
    LLMProvider: Universal OpenAI-compatible provider.
    LLMProviderError: Base exception for provider errors.
    LLMTransientError: Transient errors — caller should retry.
    LLMPermanentError: Permanent errors — do not retry.
"""

from .openai_compat import LLMProvider, LLMProviderError, LLMTransientError, LLMPermanentError

__all__ = ["LLMProvider", "LLMProviderError", "LLMTransientError", "LLMPermanentError"]
