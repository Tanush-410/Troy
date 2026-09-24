"""Build a provider from its config."""

from __future__ import annotations

from agents.providers.base import Provider, ProviderConfig


def make_provider(config: ProviderConfig) -> Provider:
    if config.kind == "anthropic":
        from agents.providers.anthropic_provider import AnthropicProvider

        return AnthropicProvider(config)
    if config.kind == "ollama":
        from agents.providers.ollama_provider import OllamaProvider

        return OllamaProvider(config)
    from agents.providers.openai_compat import OpenAICompatProvider

    return OpenAICompatProvider(config)
