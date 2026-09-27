"""Build a provider from its config."""

from __future__ import annotations

import os

from agents.providers.base import Provider, ProviderConfig


class MissingApiKey(RuntimeError):
    """`api_key_env` names a variable that is unset or empty in this environment."""


def _require_key(config: ProviderConfig) -> None:
    if config.api_key_env is None:
        return
    if not os.environ.get(config.api_key_env, "").strip():
        raise MissingApiKey(
            f"{config.api_key_env} is not set, so the {config.kind} provider cannot authenticate. "
            f"Export it in the shell, or put it in a .env file (gitignored). Never commit the key."
        )


def make_provider(config: ProviderConfig) -> Provider:
    _require_key(config)
    if config.kind == "anthropic":
        from agents.providers.anthropic_provider import AnthropicProvider

        return AnthropicProvider(config)
    if config.kind == "ollama":
        from agents.providers.ollama_provider import OllamaProvider

        return OllamaProvider(config)
    from agents.providers.openai_compat import OpenAICompatProvider

    return OpenAICompatProvider(config)
