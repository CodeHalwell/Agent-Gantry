"""
Stubbed OpenAI and Azure OpenAI embedders.

These embedders reuse the SimpleEmbedder for deterministic vectors while
respecting API metadata for health checks and model names.
"""

from __future__ import annotations

from agent_gantry.adapters.embedders.simple import SimpleEmbedder
from agent_gantry.schema.config import EmbedderConfig


class _BaseOpenAIEmbedder(SimpleEmbedder):
    """Shared logic for OpenAI-compatible embedders."""

    def __init__(
        self,
        config: EmbedderConfig,
        *,
        dimension: int = 1536,
    ) -> None:
        super().__init__(dimension=dimension)
        self._config = config

    @property
    def model_name(self) -> str:
        return self._config.model

    async def health_check(self) -> bool:
        return bool(self._config.api_key)


class OpenAIEmbedder(_BaseOpenAIEmbedder):
    """OpenAI embedder stub."""

    def __init__(self, config: EmbedderConfig) -> None:
        super().__init__(config)


class AzureOpenAIEmbedder(_BaseOpenAIEmbedder):
    """Azure OpenAI embedder stub."""

    async def health_check(self) -> bool:
        return bool(self._config.api_key)
