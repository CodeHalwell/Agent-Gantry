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
    """
    Stub implementation of an OpenAI-compatible embedder.

    This class does not make real API calls to the OpenAI embeddings endpoint.
    Instead, it delegates all embedding logic to SimpleEmbedder, producing deterministic
    vectors for testing and development purposes. It exists to provide API metadata
    compatibility and health checks, but should not be used in production where real
    OpenAI embeddings are required.
    """

    def __init__(self, config: EmbedderConfig) -> None:
        super().__init__(config)


class AzureOpenAIEmbedder(_BaseOpenAIEmbedder):
    """
    Stub implementation of an Azure OpenAI-compatible embedder.

    This class does not make real API calls to the Azure OpenAI embeddings endpoint.
    Instead, it delegates all embedding logic to SimpleEmbedder, producing deterministic
    vectors for testing and development purposes. It exists to provide API metadata
    compatibility and health checks, but should not be used in production where real
    Azure OpenAI embeddings are required.
    """
    pass
