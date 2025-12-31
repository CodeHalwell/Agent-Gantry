"""
OpenAI and Azure OpenAI embedders with production-ready implementations.

Supports multiple embedding models with configurable dimensions and
batch processing with built-in retry logic.
"""

from __future__ import annotations

import logging
import os
from typing import Any

from agent_gantry.schema.config import EmbedderConfig

logger = logging.getLogger(__name__)


class OpenAIEmbedder:
    """
    Production OpenAI embedder using the official OpenAI Python client.

    Supports models: text-embedding-3-small, text-embedding-3-large, text-embedding-ada-002
    with configurable dimensions for Matryoshka truncation.
    """

    # Default dimensions for each model
    MODEL_DIMENSIONS = {
        "text-embedding-3-small": 1536,
        "text-embedding-3-large": 3072,
        "text-embedding-ada-002": 1536,
    }

    def __init__(self, config: EmbedderConfig, *, dimension: int | None = None) -> None:
        """
        Initialize the OpenAI embedder.

        Args:
            config: Embedder configuration with API key and model
            dimension: Optional output dimension for Matryoshka truncation

        Raises:
            ImportError: If openai package is not installed
            ValueError: If API key is missing
        """
        try:
            from openai import AsyncOpenAI
        except ImportError as exc:
            raise ImportError(
                "OpenAI package is not installed. Install it with:\n"
                "  pip install agent-gantry[openai]"
            ) from exc

        api_key = config.api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OpenAI API key is required. Set it in config or OPENAI_API_KEY environment variable."
            )

        self._config = config
        self._model = config.model or "text-embedding-3-small"
        self._batch_size = config.batch_size
        self._max_retries = config.max_retries

        # Determine dimension - use specified, or config, or model default
        if dimension is not None:
            self._dimension = dimension
        elif config.dimension is not None:
            self._dimension = config.dimension
        else:
            self._dimension = self.MODEL_DIMENSIONS.get(self._model, 1536)

        # Initialize client with retry logic
        self._client = AsyncOpenAI(
            api_key=api_key,
            max_retries=self._max_retries,
        )

        logger.info(
            f"Initialized OpenAIEmbedder with model={self._model}, dimension={self._dimension}"
        )

    @property
    def dimension(self) -> int:
        """Return the embedding dimension."""
        return self._dimension

    @property
    def model_name(self) -> str:
        """Return the model name."""
        return self._model

    def get_embedder_id(self) -> str:
        """
        Return a unique identifier for this embedder configuration.

        Returns:
            Identifier combining model name and dimension
        """
        return f"{self._model}:{self._dimension}"

    async def embed_text(self, text: str) -> list[float]:
        """
        Embed a single text.

        Args:
            text: Text to embed

        Returns:
            Embedding vector
        """
        result = await self.embed_batch([text])
        return result[0]

    async def embed_batch(
        self,
        texts: list[str],
        batch_size: int | None = None,
    ) -> list[list[float]]:
        """
        Embed multiple texts with batching.

        Args:
            texts: List of texts to embed
            batch_size: Optional batch size (default: use config)

        Returns:
            List of embedding vectors
        """
        if not texts:
            return []

        batch_size = batch_size or self._batch_size
        all_embeddings: list[list[float]] = []

        # Process in batches
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]

            # Prepare parameters for embedding call
            params: dict[str, Any] = {
                "input": batch,
                "model": self._model,
            }

            # Only add dimensions parameter for models that support it
            if self._model.startswith("text-embedding-3"):
                params["dimensions"] = self._dimension

            try:
                response = await self._client.embeddings.create(**params)
                batch_embeddings = [item.embedding for item in response.data]
                all_embeddings.extend(batch_embeddings)
            except Exception as e:
                logger.error(f"Error embedding batch: {e}")
                raise

        return all_embeddings

    async def health_check(self) -> bool:
        """
        Check health by attempting a simple embedding.

        Returns:
            True if healthy
        """
        try:
            await self.embed_text("test")
            return True
        except Exception as e:
            logger.warning(f"Health check failed: {e}")
            return False


class AzureOpenAIEmbedder:
    """
    Production Azure OpenAI embedder.

    Mirrors OpenAIEmbedder functionality but uses Azure endpoints.
    """

    # Default dimensions for each model
    MODEL_DIMENSIONS = {
        "text-embedding-3-small": 1536,
        "text-embedding-3-large": 3072,
        "text-embedding-ada-002": 1536,
    }

    def __init__(self, config: EmbedderConfig, *, dimension: int | None = None) -> None:
        """
        Initialize the Azure OpenAI embedder.

        Args:
            config: Embedder configuration with API key, base URL, and model
            dimension: Optional output dimension for Matryoshka truncation

        Raises:
            ImportError: If openai package is not installed
            ValueError: If API key or api_base is missing
        """
        try:
            from openai import AsyncAzureOpenAI
        except ImportError as exc:
            raise ImportError(
                "OpenAI package is not installed. Install it with:\n"
                "  pip install agent-gantry[openai]"
            ) from exc

        api_key = config.api_key or os.getenv("AZURE_OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "Azure OpenAI API key is required. Set it in config or AZURE_OPENAI_API_KEY environment variable."
            )

        api_base = config.api_base
        if not api_base:
            raise ValueError(
                "Azure OpenAI api_base (endpoint) is required in config."
            )

        self._config = config
        self._model = config.model or "text-embedding-3-small"
        self._batch_size = config.batch_size
        self._max_retries = config.max_retries

        # Determine dimension
        if dimension is not None:
            self._dimension = dimension
        elif config.dimension is not None:
            self._dimension = config.dimension
        else:
            self._dimension = self.MODEL_DIMENSIONS.get(self._model, 1536)

        # Azure API version - use config, env var, or latest stable default
        api_version = (
            config.api_version
            or os.getenv("AZURE_OPENAI_API_VERSION")
            or "2024-10-01-preview"  # Latest stable as of Dec 2025
        )

        # Initialize Azure client with retry logic
        self._client = AsyncAzureOpenAI(
            api_key=api_key,
            azure_endpoint=api_base,
            api_version=api_version,
            max_retries=self._max_retries,
        )

        logger.info(
            f"Initialized AzureOpenAIEmbedder with model={self._model}, "
            f"dimension={self._dimension}, endpoint={api_base}"
        )

    @property
    def dimension(self) -> int:
        """Return the embedding dimension."""
        return self._dimension

    @property
    def model_name(self) -> str:
        """Return the model name."""
        return self._model

    def get_embedder_id(self) -> str:
        """
        Return a unique identifier for this embedder configuration.

        Returns:
            Identifier combining model name and dimension
        """
        return f"azure:{self._model}:{self._dimension}"

    async def embed_text(self, text: str) -> list[float]:
        """
        Embed a single text.

        Args:
            text: Text to embed

        Returns:
            Embedding vector
        """
        result = await self.embed_batch([text])
        return result[0]

    async def embed_batch(
        self,
        texts: list[str],
        batch_size: int | None = None,
    ) -> list[list[float]]:
        """
        Embed multiple texts with batching.

        Args:
            texts: List of texts to embed
            batch_size: Optional batch size (default: use config)

        Returns:
            List of embedding vectors
        """
        if not texts:
            return []

        batch_size = batch_size or self._batch_size
        all_embeddings: list[list[float]] = []

        # Process in batches
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]

            # Prepare parameters for embedding call
            params: dict[str, Any] = {
                "input": batch,
                "model": self._model,
            }

            # Only add dimensions parameter for models that support it
            if self._model.startswith("text-embedding-3"):
                params["dimensions"] = self._dimension

            try:
                response = await self._client.embeddings.create(**params)
                batch_embeddings = [item.embedding for item in response.data]
                all_embeddings.extend(batch_embeddings)
            except Exception as e:
                logger.error(f"Error embedding batch: {e}")
                raise

        return all_embeddings

    async def health_check(self) -> bool:
        """
        Check health by attempting a simple embedding.

        Returns:
            True if healthy
        """
        try:
            await self.embed_text("test")
            return True
        except Exception as e:
            logger.warning(f"Health check failed: {e}")
            return False

