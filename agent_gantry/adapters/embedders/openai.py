"""
OpenAI and Azure OpenAI embedders with production-ready implementations.

Supports multiple embedding models with configurable dimensions and
batch processing with built-in retry logic.
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Any

from agent_gantry.schema.config import EmbedderConfig

logger = logging.getLogger(__name__)


class BaseOpenAIEmbedder:
    """
    Base class for OpenAI and Azure OpenAI embedders.
    Contains shared functionality for embedding text and batches.
    """

    # Default dimensions for each model
    MODEL_DIMENSIONS = {
        "text-embedding-3-small": 1536,
        "text-embedding-3-large": 3072,
        "text-embedding-ada-002": 1536,
    }

    def __init__(self, config: EmbedderConfig, *, dimension: int | None = None) -> None:
        """
        Initialize the base OpenAI embedder.

        Args:
            config: Embedder configuration
            dimension: Optional output dimension for Matryoshka truncation
        """
        try:
            from openai import AsyncOpenAI  # noqa: F401 - availability probe for optional extra
        except ImportError as exc:
            raise ImportError(
                "OpenAI package is not installed. Install it with:\n"
                "  pip install agent-gantry[openai]"
            ) from exc

        api_key = config.api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(

                    "OpenAI API key is required. Set it in config or "
                    "OPENAI_API_KEY environment variable."

            )

        self._config = config
        self._model = config.model or "text-embedding-3-small"
        self._batch_size = config.batch_size
        self._max_retries = config.max_retries
        # Concurrent in-flight batch requests for embed_batch. Small enough to
        # stay well under provider rate limits while removing the serial
        # round-trip latency of large multi-batch syncs.
        self._max_concurrent_batches = 4

        # Determine dimension - use specified, or config, or model default
        if dimension is not None:
            self._dimension = dimension
        elif config.dimension is not None:
            self._dimension = config.dimension
        else:
            self._dimension = self.MODEL_DIMENSIONS.get(self._model, 1536)

        # Self._client needs to be set by subclasses
        self._client: Any = None

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
        raise NotImplementedError("Subclasses must implement get_embedder_id")

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

        if not self._client:
            raise RuntimeError("Client is not initialized")

        batch_size = batch_size or self._batch_size
        batches = [texts[i : i + batch_size] for i in range(0, len(texts), batch_size)]

        async def embed_one(batch: list[str], semaphore: asyncio.Semaphore) -> list[list[float]]:
            params: dict[str, Any] = {
                "input": batch,
                "model": self._model,
            }
            # Only add dimensions parameter for models that support it
            if self._model.startswith("text-embedding-3"):
                params["dimensions"] = self._dimension

            async with semaphore:
                try:
                    response = await self._client.embeddings.create(**params)
                except Exception as e:
                    logger.error(f"Error embedding batch: {e}")
                    raise
            return [item.embedding for item in response.data]

        if len(batches) == 1:
            semaphore = asyncio.Semaphore(1)
            return await embed_one(batches[0], semaphore)

        # Issue batch requests concurrently (bounded to stay under provider
        # rate limits) instead of serial round-trips; gather preserves order.
        semaphore = asyncio.Semaphore(self._max_concurrent_batches)
        tasks = [asyncio.ensure_future(embed_one(batch, semaphore)) for batch in batches]
        try:
            results = await asyncio.gather(*tasks)
        except BaseException:
            # gather propagates the first failure while sibling requests keep
            # running. Cancel and drain them so they stop consuming provider
            # quota before the error surfaces — a retrying caller would
            # otherwise overlap with the still-running originals.
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            raise

        all_embeddings: list[list[float]] = []
        for batch_embeddings in results:
            all_embeddings.extend(batch_embeddings)
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


class OpenAIEmbedder(BaseOpenAIEmbedder):
    """
    Production OpenAI embedder using the official OpenAI Python client.

    Supports models: text-embedding-3-small, text-embedding-3-large, text-embedding-ada-002
    with configurable dimensions for Matryoshka truncation.
    """

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
        super().__init__(config, dimension=dimension)
        try:
            from openai import AsyncOpenAI  # noqa: F401 - availability probe for optional extra
        except ImportError as exc:
            raise ImportError(
                "OpenAI package is not installed. Install it with:\n"
                "  pip install agent-gantry[openai]"
            ) from exc

        api_key = config.api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OpenAI API key is required. Set it in config or "
                "OPENAI_API_KEY environment variable."
            )

        # Honor an OpenAI-compatible custom endpoint (Requesty, OpenRouter,
        # Together, vLLM, …) when supplied via config.api_base or the
        # OPENAI_BASE_URL env var. Without this users have to monkey-patch
        # or globally set OPENAI_BASE_URL — both brittle.
        base_url = config.api_base or os.getenv("OPENAI_BASE_URL")
        self._api_base = base_url

        client_kwargs: dict[str, Any] = {
            "api_key": api_key,
            "max_retries": self._max_retries,
        }
        if base_url:
            client_kwargs["base_url"] = base_url

        self._client = AsyncOpenAI(**client_kwargs)

        if base_url:
            logger.info(
                f"Initialized OpenAIEmbedder with model={self._model}, "
                f"dimension={self._dimension}, base_url={base_url}"
            )
        else:
            logger.info(
                f"Initialized OpenAIEmbedder with model={self._model}, "
                f"dimension={self._dimension}"
            )

    def get_embedder_id(self) -> str:
        """
        Return a unique identifier for this embedder configuration.

        Returns:
            Identifier combining model name and dimension
        """
        base = f"{self._model}:{self._dimension}"
        # A custom OpenAI-compatible endpoint may serve a different model
        # under the same label, producing an incompatible vector space —
        # the endpoint is part of the identity, or switching endpoints
        # would skip re-embedding and corrupt retrieval.
        if self._api_base:
            return f"{base}@{self._api_base}"
        return base


class AzureOpenAIEmbedder(BaseOpenAIEmbedder):
    """
    Production Azure OpenAI embedder.

    Mirrors OpenAIEmbedder functionality but uses Azure endpoints.
    """

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
        super().__init__(config, dimension=dimension)
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
                "Azure OpenAI API key is required. Set it in config or "
                "AZURE_OPENAI_API_KEY environment variable."
            )

        api_base = config.api_base
        if not api_base:
            raise ValueError("Azure OpenAI api_base (endpoint) is required in config.")
        self._api_base = api_base

        # Azure API version - use config, env var, or latest preview default
        api_version = (
            config.api_version
            or os.getenv("AZURE_OPENAI_API_VERSION")
            or "2025-01-01-preview"  # Latest preview version as of Apr 2026
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

    def get_embedder_id(self) -> str:
        """
        Return a unique identifier for this embedder configuration.

        Returns:
            Identifier combining model name and dimension
        """
        # Endpoint included for the same reason as OpenAIEmbedder: the same
        # deployment name on different Azure resources is a different model.
        return f"azure:{self._model}:{self._dimension}@{self._api_base}"
