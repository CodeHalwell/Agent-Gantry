"""
Base embedding adapter protocol.
"""

from __future__ import annotations

from abc import abstractmethod
from typing import Protocol


class EmbeddingAdapter(Protocol):
    """
    Text embedding abstraction.

    Implementations: OpenAIEmbedder, AzureOpenAIEmbedder, CohereEmbedder,
                     HuggingFaceEmbedder, SentenceTransformerEmbedder, OllamaEmbedder.
    """

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Return the embedding dimension."""
        ...

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return the model name."""
        ...

    def get_embedder_id(self) -> str:
        """
        Return a unique identifier for this embedder configuration.

        This ID is used to track which embedder was used to create embeddings,
        enabling proper invalidation when the embedder changes.

        The default implementation combines model name and dimension.
        Implementations can override this to include additional parameters
        (e.g., task type, quantization, fine-tuning).

        Returns:
            Unique identifier string (e.g., "nomic-ai/nomic-embed-text-v1.5:768")
        """
        return f"{self.model_name}:{self.dimension}"

    @abstractmethod
    async def embed_text(self, text: str) -> list[float]:
        """
        Embed a single text.

        Args:
            text: Text to embed

        Returns:
            Embedding vector
        """
        ...

    @abstractmethod
    async def embed_batch(
        self,
        texts: list[str],
        batch_size: int | None = None,
    ) -> list[list[float]]:
        """
        Embed multiple texts.

        Args:
            texts: List of texts to embed
            batch_size: Optional batch size for processing

        Returns:
            List of embedding vectors
        """
        ...

    @abstractmethod
    async def health_check(self) -> bool:
        """
        Check health of the embedding service.

        Returns:
            True if healthy
        """
        ...
