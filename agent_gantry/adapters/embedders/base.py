"""
Base embedding adapter protocol.
"""

from __future__ import annotations

import inspect
from abc import abstractmethod
from typing import Any, Protocol


class EmbeddingAdapter(Protocol):
    """
    Text embedding abstraction.

    Implementations: SimpleEmbedder, NomicEmbedder, SentenceTransformersEmbedder,
                     OpenAIEmbedder, AzureOpenAIEmbedder.
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

    async def embed_query(self, query: str) -> list[float]:
        """
        Embed a search query, as opposed to a stored document.

        Several modern embedders are *asymmetric*: they are trained with one
        instruction for the thing being stored and another for the thing being
        searched for, and applying the wrong one measurably degrades retrieval
        on their own benchmarks. Nomic's v1.5 is the example in this tree — it
        wants ``search_document:`` on a tool and ``search_query:`` on a prompt,
        and a single embedder instance serves both here.

        The default is ``embed_text``, so a symmetric embedder needs no change
        and this stays backwards compatible for third-party adapters.

        Args:
            query: The query to embed

        Returns:
            Embedding vector
        """
        return await self.embed_text(query)

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


async def embed_query(embedder: Any, query: str) -> list[float]:
    """Embed *query* through an embedder's query side, whatever it provides.

    :class:`EmbeddingAdapter` is a ``Protocol``, so its default
    :meth:`~EmbeddingAdapter.embed_query` reaches only adapters that actually
    subclass it. An adapter that satisfies the protocol structurally — which is
    the whole point of a protocol, and what a third-party integration is most
    likely to do — would hit ``AttributeError`` on the first retrieval if the
    call site assumed the method were there.

    So the method is used when the adapter really has an awaitable one, and
    ``embed_text`` is used otherwise. That also keeps stand-ins honest without
    forcing every test double in the wild to grow a method the day this landed.

    Args:
        embedder: Any object satisfying :class:`EmbeddingAdapter`.
        query: The search query.

    Returns:
        Embedding vector.
    """
    method = getattr(embedder, "embed_query", None)
    if method is None or not inspect.iscoroutinefunction(method):
        return await embedder.embed_text(query)
    return await method(query)
