"""
Nomic Embed Text embedder with Matryoshka support.

Uses Nomic's nomic-embed-text-v1.5 model via sentence-transformers for
local, on-device embedding generation. Supports Matryoshka truncation
for efficient retrieval at various embedding dimensions.
"""

from __future__ import annotations

from typing import Any

from agent_gantry.adapters.embedders.base import EmbeddingAdapter


class NomicEmbedder(EmbeddingAdapter):
    """
    Nomic Embed Text embedder with Matryoshka truncation support.

    Uses sentence-transformers to load and run the nomic-embed-text-v1.5 model
    locally. Supports Matryoshka embedding truncation for efficient retrieval.

    Attributes:
        model_name: The Hugging Face model identifier
        dimension: Output embedding dimension (supports Matryoshka truncation)
        task_prefix: Prefix to add to texts for task-specific embeddings

    Example:
        >>> embedder = NomicEmbedder(dimension=256)
        >>> vector = await embedder.embed_text("Hello world")
        >>> assert len(vector) == 256
    """

    # Nomic's recommended task prefixes
    TASK_PREFIXES = {
        "search_document": "search_document: ",
        "search_query": "search_query: ",
        "clustering": "clustering: ",
        "classification": "classification: ",
    }

    # Default full dimension for nomic-embed-text-v1.5
    FULL_DIMENSION = 768

    # Recommended Matryoshka dimensions for efficient truncation
    MATRYOSHKA_DIMS = [768, 512, 256, 128, 64]

    def __init__(
        self,
        model: str = "nomic-ai/nomic-embed-text-v1.5",
        dimension: int | None = None,
        task_type: str = "search_document",
        device: str | None = None,
    ) -> None:
        """
        Initialize the Nomic embedder.

        Args:
            model: Hugging Face model identifier
            dimension: Output dimension (default is full 768, can truncate to 64-768)
            task_type: Task type for prefix ('search_document', 'search_query',
                      'clustering', 'classification')
            device: Device to run model on ('cpu', 'cuda', etc). Auto-detected if None.
        """
        self._model_name = model
        self._dimension = dimension or self.FULL_DIMENSION
        self._task_type = task_type
        self._task_prefix = self.TASK_PREFIXES.get(task_type, "")
        self._device = device
        self._model: Any = None
        self._initialized = False

    @property
    def dimension(self) -> int:
        """Return the embedding dimension."""
        return self._dimension

    @property
    def model_name(self) -> str:
        """Return the model name."""
        return self._model_name

    def _ensure_initialized(self) -> None:
        """Lazy-load the model on first use."""
        if self._initialized:
            return

        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            raise ImportError(
                "sentence-transformers is required for Nomic embeddings. "
                "Install with: pip install sentence-transformers"
            ) from e

        self._model = SentenceTransformer(
            self._model_name,
            trust_remote_code=True,
            device=self._device,
        )
        self._initialized = True

    def _apply_matryoshka_truncation(self, embeddings: list[list[float]]) -> list[list[float]]:
        """
        Apply Matryoshka truncation to embeddings.

        For nomic-embed-text-v1.5, we apply layer normalization before truncation
        as recommended by Nomic for optimal performance.
        """
        if self._dimension >= self.FULL_DIMENSION:
            return embeddings

        try:
            import numpy as np
        except ImportError:
            # Fallback: simple truncation without normalization
            return [emb[: self._dimension] for emb in embeddings]

        # Convert to numpy for efficient processing
        arr = np.array(embeddings, dtype=np.float32)

        # Apply layer normalization before truncation (recommended for v1.5)
        mean = np.mean(arr, axis=-1, keepdims=True)
        var = np.var(arr, axis=-1, keepdims=True)
        arr_normalized = (arr - mean) / np.sqrt(var + 1e-5)

        # Truncate to desired dimension
        arr_truncated = arr_normalized[:, : self._dimension]

        # Re-normalize to unit length for cosine similarity
        norms = np.linalg.norm(arr_truncated, axis=-1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        arr_final = arr_truncated / norms

        return arr_final.tolist()

    async def embed_text(self, text: str) -> list[float]:
        """
        Embed a single text.

        Args:
            text: Text to embed (prefix will be added automatically)

        Returns:
            Embedding vector of configured dimension
        """
        self._ensure_initialized()

        # Add task prefix
        prefixed_text = f"{self._task_prefix}{text}"

        # Generate embedding
        embedding = self._model.encode([prefixed_text], normalize_embeddings=True)
        result = embedding.tolist()

        # Apply Matryoshka truncation if needed
        truncated = self._apply_matryoshka_truncation(result)
        return truncated[0]

    async def embed_batch(
        self,
        texts: list[str],
        batch_size: int | None = None,
    ) -> list[list[float]]:
        """
        Embed multiple texts efficiently.

        Args:
            texts: List of texts to embed
            batch_size: Batch size for processing (default uses model default)

        Returns:
            List of embedding vectors
        """
        if not texts:
            return []

        self._ensure_initialized()

        # Add task prefix to all texts
        prefixed_texts = [f"{self._task_prefix}{text}" for text in texts]

        # Generate embeddings
        kwargs: dict[str, Any] = {"normalize_embeddings": True}
        if batch_size is not None:
            kwargs["batch_size"] = batch_size

        embeddings = self._model.encode(prefixed_texts, **kwargs)
        result = embeddings.tolist()

        # Apply Matryoshka truncation if needed
        return self._apply_matryoshka_truncation(result)

    async def embed_query(self, query: str) -> list[float]:
        """
        Embed a search query with the appropriate prefix.

        Uses 'search_query' prefix for optimal retrieval performance.

        Args:
            query: Search query text

        Returns:
            Query embedding vector
        """
        self._ensure_initialized()

        # Use search_query prefix for queries
        prefixed_query = f"search_query: {query}"

        embedding = self._model.encode([prefixed_query], normalize_embeddings=True)
        result = embedding.tolist()

        truncated = self._apply_matryoshka_truncation(result)
        return truncated[0]

    async def health_check(self) -> bool:
        """
        Check health of the embedder.

        Returns:
            True if the model can be loaded and used
        """
        try:
            self._ensure_initialized()
            # Quick sanity check
            test_embedding = self._model.encode(["test"], normalize_embeddings=True)
            return len(test_embedding[0]) > 0
        except Exception:
            return False

    def set_task_type(self, task_type: str) -> None:
        """
        Change the task type for embeddings.

        Args:
            task_type: New task type ('search_document', 'search_query',
                      'clustering', 'classification')
        """
        self._task_type = task_type
        self._task_prefix = self.TASK_PREFIXES.get(task_type, "")
