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

        Raises:
            ValueError: If dimension is invalid or task_type is unsupported.
        """
        self._model_name = model
        dim = self.FULL_DIMENSION if dimension is None else dimension

        # Validate dimension
        if dim < 1 or dim > self.FULL_DIMENSION:
            raise ValueError(
                f"dimension must be between 1 and {self.FULL_DIMENSION}, got {dim}"
            )
        if dim not in self.MATRYOSHKA_DIMS:
            import warnings

            warnings.warn(
                f"dimension {dim} is not a recommended Matryoshka dimension. "
                f"Recommended values: {self.MATRYOSHKA_DIMS}",
                UserWarning,
                stacklevel=2,
            )

        # Validate task_type
        if task_type not in self.TASK_PREFIXES:
            raise ValueError(
                f"Unsupported task_type '{task_type}'. "
                f"Supported types: {', '.join(self.TASK_PREFIXES.keys())}"
            )

        self._dimension = dim
        self._task_type = task_type
        self._task_prefix = self.TASK_PREFIXES[task_type]
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

        The underlying sentence-transformers model is called with
        ``normalize_embeddings=True``, so the embeddings are already
        L2-normalized. Following Nomic's Matryoshka recommendation, we
        simply truncate these normalized embeddings to the desired dimension
        without any additional normalization steps.
        """
        if self._dimension >= self.FULL_DIMENSION:
            return embeddings

        # Simple truncation of already-normalized embeddings
        return [emb[: self._dimension] for emb in embeddings]

    async def embed_text(self, text: str) -> list[float]:
        """
        Embed a single text.

        Args:
            text: Text to embed (prefix will be added automatically)

        Returns:
            Embedding vector of configured dimension
        """
        import asyncio

        self._ensure_initialized()

        # Add task prefix
        prefixed_text = f"{self._task_prefix}{text}"

        # Generate embedding (run in thread pool to avoid blocking event loop)
        loop = asyncio.get_event_loop()
        embedding = await loop.run_in_executor(
            None,
            lambda: self._model.encode([prefixed_text], normalize_embeddings=True)
        )
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
        import asyncio

        if not texts:
            return []

        self._ensure_initialized()

        # Add task prefix to all texts
        prefixed_texts = [f"{self._task_prefix}{text}" for text in texts]

        # Generate embeddings (run in thread pool to avoid blocking event loop)
        kwargs: dict[str, Any] = {"normalize_embeddings": True}
        if batch_size is not None:
            kwargs["batch_size"] = batch_size

        loop = asyncio.get_event_loop()
        embeddings = await loop.run_in_executor(
            None,
            lambda: self._model.encode(prefixed_texts, **kwargs)
        )
        result = embeddings.tolist()

        # Apply Matryoshka truncation if needed
        return self._apply_matryoshka_truncation(result)

    async def embed_query(self, query: str) -> list[float]:
        """
        Embed a search query with the appropriate prefix.

        This method always uses 'search_query' prefix regardless of the
        instance's task_type setting, as this is optimal for retrieval.
        Use embed_text() if you want to use the configured task_type prefix.

        Args:
            query: Search query text

        Returns:
            Query embedding vector
        """
        import asyncio

        self._ensure_initialized()

        # Always use search_query prefix for queries (optimal for retrieval)
        prefixed_query = f"search_query: {query}"

        # Generate embedding (run in thread pool to avoid blocking event loop)
        loop = asyncio.get_event_loop()
        embedding = await loop.run_in_executor(
            None,
            lambda: self._model.encode([prefixed_query], normalize_embeddings=True)
        )
        result = embedding.tolist()

        truncated = self._apply_matryoshka_truncation(result)
        return truncated[0]

    async def health_check(self) -> bool:
        """
        Check health of the embedder.

        Returns:
            True if the model can be loaded and used
        """
        import asyncio

        try:
            self._ensure_initialized()
            # Quick sanity check (run in thread pool to avoid blocking event loop)
            loop = asyncio.get_event_loop()
            test_embedding = await loop.run_in_executor(
                None,
                lambda: self._model.encode(["test"], normalize_embeddings=True)
            )
            return len(test_embedding[0]) > 0
        except Exception:
            return False

    def set_task_type(self, task_type: str) -> None:
        """
        Set the task type for embeddings.

        This can be used to configure the embedder before generating any embeddings.
        Changing the task type after embeddings have been created can lead to
        inconsistent prefixes between stored embeddings and new queries, which
        degrades retrieval quality. For that reason, changing the task type to a
        different value after it has been set is not allowed.

        Args:
            task_type: New task type ('search_document', 'search_query',
                'clustering', 'classification')

        Raises:
            ValueError: If an unknown task type is provided.
            RuntimeError: If attempting to change the task type after it was
                already set to a different value.
        """
        if task_type not in self.TASK_PREFIXES:
            raise ValueError(
                f"Unsupported task_type '{task_type}'. "
                f"Supported types: {', '.join(self.TASK_PREFIXES.keys())}"
            )

        # Disallow changing task type once it has been set to a different value.
        # This avoids mixing embeddings generated with different task prefixes.
        if getattr(self, "_task_type", None) is not None and self._task_type != task_type:
            raise RuntimeError(
                "Changing task_type after initialization is not supported, as it can "
                "lead to inconsistent embeddings. Create a new NomicEmbedder "
                "instance with the desired task_type instead."
            )

        self._task_type = task_type
        self._task_prefix = self.TASK_PREFIXES[task_type]
