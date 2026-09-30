"""
Nomic Embed Text embedder with Matryoshka support.

Runs nomic-embed-text-v1.5 through sentence-transformers with Nomic's task
prefixes and Matryoshka truncation (64-768 dimensions).
"""

from __future__ import annotations

import warnings

from agent_gantry.adapters.embedders.sentence_transformers import SentenceTransformersEmbedder


class NomicEmbedder(SentenceTransformersEmbedder):
    """
    Nomic Embed Text embedder with Matryoshka truncation support.

    Documents are embedded with the configured ``task_type`` prefix;
    :meth:`embed_query` always uses the ``search_query`` prefix.

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

    _model_kwargs = {"trust_remote_code": True}

    def __init__(
        self,
        model: str = "nomic-ai/nomic-embed-text-v1.5",
        dimension: int | None = None,
        task_type: str = "search_document",
        device: str | None = None,
    ) -> None:
        """
        Args:
            model: Hugging Face model identifier
            dimension: Output dimension (default is full 768, can truncate to 64-768)
            task_type: Task type for prefix ('search_document', 'search_query',
                'clustering', 'classification')
            device: Device to run model on ('cpu', 'cuda', etc). Auto-detected if None.

        Raises:
            ValueError: If dimension is invalid or task_type is unsupported.
        """
        dim = self.FULL_DIMENSION if dimension is None else dimension
        if dim < 1 or dim > self.FULL_DIMENSION:
            raise ValueError(f"dimension must be between 1 and {self.FULL_DIMENSION}, got {dim}")
        if dim not in self.MATRYOSHKA_DIMS:
            warnings.warn(
                f"dimension {dim} is not a recommended Matryoshka dimension. "
                f"Recommended values: {self.MATRYOSHKA_DIMS}",
                UserWarning,
                stacklevel=2,
            )
        if task_type not in self.TASK_PREFIXES:
            raise ValueError(
                f"Unsupported task_type '{task_type}'. "
                f"Supported types: {', '.join(self.TASK_PREFIXES.keys())}"
            )

        super().__init__(model=model, dimension=dim, device=device)
        self._task_type = task_type
        self._task_prefix = self.TASK_PREFIXES[task_type]

    def get_embedder_id(self) -> str:
        """Model, dimension and task type: a change to any of them invalidates stored vectors."""
        return f"{self._model_name}:{self._configured_dimension}:{self._task_type}"

    async def embed_batch(
        self,
        texts: list[str],
        batch_size: int | None = None,
    ) -> list[list[float]]:
        """Embed texts with the configured task prefix."""
        return await super().embed_batch([f"{self._task_prefix}{t}" for t in texts], batch_size)

    async def embed_query(self, query: str) -> list[float]:
        """Embed a search query with the ``search_query`` prefix, whatever the configured task_type."""
        batch = await SentenceTransformersEmbedder.embed_batch(self, [f"search_query: {query}"])
        return batch[0]
