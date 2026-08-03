"""
In-memory vector store implementation.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from agent_gantry.schema.tool import ToolDefinition
from agent_gantry.utils.fingerprint import compute_tool_fingerprint


class InMemoryVectorStore:
    """
    Simple in-memory vector store for development and testing.

    Uses cosine similarity for search.
    """

    def __init__(self, dimension: int = 0) -> None:
        """
        Initialize the in-memory store.

        Args:
            dimension: Vector dimension (optional, auto-detected from first embedding)
        """
        self._tools: dict[str, ToolDefinition] = {}
        self._embeddings: dict[str, list[float]] = {}
        self._fingerprints: dict[str, str] = {}
        self._metadata: dict[str, str] = {}
        self._dimension = dimension
        # Vectorized search cache: a (n, d) L2-normalized matrix plus the row
        # ordering. Rebuilt lazily on the next search after any mutation, so a
        # batch of add/delete calls only pays for one rebuild. ``None`` means
        # "stale — rebuild before use".
        self._matrix: np.ndarray | None = None
        self._matrix_keys: list[str] = []

    @property
    def dimension(self) -> int:
        """
        Return the vector dimension.

        Returns the configured dimension, or auto-detects from first stored embedding.
        """
        if self._dimension > 0:
            return self._dimension
        # Auto-detect from first embedding
        if self._embeddings:
            first_embedding = next(iter(self._embeddings.values()))
            return len(first_embedding)
        return 0

    async def initialize(self) -> None:
        """Initialize the store (no-op for in-memory)."""
        pass

    async def add_tools(
        self,
        tools: list[ToolDefinition],
        embeddings: list[list[float]],
        upsert: bool = True,
    ) -> int:
        """Add tools with their embeddings."""
        count = 0
        for tool, embedding in zip(tools, embeddings):
            key = f"{tool.namespace}.{tool.name}"
            if key not in self._tools or upsert:
                self._tools[key] = tool
                self._embeddings[key] = embedding
                # Store fingerprint for change detection. Must be the same
                # format SyncManager.detect_changes compares against
                # (compute_tool_fingerprint, "v1.0:<hash>") — storing the
                # unrelated tool.content_hash here made every sync() re-embed
                # the full registry because the comparison never matched.
                self._fingerprints[key] = compute_tool_fingerprint(tool)
                count += 1
        if count:
            self._matrix = None  # invalidate vectorized cache
        return count

    async def search(
        self,
        query_vector: list[float],
        limit: int,
        filters: dict[str, Any] | None = None,
        score_threshold: float | None = None,
        include_embeddings: bool = False,
    ) -> list[tuple[ToolDefinition, float]] | list[tuple[ToolDefinition, float, list[float]]]:
        """Search for similar tools using cosine similarity.

        Cosine scores are computed in a single vectorized matmul against a
        cached, L2-normalized embedding matrix (``query · M.T``) rather than a
        per-tool Python loop. For a registry of *n* tools this turns the hot
        path from ``n`` pure-Python dot products into one NumPy BLAS call,
        which dominates retrieval latency once *n* grows past a few dozen.
        """
        # Extract tag filters for faster set operations
        required_tags: set[str] = set()
        if filters and "tags" in filters:
            required_tags = set(filters["tags"])

        self._ensure_matrix()
        if self._matrix is None or self._matrix.shape[0] == 0:
            return []

        # Normalize the query once; stored rows are already normalized.
        q = np.asarray(query_vector, dtype=np.float32)
        # Guard against a query/stored dimension mismatch: the old per-vector
        # loop returned 0.0 similarity for mismatched lengths, so preserve that
        # graceful behaviour here instead of letting the matmul raise.
        if q.ndim != 1 or q.shape[0] != self._matrix.shape[1]:
            return []
        q_norm = float(np.linalg.norm(q))
        if q_norm == 0.0:
            # A zero query vector carries no signal. The original per-vector
            # cosine returned 0.0 for every tool (kept by a 0.0 threshold), so
            # an "empty"/zero query surfaces all tools rather than none. Preserve
            # that instead of dividing by zero.
            scores = np.zeros(self._matrix.shape[0], dtype=np.float32)
        else:
            scores = self._matrix @ (q / q_norm)  # (n,) cosine similarities

        results: list[tuple[ToolDefinition, float, str]] = []
        for idx, key in enumerate(self._matrix_keys):
            tool = self._tools.get(key)
            if tool is None:
                continue

            # Apply filters
            if filters:
                if "namespace" in filters:
                    ns_filter = filters["namespace"]
                    if isinstance(ns_filter, (list, tuple, set)):
                        if tool.namespace not in ns_filter:
                            continue
                    elif tool.namespace != ns_filter:
                        continue
                if required_tags and required_tags.isdisjoint(tool.tags):
                    continue

            score = float(scores[idx])
            if score_threshold is None or score >= score_threshold:
                results.append((tool, score, key))

        # Sort by score descending
        results.sort(key=lambda x: x[1], reverse=True)

        limited = results[:limit]
        if include_embeddings:
            return [(tool, score, self._embeddings[key]) for tool, score, key in limited]
        return [(tool, score) for tool, score, _ in limited]

    def _ensure_matrix(self) -> None:
        """(Re)build the cached normalized embedding matrix if stale."""
        if self._matrix is not None:
            return
        keys = [k for k in self._tools if self._embeddings.get(k) is not None]
        if not keys:
            self._matrix = np.zeros((0, 0), dtype=np.float32)
            self._matrix_keys = []
            return
        mat = np.asarray([self._embeddings[k] for k in keys], dtype=np.float32)
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        norms[norms == 0.0] = 1.0  # avoid divide-by-zero for zero vectors
        self._matrix = mat / norms
        self._matrix_keys = keys

    async def get_by_name(self, name: str, namespace: str = "default") -> ToolDefinition | None:
        """Get a tool by name."""
        key = f"{namespace}.{name}"
        return self._tools.get(key)

    async def delete(self, name: str, namespace: str = "default") -> bool:
        """Delete a tool."""
        key = f"{namespace}.{name}"
        if key in self._tools:
            del self._tools[key]
            self._embeddings.pop(key, None)
            self._fingerprints.pop(key, None)
            self._matrix = None  # invalidate vectorized cache
            return True
        return False

    async def list_all(
        self,
        namespace: str | None = None,
        limit: int = 1000,
        offset: int = 0,
    ) -> list[ToolDefinition]:
        """List all tools."""
        tools = list(self._tools.values())
        if namespace:
            tools = [t for t in tools if t.namespace == namespace]
        return tools[offset : offset + limit]

    async def count(self, namespace: str | None = None) -> int:
        """Count tools."""
        if namespace:
            count = 0
            for t in self._tools.values():
                if t.namespace == namespace:
                    count += 1
            return count
        return len(self._tools)

    async def health_check(self) -> bool:
        """Check health (always healthy for in-memory)."""
        return True

    @property
    def supports_metadata(self) -> bool:
        """Return True as in-memory store supports metadata storage."""
        return True

    def _cosine_similarity(self, a: list[float], b: list[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        if len(a) != len(b):
            return 0.0

        # ⚡ Bolt: Calculate dot product and norms in a single pass to reduce overhead
        dot_product = 0.0
        norm_a_sq = 0.0
        norm_b_sq = 0.0
        for x, y in zip(a, b):
            dot_product += x * y
            norm_a_sq += x * x
            norm_b_sq += y * y

        norm_a = math.sqrt(norm_a_sq)
        norm_b = math.sqrt(norm_b_sq)

        if norm_a == 0 or norm_b == 0:
            return 0.0

        return dot_product / (norm_a * norm_b)

    async def get_stored_fingerprints(self) -> dict[str, str]:
        """
        Get all stored tool fingerprints for change detection.

        Returns:
            Dictionary mapping tool_id (namespace.name) to fingerprint hash.
        """
        return dict(self._fingerprints)

    async def get_metadata(self, key: str) -> str | None:
        """
        Get a metadata value by key.

        Args:
            key: The metadata key to retrieve

        Returns:
            The metadata value if found, None otherwise.
        """
        return self._metadata.get(key)

    async def set_metadata(self, key: str, value: str) -> None:
        """
        Set a metadata value.

        Args:
            key: The metadata key
            value: The value to store
        """
        self._metadata[key] = value

    async def update_sync_metadata(self, embedder_id: str, dimension: int) -> None:
        """
        Update sync metadata after a successful sync operation.

        Args:
            embedder_id: Unique identifier for the embedder configuration
            dimension: Vector dimension
        """
        self._metadata["embedder_id"] = embedder_id
        self._metadata["dimension"] = str(dimension)
