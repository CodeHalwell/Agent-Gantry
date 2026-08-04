"""
Sync manager for Agent-Gantry.

Handles tool sync with fingerprint-based change detection.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from agent_gantry.utils.fingerprint import compute_tool_fingerprint

if TYPE_CHECKING:
    from agent_gantry.adapters.embedders.base import EmbeddingAdapter
    from agent_gantry.adapters.vector_stores.base import VectorStoreAdapter
    from agent_gantry.core.registry import ToolRegistry
    from agent_gantry.schema.tool import ToolDefinition

logger = logging.getLogger(__name__)


class SyncManager:
    """
    Manages tool sync with smart fingerprint-based change detection.

    Extracted from AgentGantry to keep the facade thin.
    """

    def __init__(
        self,
        vector_store: VectorStoreAdapter,
        embedder: EmbeddingAdapter,
        registry: ToolRegistry,
    ) -> None:
        self._vector_store = vector_store
        self._embedder = embedder
        self._registry = registry

    def get_embedder_id(self) -> str:
        """
        Get a unique identifier for the current embedder configuration.

        Returns:
            String identifier combining embedder class and key params
        """
        embedder_class = self._embedder.__class__.__name__

        # Prefer the embedder protocol's own identity: it includes the model
        # name plus any extra params the implementation adds (task type,
        # quantization, ...). The attribute fallbacks below exist only for
        # duck-typed embedders outside the base protocol — probing just
        # `model`/`_model_name` used to miss the OpenAI/Azure embedders
        # (which expose `model_name`/`_model`), so two same-dimension models
        # shared one identity and a model switch skipped re-embedding,
        # leaving queries from the new model searching the old model's
        # vectors.
        get_id = getattr(self._embedder, "get_embedder_id", None)
        if callable(get_id):
            try:
                return f"{embedder_class}-{get_id()}"
            except Exception:
                pass

        dimension = getattr(self._embedder, "dimension", None)
        if dimension is None:
            dimension = getattr(self._embedder, "_dimension", None)

        model = getattr(self._embedder, "model_name", None)
        if model is None:
            model = getattr(self._embedder, "model", None)
        if model is None:
            model = getattr(self._embedder, "_model", None)
        if model is None:
            model = getattr(self._embedder, "_model_name", None)

        parts = [embedder_class]
        if model:
            parts.append(str(model))
        if dimension:
            parts.append(f"dim{dimension}")

        return "-".join(parts)

    async def detect_changes(
        self,
        all_tools: list[ToolDefinition],
        force: bool,
    ) -> list[ToolDefinition]:
        """
        Detect which tools need to be synced based on fingerprints.

        Args:
            all_tools: List of all current tools
            force: If True, force resync of all tools

        Returns:
            List of tools that need to be synced
        """
        current_fingerprints = {
            f"{t.namespace}.{t.name}": compute_tool_fingerprint(t)
            for t in all_tools
        }

        stored_fingerprints = await self._vector_store.get_stored_fingerprints()
        embedder_id = self.get_embedder_id()
        needs_full_resync = force

        stored_embedder = await self._vector_store.get_metadata("embedder_id")
        stored_dim = await self._vector_store.get_metadata("dimension")

        if stored_embedder and stored_embedder != embedder_id:
            logger.info(
                f"Embedder changed from '{stored_embedder}' to '{embedder_id}'. "
                "Full re-sync required."
            )
            needs_full_resync = True
        elif stored_dim and int(stored_dim) != self._vector_store.dimension:
            logger.info(
                f"Dimension changed from {stored_dim} to {self._vector_store.dimension}. "
                "Full re-sync required."
            )
            needs_full_resync = True

        if needs_full_resync:
            return all_tools

        tools_to_sync = []
        for tool in all_tools:
            tool_id = f"{tool.namespace}.{tool.name}"
            current_fp = current_fingerprints[tool_id]
            stored_fp = stored_fingerprints.get(tool_id, "")

            if current_fp != stored_fp:
                tools_to_sync.append(tool)
                if stored_fp:
                    logger.debug(f"Tool '{tool_id}' changed, will re-embed")
                else:
                    logger.debug(f"Tool '{tool_id}' is new, will embed")

        return tools_to_sync

    async def sync_batches(
        self,
        tools_to_sync: list[ToolDefinition],
        batch_size: int,
    ) -> int:
        """
        Embed and save tools in batches.

        Args:
            tools_to_sync: Tools to embed and save
            batch_size: Number of tools per batch

        Returns:
            Number of tools synced
        """
        total_synced = 0
        for i in range(0, len(tools_to_sync), batch_size):
            batch = tools_to_sync[i : i + batch_size]
            texts = [t.to_searchable_text() for t in batch]
            embeddings = await self._embedder.embed_batch(texts)
            count = await self._vector_store.add_tools(batch, embeddings, upsert=True)

            for tool in batch:
                self._registry.register_tool(tool)

            total_synced += count
        return total_synced

    async def update_metadata(self) -> None:
        """Update sync metadata in the vector store."""
        await self._vector_store.update_sync_metadata(
            embedder_id=self.get_embedder_id(),
            dimension=self._vector_store.dimension,
        )
