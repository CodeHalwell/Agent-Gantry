"""
Production vector store adapters for Qdrant, Chroma, and PGVector.

Provides real implementations for remote vector databases with proper
collection management, filtering, and error handling.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import re
import uuid
from typing import TYPE_CHECKING, Any

from agent_gantry.schema.tool import ToolDefinition
from agent_gantry.utils.fingerprint import compute_tool_fingerprint

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

logger = logging.getLogger(__name__)


def _validate_sql_identifier(value: str, field_name: str) -> None:
    """
    Validate that a value is safe to use as a SQL identifier.

    Args:
        value: The identifier to validate
        field_name: Name of the field for error messages

    Raises:
        ValueError: If the identifier is invalid
    """
    if not value or len(value) > 63:  # PostgreSQL identifier length limit
        raise ValueError(f"{field_name} must be 1-63 characters")

    # Must start with letter or underscore, contain only alphanumeric and underscores
    if not re.match(r"^[a-zA-Z_][a-zA-Z0-9_]*\Z", value):
        raise ValueError(
            f"{field_name} must start with a letter or underscore and contain only "
            "alphanumeric characters and underscores"
        )


def _pg_namespace_clause(namespace: Any, param_index: int) -> tuple[str, Any]:
    """Build the pgvector namespace predicate and the value bound to it.

    The router sends namespaces as a list (``{"namespace": [...]}``); binding
    a list to ``namespace = $n`` makes asyncpg raise DataError, so lists go
    through ``= ANY($n::text[])``. Scalars keep the plain equality.
    """
    if isinstance(namespace, (list, tuple, set)):
        return f"WHERE namespace = ANY(${param_index}::text[])", list(namespace)
    return f"WHERE namespace = ${param_index}", namespace


def _selects_nothing(namespace: Any) -> bool:
    """Whether a namespace filter is an empty list, which selects nothing.

    ``if namespace:`` cannot tell ``None`` — no filter at all — from ``[]``, a
    filter that matches nothing, so ``list_all`` and ``count`` returned the
    whole collection for an empty list while ``search`` in the same adapters
    correctly returned nothing.
    """
    return isinstance(namespace, (list, tuple, set)) and not namespace


def _required_tags(filters: dict[str, Any] | None) -> set[str]:
    """Tags a result must share at least one of; empty when there is no tag filter."""
    return set(filters.get("tags") or ()) if filters else set()


def _result(
    tool_json: str, score: float, embedding: Any, include_embeddings: bool
) -> tuple[Any, ...]:
    """One search result in the shape the router expects."""
    tool = ToolDefinition.model_validate_json(tool_json)
    return (tool, score, embedding) if include_embeddings else (tool, score)


async def _search_with_tags(
    fetch: Callable[[int], Awaitable[list[Any]]], limit: int, required_tags: set[str]
) -> list[Any]:
    """Run ``fetch(size)`` and post-filter on tags, widening the window until
    ``limit`` results match or the backend runs out of rows.

    Tags live inside ``tool_json``, so none of the remote backends can apply
    the filter in the query itself.
    """
    size = limit * 4 if required_tags else limit
    while True:
        rows = await fetch(size)
        matches = [r for r in rows if not required_tags or not required_tags.isdisjoint(r[0].tags)]
        if not required_tags or len(matches) >= limit or len(rows) < size:
            return matches[:limit]
        size *= 2


class QdrantVectorStore:
    """
    Production Qdrant vector store adapter.

    Uses qdrant-client for high-performance vector search with remote or local Qdrant instances.
    """

    def __init__(
        self,
        url: str,
        api_key: str | None = None,
        collection_name: str = "agent_gantry",
        dimension: int = 1536,
        distance: str = "Cosine",
        prefer_grpc: bool = False,
        quantization: str | None = None,
    ) -> None:
        """
        Initialize the Qdrant vector store.

        Args:
            url: Qdrant server URL
            api_key: Optional API key for authentication
            collection_name: Name of the collection
            dimension: Vector dimension
            distance: Distance metric (Cosine, Euclid, Dot)
            prefer_grpc: Use gRPC if available
            quantization: Optional quantized-search mode for the collection:
                ``"scalar"`` (int8 scalar quantization — ~4x smaller vectors,
                kept in RAM, minimal recall loss) or ``"binary"`` (~32x
                smaller, fastest, best for high-dimensional embeddings such
                as OpenAI's). Applied at collection creation; searches
                rescore candidates against the original vectors so returned
                scores stay exact. Existing collections are not migrated —
                recreate the collection to change quantization.
        """
        try:
            from qdrant_client import AsyncQdrantClient
            from qdrant_client.models import Distance
        except ImportError as exc:
            raise ImportError(
                "qdrant-client is not installed. Install it with:\n"
                "  pip install agent-gantry[qdrant] (or uv add 'agent-gantry[qdrant]')"
            ) from exc

        self._collection_name = collection_name
        if quantization not in (None, "scalar", "binary"):
            raise ValueError(
                f"Unsupported quantization mode: {quantization!r} "
                f"(expected 'scalar', 'binary', or None)"
            )
        self._dimension = dimension
        self._quantization = quantization
        self._initialized = False

        # Map distance string to Qdrant Distance enum
        distance_map = {
            "Cosine": Distance.COSINE,
            "Euclid": Distance.EUCLID,
            "Dot": Distance.DOT,
        }
        self._distance = distance_map.get(distance, Distance.COSINE)

        self._client = AsyncQdrantClient(
            url=url,
            api_key=api_key,
            prefer_grpc=prefer_grpc,
        )

        logger.info(f"Initialized QdrantVectorStore with url={url}, collection={collection_name}")

    @property
    def dimension(self) -> int:
        """Return the vector dimension."""
        return self._dimension

    @staticmethod
    def _point_id(namespace: str, name: str) -> str:
        """Deterministic point id for ``namespace.name``."""
        return str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{namespace}.{name}"))

    @staticmethod
    def _build_namespace_filter(namespace: Any) -> Any:
        """Translate a namespace filter value into a Qdrant ``Filter``.

        The router sends namespaces as a list (``{"namespace": [...]}``),
        which needs ``MatchAny``; ``MatchValue`` only takes a scalar and
        rejects a list with a pydantic validation error.
        """
        from qdrant_client.models import FieldCondition, Filter, MatchAny, MatchValue

        if isinstance(namespace, (list, tuple, set)):
            match: Any = MatchAny(any=list(namespace))
        else:
            match = MatchValue(value=namespace)
        return Filter(must=[FieldCondition(key="namespace", match=match)])

    def _build_quantization_config(self) -> Any:
        """Translate the configured quantization mode to Qdrant config.

        Returns None (no quantization) when unset. int8 scalar quantization
        keeps the compressed vectors in RAM (that's where the speed win
        comes from); binary quantization compresses 32x and suits
        high-dimensional embeddings where sign bits retain enough signal.
        """
        if not self._quantization:
            return None
        # Import per mode: the binary-quantization models postdate older
        # qdrant-client releases the extras still permit, so importing them
        # unconditionally would break scalar mode (or plain construction) on
        # clients that support everything scalar needs.
        if self._quantization == "scalar":
            try:
                from qdrant_client.models import (
                    ScalarQuantization,
                    ScalarQuantizationConfig,
                    ScalarType,
                )
            except ImportError as exc:
                raise ImportError(
                    "quantization='scalar' requires a qdrant-client version with "
                    "scalar quantization support; upgrade qdrant-client."
                ) from exc
            return ScalarQuantization(
                scalar=ScalarQuantizationConfig(
                    type=ScalarType.INT8,
                    quantile=0.99,
                    always_ram=True,
                )
            )
        try:
            from qdrant_client.models import BinaryQuantization, BinaryQuantizationConfig
        except ImportError as exc:
            raise ImportError(
                "quantization='binary' requires a qdrant-client version with "
                "binary quantization support; upgrade qdrant-client."
            ) from exc
        return BinaryQuantization(binary=BinaryQuantizationConfig(always_ram=True))

    async def initialize(self) -> None:
        """Initialize the collection, creating it if needed."""
        if self._initialized:
            return

        from qdrant_client.models import VectorParams

        try:
            # Check if collection exists
            collections = await self._client.get_collections()
            exists = any(c.name == self._collection_name for c in collections.collections)

            if not exists:
                # Create collection
                await self._client.create_collection(
                    collection_name=self._collection_name,
                    vectors_config=VectorParams(
                        size=self._dimension,
                        distance=self._distance,
                    ),
                    quantization_config=self._build_quantization_config(),
                )
                logger.info(
                    f"Created Qdrant collection: {self._collection_name}"
                    + (f" (quantization={self._quantization})" if self._quantization else "")
                )

            self._initialized = True
        except Exception as e:
            logger.error(f"Failed to initialize Qdrant collection: {e}")
            raise

    async def add_tools(
        self,
        tools: list[ToolDefinition],
        embeddings: list[list[float]],
        upsert: bool = True,
    ) -> int:
        """Add tools to the vector store."""
        from qdrant_client.models import PointStruct

        await self.initialize()

        if not tools or not embeddings:
            return 0

        # Without upsert, ids already stored (or repeated within the batch)
        # are skipped and only the rows actually inserted are counted.
        skip: set[str] | None = None
        if not upsert:
            existing = await self._client.retrieve(
                collection_name=self._collection_name,
                ids=[self._point_id(t.namespace, t.name) for t in tools],
                with_payload=False,
                with_vectors=False,
            )
            skip = {str(record.id) for record in existing}

        points = []
        for tool, embedding in zip(tools, embeddings):
            point_id = self._point_id(tool.namespace, tool.name)
            if skip is not None:
                if point_id in skip:
                    continue
                skip.add(point_id)
            # The fingerprint enables incremental sync: SyncManager.detect_changes
            # compares it against compute_tool_fingerprint.
            payload = {
                "name": tool.name,
                "namespace": tool.namespace,
                "description": tool.description,
                "tool_json": tool.model_dump_json(),
                "fingerprint": compute_tool_fingerprint(tool),
            }
            points.append(PointStruct(id=point_id, vector=embedding, payload=payload))

        if points:
            await self._client.upsert(collection_name=self._collection_name, points=points)
        return len(points)

    async def search(
        self,
        query_vector: list[float],
        limit: int,
        filters: dict[str, Any] | None = None,
        score_threshold: float | None = None,
        include_embeddings: bool = False,
    ) -> list[tuple[ToolDefinition, float]] | list[tuple[ToolDefinition, float, list[float]]]:
        """Search for similar tools."""
        await self.initialize()

        # Build filter for namespace (scalar or list, see _build_namespace_filter)
        query_filter = None
        if filters and "namespace" in filters:
            query_filter = self._build_namespace_filter(filters["namespace"])

        # On a quantized collection, oversample from the compressed index and
        # rescore against the original vectors so returned scores stay exact.
        search_params = None
        if self._quantization:
            try:
                from qdrant_client.models import QuantizationSearchParams, SearchParams

                search_params = SearchParams(
                    quantization=QuantizationSearchParams(rescore=True, oversampling=2.0)
                )
            except ImportError:
                # A client this old could not have created the quantized
                # collection either; the server falls back to default rescoring.
                logger.debug("qdrant-client lacks QuantizationSearchParams; using defaults")

        # ``search`` was removed from AsyncQdrantClient in 1.19 (``query_points``
        # only); clients that predate ``query_points`` still have ``search``.
        query_points = getattr(self._client, "query_points", None)

        async def fetch(size: int) -> list[Any]:
            if query_points is not None:
                response = await query_points(
                    collection_name=self._collection_name,
                    query=query_vector,
                    limit=size,
                    query_filter=query_filter,
                    score_threshold=score_threshold,
                    with_vectors=include_embeddings,
                    search_params=search_params,
                )
                points = response.points
            else:
                points = await self._client.search(
                    collection_name=self._collection_name,
                    query_vector=query_vector,
                    limit=size,
                    query_filter=query_filter,
                    score_threshold=score_threshold,
                    with_vectors=include_embeddings,
                    search_params=search_params,
                )
            return [
                _result(
                    point.payload.get("tool_json", "{}"),
                    float(point.score),
                    list(point.vector or []),
                    include_embeddings,
                )
                for point in points
            ]

        return await _search_with_tags(fetch, limit, _required_tags(filters))

    async def get_by_name(self, name: str, namespace: str = "default") -> ToolDefinition | None:
        """Get a tool by name."""

        await self.initialize()

        point_id = self._point_id(namespace, name)

        try:
            result = await self._client.retrieve(
                collection_name=self._collection_name,
                ids=[point_id],
            )

            if result:
                tool_json = result[0].payload.get("tool_json", "{}")
                return ToolDefinition.model_validate_json(tool_json)
        except Exception as e:
            logger.debug(f"get_by_name failed for {namespace}.{name}: {e}")

        return None

    async def delete(self, name: str, namespace: str = "default") -> bool:
        """Delete a tool. Returns False if it was not stored."""
        await self.initialize()

        point_id = self._point_id(namespace, name)

        try:
            from qdrant_client.models import PointIdsList

            # Qdrant's delete is a silent no-op on a miss, so check first.
            existing = await self._client.retrieve(
                collection_name=self._collection_name,
                ids=[point_id],
                with_payload=False,
                with_vectors=False,
            )
            if not existing:
                return False
            await self._client.delete(
                collection_name=self._collection_name,
                points_selector=PointIdsList(points=[point_id]),
            )
            return True
        except Exception:
            return False

    async def list_all(
        self,
        namespace: str | None = None,
        limit: int = 1000,
        offset: int = 0,
    ) -> list[ToolDefinition]:
        """List all tools.

        ``offset`` is a row offset. Qdrant's scroll ``offset`` is a point-id
        cursor, so passing the row offset straight through made pagination a
        no-op; instead pages are walked with the cursor and the first
        ``offset`` rows are skipped.
        """
        await self.initialize()
        if _selects_nothing(namespace):
            # An empty list is a filter that matches nothing, not the
            # absence of one, which is how ``search`` reads it.
            return []

        # Build filter for namespace (scalar or list, see _build_namespace_filter)
        query_filter = self._build_namespace_filter(namespace) if namespace else None

        tools: list[ToolDefinition] = []
        to_skip = max(offset, 0)
        cursor: Any = None
        while len(tools) < limit:
            page_size = min(256, to_skip + limit - len(tools))
            records, cursor = await self._client.scroll(
                collection_name=self._collection_name,
                scroll_filter=query_filter,
                limit=page_size,
                offset=cursor,
                with_payload=True,
                with_vectors=False,
            )
            for record in records:
                if to_skip:
                    to_skip -= 1
                    continue
                tool_json = (record.payload or {}).get("tool_json", "{}")
                tools.append(ToolDefinition.model_validate_json(tool_json))
                if len(tools) >= limit:
                    break
            if cursor is None or not records:
                break

        return tools

    async def count(self, namespace: str | None = None) -> int:
        """Count tools."""
        await self.initialize()
        if _selects_nothing(namespace):
            # An empty list is a filter that matches nothing, not the
            # absence of one, which is how ``search`` reads it.
            return 0

        if namespace:
            # Count with filter (scalar or list, see _build_namespace_filter)
            query_filter = self._build_namespace_filter(namespace)
            result = await self._client.count(
                collection_name=self._collection_name,
                count_filter=query_filter,
            )
        else:
            # Count all
            result = await self._client.count(collection_name=self._collection_name)

        return result.count

    async def health_check(self) -> bool:
        """Check health of Qdrant connection."""
        try:
            await self._client.get_collections()
            return True
        except Exception:
            return False

    @property
    def supports_metadata(self) -> bool:
        """Qdrant supports metadata storage (via a side collection)."""
        return True

    async def get_stored_fingerprints(self) -> dict[str, str]:
        """
        Get all stored tool fingerprints for incremental sync.

        Scrolls the collection fetching only payload fields (no vectors), so
        unchanged tools can skip re-embedding on subsequent syncs.
        """
        await self.initialize()

        fingerprints: dict[str, str] = {}
        offset = None
        while True:
            records, offset = await self._client.scroll(
                collection_name=self._collection_name,
                limit=256,
                offset=offset,
                with_payload=["name", "namespace", "fingerprint"],
                with_vectors=False,
            )
            for record in records:
                payload = record.payload or {}
                fingerprint = payload.get("fingerprint")
                name = payload.get("name")
                namespace = payload.get("namespace", "default")
                if fingerprint and name:
                    fingerprints[f"{namespace}.{name}"] = fingerprint
            if offset is None:
                break
        return fingerprints

    async def _ensure_meta_collection(self) -> str:
        """Create the side collection holding sync metadata key/values."""
        from qdrant_client.models import VectorParams

        meta_name = f"{self._collection_name}__meta"
        collections = await self._client.get_collections()
        exists = any(c.name == meta_name for c in collections.collections)
        if not exists:
            await self._client.create_collection(
                collection_name=meta_name,
                vectors_config=VectorParams(size=1, distance=self._distance),
            )
        return meta_name

    async def get_metadata(self, key: str) -> str | None:
        """Get a sync-metadata value by key."""
        try:
            meta_name = await self._ensure_meta_collection()
            point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"__gantry_meta__.{key}"))
            result = await self._client.retrieve(
                collection_name=meta_name,
                ids=[point_id],
            )
            if result:
                payload = result[0].payload or {}
                value = payload.get("value")
                return str(value) if value is not None else None
        except Exception as e:
            logger.debug(f"get_metadata failed for {key}: {e}")
        return None

    async def set_metadata(self, key: str, value: str) -> None:
        """Set a sync-metadata value."""
        from qdrant_client.models import PointStruct

        meta_name = await self._ensure_meta_collection()
        point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"__gantry_meta__.{key}"))
        await self._client.upsert(
            collection_name=meta_name,
            points=[
                PointStruct(
                    id=point_id,
                    vector=[0.0],
                    payload={"key": key, "value": value},
                )
            ],
        )

    async def update_sync_metadata(self, embedder_id: str, dimension: int) -> None:
        """Update sync metadata after a successful sync."""
        await self.set_metadata("embedder_id", embedder_id)
        await self.set_metadata("dimension", str(dimension))

    async def close(self) -> None:
        """Close the underlying client (``AgentGantry.close()`` calls this)."""
        await self._client.close()


class ChromaVectorStore:
    """
    Production Chroma vector store adapter.

    Supports remote, persistent, and in-memory modes.
    """

    def __init__(
        self,
        url: str | None = None,
        collection_name: str = "agent_gantry",
        persist_directory: str | None = None,
        api_key: str | None = None,
        dimension: int = 0,
    ) -> None:
        """
        Initialize the Chroma vector store.

        Args:
            url: Remote Chroma server URL (for remote mode)
            collection_name: Name of the collection
            persist_directory: Local persistence directory (for persistent mode)
            api_key: Optional API key for authentication
            dimension: Vector dimension for tracking purposes only. This parameter
                      is not used for validation or dimension enforcement by Chroma,
                      but provides a way to track the expected dimension externally.
        """
        try:
            import chromadb
        except ImportError as exc:
            raise ImportError(
                "chromadb is not installed. Install it with:\n  pip install agent-gantry[chroma] (or uv add 'agent-gantry[chroma]')"
            ) from exc

        self._collection_name = collection_name
        self._dimension = dimension
        self._initialized = False

        # Determine client mode
        if url:
            # Remote mode
            self._client = chromadb.HttpClient(
                host=url, headers={"Authorization": api_key} if api_key else None
            )
            logger.info(f"Initialized ChromaVectorStore in remote mode: {url}")
        elif persist_directory:
            # Persistent mode
            self._client = chromadb.PersistentClient(path=persist_directory)
            logger.info(f"Initialized ChromaVectorStore in persistent mode: {persist_directory}")
        else:
            # In-memory mode
            self._client = chromadb.Client()
            logger.info("Initialized ChromaVectorStore in memory mode")

        self._collection = None

    @property
    def dimension(self) -> int:
        """
        Return the vector dimension for tracking purposes.

        Note: This dimension is not enforced by Chroma and is used for
        external tracking and consistency checks only.
        """
        return self._dimension

    @staticmethod
    def _build_namespace_where(namespace: Any) -> dict[str, Any] | None:
        """Translate a namespace filter value into a Chroma ``where`` clause.

        The router sends namespaces as a list (``{"namespace": [...]}``),
        which Chroma only accepts through the ``$in`` operator — a bare list
        is rejected with ValueError. Returns None for an empty list, which
        can match nothing (and ``$in`` requires a non-empty list).
        """
        if isinstance(namespace, (list, tuple, set)):
            values = list(namespace)
            if not values:
                return None
            return {"namespace": {"$in": values}}
        return {"namespace": namespace}

    async def initialize(self) -> None:
        """Initialize the collection."""
        if self._initialized:
            return

        # Get or create collection with cosine similarity
        # Wrap synchronous operation to avoid blocking event loop
        self._collection = await asyncio.to_thread(
            self._client.get_or_create_collection,
            name=self._collection_name,
            metadata={"hnsw:space": "cosine"},
        )

        self._initialized = True
        logger.info(f"Initialized Chroma collection: {self._collection_name}")

    async def add_tools(
        self,
        tools: list[ToolDefinition],
        embeddings: list[list[float]],
        upsert: bool = True,
    ) -> int:
        """Add tools to the vector store."""
        await self.initialize()

        if not tools or not embeddings:
            return 0

        # Without upsert, ids already stored (or repeated within the batch)
        # are skipped and only the rows actually inserted are counted.
        skip: set[str] | None = None
        if not upsert:
            existing = await asyncio.to_thread(
                self._collection.get,
                ids=[f"{t.namespace}.{t.name}" for t in tools],
                include=[],
            )
            skip = set(existing.get("ids") or [])

        ids: list[str] = []
        documents: list[str] = []
        metadatas: list[dict[str, Any]] = []
        vectors: list[list[float]] = []
        for tool, embedding in zip(tools, embeddings):
            tool_id = f"{tool.namespace}.{tool.name}"
            if skip is not None:
                if tool_id in skip:
                    continue
                skip.add(tool_id)
            ids.append(tool_id)
            documents.append(tool.description)
            # Full tool JSON plus the sync fingerprint SyncManager.detect_changes compares
            metadatas.append(
                {
                    "name": tool.name,
                    "namespace": tool.namespace,
                    "tool_json": tool.model_dump_json(),
                    "fingerprint": compute_tool_fingerprint(tool),
                }
            )
            vectors.append(embedding)

        if ids:
            await asyncio.to_thread(
                self._collection.upsert,
                ids=ids,
                embeddings=vectors,
                documents=documents,
                metadatas=metadatas,
            )
        return len(ids)

    async def search(
        self,
        query_vector: list[float],
        limit: int,
        filters: dict[str, Any] | None = None,
        score_threshold: float | None = None,
        include_embeddings: bool = False,
    ) -> list[tuple[ToolDefinition, float]] | list[tuple[ToolDefinition, float, list[float]]]:
        """Search for similar tools."""
        await self.initialize()

        # Build where filter for namespace (scalar or list)
        where = None
        if filters and "namespace" in filters:
            where = self._build_namespace_where(filters["namespace"])
            if where is None:
                return []  # empty namespace list matches nothing

        include = ["metadatas", "distances"] + (["embeddings"] if include_embeddings else [])

        async def fetch(size: int) -> list[Any]:
            results = await asyncio.to_thread(
                self._collection.query,
                query_embeddings=[query_vector],
                n_results=size,
                where=where,
                include=include,
            )
            metadatas = (results.get("metadatas") or [[]])[0]
            distances = (results.get("distances") or [[]])[0]
            embeddings = (results.get("embeddings") or [[]])[0] if include_embeddings else []
            out: list[Any] = []
            for index, (metadata, distance) in enumerate(zip(metadatas, distances)):
                score = 1.0 - float(distance)  # cosine distance -> similarity
                if score_threshold is not None and score < score_threshold:
                    continue
                embedding = list(embeddings[index]) if include_embeddings else None
                out.append(
                    _result(metadata.get("tool_json", "{}"), score, embedding, include_embeddings)
                )
            return out

        return await _search_with_tags(fetch, limit, _required_tags(filters))

    async def get_by_name(self, name: str, namespace: str = "default") -> ToolDefinition | None:
        """Get a tool by name."""
        await self.initialize()

        tool_id = f"{namespace}.{name}"

        try:
            # Wrap synchronous operation to avoid blocking event loop
            result = await asyncio.to_thread(self._collection.get, ids=[tool_id])

            if result["metadatas"]:
                tool_json = result["metadatas"][0].get("tool_json", "{}")
                return ToolDefinition.model_validate_json(tool_json)
        except Exception as e:
            logger.debug(f"get_by_name failed for {namespace}.{name}: {e}")

        return None

    async def delete(self, name: str, namespace: str = "default") -> bool:
        """Delete a tool. Returns False if it was not stored."""
        await self.initialize()

        tool_id = f"{namespace}.{name}"

        try:
            # Chroma's delete is a silent no-op on a miss, so check first.
            existing = await asyncio.to_thread(self._collection.get, ids=[tool_id], include=[])
            if not existing.get("ids"):
                return False
            await asyncio.to_thread(self._collection.delete, ids=[tool_id])
            return True
        except Exception:
            return False

    async def list_all(
        self,
        namespace: str | None = None,
        limit: int = 1000,
        offset: int = 0,
    ) -> list[ToolDefinition]:
        """List all tools."""
        await self.initialize()
        if _selects_nothing(namespace):
            # An empty list is a filter that matches nothing, not the
            # absence of one, which is how ``search`` reads it.
            return []

        # Build where filter for namespace (scalar or list)
        where = None
        if namespace:
            where = self._build_namespace_where(namespace)

        try:
            # Wrap synchronous operation to avoid blocking event loop
            result = await asyncio.to_thread(
                self._collection.get,
                where=where,
                limit=limit,
                offset=offset,
            )

            tools = []
            if result["metadatas"]:
                for metadata in result["metadatas"]:
                    tool_json = metadata.get("tool_json", "{}")
                    tools.append(ToolDefinition.model_validate_json(tool_json))

            return tools
        except Exception as e:
            logger.warning(f"list_all failed: {e}")
            return []

    async def count(self, namespace: str | None = None) -> int:
        """Count tools."""
        await self.initialize()
        if _selects_nothing(namespace):
            # An empty list is a filter that matches nothing, not the
            # absence of one, which is how ``search`` reads it.
            return 0

        try:
            where = None
            if namespace:
                where = self._build_namespace_where(namespace)

            # Wrap synchronous operation to avoid blocking event loop
            if where:
                # Collection.count() takes no filter (passing one raised
                # TypeError, swallowed below into a permanent 0). Fetch the
                # matching ids only and count those.
                result = await asyncio.to_thread(
                    self._collection.get,
                    where=where,
                    include=[],
                )
                return len(result.get("ids") or [])
            return await asyncio.to_thread(self._collection.count)
        except Exception:
            return 0

    async def health_check(self) -> bool:
        """Check health of Chroma connection."""
        try:
            await self.initialize()
            return True
        except Exception:
            return False

    @property
    def supports_metadata(self) -> bool:
        """Chroma supports metadata storage (via a side collection)."""
        return True

    async def get_stored_fingerprints(self) -> dict[str, str]:
        """Get all stored tool fingerprints for incremental sync."""
        await self.initialize()

        try:
            result = await asyncio.to_thread(
                self._collection.get,
                include=["metadatas"],
            )
        except Exception as e:
            logger.debug(f"get_stored_fingerprints failed: {e}")
            return {}

        fingerprints: dict[str, str] = {}
        ids = result.get("ids") or []
        metadatas = result.get("metadatas") or []
        for tool_id, metadata in zip(ids, metadatas):
            if metadata:
                fingerprint = metadata.get("fingerprint")
                if fingerprint:
                    fingerprints[tool_id] = fingerprint
        return fingerprints

    async def _get_meta_collection(self) -> Any:
        """Get or create the side collection holding sync metadata."""
        return await asyncio.to_thread(
            self._client.get_or_create_collection,
            name=f"{self._collection_name}__meta",
        )

    async def get_metadata(self, key: str) -> str | None:
        """Get a sync-metadata value by key."""
        try:
            meta = await self._get_meta_collection()
            result = await asyncio.to_thread(meta.get, ids=[key])
            documents = result.get("documents") or []
            if documents:
                return documents[0]
        except Exception as e:
            logger.debug(f"get_metadata failed for {key}: {e}")
        return None

    async def set_metadata(self, key: str, value: str) -> None:
        """Set a sync-metadata value."""
        meta = await self._get_meta_collection()
        await asyncio.to_thread(
            meta.upsert,
            ids=[key],
            embeddings=[[0.0]],
            documents=[value],
        )

    async def update_sync_metadata(self, embedder_id: str, dimension: int) -> None:
        """Update sync metadata after a successful sync."""
        await self.set_metadata("embedder_id", embedder_id)
        await self.set_metadata("dimension", str(dimension))


class PGVectorStore:
    """
    Production PGVector vector store adapter.

    Uses asyncpg for async PostgreSQL operations with pgvector extension.
    """

    def __init__(
        self,
        url: str,
        table_name: str = "agent_gantry_tools",
        dimension: int = 1536,
    ) -> None:
        """
        Initialize the PGVector store.

        Args:
            url: PostgreSQL connection string
            table_name: Name of the tools table
            dimension: Vector dimension
        """
        try:
            import asyncpg  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "asyncpg is not installed. Install it with:\n  pip install agent-gantry[pgvector] (or uv add 'agent-gantry[pgvector]')"
            ) from exc

        if not url:
            raise ValueError("PGVector requires a connection string (url)")

        # Validate table name to prevent SQL injection
        _validate_sql_identifier(table_name, "table_name")

        self._url = url
        self._table_name = table_name
        # PostgreSQL silently truncates identifiers to 63 bytes, so for a
        # table_name near the limit "<table>__meta" would truncate back to
        # the tools table's own identifier and CREATE TABLE IF NOT EXISTS
        # would treat the tools table as the metadata table. Derive a
        # shortened, hash-distinguished name in that case.
        meta_name = f"{table_name}__meta"
        if len(meta_name) > 63:
            digest = hashlib.sha256(table_name.encode()).hexdigest()[:10]
            meta_name = f"{table_name[:46]}_{digest}__meta"
        self._meta_table_name = meta_name
        self._dimension = dimension
        self._pool = None
        self._initialized = False

        logger.info(f"Initialized PGVectorStore with table={table_name}")

    @property
    def dimension(self) -> int:
        """Return the vector dimension."""
        return self._dimension

    async def initialize(self) -> None:
        """Initialize database connection and create table if needed."""
        if self._initialized:
            return

        import asyncpg

        # Create connection pool
        self._pool = await asyncpg.create_pool(self._url)

        async with self._pool.acquire() as conn:
            # Enable vector extension
            await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")

            # Create table
            await conn.execute(f"""
                CREATE TABLE IF NOT EXISTS "{self._table_name}" (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    namespace TEXT NOT NULL,
                    description TEXT,
                    tool_json TEXT NOT NULL,
                    fingerprint TEXT,
                    embedding vector({self._dimension}),
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
                )
            """)

            # Migrate pre-fingerprint tables in place (no-op when present).
            # Tools without a fingerprint simply re-embed on the next sync.
            await conn.execute(f"""
                ALTER TABLE "{self._table_name}"
                ADD COLUMN IF NOT EXISTS fingerprint TEXT
            """)

            # Sync metadata table (embedder_id / dimension tracking)
            await conn.execute(f"""
                CREATE TABLE IF NOT EXISTS "{self._meta_table_name}" (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL,
                    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
                )
            """)

            # Create IVFFlat index for fast vector search
            await conn.execute(f"""
                CREATE INDEX IF NOT EXISTS "{self._table_name}_embedding_idx"
                ON "{self._table_name}"
                USING ivfflat (embedding vector_cosine_ops)
                WITH (lists = 100)
            """)

            # Create namespace index for filtering
            await conn.execute(f"""
                CREATE INDEX IF NOT EXISTS "{self._table_name}_namespace_idx"
                ON "{self._table_name}" (namespace)
            """)

        self._initialized = True
        logger.info(f"Initialized PGVector table: {self._table_name}")

    async def add_tools(
        self,
        tools: list[ToolDefinition],
        embeddings: list[list[float]],
        upsert: bool = True,
    ) -> int:
        """Add tools to the vector store."""
        await self.initialize()

        if not tools or not embeddings:
            return 0

        records = [
            (
                f"{tool.namespace}.{tool.name}",
                tool.name,
                tool.namespace,
                tool.description,
                tool.model_dump_json(),
                compute_tool_fingerprint(tool),
                json.dumps(embedding),
            )
            for tool, embedding in zip(tools, embeddings)
        ]
        insert = (
            f'INSERT INTO "{self._table_name}" '
            "(id, name, namespace, description, tool_json, fingerprint, embedding) "
            "VALUES ($1, $2, $3, $4, $5, $6, $7)"
        )

        async with self._pool.acquire() as conn:
            if upsert:
                await conn.executemany(
                    f"""
                    {insert}
                    ON CONFLICT (id) DO UPDATE SET
                        name = EXCLUDED.name,
                        namespace = EXCLUDED.namespace,
                        description = EXCLUDED.description,
                        tool_json = EXCLUDED.tool_json,
                        fingerprint = EXCLUDED.fingerprint,
                        embedding = EXCLUDED.embedding,
                        updated_at = NOW()
                    """,
                    records,
                )
                return len(records)

            # Without upsert, ids already stored are skipped and only the rows
            # actually inserted are counted.
            inserted = 0
            async with conn.transaction():
                for record in records:
                    row = await conn.fetchrow(
                        f"{insert} ON CONFLICT (id) DO NOTHING RETURNING id", *record
                    )
                    inserted += row is not None
            return inserted

    async def search(
        self,
        query_vector: list[float],
        limit: int,
        filters: dict[str, Any] | None = None,
        score_threshold: float | None = None,
        include_embeddings: bool = False,
    ) -> list[tuple[ToolDefinition, float]] | list[tuple[ToolDefinition, float, list[float]]]:
        """Search for similar tools."""
        await self.initialize()

        namespace_clause = ""
        namespace_params: list[Any] = []
        if filters and "namespace" in filters:
            namespace_clause, ns_param = _pg_namespace_clause(filters["namespace"], 3)
            namespace_params.append(ns_param)

        select_cols = "tool_json, 1 - (embedding <=> $1::vector) AS similarity"
        if include_embeddings:
            select_cols += ", embedding"
        query = f"""
            SELECT {select_cols}
            FROM "{self._table_name}"
            {namespace_clause}
            ORDER BY embedding <=> $1::vector
            LIMIT $2
        """
        embedding_str = json.dumps(query_vector)

        async def fetch(size: int) -> list[Any]:
            async with self._pool.acquire() as conn:
                rows = await conn.fetch(query, embedding_str, size, *namespace_params)
            out: list[Any] = []
            for row in rows:
                score = float(row["similarity"])
                if score_threshold is not None and score < score_threshold:
                    continue
                embedding = None
                if include_embeddings:
                    # asyncpg returns the pgvector value as text unless a codec is registered
                    raw = row["embedding"]
                    embedding = json.loads(raw) if isinstance(raw, str) else list(raw)
                out.append(_result(row["tool_json"], score, embedding, include_embeddings))
            return out

        return await _search_with_tags(fetch, limit, _required_tags(filters))

    async def get_by_name(self, name: str, namespace: str = "default") -> ToolDefinition | None:
        """Get a tool by name."""
        await self.initialize()

        tool_id = f"{namespace}.{name}"

        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                f'SELECT tool_json FROM "{self._table_name}" WHERE id = $1',
                tool_id,
            )

            if row:
                return ToolDefinition.model_validate_json(row["tool_json"])

        return None

    async def delete(self, name: str, namespace: str = "default") -> bool:
        """Delete a tool."""
        await self.initialize()

        tool_id = f"{namespace}.{name}"

        async with self._pool.acquire() as conn:
            result = await conn.execute(
                f'DELETE FROM "{self._table_name}" WHERE id = $1',
                tool_id,
            )

            # Check if any rows were deleted
            return result.split()[-1] != "0"

    async def list_all(
        self,
        namespace: str | None = None,
        limit: int = 1000,
        offset: int = 0,
    ) -> list[ToolDefinition]:
        """List all tools."""
        await self.initialize()
        if _selects_nothing(namespace):
            # An empty list is a filter that matches nothing, not the
            # absence of one, which is how ``search`` reads it.
            return []

        namespace_clause = ""
        params: list[Any] = [limit, offset]

        if namespace:
            namespace_clause, ns_param = _pg_namespace_clause(namespace, 3)
            params.append(ns_param)

        query = f"""
            SELECT tool_json FROM "{self._table_name}"
            {namespace_clause}
            ORDER BY created_at DESC, id
            LIMIT $1 OFFSET $2
        """

        async with self._pool.acquire() as conn:
            rows = await conn.fetch(query, *params)

            tools = []
            for row in rows:
                tools.append(ToolDefinition.model_validate_json(row["tool_json"]))

            return tools

    async def count(self, namespace: str | None = None) -> int:
        """Count tools."""
        await self.initialize()
        if _selects_nothing(namespace):
            # An empty list is a filter that matches nothing, not the
            # absence of one, which is how ``search`` reads it.
            return 0

        namespace_clause = ""
        params: list[Any] = []

        if namespace:
            namespace_clause, ns_param = _pg_namespace_clause(namespace, 1)
            params.append(ns_param)

        query = f'SELECT COUNT(*) FROM "{self._table_name}" {namespace_clause}'

        async with self._pool.acquire() as conn:
            result = await conn.fetchval(query, *params)
            return result

    async def health_check(self) -> bool:
        """Check health of PostgreSQL connection."""
        try:
            await self.initialize()
            async with self._pool.acquire() as conn:
                await conn.fetchval("SELECT 1")
            return True
        except Exception:
            return False

    @property
    def supports_metadata(self) -> bool:
        """PGVector supports metadata storage (via a side table)."""
        return True

    async def get_stored_fingerprints(self) -> dict[str, str]:
        """Get all stored tool fingerprints for incremental sync."""
        await self.initialize()

        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                f'SELECT id, fingerprint FROM "{self._table_name}" '
                "WHERE fingerprint IS NOT NULL"
            )
            return {row["id"]: row["fingerprint"] for row in rows}

    async def get_metadata(self, key: str) -> str | None:
        """Get a sync-metadata value by key."""
        await self.initialize()

        async with self._pool.acquire() as conn:
            return await conn.fetchval(
                f'SELECT value FROM "{self._meta_table_name}" WHERE key = $1',
                key,
            )

    async def set_metadata(self, key: str, value: str) -> None:
        """Set a sync-metadata value."""
        await self.initialize()

        async with self._pool.acquire() as conn:
            await conn.execute(
                f"""
                INSERT INTO "{self._meta_table_name}" (key, value, updated_at)
                VALUES ($1, $2, NOW())
                ON CONFLICT (key) DO UPDATE SET
                    value = EXCLUDED.value,
                    updated_at = NOW()
                """,
                key,
                value,
            )

    async def update_sync_metadata(self, embedder_id: str, dimension: int) -> None:
        """Update sync metadata after a successful sync."""
        await self.set_metadata("embedder_id", embedder_id)
        await self.set_metadata("dimension", str(dimension))

    async def close(self) -> None:
        """Close the connection pool."""
        if self._pool:
            await self._pool.close()
