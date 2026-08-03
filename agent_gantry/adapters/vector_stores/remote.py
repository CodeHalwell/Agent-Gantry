"""
Production vector store adapters for Qdrant, Chroma, and PGVector.

Provides real implementations for remote vector databases with proper
collection management, filtering, and error handling.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import uuid
from typing import Any

from agent_gantry.schema.tool import ToolDefinition
from agent_gantry.utils.fingerprint import compute_tool_fingerprint

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
        """
        try:
            from qdrant_client import AsyncQdrantClient
            from qdrant_client.models import Distance, VectorParams
        except ImportError as exc:
            raise ImportError(
                "qdrant-client is not installed. Install it with:\n"
                "  pip install agent-gantry[qdrant]"
            ) from exc

        self._url = url
        self._api_key = api_key
        self._collection_name = collection_name
        self._dimension = dimension
        self._prefer_grpc = prefer_grpc
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
        self._VectorParams = VectorParams

        logger.info(f"Initialized QdrantVectorStore with url={url}, collection={collection_name}")

    @property
    def dimension(self) -> int:
        """Return the vector dimension."""
        return self._dimension

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
                )
                logger.info(f"Created Qdrant collection: {self._collection_name}")

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

        points = []
        for tool, embedding in zip(tools, embeddings):
            # Generate deterministic ID
            point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{tool.namespace}.{tool.name}"))

            # Create payload with tool data. The fingerprint enables
            # incremental sync: SyncManager.detect_changes compares it against
            # compute_tool_fingerprint so unchanged tools skip re-embedding.
            payload = {
                "name": tool.name,
                "namespace": tool.namespace,
                "description": tool.description,
                "tool_json": tool.model_dump_json(),
                "fingerprint": compute_tool_fingerprint(tool),
            }

            points.append(
                PointStruct(
                    id=point_id,
                    vector=embedding,
                    payload=payload,
                )
            )

        # Upsert points
        await self._client.upsert(
            collection_name=self._collection_name,
            points=points,
        )

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
        from qdrant_client.models import FieldCondition, Filter, MatchValue

        await self.initialize()

        # Build filter for namespace
        query_filter = None
        if filters and "namespace" in filters:
            namespace = filters["namespace"]
            query_filter = Filter(
                must=[
                    FieldCondition(
                        key="namespace",
                        match=MatchValue(value=namespace),
                    )
                ]
            )

        # Search with optional vector retrieval
        results = await self._client.search(
            collection_name=self._collection_name,
            query_vector=query_vector,
            limit=limit,
            query_filter=query_filter,
            score_threshold=score_threshold,
            with_vectors=include_embeddings,  # Request vectors if needed
        )

        # Convert results to tools
        if include_embeddings:
            tools_with_embeddings: list[tuple[ToolDefinition, float, list[float]]] = []
            for result in results:
                tool_json = result.payload.get("tool_json", "{}")
                tool = ToolDefinition.model_validate_json(tool_json)
                # Extract vector from result
                embedding = list(result.vector) if result.vector else []
                tools_with_embeddings.append((tool, float(result.score), embedding))
            return tools_with_embeddings
        else:
            tools_without_embeddings: list[tuple[ToolDefinition, float]] = []
            for result in results:
                tool_json = result.payload.get("tool_json", "{}")
                tool = ToolDefinition.model_validate_json(tool_json)
                tools_without_embeddings.append((tool, float(result.score)))
            return tools_without_embeddings

    async def get_by_name(self, name: str, namespace: str = "default") -> ToolDefinition | None:
        """Get a tool by name."""

        await self.initialize()

        # Generate the same deterministic ID
        point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{namespace}.{name}"))

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
        """Delete a tool."""
        await self.initialize()

        point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{namespace}.{name}"))

        try:
            from qdrant_client.models import PointIdsList

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
        """List all tools."""
        from qdrant_client.models import FieldCondition, Filter, MatchValue

        await self.initialize()

        # Build filter for namespace
        query_filter = None
        if namespace:
            query_filter = Filter(
                must=[
                    FieldCondition(
                        key="namespace",
                        match=MatchValue(value=namespace),
                    )
                ]
            )

        # Scroll through results
        results, _ = await self._client.scroll(
            collection_name=self._collection_name,
            scroll_filter=query_filter,
            limit=limit,
            offset=offset,
        )

        tools = []
        for record in results:
            tool_json = record.payload.get("tool_json", "{}")
            tools.append(ToolDefinition.model_validate_json(tool_json))

        return tools

    async def count(self, namespace: str | None = None) -> int:
        """Count tools."""
        from qdrant_client.models import FieldCondition, Filter, MatchValue

        await self.initialize()

        if namespace:
            # Count with filter
            query_filter = Filter(
                must=[
                    FieldCondition(
                        key="namespace",
                        match=MatchValue(value=namespace),
                    )
                ]
            )
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
                "chromadb is not installed. Install it with:\n  pip install agent-gantry[chroma]"
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

        ids = []
        documents = []
        metadatas = []
        vectors = []

        for tool, embedding in zip(tools, embeddings):
            # Generate deterministic ID
            tool_id = f"{tool.namespace}.{tool.name}"
            ids.append(tool_id)

            # Document is the searchable text
            documents.append(tool.description)

            # Metadata includes full tool JSON and the sync fingerprint
            # (compared by SyncManager.detect_changes for incremental sync)
            metadatas.append(
                {
                    "name": tool.name,
                    "namespace": tool.namespace,
                    "tool_json": tool.model_dump_json(),
                    "fingerprint": compute_tool_fingerprint(tool),
                }
            )

            vectors.append(embedding)

        # Upsert to collection
        # Wrap synchronous operations to avoid blocking event loop
        if upsert:
            await asyncio.to_thread(
                self._collection.upsert,
                ids=ids,
                embeddings=vectors,
                documents=documents,
                metadatas=metadatas,
            )
        else:
            await asyncio.to_thread(
                self._collection.add,
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

        # Build where filter for namespace
        where = None
        if filters and "namespace" in filters:
            where = {"namespace": filters["namespace"]}

        # Query collection with optional embeddings
        # Wrap synchronous operation to avoid blocking event loop
        results = await asyncio.to_thread(
            self._collection.query,
            query_embeddings=[query_vector],
            n_results=limit,
            where=where,
            include=["metadatas", "distances", "embeddings"]
            if include_embeddings
            else ["metadatas", "distances"],
        )

        # Convert results to tools
        if include_embeddings:
            tools_with_embeddings: list[tuple[ToolDefinition, float, list[float]]] = []

            if results["metadatas"] and results["distances"] and results.get("embeddings"):
                for metadata, distance, embedding in zip(
                    results["metadatas"][0], results["distances"][0], results["embeddings"][0]
                ):
                    tool_json = metadata.get("tool_json", "{}")
                    tool = ToolDefinition.model_validate_json(tool_json)

                    # Convert distance to similarity score (1 - distance for cosine)
                    score = 1.0 - float(distance)

                    # Apply score threshold if specified
                    if score_threshold is None or score >= score_threshold:
                        tools_with_embeddings.append((tool, score, list(embedding)))

            return tools_with_embeddings
        else:
            tools_without_embeddings: list[tuple[ToolDefinition, float]] = []

            if results["metadatas"] and results["distances"]:
                for metadata, distance in zip(results["metadatas"][0], results["distances"][0]):
                    tool_json = metadata.get("tool_json", "{}")
                    tool = ToolDefinition.model_validate_json(tool_json)

                    # Convert distance to similarity score (1 - distance for cosine)
                    score = 1.0 - float(distance)

                    # Apply score threshold if specified
                    if score_threshold is None or score >= score_threshold:
                        tools_without_embeddings.append((tool, score))

            return tools_without_embeddings

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
        """Delete a tool."""
        await self.initialize()

        tool_id = f"{namespace}.{name}"

        try:
            # Wrap synchronous operation to avoid blocking event loop
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

        # Build where filter for namespace
        where = None
        if namespace:
            where = {"namespace": namespace}

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

        try:
            where = None
            if namespace:
                where = {"namespace": namespace}

            # Wrap synchronous operation to avoid blocking event loop
            if where:
                result = await asyncio.to_thread(self._collection.count, where=where)
            else:
                result = await asyncio.to_thread(self._collection.count)
            return result
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
                "asyncpg is not installed. Install it with:\n  pip install agent-gantry[pgvector]"
            ) from exc

        if not url:
            raise ValueError("PGVector requires a connection string (url)")

        # Validate table name to prevent SQL injection
        _validate_sql_identifier(table_name, "table_name")

        self._url = url
        self._table_name = table_name
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
                CREATE TABLE IF NOT EXISTS "{self._table_name}__meta" (
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

        async with self._pool.acquire() as conn:
            records = []
            for tool, embedding in zip(tools, embeddings):
                tool_id = f"{tool.namespace}.{tool.name}"
                embedding_str = json.dumps(embedding)
                records.append(
                    (
                        tool_id,
                        tool.name,
                        tool.namespace,
                        tool.description,
                        tool.model_dump_json(),
                        compute_tool_fingerprint(tool),
                        embedding_str,
                    )
                )

            if upsert:
                await conn.executemany(
                    f"""
                    INSERT INTO "{self._table_name}"
                    (id, name, namespace, description, tool_json, fingerprint, embedding, updated_at)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, NOW())
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
            else:
                await conn.executemany(
                    f"""
                    INSERT INTO "{self._table_name}"
                    (id, name, namespace, description, tool_json, fingerprint, embedding)
                    VALUES ($1, $2, $3, $4, $5, $6, $7)
                    """,
                    records,
                )

        return len(tools)

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

        embedding_str = json.dumps(query_vector)

        # Build query with optional namespace filter
        namespace_clause = ""
        params = [embedding_str, limit]

        if filters and "namespace" in filters:
            namespace_clause = "WHERE namespace = $3"
            params.append(filters["namespace"])

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

        async with self._pool.acquire() as conn:
            rows = await conn.fetch(query, *params)

            if include_embeddings:
                tools_with_embeddings: list[tuple[ToolDefinition, float, list[float]]] = []
                for row in rows:
                    score = float(row["similarity"])

                    # Apply score threshold if specified
                    if score_threshold is None or score >= score_threshold:
                        tool = ToolDefinition.model_validate_json(row["tool_json"])
                        # Parse embedding depending on asyncpg return type
                        # It may be returned as a list-like type or a string, depending on type setup
                        embedding_val = row["embedding"]
                        if isinstance(embedding_val, str):
                            # It's a string like "[1.0, 2.0, ...]"
                            parsed_embedding = json.loads(embedding_val)
                        else:
                            # Try to coerce it to list (pgvector type returned by asyncpg/pgvector)
                            parsed_embedding = list(embedding_val)

                        tools_with_embeddings.append((tool, score, parsed_embedding))

                return tools_with_embeddings
            else:
                tools: list[tuple[ToolDefinition, float]] = []
                for row in rows:
                    score = float(row["similarity"])

                    # Apply score threshold if specified
                    if score_threshold is None or score >= score_threshold:
                        tool = ToolDefinition.model_validate_json(row["tool_json"])
                        tools.append((tool, score))

                return tools

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

        namespace_clause = ""
        params = [limit, offset]

        if namespace:
            namespace_clause = "WHERE namespace = $3"
            params.append(namespace)

        query = f"""
            SELECT tool_json FROM "{self._table_name}"
            {namespace_clause}
            ORDER BY created_at DESC
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

        namespace_clause = ""
        params = []

        if namespace:
            namespace_clause = "WHERE namespace = $1"
            params.append(namespace)

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
                f'SELECT value FROM "{self._table_name}__meta" WHERE key = $1',
                key,
            )

    async def set_metadata(self, key: str, value: str) -> None:
        """Set a sync-metadata value."""
        await self.initialize()

        async with self._pool.acquire() as conn:
            await conn.execute(
                f"""
                INSERT INTO "{self._table_name}__meta" (key, value, updated_at)
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
