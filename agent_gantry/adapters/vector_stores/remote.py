"""
Production vector store adapters for Qdrant, Chroma, and PGVector.

Provides real implementations for remote vector databases with proper
collection management, filtering, and error handling.
"""

from __future__ import annotations

import logging
import uuid
from typing import Any

from agent_gantry.schema.tool import ToolDefinition

logger = logging.getLogger(__name__)


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

        logger.info(
            f"Initialized QdrantVectorStore with url={url}, collection={collection_name}"
        )

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
            point_id = str(
                uuid.uuid5(uuid.NAMESPACE_DNS, f"{tool.namespace}.{tool.name}")
            )

            # Create payload with tool data
            payload = {
                "name": tool.name,
                "namespace": tool.namespace,
                "description": tool.description,
                "tool_json": tool.model_dump_json(),
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
    ) -> list[tuple[ToolDefinition, float]]:
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

        # Search
        results = await self._client.search(
            collection_name=self._collection_name,
            query_vector=query_vector,
            limit=limit,
            query_filter=query_filter,
            score_threshold=score_threshold,
        )

        # Convert results to tools
        tools: list[tuple[ToolDefinition, float]] = []
        for result in results:
            tool_json = result.payload.get("tool_json", "{}")
            tool = ToolDefinition.model_validate_json(tool_json)
            tools.append((tool, result.score))

        return tools

    async def get_by_name(
        self, name: str, namespace: str = "default"
    ) -> ToolDefinition | None:
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
            await self._client.delete(
                collection_name=self._collection_name,
                points_selector=[point_id],
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
    ) -> None:
        """
        Initialize the Chroma vector store.

        Args:
            url: Remote Chroma server URL (for remote mode)
            collection_name: Name of the collection
            persist_directory: Local persistence directory (for persistent mode)
            api_key: Optional API key for authentication
        """
        try:
            import chromadb
        except ImportError as exc:
            raise ImportError(
                "chromadb is not installed. Install it with:\n"
                "  pip install agent-gantry[chroma]"
            ) from exc

        self._collection_name = collection_name
        self._initialized = False

        # Determine client mode
        if url:
            # Remote mode
            self._client = chromadb.HttpClient(host=url, headers={"Authorization": api_key} if api_key else None)
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

    async def initialize(self) -> None:
        """Initialize the collection."""
        if self._initialized:
            return

        # Get or create collection with cosine similarity
        self._collection = self._client.get_or_create_collection(
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

            # Metadata includes full tool JSON
            metadatas.append({
                "name": tool.name,
                "namespace": tool.namespace,
                "tool_json": tool.model_dump_json(),
            })

            vectors.append(embedding)

        # Upsert to collection
        if upsert:
            self._collection.upsert(
                ids=ids,
                embeddings=vectors,
                documents=documents,
                metadatas=metadatas,
            )
        else:
            self._collection.add(
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
    ) -> list[tuple[ToolDefinition, float]]:
        """Search for similar tools."""
        await self.initialize()

        # Build where filter for namespace
        where = None
        if filters and "namespace" in filters:
            where = {"namespace": filters["namespace"]}

        # Query collection
        results = self._collection.query(
            query_embeddings=[query_vector],
            n_results=limit,
            where=where,
        )

        # Convert results to tools
        tools: list[tuple[ToolDefinition, float]] = []

        if results["metadatas"] and results["distances"]:
            for metadata, distance in zip(results["metadatas"][0], results["distances"][0]):
                tool_json = metadata.get("tool_json", "{}")
                tool = ToolDefinition.model_validate_json(tool_json)

                # Convert distance to similarity score (1 - distance for cosine)
                score = 1.0 - distance

                # Apply score threshold if specified
                if score_threshold is None or score >= score_threshold:
                    tools.append((tool, score))

        return tools

    async def get_by_name(
        self, name: str, namespace: str = "default"
    ) -> ToolDefinition | None:
        """Get a tool by name."""
        await self.initialize()

        tool_id = f"{namespace}.{name}"

        try:
            result = self._collection.get(ids=[tool_id])

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
            self._collection.delete(ids=[tool_id])
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
            result = self._collection.get(
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

            result = self._collection.count(where=where) if where else self._collection.count()
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
                "asyncpg is not installed. Install it with:\n"
                "  pip install agent-gantry[pgvector]"
            ) from exc

        if not url:
            raise ValueError("PGVector requires a connection string (url)")

        self._url = url
        self._table_name = table_name
        self._dimension = dimension
        self._pool = None
        self._initialized = False

        logger.info(f"Initialized PGVectorStore with table={table_name}")

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
                CREATE TABLE IF NOT EXISTS {self._table_name} (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    namespace TEXT NOT NULL,
                    description TEXT,
                    tool_json TEXT NOT NULL,
                    embedding vector({self._dimension}),
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
                )
            """)

            # Create IVFFlat index for fast vector search
            await conn.execute(f"""
                CREATE INDEX IF NOT EXISTS {self._table_name}_embedding_idx
                ON {self._table_name}
                USING ivfflat (embedding vector_cosine_ops)
                WITH (lists = 100)
            """)

            # Create namespace index for filtering
            await conn.execute(f"""
                CREATE INDEX IF NOT EXISTS {self._table_name}_namespace_idx
                ON {self._table_name} (namespace)
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
            for tool, embedding in zip(tools, embeddings):
                tool_id = f"{tool.namespace}.{tool.name}"
                embedding_str = "[" + ",".join(str(x) for x in embedding) + "]"

                if upsert:
                    await conn.execute(
                        f"""
                        INSERT INTO {self._table_name}
                        (id, name, namespace, description, tool_json, embedding, updated_at)
                        VALUES ($1, $2, $3, $4, $5, $6, NOW())
                        ON CONFLICT (id) DO UPDATE SET
                            name = EXCLUDED.name,
                            namespace = EXCLUDED.namespace,
                            description = EXCLUDED.description,
                            tool_json = EXCLUDED.tool_json,
                            embedding = EXCLUDED.embedding,
                            updated_at = NOW()
                        """,
                        tool_id,
                        tool.name,
                        tool.namespace,
                        tool.description,
                        tool.model_dump_json(),
                        embedding_str,
                    )
                else:
                    await conn.execute(
                        f"""
                        INSERT INTO {self._table_name}
                        (id, name, namespace, description, tool_json, embedding)
                        VALUES ($1, $2, $3, $4, $5, $6)
                        """,
                        tool_id,
                        tool.name,
                        tool.namespace,
                        tool.description,
                        tool.model_dump_json(),
                        embedding_str,
                    )

        return len(tools)

    async def search(
        self,
        query_vector: list[float],
        limit: int,
        filters: dict[str, Any] | None = None,
        score_threshold: float | None = None,
    ) -> list[tuple[ToolDefinition, float]]:
        """Search for similar tools."""
        await self.initialize()

        embedding_str = "[" + ",".join(str(x) for x in query_vector) + "]"

        # Build query with optional namespace filter
        namespace_clause = ""
        params = [embedding_str, limit]

        if filters and "namespace" in filters:
            namespace_clause = "WHERE namespace = $3"
            params.append(filters["namespace"])

        query = f"""
            SELECT tool_json, 1 - (embedding <=> $1::vector) AS similarity
            FROM {self._table_name}
            {namespace_clause}
            ORDER BY embedding <=> $1::vector
            LIMIT $2
        """

        async with self._pool.acquire() as conn:
            rows = await conn.fetch(query, *params)

            tools: list[tuple[ToolDefinition, float]] = []
            for row in rows:
                score = float(row["similarity"])

                # Apply score threshold if specified
                if score_threshold is None or score >= score_threshold:
                    tool = ToolDefinition.model_validate_json(row["tool_json"])
                    tools.append((tool, score))

            return tools

    async def get_by_name(
        self, name: str, namespace: str = "default"
    ) -> ToolDefinition | None:
        """Get a tool by name."""
        await self.initialize()

        tool_id = f"{namespace}.{name}"

        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                f"SELECT tool_json FROM {self._table_name} WHERE id = $1",
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
                f"DELETE FROM {self._table_name} WHERE id = $1",
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
            SELECT tool_json FROM {self._table_name}
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

        query = f"SELECT COUNT(*) FROM {self._table_name} {namespace_clause}"

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

    async def close(self) -> None:
        """Close the connection pool."""
        if self._pool:
            await self._pool.close()
