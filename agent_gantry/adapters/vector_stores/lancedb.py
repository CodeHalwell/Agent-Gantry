"""
LanceDB vector store adapter for Agent-Gantry.

Provides on-device, zero-config persistence with local LanceDB files,
supporting both tools and skills collections for semantic retrieval.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from agent_gantry.adapters.vector_stores.lancedb_mixins import (
    LanceDBMetadataMixin,
    LanceDBToolsMixin,
    _escape_sql_string,
    _validate_identifier,
)
from agent_gantry.schema.skill import Skill
from agent_gantry.schema.tool import ToolDefinition
from agent_gantry.utils.fingerprint import compute_tool_fingerprint

__all__ = ["LanceDBVectorStore", "_escape_sql_string", "_validate_identifier"]

logger = logging.getLogger(__name__)


def _tool_record(tool: ToolDefinition, embedding: list[float], now: str) -> dict[str, Any]:
    return {
        "id": f"{tool.namespace}.{tool.name}",
        "name": tool.name,
        "namespace": tool.namespace,
        "description": tool.description,
        "tool_json": tool.model_dump_json(),
        "fingerprint": compute_tool_fingerprint(tool),
        "vector": embedding,
        "created_at": now,
        "updated_at": now,
    }


def _skill_record(skill: Skill, embedding: list[float], now: str) -> dict[str, Any]:
    return {
        "id": f"{skill.namespace}.{skill.name}",
        "name": skill.name,
        "namespace": skill.namespace,
        "description": skill.description,
        "category": skill.category.value,
        "skill_json": skill.model_dump_json(),
        "vector": embedding,
        "created_at": now,
        "updated_at": now,
    }


def _namespace_predicate(ns_filter: Any) -> str | None:
    """SQL predicate for a namespace filter; None for an empty collection (matches nothing)."""
    if not isinstance(ns_filter, (list, tuple, set)):
        return f"namespace = '{_escape_sql_string(ns_filter)}'"
    values = list(ns_filter)
    if not values:
        return None
    if len(values) == 1:
        return f"namespace = '{_escape_sql_string(values[0])}'"
    return "namespace IN ({})".format(", ".join(f"'{_escape_sql_string(v)}'" for v in values))


class LanceDBVectorStore(LanceDBToolsMixin, LanceDBMetadataMixin):
    """
    LanceDB vector store for on-device semantic indexing.

    Provides SQLite-like local persistence for tools and skills with
    high-speed, low-memory vector search. Supports zero-config setup
    with automatic database creation.

    Multi-Process Limitations:
        LanceDB uses file-based storage and does not provide built-in locking
        mechanisms for concurrent writes. To ensure data consistency:

        * **Single Writer**: Only one process should write to a database at a time
        * **Multiple Readers**: Multiple processes can safely read from the same database
        * **Coordination**: Use external locks (e.g., file locks, distributed locks)
          if you need concurrent writes from multiple processes
        * **Alternatives**: For true multi-process write support, consider using
          Qdrant or PostgreSQL with pgvector adapters

        Example with file locking:
        ```python
        import fcntl
        with open('.agent_gantry/lancedb.lock', 'w') as lock_file:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            await store.add_tools(tools, embeddings)
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
        ```

    Security Note:
        SQL injection protection is implemented through a defense-in-depth approach:
        1. Input validation via _validate_identifier() (length limits, control char rejection)
        2. SQL escaping via _escape_sql_string() (single-quote doubling)
        3. Limited scope - only metadata key lookups use WHERE clauses

        LanceDB does not currently support parameterized queries for WHERE clauses.
        All SQL injection test cases in the test suite verify this protection is effective.

    Attributes:
        db_path: Path to the LanceDB database directory
        tools_table: Name of the tools collection
        skills_table: Name of the skills collection
        dimension: Vector dimension (supports Matryoshka truncation)

    Example:
        >>> store = LanceDBVectorStore()
        >>> await store.initialize()
        >>> await store.add_tools(tools, embeddings)
        >>> results = await store.search(query_vector, limit=5)
    """

    # Default database location (SQLite-like behavior)
    DEFAULT_DB_PATH = ".agent_gantry/lancedb"

    def __init__(
        self,
        db_path: str | None = None,
        tools_table: str = "tools",
        skills_table: str = "skills",
        dimension: int = 768,
    ) -> None:
        """
        Initialize the LanceDB vector store.

        Args:
            db_path: Path to database directory. If None, uses ~/.agent_gantry/lancedb
                    or current directory's .agent_gantry/lancedb
            tools_table: Name of the tools table
            skills_table: Name of the skills table
            dimension: Vector dimension for embeddings
        """
        self._db_path = self._resolve_db_path(db_path)
        self._tools_table_name = tools_table
        self._skills_table_name = skills_table
        self._metadata_table_name = "_gantry_metadata"
        self._dimension = dimension
        self._db: Any = None
        self._tools_table: Any = None
        self._skills_table: Any = None
        self._metadata_table: Any = None
        self._initialized = False

    def _resolve_db_path(self, db_path: str | None) -> str:
        """Resolve database path with zero-config defaults."""
        if db_path:
            return db_path

        # Try current directory first, then user home
        cwd_path = Path.cwd() / self.DEFAULT_DB_PATH
        home_path = Path.home() / self.DEFAULT_DB_PATH

        # Prefer existing database, otherwise use current directory
        if home_path.exists():
            return str(home_path)
        return str(cwd_path)

    async def initialize(self) -> None:
        """
        Initialize the database and create tables if needed.

        Creates the database directory and tables on first run.
        Idempotent - safe to call multiple times.
        """
        if self._initialized:
            return

        try:
            import lancedb  # type: ignore[import-untyped]
            import pyarrow as pa  # type: ignore[import-untyped]
        except ImportError as e:
            raise ImportError(
                "lancedb and pyarrow are required. Install with: pip install lancedb pyarrow (or uv add lancedb pyarrow)"
            ) from e

        # Create database directory
        db_dir = Path(self._db_path)
        db_dir.mkdir(parents=True, exist_ok=True)

        # Connect to database (blocking file I/O — keep it off the event loop)
        self._db = await asyncio.to_thread(lancedb.connect, str(db_dir))

        # Create tools table schema
        tools_schema = pa.schema(
            [
                pa.field("id", pa.string()),
                pa.field("name", pa.string()),
                pa.field("namespace", pa.string()),
                pa.field("description", pa.string()),
                pa.field("tool_json", pa.string()),  # Full serialized ToolDefinition
                pa.field("fingerprint", pa.string()),  # Hash of tool for change detection
                pa.field("vector", pa.list_(pa.float32(), self._dimension)),
                pa.field("created_at", pa.string()),
                pa.field("updated_at", pa.string()),
            ]
        )

        # Create skills table schema
        skills_schema = pa.schema(
            [
                pa.field("id", pa.string()),
                pa.field("name", pa.string()),
                pa.field("namespace", pa.string()),
                pa.field("description", pa.string()),
                pa.field("category", pa.string()),
                pa.field("skill_json", pa.string()),  # Full serialized Skill
                pa.field("vector", pa.list_(pa.float32(), self._dimension)),
                pa.field("created_at", pa.string()),
                pa.field("updated_at", pa.string()),
            ]
        )

        # Create metadata table schema (stores sync state)
        metadata_schema = pa.schema(
            [
                pa.field("key", pa.string()),
                pa.field("value", pa.string()),
                pa.field("updated_at", pa.string()),
            ]
        )

        # Open or create the tables off the event loop (file I/O)
        def open_tables() -> tuple[Any, Any, Any, bool]:
            # list_tables() returns a TableListResult (with .tables) on newer versions
            listed = self._db.list_tables()
            existing = set(getattr(listed, "tables", listed))

            def open_or_create(name: str, schema: Any) -> Any:
                if name in existing:
                    return self._db.open_table(name)
                return self._db.create_table(name, schema=schema)

            return (
                open_or_create(self._tools_table_name, tools_schema),
                open_or_create(self._skills_table_name, skills_schema),
                open_or_create(self._metadata_table_name, metadata_schema),
                self._tools_table_name in existing,
            )

        (
            self._tools_table,
            self._skills_table,
            self._metadata_table,
            tools_existed,
        ) = await asyncio.to_thread(open_tables)
        if tools_existed:
            await self._migrate_tools_schema(tools_schema)

        self._initialized = True

    def _collection(self, skills: bool) -> tuple[Any, str, Any]:
        """(table, JSON column, model) of the skills or the tools collection."""
        if skills:
            return self._skills_table, "skill_json", Skill
        return self._tools_table, "tool_json", ToolDefinition

    async def _add_items(
        self,
        items: list[Any],
        embeddings: list[list[float]],
        upsert: bool,
        *,
        skills: bool,
    ) -> int:
        """Add tools or skills with their embeddings; returns the number written."""
        if not items:
            return 0

        kind = "Skills" if skills else "Tools"
        if len(items) != len(embeddings):
            raise ValueError(
                f"{kind} and embeddings must have same length: "
                f"got {len(items)} {kind.lower()} and {len(embeddings)} embeddings"
            )
        for i, emb in enumerate(embeddings):
            if len(emb) != self._dimension:
                raise ValueError(
                    f"Embedding {i} has dimension {len(emb)}, expected {self._dimension}"
                )

        await self._ensure_initialized()
        table, _, _ = self._collection(skills)

        now = datetime.now(timezone.utc).isoformat()
        to_record = _skill_record if skills else _tool_record
        records = [to_record(item, embedding, now) for item, embedding in zip(items, embeddings)]

        if upsert:
            # One row per id, the last occurrence winning, as the in-memory
            # store, pgvector (ON CONFLICT), Qdrant and Chroma all behave. The
            # delete below clears the id once, so a repeat within the batch
            # was written twice and then returned twice from every search.
            records = list({record["id"]: record for record in records}.values())

        # Predicate over the batch's ids (escape for SQL safety)
        ids = [_escape_sql_string(record["id"]) for record in records]
        if len(ids) > 1:
            id_predicate = "id IN ({})".format(", ".join(f"'{id_}'" for id_ in ids))
        else:
            id_predicate = f"id = '{ids[0]}'"

        if upsert:
            try:
                await asyncio.to_thread(table.delete, id_predicate)
            except RuntimeError as e:
                # LanceDB raises RuntimeError when nothing matches
                logger.debug(f"Delete during upsert (expected if records don't exist): {e}")
        else:
            # Without upsert, ids already present (in the table or earlier in
            # this batch) are skipped and only the inserted rows are counted.
            existing = await asyncio.to_thread(
                table.search().select(["id"]).where(id_predicate).limit(None).to_list
            )
            seen_ids = {row["id"] for row in existing}
            deduped = []
            for record in records:
                if record["id"] in seen_ids:
                    continue
                seen_ids.add(record["id"])
                deduped.append(record)
            records = deduped
            if not records:
                return 0

        await asyncio.to_thread(table.add, records)
        return len(records)

    async def add_tools(
        self,
        tools: list[ToolDefinition],
        embeddings: list[list[float]],
        upsert: bool = True,
    ) -> int:
        """
        Add tools with their embeddings.

        Raises:
            ValueError: If tools and embeddings differ in length or an
                embedding does not match the configured dimension
        """
        return await self._add_items(tools, embeddings, upsert, skills=False)

    async def add_skills(
        self,
        skills: list[Skill],
        embeddings: list[list[float]],
        upsert: bool = True,
    ) -> int:
        """Add skills with their embeddings (same contract as :meth:`add_tools`)."""
        return await self._add_items(skills, embeddings, upsert, skills=True)

    async def search(
        self,
        query_vector: list[float],
        limit: int,
        filters: dict[str, Any] | None = None,
        score_threshold: float | None = None,
        include_embeddings: bool = False,
    ) -> list[tuple[ToolDefinition, float]] | list[tuple[ToolDefinition, float, list[float]]]:
        """
        Search for tools similar to the query vector.

        Args:
            query_vector: Query embedding vector
            limit: Maximum number of results
            filters: Optional filters (namespace, tags)
            score_threshold: Minimum similarity score (0-1, higher is better)
            include_embeddings: If True, return embeddings along with tools

        Returns:
            List of (tool, score) tuples if include_embeddings=False
            List of (tool, score, embedding) tuples if include_embeddings=True
        """
        await self._ensure_initialized()

        # Build search query. Only materialize the columns we need — without
        # .select() every row also deserializes the full embedding vector into
        # Python objects just to be discarded. `_distance` must be listed
        # explicitly: newer Lance versions stop auto-projecting it when output
        # columns are specified.
        columns = (
            ["tool_json", "vector", "_distance"]
            if include_embeddings
            else ["tool_json", "_distance"]
        )
        # The namespace predicate is needed both for the query and to size the
        # tag over-fetch below.
        where_clause: str | None = None
        if filters and "namespace" in filters:
            where_clause = _namespace_predicate(filters["namespace"])
            if where_clause is None:
                return []  # empty namespace list matches nothing

        # Pre-calculate required tags for faster set operations
        required_tags: set[str] = set()
        if filters and "tags" in filters:
            required_tags = set(filters["tags"])

        # Over-fetch for post-filtering. Tags live inside tool_json (there is
        # no tags column to push into the predicate), so a tag filter can only
        # be applied after deserialising a candidate — and a fixed 2x window
        # silently dropped tagged tools ranked below it. The window is widened
        # instead, doubling until enough tagged rows are found or the
        # namespace is exhausted, so the common case (tagged tools ranked near
        # the top) still costs one query rather than a full scan.
        fetch_limit = limit * 2
        max_rows = fetch_limit
        if required_tags:
            max_rows = int(await asyncio.to_thread(self._tools_table.count_rows, where_clause))
            if max_rows == 0:
                return []  # LanceDB rejects limit(0) on a vector query
            fetch_limit = min(max(limit * 4, 1), max_rows)

        def _build_search(size: int) -> Any:
            search = (
                self._tools_table.search(query_vector).metric("cosine").select(columns).limit(size)
            )
            return search.where(where_clause) if where_clause else search

        def _collect(rows: list[Any]) -> list[Any]:
            """Score, deserialise and tag-filter a page of rows, up to ``limit``."""
            collected: list[Any] = []
            for row in rows:
                # With the cosine metric LanceDB's ``_distance`` is 1 - cosine
                # similarity, whatever the vectors' length, so the score is the
                # cosine itself -- what every other store compares
                # ``score_threshold`` against. It used to be ``1 - d/2`` on the
                # default squared-L2 distance, clamped at 0.0: exact only for
                # unit vectors, and a clamp that made an anti-correlated tool
                # score 0 and so survive the ``score_threshold=0.0`` the
                # convenience layers use, where the other stores drop it.
                distance = row.get("_distance", 0)
                score = 1.0 - distance

                if score_threshold is not None and score < score_threshold:
                    continue

                # Deserialize tool (validate once; tags are checked on the model)
                tool_json_str = row.get("tool_json")
                if not tool_json_str:
                    logger.warning("Skipping row with missing tool_json field")
                    continue

                try:
                    tool = ToolDefinition.model_validate_json(tool_json_str)
                except Exception as e:
                    logger.warning(f"Failed to deserialize tool: {e}")
                    continue

                # Filter by tags if specified
                if required_tags and required_tags.isdisjoint(tool.tags):
                    continue

                if include_embeddings:
                    vector = row.get("vector")
                    embedding = list(vector) if vector is not None else []
                    collected.append((tool, score, embedding))
                else:
                    collected.append((tool, score))

                if len(collected) >= limit:
                    break
            return collected

        output: list[Any] = []
        while True:
            # Execute search off the event loop — LanceDB queries are
            # synchronous Rust/file I/O and would otherwise block every
            # concurrent coroutine.
            rows = await asyncio.to_thread(_build_search(fetch_limit).to_list)
            output = _collect(rows)
            if (
                len(output) >= limit
                or not required_tags
                or fetch_limit >= max_rows
                or len(rows) < fetch_limit
            ):
                # Enough matches, no tag filter to widen for, or the namespace
                # is exhausted — either way a wider window cannot add anything.
                break
            fetch_limit = min(fetch_limit * 2, max_rows)

        return output

    async def search_skills(
        self,
        query_vector: list[float],
        limit: int,
        filters: dict[str, Any] | None = None,
        score_threshold: float | None = None,
    ) -> list[tuple[Skill, float]]:
        """
        Search for skills similar to the query vector.

        Args:
            query_vector: Query embedding vector
            limit: Maximum number of results
            filters: Optional filters (namespace, category)
            score_threshold: Minimum similarity score

        Returns:
            List of (skill, score) tuples sorted by relevance
        """
        await self._ensure_initialized()

        # Project only the columns used below. Without .select() every row also
        # materializes its full embedding vector just to be discarded --
        # the same fix already applied to the tools search above. `_distance`
        # must be listed explicitly: newer Lance versions stop auto-projecting
        # it once output columns are specified.
        search = (
            self._skills_table.search(query_vector)
            .metric("cosine")
            .select(["skill_json", "_distance"])
            .limit(limit * 2)
        )

        # ONE combined predicate: LanceDB's .where() is a setter, not an
        # accumulator, so a second call would drop the first constraint.
        where_clauses: list[str] = []
        if filters and "namespace" in filters:
            predicate = _namespace_predicate(filters["namespace"])
            if predicate is None:
                return []  # empty namespace list matches nothing
            where_clauses.append(predicate)
        if filters and "category" in filters:
            escaped_cat = _escape_sql_string(filters["category"])
            where_clauses.append(f"category = '{escaped_cat}'")
        if where_clauses:
            search = search.where(" AND ".join(where_clauses))

        results = await asyncio.to_thread(search.to_list)

        output: list[tuple[Skill, float]] = []
        for row in results:
            distance = row.get("_distance", 0)
            score = 1.0 - distance  # cosine similarity; see the tools search

            if score_threshold is not None and score < score_threshold:
                continue

            # Deserialize skill with None check
            skill_json_str = row.get("skill_json")
            if not skill_json_str:
                logger.warning("Skipping row with missing skill_json field")
                continue

            try:
                skill = Skill.model_validate_json(skill_json_str)
            except Exception as e:
                logger.warning(f"Failed to deserialize skill: {e}")
                continue

            output.append((skill, score))

            if len(output) >= limit:
                break

        return output

    async def _get_item(self, name: str, namespace: str, *, skills: bool) -> Any:
        await self._ensure_initialized()
        _validate_identifier(name, "name")
        _validate_identifier(namespace, "namespace")
        table, json_field, model = self._collection(skills)
        item_id = _escape_sql_string(f"{namespace}.{name}")
        rows = await asyncio.to_thread(
            table.search().select([json_field]).where(f"id = '{item_id}'").limit(1).to_list
        )
        if not rows:
            return None
        raw = rows[0].get(json_field)
        if not raw:
            logger.warning(f"{namespace}.{name} has missing {json_field} field")
            return None
        return model.model_validate_json(raw)

    async def _delete_item(self, name: str, namespace: str, *, skills: bool) -> bool:
        await self._ensure_initialized()
        _validate_identifier(name, "name")
        _validate_identifier(namespace, "namespace")
        table, _, _ = self._collection(skills)
        predicate = f"id = '{_escape_sql_string(f'{namespace}.{name}')}'"
        # LanceDB's delete is a silent no-op on a miss, so count first.
        if not await asyncio.to_thread(table.count_rows, predicate):
            return False
        await asyncio.to_thread(table.delete, predicate)
        return True

    async def _count_items(self, namespace: str | None, *, skills: bool) -> int:
        await self._ensure_initialized()
        table, _, _ = self._collection(skills)
        if namespace is None:
            return int(await asyncio.to_thread(table.count_rows))
        _validate_identifier(namespace, "namespace")
        predicate = f"namespace = '{_escape_sql_string(namespace)}'"
        return int(await asyncio.to_thread(table.count_rows, predicate))

    async def _list_items(
        self,
        namespace: str | None,
        category: str | None,
        limit: int,
        offset: int,
        *,
        skills: bool,
    ) -> list[Any]:
        """List rows; table errors propagate, malformed rows are skipped."""
        await self._ensure_initialized()
        table, json_field, model = self._collection(skills)
        where_clauses = []
        if namespace is not None:
            _validate_identifier(namespace, "namespace")
            where_clauses.append(f"namespace = '{_escape_sql_string(namespace)}'")
        if category is not None:
            _validate_identifier(category, "category")
            where_clauses.append(f"category = '{_escape_sql_string(category)}'")
        # Only the JSON column is read; projecting it keeps the vector out of the scan.
        query = table.search().select([json_field])
        if where_clauses:
            query = query.where(" AND ".join(where_clauses))
        result = await asyncio.to_thread(query.limit(limit).offset(offset).to_arrow)

        items: list[Any] = []
        for raw in result[json_field].to_pylist():
            if not raw:
                continue
            try:
                items.append(model.model_validate_json(raw))
            except Exception as e:
                logger.warning(f"Skipping malformed {json_field} record: {e}")
        return items

    async def get_by_name(self, name: str, namespace: str = "default") -> ToolDefinition | None:
        """Get a tool by name, or None if not found."""
        return await self._get_item(name, namespace, skills=False)

    async def get_skill_by_name(self, name: str, namespace: str = "default") -> Skill | None:
        """Get a skill by name, or None if not found."""
        return await self._get_item(name, namespace, skills=True)

    async def delete(self, name: str, namespace: str = "default") -> bool:
        """Delete a tool; False if it was not found."""
        return await self._delete_item(name, namespace, skills=False)

    async def delete_skill(self, name: str, namespace: str = "default") -> bool:
        """Delete a skill; False if it was not found."""
        return await self._delete_item(name, namespace, skills=True)

    async def list_all(
        self,
        namespace: str | None = None,
        limit: int = 1000,
        offset: int = 0,
    ) -> list[ToolDefinition]:
        """List tools, optionally filtered by namespace."""
        return await self._list_items(namespace, None, limit, offset, skills=False)

    async def list_all_skills(
        self,
        namespace: str | None = None,
        category: str | None = None,
        limit: int = 1000,
        offset: int = 0,
    ) -> list[Skill]:
        """List skills, optionally filtered by namespace and category."""
        return await self._list_items(namespace, category, limit, offset, skills=True)

    async def count(self, namespace: str | None = None) -> int:
        """Count tools, optionally within a namespace."""
        return await self._count_items(namespace, skills=False)

    async def count_skills(self, namespace: str | None = None) -> int:
        """Count skills, optionally within a namespace."""
        return await self._count_items(namespace, skills=True)

    async def health_check(self) -> bool:
        """
        Check health of the vector store.

        Returns:
            True if database is accessible and operational

        Note:
            For detailed health information including migration status,
            use get_health_status() instead.
        """
        try:
            await self._ensure_initialized()
            # Verify tables exist and are queryable
            _ = await asyncio.to_thread(self._tools_table.count_rows)
            _ = await asyncio.to_thread(self._skills_table.count_rows)
            return True
        except Exception:
            return False

    async def get_health_status(self) -> dict[str, Any]:
        """
        Get detailed health status of the vector store.

        Returns detailed information about database health, including:
        - Basic health check (is database accessible)
        - Tool and skill counts
        - Schema migration status
        - Metadata consistency

        Returns:
            Dictionary with health status information:
            - healthy: bool - Overall health status
            - tool_count: int - Number of tools in database
            - skill_count: int - Number of skills in database
            - migration_needed: bool - Whether schema migration is needed
            - migration_status: str - "unknown", "up_to_date", "pending", or "failed"
            - schema_version: str - Current schema version info
            - embedder_id: str (optional) - Embedder ID from metadata if available
            - issues: list[str] - List of any detected issues

        Example:
            >>> status = await store.get_health_status()
            >>> if status["migration_needed"]:
            ...     print(f"Migration status: {status['migration_status']}")
        """
        status: dict[str, Any] = {
            "healthy": False,
            "tool_count": 0,
            "skill_count": 0,
            "migration_needed": False,
            "migration_status": "unknown",
            "schema_version": "v1.0",
            "issues": [],
        }

        try:
            await self._ensure_initialized()

            # Check basic health
            status["healthy"] = await self.health_check()
            if not status["healthy"]:
                status["issues"].append("Database is not accessible")
                return status

            # Get counts
            status["tool_count"] = await self.count()
            status["skill_count"] = await self.count_skills()

            # Check schema migration status
            try:
                current_schema = self._tools_table.schema
                current_field_names = {field.name for field in current_schema}

                # Expected fields in current schema version
                expected_fields = {
                    "id",
                    "name",
                    "namespace",
                    "description",
                    "tool_json",
                    "fingerprint",
                    "vector",
                    "created_at",
                    "updated_at",
                }

                missing_fields = expected_fields - current_field_names
                if missing_fields:
                    status["migration_needed"] = True
                    status["migration_status"] = "pending"
                    status["issues"].append(
                        f"Schema migration needed: missing fields {missing_fields}"
                    )
                else:
                    status["migration_status"] = "up_to_date"

            except Exception as e:
                status["migration_status"] = "failed"
                status["issues"].append(f"Schema check failed: {e}")

            # Check metadata consistency
            try:
                embedder_id = await self.get_metadata("embedder_id")
                stored_dimension = await self.get_metadata("dimension")

                if stored_dimension:
                    try:
                        stored_dim_int = int(stored_dimension)
                        if stored_dim_int <= 0:
                            status["issues"].append(
                                f"Invalid dimension metadata: '{stored_dimension}' "
                                f"must be a positive integer"
                            )
                        elif stored_dim_int != self._dimension:
                            status["issues"].append(
                                f"Dimension mismatch: stored={stored_dimension}, "
                                f"configured={self._dimension}"
                            )
                    except ValueError:
                        status["issues"].append(
                            f"Invalid dimension metadata: '{stored_dimension}' must be an integer"
                        )

                if embedder_id:
                    status["embedder_id"] = embedder_id
            except Exception as e:
                status["issues"].append(f"Metadata check failed: {e}")

        except Exception as e:
            status["healthy"] = False
            status["issues"].append(f"Health check error: {e}")

        return status

    async def _ensure_initialized(self) -> None:
        """Ensure the database is initialized."""
        if not self._initialized:
            await self.initialize()

    @property
    def db_path(self) -> str:
        """Return the database path."""
        return self._db_path

    @property
    def dimension(self) -> int:
        """Return the vector dimension."""
        return self._dimension
