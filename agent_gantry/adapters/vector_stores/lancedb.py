"""
LanceDB vector store adapter for Agent-Gantry.

Provides on-device, zero-config persistence with local LanceDB files,
supporting both tools and skills collections for semantic retrieval.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from agent_gantry.schema.skill import Skill
from agent_gantry.schema.tool import ToolDefinition


def _escape_sql_string(value: str) -> str:
    """Escape single quotes in SQL strings to prevent SQL injection."""
    return value.replace("'", "''")


class LanceDBVectorStore:
    """
    LanceDB vector store for on-device semantic indexing.

    Provides SQLite-like local persistence for tools and skills with
    high-speed, low-memory vector search. Supports zero-config setup
    with automatic database creation.

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
        self._dimension = dimension
        self._db: Any = None
        self._tools_table: Any = None
        self._skills_table: Any = None
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
            import lancedb
            import pyarrow as pa
        except ImportError as e:
            raise ImportError(
                "lancedb and pyarrow are required. "
                "Install with: pip install lancedb pyarrow"
            ) from e

        # Create database directory
        db_dir = Path(self._db_path)
        db_dir.mkdir(parents=True, exist_ok=True)

        # Connect to database
        self._db = lancedb.connect(str(db_dir))

        # Create tools table schema
        tools_schema = pa.schema([
            pa.field("id", pa.string()),
            pa.field("name", pa.string()),
            pa.field("namespace", pa.string()),
            pa.field("description", pa.string()),
            pa.field("tool_json", pa.string()),  # Full serialized ToolDefinition
            pa.field("vector", pa.list_(pa.float32(), self._dimension)),
            pa.field("created_at", pa.string()),
            pa.field("updated_at", pa.string()),
        ])

        # Create skills table schema
        skills_schema = pa.schema([
            pa.field("id", pa.string()),
            pa.field("name", pa.string()),
            pa.field("namespace", pa.string()),
            pa.field("description", pa.string()),
            pa.field("category", pa.string()),
            pa.field("skill_json", pa.string()),  # Full serialized Skill
            pa.field("vector", pa.list_(pa.float32(), self._dimension)),
            pa.field("created_at", pa.string()),
            pa.field("updated_at", pa.string()),
        ])

        # Create or open tables
        existing_tables = self._db.list_tables()

        if self._tools_table_name in existing_tables:
            self._tools_table = self._db.open_table(self._tools_table_name)
        else:
            self._tools_table = self._db.create_table(
                self._tools_table_name,
                schema=tools_schema,
            )

        if self._skills_table_name in existing_tables:
            self._skills_table = self._db.open_table(self._skills_table_name)
        else:
            self._skills_table = self._db.create_table(
                self._skills_table_name,
                schema=skills_schema,
            )

        self._initialized = True

    async def add_tools(
        self,
        tools: list[ToolDefinition],
        embeddings: list[list[float]],
        upsert: bool = True,
    ) -> int:
        """
        Add tools with their embeddings.

        Args:
            tools: List of tool definitions
            embeddings: List of embedding vectors
            upsert: Whether to update existing tools (default True)

        Returns:
            Number of tools added/updated
        """
        if not tools:
            return 0

        await self._ensure_initialized()

        now = datetime.now(timezone.utc).isoformat()
        records = []

        for tool, embedding in zip(tools, embeddings):
            tool_id = f"{tool.namespace}.{tool.name}"
            record = {
                "id": tool_id,
                "name": tool.name,
                "namespace": tool.namespace,
                "description": tool.description,
                "tool_json": tool.model_dump_json(),
                "vector": embedding,
                "created_at": now,
                "updated_at": now,
            }
            records.append(record)

        if upsert:
            # Delete existing records with same IDs (escape for SQL safety)
            ids = [_escape_sql_string(f"{t.namespace}.{t.name}") for t in tools]
            try:
                if len(ids) > 1:
                    escaped_ids = ", ".join(f"'{id_}'" for id_ in ids)
                    self._tools_table.delete(f"id IN ({escaped_ids})")
                else:
                    self._tools_table.delete(f"id = '{ids[0]}'")
            except Exception:
                pass  # Table might be empty

        self._tools_table.add(records)
        return len(records)

    async def add_skills(
        self,
        skills: list[Skill],
        embeddings: list[list[float]],
        upsert: bool = True,
    ) -> int:
        """
        Add skills with their embeddings.

        Args:
            skills: List of skill definitions
            embeddings: List of embedding vectors
            upsert: Whether to update existing skills (default True)

        Returns:
            Number of skills added/updated
        """
        if not skills:
            return 0

        await self._ensure_initialized()

        now = datetime.now(timezone.utc).isoformat()
        records = []

        for skill, embedding in zip(skills, embeddings):
            skill_id = f"{skill.namespace}.{skill.name}"
            record = {
                "id": skill_id,
                "name": skill.name,
                "namespace": skill.namespace,
                "description": skill.description,
                "category": skill.category.value,
                "skill_json": skill.model_dump_json(),
                "vector": embedding,
                "created_at": now,
                "updated_at": now,
            }
            records.append(record)

        if upsert:
            # Delete existing records with same IDs (escape for SQL safety)
            ids = [_escape_sql_string(f"{s.namespace}.{s.name}") for s in skills]
            try:
                if len(ids) > 1:
                    escaped_ids = ", ".join(f"'{id_}'" for id_ in ids)
                    self._skills_table.delete(f"id IN ({escaped_ids})")
                else:
                    self._skills_table.delete(f"id = '{ids[0]}'")
            except Exception:
                pass

        self._skills_table.add(records)
        return len(records)

    async def search(
        self,
        query_vector: list[float],
        limit: int,
        filters: dict[str, Any] | None = None,
        score_threshold: float | None = None,
    ) -> list[tuple[ToolDefinition, float]]:
        """
        Search for tools similar to the query vector.

        Args:
            query_vector: Query embedding vector
            limit: Maximum number of results
            filters: Optional filters (namespace, tags)
            score_threshold: Minimum similarity score (0-1, higher is better)

        Returns:
            List of (tool, score) tuples sorted by relevance
        """
        await self._ensure_initialized()

        # Build search query
        search = self._tools_table.search(query_vector).limit(limit * 2)  # Over-fetch for filtering

        # Apply namespace filter if specified (escape for SQL safety)
        if filters and "namespace" in filters:
            ns_filter = filters["namespace"]
            if isinstance(ns_filter, (list, tuple, set)):
                ns_list = list(ns_filter)
                if len(ns_list) == 1:
                    escaped_ns = _escape_sql_string(ns_list[0])
                    search = search.where(f"namespace = '{escaped_ns}'")
                else:
                    escaped_values = ", ".join(f"'{_escape_sql_string(ns)}'" for ns in ns_list)
                    search = search.where(f"namespace IN ({escaped_values})")
            else:
                escaped_ns = _escape_sql_string(ns_filter)
                search = search.where(f"namespace = '{escaped_ns}'")

        # Execute search
        results = search.to_list()

        # Process results
        output: list[tuple[ToolDefinition, float]] = []
        for row in results:
            # LanceDB returns distance (lower is better), convert to similarity
            distance = row.get("_distance", 0)
            # Convert L2 distance to cosine similarity approximation
            score = max(0.0, 1.0 - (distance / 2.0))

            if score_threshold is not None and score < score_threshold:
                continue

            # Filter by tags if specified
            if filters and "tags" in filters:
                tool_json = json.loads(row["tool_json"])
                tool_tags = tool_json.get("tags", [])
                if not any(tag in tool_tags for tag in filters["tags"]):
                    continue

            # Deserialize tool
            tool = ToolDefinition.model_validate_json(row["tool_json"])
            output.append((tool, score))

            if len(output) >= limit:
                break

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

        search = self._skills_table.search(query_vector).limit(limit * 2)

        # Apply namespace filter (escape for SQL safety)
        if filters and "namespace" in filters:
            ns_filter = filters["namespace"]
            if isinstance(ns_filter, (list, tuple, set)):
                ns_list = list(ns_filter)
                if len(ns_list) == 1:
                    escaped_ns = _escape_sql_string(ns_list[0])
                    search = search.where(f"namespace = '{escaped_ns}'")
                else:
                    escaped_values = ", ".join(f"'{_escape_sql_string(ns)}'" for ns in ns_list)
                    search = search.where(f"namespace IN ({escaped_values})")
            else:
                escaped_ns = _escape_sql_string(ns_filter)
                search = search.where(f"namespace = '{escaped_ns}'")

        # Apply category filter (escape for SQL safety)
        if filters and "category" in filters:
            escaped_cat = _escape_sql_string(filters["category"])
            search = search.where(f"category = '{escaped_cat}'")

        results = search.to_list()

        output: list[tuple[Skill, float]] = []
        for row in results:
            distance = row.get("_distance", 0)
            score = max(0.0, 1.0 - (distance / 2.0))

            if score_threshold is not None and score < score_threshold:
                continue

            skill = Skill.model_validate_json(row["skill_json"])
            output.append((skill, score))

            if len(output) >= limit:
                break

        return output

    async def get_by_name(
        self, name: str, namespace: str = "default"
    ) -> ToolDefinition | None:
        """
        Get a tool by name.

        Args:
            name: Tool name
            namespace: Tool namespace

        Returns:
            Tool definition if found, None otherwise
        """
        await self._ensure_initialized()

        # Escape ID for SQL safety
        tool_id = _escape_sql_string(f"{namespace}.{name}")
        try:
            results = self._tools_table.search().where(f"id = '{tool_id}'").limit(1).to_list()
            if results:
                return ToolDefinition.model_validate_json(results[0]["tool_json"])
        except Exception:
            pass
        return None

    async def get_skill_by_name(
        self, name: str, namespace: str = "default"
    ) -> Skill | None:
        """
        Get a skill by name.

        Args:
            name: Skill name
            namespace: Skill namespace

        Returns:
            Skill definition if found, None otherwise
        """
        await self._ensure_initialized()

        # Escape ID for SQL safety
        skill_id = _escape_sql_string(f"{namespace}.{name}")
        try:
            results = self._skills_table.search().where(f"id = '{skill_id}'").limit(1).to_list()
            if results:
                return Skill.model_validate_json(results[0]["skill_json"])
        except Exception:
            pass
        return None

    async def delete(self, name: str, namespace: str = "default") -> bool:
        """
        Delete a tool.

        Args:
            name: Tool name
            namespace: Tool namespace

        Returns:
            True if deleted, False if not found
        """
        await self._ensure_initialized()

        # Escape ID for SQL safety
        tool_id = _escape_sql_string(f"{namespace}.{name}")
        try:
            self._tools_table.delete(f"id = '{tool_id}'")
            return True
        except Exception:
            return False

    async def delete_skill(self, name: str, namespace: str = "default") -> bool:
        """
        Delete a skill.

        Args:
            name: Skill name
            namespace: Skill namespace

        Returns:
            True if deleted, False if not found
        """
        await self._ensure_initialized()

        # Escape ID for SQL safety
        skill_id = _escape_sql_string(f"{namespace}.{name}")
        try:
            self._skills_table.delete(f"id = '{skill_id}'")
            return True
        except Exception:
            return False

    async def list_all(
        self,
        namespace: str | None = None,
        limit: int = 1000,
        offset: int = 0,
    ) -> list[ToolDefinition]:
        """
        List all tools.

        Args:
            namespace: Filter by namespace (None for all)
            limit: Maximum results
            offset: Pagination offset

        Returns:
            List of tool definitions
        """
        await self._ensure_initialized()

        try:
            # Use to_arrow for listing (doesn't require pandas)
            table = self._tools_table.to_arrow()
            records = table.to_pylist()

            # Filter by namespace if specified
            if namespace:
                records = [r for r in records if r.get("namespace") == namespace]

            # Apply pagination
            records = records[offset : offset + limit]

            return [
                ToolDefinition.model_validate_json(r["tool_json"])
                for r in records
            ]
        except Exception:
            return []

    async def list_all_skills(
        self,
        namespace: str | None = None,
        category: str | None = None,
        limit: int = 1000,
        offset: int = 0,
    ) -> list[Skill]:
        """
        List all skills.

        Args:
            namespace: Filter by namespace
            category: Filter by category
            limit: Maximum results
            offset: Pagination offset

        Returns:
            List of skill definitions
        """
        await self._ensure_initialized()

        try:
            table = self._skills_table.to_arrow()
            records = table.to_pylist()

            # Filter by namespace and category
            if namespace:
                records = [r for r in records if r.get("namespace") == namespace]
            if category:
                records = [r for r in records if r.get("category") == category]

            # Apply pagination
            records = records[offset : offset + limit]

            return [
                Skill.model_validate_json(r["skill_json"])
                for r in records
            ]
        except Exception:
            return []

    async def count(self, namespace: str | None = None) -> int:
        """
        Count tools.

        Args:
            namespace: Filter by namespace

        Returns:
            Number of tools
        """
        await self._ensure_initialized()

        try:
            if namespace:
                # For namespace filtering, we need to scan records
                table = self._tools_table.to_arrow()
                records = table.to_pylist()
                return len([r for r in records if r.get("namespace") == namespace])
            # Use count_rows() for efficient counting when no filter
            return self._tools_table.count_rows()
        except Exception:
            return 0

    async def count_skills(self, namespace: str | None = None) -> int:
        """
        Count skills.

        Args:
            namespace: Filter by namespace

        Returns:
            Number of skills
        """
        await self._ensure_initialized()

        try:
            if namespace:
                table = self._skills_table.to_arrow()
                records = table.to_pylist()
                return len([r for r in records if r.get("namespace") == namespace])
            return self._skills_table.count_rows()
        except Exception:
            return 0

    async def health_check(self) -> bool:
        """
        Check health of the vector store.

        Returns:
            True if database is accessible and operational
        """
        try:
            await self._ensure_initialized()
            # Verify tables exist and are queryable
            _ = self._tools_table.count_rows()
            _ = self._skills_table.count_rows()
            return True
        except Exception:
            return False

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
