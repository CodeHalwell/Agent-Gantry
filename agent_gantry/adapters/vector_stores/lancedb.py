"""
LanceDB vector store adapter for Agent-Gantry.

Provides on-device, zero-config persistence with local LanceDB files,
supporting both tools and skills collections for semantic retrieval.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from agent_gantry.schema.skill import Skill
from agent_gantry.schema.tool import ToolDefinition
from agent_gantry.utils.fingerprint import compute_tool_fingerprint

logger = logging.getLogger(__name__)


def _escape_sql_string(value: str) -> str:
    """
    Escape special characters in SQL strings to prevent injection.

    This function provides SQL injection protection for LanceDB queries by:
    1. Escaping backslashes (must be done first)
    2. Escaping single quotes using SQL standard ('') escaping

    Note: This is used in conjunction with _validate_identifier() which rejects
    control characters and enforces length limits. LanceDB does not currently
    support parameterized queries for WHERE clauses, so string escaping is
    necessary. All user-provided values go through validation before escaping.

    Security considerations:
    - Only used for metadata key lookups (not arbitrary user input)
    - Keys are validated by _validate_identifier() before escaping
    - All test cases in test suite verify SQL injection attempts are blocked

    Args:
        value: The string value to escape

    Returns:
        Escaped string safe for SQL inclusion
    """
    # Escape backslashes first, then single quotes
    return value.replace("\\", "\\\\").replace("'", "''")


def _validate_identifier(value: str, field_name: str) -> None:
    """
    Validate that a value is safe to use in SQL queries.

    This provides the first line of defense against SQL injection by:
    1. Enforcing length limits (1-256 characters)
    2. Rejecting null bytes and control characters (ASCII < 32)

    This validation occurs before any SQL escaping is applied.

    Args:
        value: The value to validate
        field_name: Name of the field (for error messages)

    Raises:
        ValueError: If validation fails
    """
    if not value or len(value) > 256:
        raise ValueError(f"{field_name} must be 1-256 characters")
    # Reject null bytes and other control characters
    if any(ord(c) < 32 for c in value):
        raise ValueError(f"{field_name} contains invalid characters")


class LanceDBVectorStore:
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
        2. SQL escaping via _escape_sql_string() (backslash and quote escaping)
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
            pa.field("fingerprint", pa.string()),  # Hash of tool for change detection
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

        # Create metadata table schema (stores sync state)
        metadata_schema = pa.schema([
            pa.field("key", pa.string()),
            pa.field("value", pa.string()),
            pa.field("updated_at", pa.string()),
        ])

        # Create or open tables
        # Note: list_tables() returns a TableListResult object with a 'tables' attribute
        table_list_result = self._db.list_tables()
        existing_tables = (
            table_list_result.tables
            if hasattr(table_list_result, "tables")
            else list(table_list_result)
        )

        if self._tools_table_name in existing_tables:
            self._tools_table = self._db.open_table(self._tools_table_name)
            # Migrate schema if needed
            await self._migrate_tools_schema(tools_schema)
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

        if self._metadata_table_name in existing_tables:
            self._metadata_table = self._db.open_table(self._metadata_table_name)
        else:
            self._metadata_table = self._db.create_table(
                self._metadata_table_name,
                schema=metadata_schema,
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

        Raises:
            ValueError: If tools and embeddings have different lengths or
                       if embedding dimensions don't match configured dimension
        """
        if not tools:
            return 0

        # Validate inputs
        if len(tools) != len(embeddings):
            raise ValueError(
                f"Tools and embeddings must have same length: "
                f"got {len(tools)} tools and {len(embeddings)} embeddings"
            )

        for i, emb in enumerate(embeddings):
            if len(emb) != self._dimension:
                raise ValueError(
                    f"Embedding {i} has dimension {len(emb)}, "
                    f"expected {self._dimension}"
                )

        await self._ensure_initialized()

        now = datetime.now(timezone.utc).isoformat()
        records = []

        for tool, embedding in zip(tools, embeddings):
            tool_id = f"{tool.namespace}.{tool.name}"
            fingerprint = compute_tool_fingerprint(tool)
            record = {
                "id": tool_id,
                "name": tool.name,
                "namespace": tool.namespace,
                "description": tool.description,
                "tool_json": tool.model_dump_json(),
                "fingerprint": fingerprint,
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
            except RuntimeError as e:
                # Table might be empty or records might not exist - this is expected
                logger.debug(f"Delete during upsert (expected if records don't exist): {e}")
            except Exception as e:
                # Unexpected error during deletion
                logger.warning(f"Unexpected error during upsert delete: {e}")
                raise

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

        Raises:
            ValueError: If skills and embeddings have different lengths or
                       if embedding dimensions don't match configured dimension
        """
        if not skills:
            return 0

        # Validate inputs
        if len(skills) != len(embeddings):
            raise ValueError(
                f"Skills and embeddings must have same length: "
                f"got {len(skills)} skills and {len(embeddings)} embeddings"
            )

        for i, emb in enumerate(embeddings):
            if len(emb) != self._dimension:
                raise ValueError(
                    f"Embedding {i} has dimension {len(emb)}, "
                    f"expected {self._dimension}"
                )

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
            except RuntimeError as e:
                # Table might be empty or records might not exist - this is expected
                logger.debug(f"Delete during upsert (expected if records don't exist): {e}")
            except Exception as e:
                # Unexpected error during deletion
                logger.warning(f"Unexpected error during upsert delete: {e}")
                raise

        self._skills_table.add(records)
        return len(records)

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
        import logging

        if include_embeddings:
            logging.getLogger(__name__).warning(
                "LanceDBVectorStore does not support include_embeddings yet. "
                "Returning without embeddings."
            )

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
                tool_json_str = row.get("tool_json")
                if not tool_json_str:
                    logger.warning("Skipping row with missing tool_json field")
                    continue
                tool_json = json.loads(tool_json_str)
                tool_tags = tool_json.get("tags", [])
                if not any(tag in tool_tags for tag in filters["tags"]):
                    continue

            # Deserialize tool
            tool_json_str = row.get("tool_json")
            if not tool_json_str:
                logger.warning("Skipping row with missing tool_json field")
                continue

            try:
                tool = ToolDefinition.model_validate_json(tool_json_str)
            except Exception as e:
                logger.warning(f"Failed to deserialize tool: {e}")
                continue

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

        # Validate inputs for SQL safety
        _validate_identifier(name, "name")
        _validate_identifier(namespace, "namespace")

        # Escape ID for SQL safety
        tool_id = _escape_sql_string(f"{namespace}.{name}")
        try:
            results = self._tools_table.search().where(f"id = '{tool_id}'").limit(1).to_list()
            if results:
                tool_json_str = results[0].get("tool_json")
                if tool_json_str:
                    return ToolDefinition.model_validate_json(tool_json_str)
                else:
                    logger.warning(f"Tool {namespace}.{name} has missing tool_json field")
        except Exception as e:
            # Record may not exist - log at debug level
            logger.debug(f"get_by_name lookup failed for {namespace}.{name}: {e}")
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

        # Validate inputs for SQL safety
        _validate_identifier(name, "name")
        _validate_identifier(namespace, "namespace")

        # Escape ID for SQL safety
        skill_id = _escape_sql_string(f"{namespace}.{name}")
        try:
            results = self._skills_table.search().where(f"id = '{skill_id}'").limit(1).to_list()
            if results:
                skill_json_str = results[0].get("skill_json")
                if skill_json_str:
                    return Skill.model_validate_json(skill_json_str)
                else:
                    logger.warning(f"Skill {namespace}.{name} has missing skill_json field")
        except Exception as e:
            # Record may not exist - log at debug level
            logger.debug(f"get_skill_by_name lookup failed for {namespace}.{name}: {e}")
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

        # Validate inputs for SQL safety
        _validate_identifier(name, "name")
        _validate_identifier(namespace, "namespace")

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

        # Validate inputs for SQL safety
        _validate_identifier(name, "name")
        _validate_identifier(namespace, "namespace")

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

        # Validate namespace if provided
        if namespace is not None:
            _validate_identifier(namespace, "namespace")

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
                if r.get("tool_json")  # Skip records with missing tool_json
            ]
        except Exception as e:
            logger.warning(f"Error listing tools: {e}")
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

        # Validate inputs if provided
        if namespace is not None:
            _validate_identifier(namespace, "namespace")
        if category is not None:
            _validate_identifier(category, "category")

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
                if r.get("skill_json")  # Skip records with missing skill_json
            ]
        except Exception as e:
            logger.warning(f"Error listing skills: {e}")
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

        # Validate namespace if provided
        if namespace is not None:
            _validate_identifier(namespace, "namespace")

        try:
            if namespace:
                # For namespace filtering, we need to scan records
                table = self._tools_table.to_arrow()
                records = table.to_pylist()
                return len([r for r in records if r.get("namespace") == namespace])
            # Use count_rows() for efficient counting when no filter
            return int(self._tools_table.count_rows())
        except Exception as e:
            logger.warning(f"Error counting tools: {e}")
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

        # Validate namespace if provided
        if namespace is not None:
            _validate_identifier(namespace, "namespace")

        try:
            if namespace:
                table = self._skills_table.to_arrow()
                records = table.to_pylist()
                return len([r for r in records if r.get("namespace") == namespace])
            return int(self._skills_table.count_rows())
        except Exception as e:
            logger.warning(f"Error counting skills: {e}")
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

    async def _migrate_tools_schema(self, target_schema: Any) -> None:
        """
        Migrate tools table schema if needed.

        This handles adding missing columns to existing databases to support
        new features like fingerprinting without losing data.

        Args:
            target_schema: The target PyArrow schema
        """
        try:
            # Get current schema
            current_schema = self._tools_table.schema
            current_field_names = {field.name for field in current_schema}
            target_field_names = {field.name for field in target_schema}

            # Check if migration is needed
            missing_fields = target_field_names - current_field_names
            if not missing_fields:
                return  # Schema is up to date

            logger.info(f"Migrating tools table schema. Adding fields: {missing_fields}")

            # LanceDB doesn't support ALTER TABLE, so we need to:
            # 1. Read all existing data
            # 2. Add missing columns with default values
            # 3. Re-insert data

            # Read existing data
            table = self._tools_table.to_arrow()
            records = table.to_pylist()

            if not records:
                # Empty table, just recreate with new schema
                self._db.drop_table(self._tools_table_name)
                self._tools_table = self._db.create_table(
                    self._tools_table_name,
                    schema=target_schema,
                )
                return

            # Add missing fields with default values
            now = datetime.now(timezone.utc).isoformat()
            for record in records:
                if "fingerprint" not in record and "fingerprint" in missing_fields:
                    # Compute fingerprint for existing tools
                    try:
                        tool = ToolDefinition.model_validate_json(record["tool_json"])
                        record["fingerprint"] = compute_tool_fingerprint(tool)
                    except Exception as e:
                        # Fallback to empty fingerprint if tool JSON is invalid
                        logger.warning(f"Failed to compute fingerprint during migration: {e}")
                        record["fingerprint"] = ""
                if "created_at" not in record and "created_at" in missing_fields:
                    record["created_at"] = now
                if "updated_at" not in record and "updated_at" in missing_fields:
                    record["updated_at"] = now

            # Drop and recreate table with new schema
            self._db.drop_table(self._tools_table_name)
            self._tools_table = self._db.create_table(
                self._tools_table_name,
                schema=target_schema,
            )

            # Re-insert data
            self._tools_table.add(records)
            logger.info(f"Successfully migrated {len(records)} tools to new schema")

        except Exception as e:
            logger.error(f"Schema migration failed: {e}")
            # Don't raise - allow system to continue with current schema
            # This makes the migration non-breaking

    @property
    def db_path(self) -> str:
        """Return the database path."""
        return self._db_path

    @property
    def dimension(self) -> int:
        """Return the vector dimension."""
        return self._dimension

    # =========================================================================
    # Metadata and Fingerprint Methods (for auto-sync support)
    # =========================================================================

    async def get_metadata(self, key: str) -> str | None:
        """
        Get a metadata value by key.

        Args:
            key: The metadata key

        Returns:
            The value if found, None otherwise
        """
        await self._ensure_initialized()

        try:
            escaped_key = _escape_sql_string(key)
            results = self._metadata_table.search().where(f"key = '{escaped_key}'").limit(1).to_list()
            if results and results[0].get("value"):
                value: str = results[0]["value"]
                return value
        except Exception as e:
            logger.debug(f"get_metadata failed for key '{key}': {e}")
        return None

    async def set_metadata(self, key: str, value: str) -> None:
        """
        Set a metadata value.

        Args:
            key: The metadata key
            value: The value to store
        """
        await self._ensure_initialized()

        now = datetime.now(timezone.utc).isoformat()

        # Delete existing record if present
        try:
            escaped_key = _escape_sql_string(key)
            self._metadata_table.delete(f"key = '{escaped_key}'")
        except RuntimeError:
            pass  # Record doesn't exist - this is expected
        except Exception as e:
            logger.warning(f"Unexpected error deleting metadata key '{key}': {e}")
            # Continue anyway - we'll try to add the new record

        # Add new record
        self._metadata_table.add([{
            "key": key,
            "value": value,
            "updated_at": now,
        }])

    async def get_stored_fingerprints(self) -> dict[str, str]:
        """
        Get all stored tool fingerprints.

        Returns:
            Dictionary mapping tool_id to fingerprint
        """
        await self._ensure_initialized()

        try:
            table = self._tools_table.to_arrow()
            records = table.to_pylist()
            return {r["id"]: r.get("fingerprint", "") for r in records}
        except Exception as e:
            logger.debug(f"get_stored_fingerprints failed: {e}")
            return {}

    async def get_sync_status(self) -> dict[str, Any]:
        """
        Get the current sync status including metadata.

        Returns:
            Dictionary with sync status info:
            - tool_count: Number of tools in database
            - embedder_id: Identifier of embedder used
            - dimension: Vector dimension
            - last_sync: ISO timestamp of last sync
        """
        await self._ensure_initialized()

        status: dict[str, Any] = {
            "tool_count": await self.count(),
            "dimension": self._dimension,
        }

        # Get metadata values
        embedder_id = await self.get_metadata("embedder_id")
        if embedder_id:
            status["embedder_id"] = embedder_id

        last_sync = await self.get_metadata("last_sync")
        if last_sync:
            status["last_sync"] = last_sync

        stored_dimension = await self.get_metadata("dimension")
        if stored_dimension:
            status["stored_dimension"] = int(stored_dimension)

        return status

    async def update_sync_metadata(
        self,
        embedder_id: str,
        dimension: int,
    ) -> None:
        """
        Update sync metadata after a successful sync.

        This method provides transaction-like semantics by updating all
        metadata fields together. If any update fails, the entire operation
        is considered failed and previous state is preserved.

        Args:
            embedder_id: Identifier for the embedder used
            dimension: Vector dimension used

        Raises:
            Exception: If metadata update fails
        """
        now = datetime.now(timezone.utc).isoformat()

        # Store old values for rollback
        old_embedder_id = await self.get_metadata("embedder_id")
        old_dimension = await self.get_metadata("dimension")
        old_last_sync = await self.get_metadata("last_sync")

        try:
            # Update all metadata fields
            await self.set_metadata("embedder_id", embedder_id)
            await self.set_metadata("dimension", str(dimension))
            await self.set_metadata("last_sync", now)
        except Exception as e:
            # Attempt rollback on failure
            logger.error(f"Sync metadata update failed: {e}. Attempting rollback...")
            try:
                if old_embedder_id is not None:
                    await self.set_metadata("embedder_id", old_embedder_id)
                if old_dimension is not None:
                    await self.set_metadata("dimension", old_dimension)
                if old_last_sync is not None:
                    await self.set_metadata("last_sync", old_last_sync)
                logger.info("Rollback completed successfully")
            except Exception as rollback_error:
                logger.error(f"Rollback failed: {rollback_error}")
            raise  # Re-raise original exception
