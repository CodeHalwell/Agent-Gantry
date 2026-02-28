"""
LanceDB vector store adapter for Agent-Gantry.

Provides on-device, zero-config persistence with local LanceDB files,
supporting both tools and skills collections for semantic retrieval.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from agent_gantry.adapters.vector_stores.lancedb_mixins import (
    LanceDBMetadataMixin,
    LanceDBSkillsMixin,
    LanceDBToolsMixin,
    _escape_sql_string,
    _validate_identifier,
)

__all__ = ["LanceDBVectorStore", "_escape_sql_string", "_validate_identifier"]

logger = logging.getLogger(__name__)


class LanceDBVectorStore(LanceDBToolsMixin, LanceDBSkillsMixin, LanceDBMetadataMixin):
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
            import lancedb  # type: ignore[import-untyped]
            import pyarrow as pa  # type: ignore[import-untyped]
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
            _ = self._tools_table.count_rows()
            _ = self._skills_table.count_rows()
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
                    "id", "name", "namespace", "description", "tool_json",
                    "fingerprint", "vector", "created_at", "updated_at"
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
                            f"Invalid dimension metadata: '{stored_dimension}' "
                            f"must be an integer"
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

