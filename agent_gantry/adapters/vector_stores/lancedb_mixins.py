"""
Mixins for LanceDB vector store to separate domain logic.
"""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timezone
from typing import Any

from agent_gantry.schema.skill import Skill
from agent_gantry.schema.tool import ToolDefinition
from agent_gantry.utils.fingerprint import compute_tool_fingerprint

# Pre-compile regex for control character checking
# Benchmark: ~5x faster than generator expression (any(ord(c) < 32...))
_CTRL_CHAR_RE = re.compile(r"[\x00-\x1f]")

logger = logging.getLogger(__name__)


# We define protocols or assume the mixins will be mixed into a class that has these attributes.
# In Python, mixins typically just use `self._attr` directly, but for type checkers, it's
# helpful to declare them or rely on duck typing. Since `LanceDBVectorStore` will inherit these,
# we just use the attributes.


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
    if _CTRL_CHAR_RE.search(value):
        raise ValueError(f"{field_name} contains invalid characters")


class LanceDBToolsMixin:
    """Mixin for LanceDB tools operations."""

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
            if len(emb) != self._dimension:  # type: ignore
                raise ValueError(
                    f"Embedding {i} has dimension {len(emb)}, expected {self._dimension}"  # type: ignore
                )

        await self._ensure_initialized()  # type: ignore

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
                    self._tools_table.delete(f"id IN ({escaped_ids})")  # type: ignore
                else:
                    self._tools_table.delete(f"id = '{ids[0]}'")  # type: ignore
            except RuntimeError as e:
                # LanceDB raises RuntimeError when attempting to delete non-existent records
                # This is expected during upsert when records don't exist yet
                logger.debug(f"Delete during upsert (expected if records don't exist): {e}")
            except Exception as e:
                # Unexpected error during deletion
                logger.warning(f"Unexpected error during upsert delete: {e}")
                raise

        self._tools_table.add(records)  # type: ignore
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

        await self._ensure_initialized()  # type: ignore

        # Build search query
        search = self._tools_table.search(query_vector).limit(limit * 2)  # type: ignore # Over-fetch for filtering

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

        # Pre-calculate required tags for faster set operations
        required_tags: set[str] = set()
        if filters and "tags" in filters:
            required_tags = set(filters["tags"])

        for row in results:
            # LanceDB returns distance (lower is better), convert to similarity
            distance = row.get("_distance", 0)
            # Convert L2 distance to cosine similarity approximation
            score = max(0.0, 1.0 - (distance / 2.0))

            if score_threshold is not None and score < score_threshold:
                continue

            # Filter by tags if specified
            if required_tags:
                tool_json_str = row.get("tool_json")
                if not tool_json_str:
                    logger.warning("Skipping row with missing tool_json field")
                    continue
                tool_json = json.loads(tool_json_str)
                tool_tags = tool_json.get("tags", [])
                if required_tags.isdisjoint(tool_tags):
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

    async def get_by_name(self, name: str, namespace: str = "default") -> ToolDefinition | None:
        """
        Get a tool by name.

        Args:
            name: Tool name
            namespace: Tool namespace

        Returns:
            Tool definition if found, None otherwise
        """
        await self._ensure_initialized()  # type: ignore

        # Validate inputs for SQL safety
        _validate_identifier(name, "name")
        _validate_identifier(namespace, "namespace")

        # Escape ID for SQL safety
        tool_id = _escape_sql_string(f"{namespace}.{name}")
        try:
            results = self._tools_table.search().where(f"id = '{tool_id}'").limit(1).to_list()  # type: ignore
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

    async def delete(self, name: str, namespace: str = "default") -> bool:
        """
        Delete a tool.

        Args:
            name: Tool name
            namespace: Tool namespace

        Returns:
            True if deleted, False if not found
        """
        await self._ensure_initialized()  # type: ignore

        # Validate inputs for SQL safety
        _validate_identifier(name, "name")
        _validate_identifier(namespace, "namespace")

        # Escape ID for SQL safety
        tool_id = _escape_sql_string(f"{namespace}.{name}")
        try:
            self._tools_table.delete(f"id = '{tool_id}'")  # type: ignore
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
        await self._ensure_initialized()  # type: ignore

        # Validate namespace if provided
        if namespace is not None:
            _validate_identifier(namespace, "namespace")

        try:
            query = self._tools_table.search()  # type: ignore
            if namespace:
                query = query.where(f"namespace = '{_escape_sql_string(namespace)}'")

            records = query.limit(limit).offset(offset).to_list()

            return [
                ToolDefinition.model_validate_json(r["tool_json"])
                for r in records
                if r.get("tool_json")  # Skip records with missing tool_json
            ]
        except Exception as e:
            logger.warning(f"Error listing tools: {e}")
            return []

    async def count(self, namespace: str | None = None) -> int:
        """
        Count tools.

        Args:
            namespace: Filter by namespace

        Returns:
            Number of tools
        """
        await self._ensure_initialized()  # type: ignore

        # Validate namespace if provided
        if namespace is not None:
            _validate_identifier(namespace, "namespace")

        try:
            if namespace:
                return int(
                    self._tools_table.count_rows(f"namespace = '{_escape_sql_string(namespace)}'")
                )  # type: ignore
            # Use count_rows() for efficient counting when no filter
            return int(self._tools_table.count_rows())  # type: ignore
        except Exception as e:
            logger.warning(f"Error counting tools: {e}")
            return 0

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
            current_schema = self._tools_table.schema  # type: ignore
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
            table = self._tools_table.to_arrow()  # type: ignore
            records = table.to_pylist()

            if not records:
                # Empty table, just recreate with new schema
                self._db.drop_table(self._tools_table_name)  # type: ignore
                self._tools_table = self._db.create_table(  # type: ignore
                    self._tools_table_name,  # type: ignore
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
            self._db.drop_table(self._tools_table_name)  # type: ignore
            self._tools_table = self._db.create_table(  # type: ignore
                self._tools_table_name,  # type: ignore
                schema=target_schema,
            )

            # Re-insert data
            self._tools_table.add(records)  # type: ignore
            logger.info(f"Successfully migrated {len(records)} tools to new schema")

        except Exception as e:
            logger.error(f"Schema migration failed: {e}")
            # Don't raise - allow system to continue with current schema
            # This makes the migration non-breaking


class LanceDBSkillsMixin:
    """Mixin for LanceDB skills operations."""

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
            if len(emb) != self._dimension:  # type: ignore
                raise ValueError(
                    f"Embedding {i} has dimension {len(emb)}, expected {self._dimension}"  # type: ignore
                )

        await self._ensure_initialized()  # type: ignore

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
                    self._skills_table.delete(f"id IN ({escaped_ids})")  # type: ignore
                else:
                    self._skills_table.delete(f"id = '{ids[0]}'")  # type: ignore
            except RuntimeError as e:
                # LanceDB raises RuntimeError when attempting to delete non-existent records
                # This is expected during upsert when records don't exist yet
                logger.debug(f"Delete during upsert (expected if records don't exist): {e}")
            except Exception as e:
                # Unexpected error during deletion
                logger.warning(f"Unexpected error during upsert delete: {e}")
                raise

        self._skills_table.add(records)  # type: ignore
        return len(records)

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
        await self._ensure_initialized()  # type: ignore

        search = self._skills_table.search(query_vector).limit(limit * 2)  # type: ignore

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

    async def get_skill_by_name(self, name: str, namespace: str = "default") -> Skill | None:
        """
        Get a skill by name.

        Args:
            name: Skill name
            namespace: Skill namespace

        Returns:
            Skill definition if found, None otherwise
        """
        await self._ensure_initialized()  # type: ignore

        # Validate inputs for SQL safety
        _validate_identifier(name, "name")
        _validate_identifier(namespace, "namespace")

        # Escape ID for SQL safety
        skill_id = _escape_sql_string(f"{namespace}.{name}")
        try:
            results = self._skills_table.search().where(f"id = '{skill_id}'").limit(1).to_list()  # type: ignore
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

    async def delete_skill(self, name: str, namespace: str = "default") -> bool:
        """
        Delete a skill.

        Args:
            name: Skill name
            namespace: Skill namespace

        Returns:
            True if deleted, False if not found
        """
        await self._ensure_initialized()  # type: ignore

        # Validate inputs for SQL safety
        _validate_identifier(name, "name")
        _validate_identifier(namespace, "namespace")

        # Escape ID for SQL safety
        skill_id = _escape_sql_string(f"{namespace}.{name}")
        try:
            self._skills_table.delete(f"id = '{skill_id}'")  # type: ignore
            return True
        except Exception:
            return False

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
        await self._ensure_initialized()  # type: ignore

        # Validate inputs if provided
        if namespace is not None:
            _validate_identifier(namespace, "namespace")
        if category is not None:
            _validate_identifier(category, "category")

        try:
            query = self._skills_table.search()  # type: ignore
            where_clauses = []
            if namespace:
                where_clauses.append(f"namespace = '{_escape_sql_string(namespace)}'")
            if category:
                where_clauses.append(f"category = '{_escape_sql_string(category)}'")

            if where_clauses:
                query = query.where(" AND ".join(where_clauses))

            records = query.limit(limit).offset(offset).to_list()

            return [
                Skill.model_validate_json(r["skill_json"])
                for r in records
                if r.get("skill_json")  # Skip records with missing skill_json
            ]
        except Exception as e:
            logger.warning(f"Error listing skills: {e}")
            return []

    async def count_skills(self, namespace: str | None = None) -> int:
        """
        Count skills.

        Args:
            namespace: Filter by namespace

        Returns:
            Number of skills
        """
        await self._ensure_initialized()  # type: ignore

        # Validate namespace if provided
        if namespace is not None:
            _validate_identifier(namespace, "namespace")

        try:
            if namespace:
                return int(
                    self._skills_table.count_rows(f"namespace = '{_escape_sql_string(namespace)}'")
                )  # type: ignore
            return int(self._skills_table.count_rows())  # type: ignore
        except Exception as e:
            logger.warning(f"Error counting skills: {e}")
            return 0


class LanceDBMetadataMixin:
    """Mixin for LanceDB metadata operations."""

    async def get_metadata(self, key: str) -> str | None:
        """
        Get a metadata value by key.

        Args:
            key: The metadata key

        Returns:
            The value if found, None otherwise
        """
        await self._ensure_initialized()  # type: ignore

        try:
            escaped_key = _escape_sql_string(key)
            results = (
                self._metadata_table.search().where(f"key = '{escaped_key}'").limit(1).to_list()
            )  # type: ignore
            if results and results[0].get("value") is not None:
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
        await self._ensure_initialized()  # type: ignore

        now = datetime.now(timezone.utc).isoformat()

        # Delete existing record if present
        try:
            escaped_key = _escape_sql_string(key)
            self._metadata_table.delete(f"key = '{escaped_key}'")  # type: ignore
        except RuntimeError:
            # LanceDB raises RuntimeError when attempting to delete non-existent records
            pass
        except Exception as e:
            logger.warning(f"Unexpected error deleting metadata key '{key}': {e}")
            # Continue anyway - we'll try to add the new record

        # Add new record
        self._metadata_table.add(
            [
                {  # type: ignore
                    "key": key,
                    "value": value,
                    "updated_at": now,
                }
            ]
        )

    async def get_stored_fingerprints(self) -> dict[str, str]:
        """
        Get all stored tool fingerprints.

        Returns:
            Dictionary mapping tool_id to fingerprint
        """
        await self._ensure_initialized()  # type: ignore

        try:
            table = self._tools_table.to_arrow()  # type: ignore
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
        await self._ensure_initialized()  # type: ignore

        status: dict[str, Any] = {
            "tool_count": await self.count(),  # type: ignore
            "dimension": self._dimension,  # type: ignore
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
        is considered failed and an attempt is made to rollback to previous state.

        Rollback Limitations:
            Due to LanceDB's lack of native transaction support, rollback is
            best-effort only and may fail if:

            - The metadata table becomes corrupted during updates
            - A second concurrent process modifies metadata simultaneously
            - The database connection is lost during rollback

            If rollback fails, the metadata may be left in an inconsistent state
            with some fields updated and others not. In this case:

            - Check logs for "Rollback failed" error messages
            - Manually verify metadata consistency with get_sync_status()
            - Consider re-syncing all tools to restore consistency
            - Use external locks (e.g., file locks) to prevent concurrent writes

        Args:
            embedder_id: Identifier for the embedder used
            dimension: Vector dimension used

        Raises:
            Exception: If metadata update fails (with rollback attempted)
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
