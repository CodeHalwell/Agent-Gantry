"""
Mixins and SQL-safety helpers for the LanceDB vector store.

The tools/skills operations live in ``lancedb.py``; this module holds what is
inherited from it: the tools schema migration and the sync-metadata API.
"""

from __future__ import annotations

import asyncio
import logging
import re
from datetime import datetime, timezone
from typing import Any

from agent_gantry.schema.tool import ToolDefinition
from agent_gantry.utils.fingerprint import compute_tool_fingerprint

# Pre-compiled: ~5x faster than a generator over ord(c) < 32
_CTRL_CHAR_RE = re.compile(r"[\x00-\x1f]")

logger = logging.getLogger(__name__)


def _escape_sql_string(value: str) -> str:
    """
    Escape a string for a LanceDB (DataFusion) SQL literal.

    Single quotes are doubled — the only escape DataFusion string literals
    recognise. Backslashes are left alone: DataFusion has no backslash escapes,
    so doubling them produced literals that never matched the stored value.

    LanceDB has no parameterised WHERE clauses, so this is the escaping layer;
    caller-supplied identifiers additionally go through ``_validate_identifier``.
    """
    return value.replace("'", "''")


def _validate_identifier(value: str, field_name: str) -> None:
    """
    Reject a value unsafe for a SQL predicate: empty, over 256 characters, or
    containing control characters.

    Raises:
        ValueError: If validation fails
    """
    if not value or len(value) > 256:
        raise ValueError(f"{field_name} must be 1-256 characters")
    if _CTRL_CHAR_RE.search(value):
        raise ValueError(f"{field_name} contains invalid characters")


class LanceDBToolsMixin:
    """Schema-migration support for the LanceDB tools table."""

    async def _migrate_tools_schema(self, target_schema: Any) -> None:
        """
        Add columns missing from an existing tools table (e.g. ``fingerprint``).

        LanceDB has no ALTER TABLE, so the rows are copied into
        ``<table>__migrating`` first and the tools table is only dropped once
        that copy exists; a failure leaves the data there and raises.
        """
        current_fields = {field.name for field in self._tools_table.schema}  # type: ignore
        missing_fields = {field.name for field in target_schema} - current_fields
        if not missing_fields:
            return
        logger.info(f"Migrating tools table schema. Adding fields: {missing_fields}")

        records = await asyncio.to_thread(
            lambda: self._tools_table.to_arrow().to_pylist()  # type: ignore
        )
        now = datetime.now(timezone.utc).isoformat()
        for record in records:
            if "fingerprint" in missing_fields:
                try:
                    tool = ToolDefinition.model_validate_json(record["tool_json"])
                    record["fingerprint"] = compute_tool_fingerprint(tool)
                except Exception as e:
                    logger.warning(f"Failed to compute fingerprint during migration: {e}")
                    record["fingerprint"] = ""
            for field in ("created_at", "updated_at"):
                if field in missing_fields:
                    record[field] = now

        def recreate() -> Any:
            db, name = self._db, self._tools_table_name  # type: ignore
            if not records:
                db.drop_table(name)
                return db.create_table(name, schema=target_schema)
            backup = f"{name}__migrating"
            db.create_table(backup, data=records, schema=target_schema, mode="overwrite")
            db.drop_table(name)
            table = db.create_table(name, data=records, schema=target_schema)
            db.drop_table(backup)
            return table

        self._tools_table = await asyncio.to_thread(recreate)
        logger.info(f"Migrated {len(records)} tools to new schema")


class LanceDBMetadataMixin:
    """Mixin for LanceDB metadata operations."""

    async def get_metadata(self, key: str) -> str | None:
        """Get a metadata value by key, or None if absent."""
        await self._ensure_initialized()  # type: ignore

        try:
            escaped_key = _escape_sql_string(key)
            query = self._metadata_table.search().where(f"key = '{escaped_key}'").limit(1)  # type: ignore
            results = await asyncio.to_thread(query.to_list)
            if results and results[0].get("value") is not None:
                value: str = results[0]["value"]
                return value
        except Exception as e:
            logger.debug(f"get_metadata failed for key '{key}': {e}")
        return None

    async def set_metadata(self, key: str, value: str) -> None:
        """Set a metadata value (replacing any existing one)."""
        await self._ensure_initialized()  # type: ignore

        now = datetime.now(timezone.utc).isoformat()

        try:
            escaped_key = _escape_sql_string(key)
            await asyncio.to_thread(self._metadata_table.delete, f"key = '{escaped_key}'")  # type: ignore
        except RuntimeError:
            pass  # LanceDB raises when nothing matches
        except Exception as e:
            logger.warning(f"Unexpected error deleting metadata key '{key}': {e}")

        await asyncio.to_thread(
            self._metadata_table.add,  # type: ignore
            [{"key": key, "value": value, "updated_at": now}],
        )

    async def get_stored_fingerprints(self) -> dict[str, str]:
        """Get all stored tool fingerprints, keyed by ``namespace.name``."""
        await self._ensure_initialized()  # type: ignore

        try:
            query = self._tools_table.search().select(["id", "fingerprint"]).limit(None)  # type: ignore
            table = await asyncio.to_thread(query.to_arrow)
            ids = table["id"].to_pylist()
            fingerprints = [f if f is not None else "" for f in table["fingerprint"].to_pylist()]
            return dict(zip(ids, fingerprints))
        except Exception as e:
            logger.debug(f"get_stored_fingerprints failed: {e}")
            return {}

    async def update_sync_metadata(self, embedder_id: str, dimension: int) -> None:
        """Record the embedder and dimension used by the last sync."""
        await self.set_metadata("embedder_id", embedder_id)
        await self.set_metadata("dimension", str(dimension))
