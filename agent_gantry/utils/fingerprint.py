"""
Tool fingerprinting utilities for change detection.

This module provides fingerprinting functionality to detect when tools have
changed and need re-embedding in vector stores.
"""

from __future__ import annotations

import hashlib
import json

from agent_gantry.schema.tool import ToolDefinition

# Hash length for fingerprints (16 hex chars = 64 bits)
# Provides good collision resistance while keeping storage compact
FINGERPRINT_LENGTH = 16

# Fingerprint version - increment when fingerprint algorithm changes
# Format: v{major}.{minor}
# - Major: Breaking changes (all fingerprints must be recomputed)
# - Minor: Non-breaking enhancements
FINGERPRINT_VERSION = "v1.1"


def compute_tool_fingerprint(tool: ToolDefinition, version: str | None = None) -> str:
    """
    Compute a fingerprint hash for a tool definition.

    The fingerprint is based on the tool's semantic content (name, namespace,
    description, parameters schema, tags, examples) AND security-critical fields
    (capabilities, requires_confirmation). This ensures that changes to tool
    permissions trigger re-embedding.

    Args:
        tool: The tool definition
        version: Fingerprint version to use (defaults to current FINGERPRINT_VERSION)

    Returns:
        Versioned fingerprint string in format: {version}:{hash}
        Example: "v1.0:a1b2c3d4e5f67890"

    Raises:
        ValueError: If unsupported version is requested
    """
    version = version or FINGERPRINT_VERSION

    if version not in ("v1.0", "v1.1"):
        raise ValueError(f"Unsupported fingerprint version: {version}")

    # SHA256 of sorted JSON. Includes security-critical fields (capabilities,
    # requires_confirmation) so permission changes trigger re-embedding.
    payload: dict = {
        "name": tool.name,
        "namespace": tool.namespace,
        "description": tool.description,
        "parameters_schema": tool.parameters_schema,
        "tags": sorted(tool.tags),
        "examples": sorted(tool.examples),
        "capabilities": sorted([str(cap) for cap in tool.capabilities]),
        "requires_confirmation": tool.requires_confirmation,
    }
    if version == "v1.1":
        # v1.1 additionally covers every persisted routing/lifecycle field:
        # stores serve the stored ToolDefinition back to the router, so a
        # definition-only change (e.g. flipping `deprecated`) must re-sync
        # the stored copy or filters like exclude_deprecated keep acting on
        # stale data. Runtime health and the per-instantiation created_at
        # timestamp stay excluded — including them would defeat incremental
        # sync. Costs one re-embed per actually-changed tool.
        payload.update(
            {
                "version": tool.version,
                "extended_description": tool.extended_description,
                "returns_schema": tool.returns_schema,
                "source": tool.source.value,
                "source_uri": tool.source_uri,
                "cost": tool.cost.model_dump(mode="json"),
                "metadata": tool.metadata,
                "deprecated": tool.deprecated,
                "deprecation_message": tool.deprecation_message,
                "superseded_by": tool.superseded_by,
            }
        )
    content = json.dumps(payload, sort_keys=True, default=str)
    hash_value = hashlib.sha256(content.encode()).hexdigest()[:FINGERPRINT_LENGTH]
    return f"{version}:{hash_value}"


def parse_fingerprint(fingerprint: str) -> tuple[str, str]:
    """
    Parse a versioned fingerprint into version and hash components.

    Args:
        fingerprint: Versioned fingerprint string (e.g., "v1.0:a1b2c3d4e5f67890")

    Returns:
        Tuple of (version, hash)

    Raises:
        ValueError: If fingerprint format is invalid
    """
    if ":" not in fingerprint:
        # Legacy fingerprint without version
        return ("v1.0", fingerprint)

    parts = fingerprint.split(":", 1)
    if len(parts) != 2:
        raise ValueError(f"Invalid fingerprint format: {fingerprint}")

    return (parts[0], parts[1])
