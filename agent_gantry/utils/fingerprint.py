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
FINGERPRINT_VERSION = "v1.0"


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

    if version != "v1.0":
        raise ValueError(f"Unsupported fingerprint version: {version}")

    # v1.0 algorithm: SHA256 of sorted JSON
    # Includes security-critical fields (capabilities, requires_confirmation)
    # to ensure permission changes trigger re-embedding
    content = json.dumps(
        {
            "name": tool.name,
            "namespace": tool.namespace,
            "description": tool.description,
            "parameters_schema": tool.parameters_schema,
            "tags": sorted(tool.tags),
            "examples": sorted(tool.examples),
            "capabilities": sorted([str(cap) for cap in tool.capabilities]),
            "requires_confirmation": tool.requires_confirmation,
        },
        sort_keys=True,
    )
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

