"""
Tool fingerprinting for change detection.

A fingerprint is a versioned hash of a tool's persisted definition; vector
stores keep it next to the embedding so ``sync()`` only re-embeds tools that
actually changed.
"""

from __future__ import annotations

import hashlib
import json

from agent_gantry.schema.tool import ToolDefinition

# 16 hex chars = 64 bits: compact with good collision resistance
FINGERPRINT_LENGTH = 16

# Bump when the hashed payload changes; every stored fingerprint then mismatches
# and the next sync re-embeds everything once.
FINGERPRINT_VERSION = "v1.1"


def compute_tool_fingerprint(tool: ToolDefinition) -> str:
    """
    Compute the fingerprint of a tool definition.

    Covers the semantic content (name, description, schema, tags, examples),
    the security-critical fields (capabilities, requires_confirmation) and
    every other persisted routing/lifecycle field, because stores serve the
    stored definition back to the router. Runtime health and the
    per-instantiation ``created_at`` stay excluded so incremental sync works.

    Returns:
        ``"{version}:{hash}"``, e.g. ``"v1.1:a1b2c3d4e5f67890"``
    """
    payload: dict = {
        "name": tool.name,
        "namespace": tool.namespace,
        "description": tool.description,
        "parameters_schema": tool.parameters_schema,
        "tags": sorted(tool.tags),
        "examples": sorted(tool.examples),
        "capabilities": sorted([str(cap) for cap in tool.capabilities]),
        "requires_confirmation": tool.requires_confirmation,
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
    content = json.dumps(payload, sort_keys=True, default=str)
    hash_value = hashlib.sha256(content.encode()).hexdigest()[:FINGERPRINT_LENGTH]
    return f"{FINGERPRINT_VERSION}:{hash_value}"
