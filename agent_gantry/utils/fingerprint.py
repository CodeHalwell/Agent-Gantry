"""
Tool fingerprinting utilities for change detection.

This module provides fingerprinting functionality to detect when tools have
changed and need re-embedding in vector stores.
"""

from __future__ import annotations

import hashlib
import json

from agent_gantry.schema.tool import ToolDefinition


def compute_tool_fingerprint(tool: ToolDefinition) -> str:
    """
    Compute a fingerprint hash for a tool definition.

    The fingerprint is based on the tool's semantic content (name, namespace,
    description, parameters schema, tags, and examples). This allows detecting
    when a tool has changed and needs re-embedding.

    Args:
        tool: The tool definition

    Returns:
        SHA256 hash (first 16 chars) of the tool's semantic content
    """
    content = json.dumps(
        {
            "name": tool.name,
            "namespace": tool.namespace,
            "description": tool.description,
            "parameters_schema": tool.parameters_schema,
            "tags": sorted(tool.tags),
            "examples": sorted(tool.examples),
        },
        sort_keys=True,
    )
    return hashlib.sha256(content.encode()).hexdigest()[:16]
