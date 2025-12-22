"""
Tool specification adapters for Agent-Gantry.

Provides adapters to convert ToolDefinition to provider-specific formats
and map provider tool-call payloads to unified ToolCall objects.
"""

from agent_gantry.adapters.tool_spec.base import (
    ToolCallPayload,
    ToolSpecAdapter,
)
from agent_gantry.adapters.tool_spec.registry import DialectRegistry, get_adapter

__all__ = [
    "DialectRegistry",
    "ToolCallPayload",
    "ToolSpecAdapter",
    "get_adapter",
]
