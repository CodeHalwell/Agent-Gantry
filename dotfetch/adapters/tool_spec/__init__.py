"""
Tool specification adapters for dotfetch.

Provides adapters to convert ToolDefinition to provider-specific formats
and map provider tool-call payloads to unified ToolCall objects.
"""

from dotfetch.adapters.tool_spec.base import (
    ToolCallPayload,
    ToolSpecAdapter,
)
from dotfetch.adapters.tool_spec.registry import DialectRegistry, get_adapter

__all__ = [
    "DialectRegistry",
    "ToolCallPayload",
    "ToolSpecAdapter",
    "get_adapter",
]
