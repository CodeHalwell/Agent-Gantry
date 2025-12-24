"""Shared utility functions for Agent-Gantry core."""

from agent_gantry.schema.tool import ToolDefinition


def tool_to_searchable_text(tool: ToolDefinition) -> str:
    """
    Convert tool metadata to searchable text for embedding.

    Args:
        tool: The tool definition

    Returns:
        Concatenated string of tool metadata
    """
    return tool.to_searchable_text()
