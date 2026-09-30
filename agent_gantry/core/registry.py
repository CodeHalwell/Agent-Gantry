"""
Tool registry for Agent-Gantry.

In-memory map of tool definitions and their execution handlers, keyed by
``namespace.name``. Registration itself (decorators, introspection, pending
buffers) lives in :class:`~agent_gantry.core.gantry.AgentGantry`; this class
only stores what the executor needs to look up.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from agent_gantry.schema.tool import ToolDefinition


class ToolRegistry:
    """Registry of tool definitions and handlers for execution lookup."""

    def __init__(self) -> None:
        """Initialize the registry."""
        self._tools: dict[str, ToolDefinition] = {}
        self._handlers: dict[str, Callable[..., Any]] = {}
        # Secondary index: bare tool name -> first registered qualified key.
        # Keeps get_tool_by_name O(1); executed on every ExecutionEngine call.
        self._name_index: dict[str, str] = {}

    def _index_tool(self, key: str, tool: ToolDefinition) -> None:
        """Record a tool in the name index (first registration wins)."""
        self._name_index.setdefault(tool.name, key)

    def _unindex_tool(self, key: str, name: str) -> None:
        """Drop a deleted tool from the name index, re-pointing to another namespace if any."""
        if self._name_index.get(name) != key:
            return
        del self._name_index[name]
        for other_key, other_tool in self._tools.items():
            if other_tool.name == name:
                self._name_index[name] = other_key
                break

    def get_tool(self, name: str, namespace: str = "default") -> ToolDefinition | None:
        """
        Get a tool by name and namespace.

        Args:
            name: Tool name
            namespace: Tool namespace

        Returns:
            The tool definition if found
        """
        key = f"{namespace}.{name}"
        return self._tools.get(key)

    def get_tool_by_name(self, name: str) -> ToolDefinition | None:
        """
        Get a tool by name, searching across all namespaces.

        Returns the first match found. Useful when the caller doesn't know
        which namespace a tool belongs to.

        Args:
            name: Tool name to search for

        Returns:
            The tool definition if found, None otherwise
        """
        # Try default namespace first for speed
        default_key = f"default.{name}"
        if default_key in self._tools:
            return self._tools[default_key]

        # O(1) lookup across namespaces via the name index
        key = self._name_index.get(name)
        if key is not None:
            return self._tools.get(key)
        return None

    def namespaces_for_name(self, name: str) -> list[str]:
        """Return every namespace registering a tool called ``name``.

        Used to tell an unambiguous bare-name lookup apart from one that
        silently picked a winner among same-named tools.
        """
        return [tool.namespace for tool in self._tools.values() if tool.name == name]

    def get_handler(self, key: str) -> Callable[..., Any] | None:
        """
        Get the handler for a tool by its full key (namespace.name).

        Args:
            key: Full tool key (namespace.name)

        Returns:
            The handler callable if found
        """
        return self._handlers.get(key)

    def register_tool(
        self, tool: ToolDefinition, handler: Callable[..., Any] | None = None
    ) -> None:
        """
        Register a tool definition, optionally with its execution handler.

        Args:
            tool: The tool definition to register
            handler: The callable that executes the tool. ``None`` leaves any
                existing handler untouched, which is what a re-registration
                from a sync wants.
        """
        key = f"{tool.namespace}.{tool.name}"
        self._tools[key] = tool
        self._index_tool(key, tool)
        if handler is not None:
            self._handlers[key] = handler

    def register_handler(self, key: str, handler: Callable[..., Any]) -> None:
        """
        Register a handler for a tool.

        Args:
            key: Full tool key (namespace.name)
            handler: The callable to execute
        """
        self._handlers[key] = handler

    def list_tools(self, namespace: str | None = None) -> list[ToolDefinition]:
        """
        List all registered tools.

        Args:
            namespace: Filter by namespace

        Returns:
            List of tool definitions
        """
        tools = list(self._tools.values())
        if namespace:
            tools = [t for t in tools if t.namespace == namespace]
        return tools

    def delete_tool(self, name: str, namespace: str = "default") -> Callable[..., Any] | None:
        """
        Delete a tool and its handler from the registry.

        Args:
            name: Tool name
            namespace: Tool namespace

        Returns:
            The handler that was registered for the tool, or ``None`` when the
            tool was unknown or had no handler.
        """
        key = f"{namespace}.{name}"
        if key not in self._tools:
            return None
        del self._tools[key]
        self._unindex_tool(key, name)
        return self._handlers.pop(key, None)

    def has_tool(self, name: str, namespace: str = "default") -> bool:
        """Whether a tool is registered under ``namespace.name``."""
        return f"{namespace}.{name}" in self._tools

    @property
    def tool_count(self) -> int:
        """Return the number of registered tools."""
        return len(self._tools)

    @property
    def handler_count(self) -> int:
        """Return the number of tools with an execution handler."""
        return len(self._handlers)
