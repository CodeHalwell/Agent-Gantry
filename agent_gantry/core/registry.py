"""
Tool registry for Agent-Gantry.

Manages tool registration and lifecycle.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from agent_gantry.schema.tool import ToolCapability, ToolDefinition


class ToolRegistry:
    """
    Registry for managing tool definitions and handlers.

    Handles:
    - Python function registration via decorators
    - Tool lifecycle management
    - Handler lookup for execution
    """

    def __init__(self) -> None:
        """Initialize the registry."""
        self._tools: dict[str, ToolDefinition] = {}
        self._handlers: dict[str, Callable[..., Any]] = {}
        self._pending: list[ToolDefinition] = []

    def register(
        self,
        func: Callable[..., Any] | None = None,
        *,
        name: str | None = None,
        namespace: str = "default",
        capabilities: list[ToolCapability] | None = None,
        requires_confirmation: bool = False,
        tags: list[str] | None = None,
        examples: list[str] | None = None,
    ) -> Callable[..., Any]:
        """
        Decorator to register a Python function as a tool.

        Args:
            func: The function to register
            name: Custom name for the tool
            namespace: Namespace for organizing tools
            capabilities: Tool capabilities for permission checks
            requires_confirmation: Whether to require human confirmation
            tags: Tags for categorizing the tool
            examples: Example queries for the tool

        Returns:
            The decorated function
        """

        def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
            tool_name = name or fn.__name__
            tool_description = fn.__doc__ or f"Tool: {tool_name}"
            parameters_schema = self._build_parameters_schema(fn)

            tool = ToolDefinition(
                name=tool_name,
                namespace=namespace,
                description=tool_description.strip(),
                parameters_schema=parameters_schema,
                capabilities=capabilities or [],
                requires_confirmation=requires_confirmation,
                tags=tags or [],
                examples=examples or [],
            )

            key = f"{namespace}.{tool_name}"
            self._tools[key] = tool
            self._handlers[key] = fn
            self._pending.append(tool)

            return fn

        if func is not None:
            return decorator(func)
        return decorator

    def _build_parameters_schema(self, func: Callable[..., Any]) -> dict[str, Any]:
        """Build JSON Schema for function parameters."""
        import inspect
        from typing import get_type_hints

        sig = inspect.signature(func)
        try:
            type_hints = get_type_hints(func)
        except Exception:
            type_hints = {}

        properties: dict[str, Any] = {}
        required: list[str] = []

        for param_name, param in sig.parameters.items():
            if param_name in ("self", "cls"):
                continue

            param_schema: dict[str, Any] = {}
            param_type = type_hints.get(param_name)

            # Check for exact type matches
            if param_type == int or param_type == "int":
                param_schema["type"] = "integer"
            elif param_type == float or param_type == "float":
                param_schema["type"] = "number"
            elif param_type == bool or param_type == "bool":
                param_schema["type"] = "boolean"
            elif param_type == str or param_type == "str":
                param_schema["type"] = "string"
            else:
                # Default to string for unknown types
                param_schema["type"] = "string"

            properties[param_name] = param_schema

            if param.default is inspect.Parameter.empty:
                required.append(param_name)

        return {
            "type": "object",
            "properties": properties,
            "required": required,
        }

    def add_tool(self, tool: ToolDefinition, handler: Callable[..., Any]) -> None:
        """
        Add a tool directly with its handler.

        Args:
            tool: The tool definition
            handler: The callable to execute
        """
        key = f"{tool.namespace}.{tool.name}"
        self._tools[key] = tool
        self._handlers[key] = handler
        self._pending.append(tool)

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

    def get_handler(self, key: str) -> Callable[..., Any] | None:
        """
        Get the handler for a tool by its full key (namespace.name).

        Args:
            key: Full tool key (namespace.name)

        Returns:
            The handler callable if found
        """
        return self._handlers.get(key)

    def register_tool(self, tool: ToolDefinition) -> None:
        """
        Register a tool definition.

        Args:
            tool: The tool definition to register
        """
        key = f"{tool.namespace}.{tool.name}"
        self._tools[key] = tool

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

    def delete_tool(self, name: str, namespace: str = "default") -> bool:
        """
        Delete a tool from the registry.

        Args:
            name: Tool name
            namespace: Tool namespace

        Returns:
            True if the tool was deleted
        """
        key = f"{namespace}.{name}"
        if key in self._tools:
            del self._tools[key]
            self._handlers.pop(key, None)
            return True
        return False

    def get_pending(self) -> list[ToolDefinition]:
        """
        Get tools pending sync to vector store.

        Returns:
            List of pending tool definitions
        """
        return self._pending.copy()

    def clear_pending(self) -> None:
        """Clear the pending tools list."""
        self._pending = []

    @property
    def tool_count(self) -> int:
        """Return the number of registered tools."""
        return len(self._tools)
