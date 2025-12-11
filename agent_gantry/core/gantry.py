"""
Main AgentGantry facade.

Primary entry point for the Agent-Gantry library.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from agent_gantry.schema.config import AgentGantryConfig
from agent_gantry.schema.query import RetrievalResult, ToolQuery
from agent_gantry.schema.tool import ToolCapability, ToolDefinition

if TYPE_CHECKING:
    from agent_gantry.adapters.embedders.base import EmbeddingAdapter
    from agent_gantry.adapters.rerankers.base import RerankerAdapter
    from agent_gantry.adapters.vector_stores.base import VectorStoreAdapter
    from agent_gantry.observability.telemetry import TelemetryAdapter
    from agent_gantry.schema.execution import BatchToolCall, BatchToolResult, ToolCall, ToolResult


class AgentGantry:
    """
    Main facade for Agent-Gantry.

    Provides intelligent, secure tool orchestration for LLM-based agent systems.

    Example:
        gantry = AgentGantry()

        @gantry.register
        def my_tool(x: int) -> str:
            '''Does something useful.'''
            return str(x * 2)

        tools = await gantry.retrieve_tools("double a number")
    """

    def __init__(
        self,
        config: AgentGantryConfig | None = None,
        vector_store: VectorStoreAdapter | None = None,
        embedder: EmbeddingAdapter | None = None,
        reranker: RerankerAdapter | None = None,
        telemetry: TelemetryAdapter | None = None,
    ) -> None:
        """
        Initialize AgentGantry.

        Args:
            config: Configuration for the gantry instance
            vector_store: Custom vector store adapter
            embedder: Custom embedding adapter
            reranker: Custom reranker adapter
            telemetry: Custom telemetry adapter
        """
        self._config = config or AgentGantryConfig()
        self._vector_store = vector_store
        self._embedder = embedder
        self._reranker = reranker
        self._telemetry = telemetry
        self._pending_tools: list[ToolDefinition] = []
        self._tool_handlers: dict[str, Callable[..., Any]] = {}
        self._initialized = False

    @classmethod
    def from_config(cls, path: str) -> AgentGantry:
        """
        Create an AgentGantry instance from a YAML config file.

        Args:
            path: Path to the YAML configuration file

        Returns:
            Configured AgentGantry instance
        """
        config = AgentGantryConfig.from_yaml(path)
        return cls(config=config)

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
        Decorator to register Python functions as tools.

        Args:
            func: The function to register (when used without parentheses)
            name: Custom name for the tool (defaults to function name)
            namespace: Namespace for organizing tools
            capabilities: List of capabilities this tool has
            requires_confirmation: Whether to require human confirmation
            tags: Tags for categorizing the tool
            examples: Example queries that this tool handles

        Returns:
            The decorated function
        """

        def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
            tool_name = name or fn.__name__
            tool_description = fn.__doc__ or f"Tool: {tool_name}"

            # Build parameters schema from function signature
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

            self._pending_tools.append(tool)
            self._tool_handlers[tool_name] = fn

            return fn

        if func is not None:
            return decorator(func)
        return decorator

    def _build_parameters_schema(self, func: Callable[..., Any]) -> dict[str, Any]:
        """Build JSON Schema for function parameters."""
        import inspect
        return build_parameters_schema(func)


def build_parameters_schema(func: Callable[..., Any]) -> dict[str, Any]:
    """Build JSON Schema for function parameters."""
    import inspect

    sig = inspect.signature(func)
    type_hints = {}
    try:
        type_hints = func.__annotations__
    except AttributeError:
        pass

    properties: dict[str, Any] = {}
    required: list[str] = []

    for param_name, param in sig.parameters.items():
        if param_name in ("self", "cls"):
            continue

        param_schema: dict[str, Any] = {}
        param_type = type_hints.get(param_name, Any)

        # Map Python types to JSON Schema types
        if param_type is int:
            param_schema["type"] = "integer"
        elif param_type is float:
            param_schema["type"] = "number"
        elif param_type is bool:
            param_schema["type"] = "boolean"
        elif param_type is str:
            param_schema["type"] = "string"
        else:
            param_schema["type"] = "string"  # Default fallback

        properties[param_name] = param_schema

        if param.default is inspect.Parameter.empty:
            required.append(param_name)

    return {
        "type": "object",
        "properties": properties,
        "required": required,
    }
    async def add_tool(self, tool: ToolDefinition) -> None:
        """
        Add a tool definition directly.

        Args:
            tool: The tool definition to add
        """
        self._pending_tools.append(tool)

    async def sync(self) -> int:
        """
        Sync pending registrations to vector store.

        Returns:
            Number of tools synced
        """
        if not self._pending_tools:
            return 0

        # TODO: Implement actual syncing to vector store
        count = len(self._pending_tools)
        self._pending_tools = []
        return count

    async def retrieve(self, query: ToolQuery) -> RetrievalResult:
        """
        Core semantic routing function.

        Args:
            query: The tool query with context and filters

        Returns:
            RetrievalResult with scored tools
        """
        # TODO: Implement actual retrieval logic
        raise NotImplementedError("Retrieval not yet implemented")

    async def retrieve_tools(
        self,
        query: str,
        limit: int = 5,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        """
        Convenience wrapper: returns OpenAI-compatible schemas.

        Args:
            query: The natural language query
            limit: Maximum number of tools to return
            **kwargs: Additional query parameters

        Returns:
            List of OpenAI-compatible tool schemas
        """
        from agent_gantry.schema.query import ConversationContext, ToolQuery

        context = ConversationContext(query=query)
        tool_query = ToolQuery(context=context, limit=limit, **kwargs)
        result = await self.retrieve(tool_query)
        return result.to_openai_tools()

    async def execute(self, call: ToolCall) -> ToolResult:
        """
        Execute a tool call with full protections.

        Args:
            call: The tool call to execute

        Returns:
            Result of the tool execution
        """
        # TODO: Implement actual execution logic
        raise NotImplementedError("Execution not yet implemented")

    async def execute_batch(self, batch: BatchToolCall) -> BatchToolResult:
        """
        Execute multiple tool calls.

        Args:
            batch: The batch of tool calls

        Returns:
            Results of all tool executions
        """
        # TODO: Implement batch execution
        raise NotImplementedError("Batch execution not yet implemented")

    def serve_mcp(self, transport: str = "stdio") -> None:
        """
        Start serving as an MCP server.

        Args:
            transport: Transport type ("stdio" or "sse")
        """
        # TODO: Implement MCP server
        raise NotImplementedError("MCP server not yet implemented")

    def serve_a2a(self, host: str = "0.0.0.0", port: int = 8080) -> None:
        """
        Start serving as an A2A agent.

        Args:
            host: Host to bind to
            port: Port to listen on
        """
        # TODO: Implement A2A server
        raise NotImplementedError("A2A server not yet implemented")

    @property
    def tool_count(self) -> int:
        """Return the number of registered tools."""
        return len(self._tool_handlers)

    async def get_tool(
        self, name: str, namespace: str = "default"
    ) -> ToolDefinition | None:
        """
        Get a tool by name.

        Args:
            name: Tool name
            namespace: Tool namespace

        Returns:
            The tool definition if found
        """
        # TODO: Implement actual lookup
        return None

    async def list_tools(
        self,
        namespace: str | None = None,
    ) -> list[ToolDefinition]:
        """
        List all registered tools.

        Args:
            namespace: Filter by namespace

        Returns:
            List of tool definitions
        """
        # TODO: Implement actual listing
        return []

    async def delete_tool(self, name: str, namespace: str = "default") -> bool:
        """
        Delete a tool.

        Args:
            name: Tool name
            namespace: Tool namespace

        Returns:
            True if tool was deleted
        """
        # TODO: Implement deletion
        return False

    async def health_check(self) -> dict[str, bool]:
        """
        Check health of all components.

        Returns:
            Dictionary of component health status
        """
        return {
            "vector_store": self._vector_store is not None,
            "embedder": self._embedder is not None,
            "telemetry": self._telemetry is not None,
        }
