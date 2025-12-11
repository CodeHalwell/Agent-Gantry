"""
Main AgentGantry facade.

Primary entry point for the Agent-Gantry library.
"""

from __future__ import annotations

import uuid
from collections.abc import Callable
from time import perf_counter
from typing import TYPE_CHECKING, Any

from agent_gantry.adapters.embedders.simple import SimpleEmbedder
from agent_gantry.adapters.vector_stores.memory import InMemoryVectorStore
from agent_gantry.core.executor import ExecutionEngine
from agent_gantry.core.registry import ToolRegistry
from agent_gantry.core.security import SecurityPolicy
from agent_gantry.observability.console import ConsoleTelemetryAdapter
from agent_gantry.schema.config import AgentGantryConfig
from agent_gantry.schema.query import RetrievalResult, ScoredTool, ToolQuery
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
        security_policy: SecurityPolicy | None = None,
    ) -> None:
        """
        Initialize AgentGantry.

        Args:
            config: Configuration for the gantry instance
            vector_store: Custom vector store adapter
            embedder: Custom embedding adapter
            reranker: Custom reranker adapter
            telemetry: Custom telemetry adapter
            security_policy: Security policy for permission checks
        """
        self._config = config or AgentGantryConfig()
        self._vector_store = vector_store or InMemoryVectorStore()
        self._embedder = embedder or SimpleEmbedder()
        self._reranker = reranker
        self._telemetry = telemetry
        self._security_policy = security_policy or SecurityPolicy()
        self._registry = ToolRegistry()
        self._executor = ExecutionEngine(
            registry=self._registry,
            security_policy=self._security_policy,
            telemetry=self._telemetry,
        )
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
            self._registry.register_handler(f"{namespace}.{tool_name}", fn)

            return fn

        if func is not None:
            return decorator(func)
        return decorator

    def _build_parameters_schema(self, func: Callable[..., Any]) -> dict[str, Any]:
        """Build JSON Schema for function parameters."""
        return build_parameters_schema(func)

    async def _ensure_initialized(self) -> None:
        """Initialize backing services once."""
        if not self._initialized:
            await self._vector_store.initialize()
            self._initialized = True

    async def add_tool(self, tool: ToolDefinition) -> None:
        """
        Add a tool definition directly.

        Args:
            tool: The tool definition to add
        """
        await self._ensure_initialized()
        embedding = await self._embedder.embed_text(self._tool_to_text(tool))
        await self._vector_store.add_tools([tool], [embedding], upsert=True)
        self._registry.register_tool(tool)

    async def sync(self) -> int:
        """
        Sync pending registrations to vector store.

        Returns:
            Number of tools synced
        """
        if not self._pending_tools:
            return 0

        await self._ensure_initialized()
        texts = [self._tool_to_text(t) for t in self._pending_tools]
        embeddings = await self._embedder.embed_batch(texts)
        count = await self._vector_store.add_tools(self._pending_tools, embeddings, upsert=True)
        
        # Register tools in registry
        for tool in self._pending_tools:
            self._registry.register_tool(tool)
        
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
        await self._ensure_initialized()
        if self._config.auto_sync:
            await self.sync()

        overall_start = perf_counter()
        embed_start = perf_counter()
        query_vector = await self._embedder.embed_text(query.context.query)
        query_embedding_time_ms = (perf_counter() - embed_start) * 1000

        filters: dict[str, Any] | None = None
        if query.namespaces:
            filters = {"namespace": query.namespaces}

        search_start = perf_counter()
        raw_results = await self._vector_store.search(
            query_vector,
            limit=query.limit,
            filters=filters,
            score_threshold=query.score_threshold,
        )
        vector_search_time_ms = (perf_counter() - search_start) * 1000

        scored: list[ScoredTool] = []
        for tool, score in raw_results:
            if query.exclude_deprecated and tool.deprecated:
                continue
            if query.namespaces and tool.namespace not in query.namespaces:
                continue
            if query.required_capabilities and not all(
                cap in tool.capabilities for cap in query.required_capabilities
            ):
                continue
            if query.excluded_capabilities and any(
                cap in tool.capabilities for cap in query.excluded_capabilities
            ):
                continue
            if query.sources and tool.source not in query.sources:
                continue
            scored.append(ScoredTool(tool=tool, semantic_score=score))

        scored.sort(key=lambda s: s.final_score, reverse=True)
        scored = scored[: query.limit]

        total_time_ms = (perf_counter() - overall_start) * 1000
        return RetrievalResult(
            tools=scored,
            query_embedding_time_ms=query_embedding_time_ms,
            vector_search_time_ms=vector_search_time_ms,
            rerank_time_ms=None,
            total_time_ms=total_time_ms,
            candidate_count=len(raw_results),
            filtered_count=len(scored),
            trace_id=str(uuid.uuid4()),
        )

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
        await self._ensure_initialized()
        if self._config.auto_sync:
            await self.sync()
        
        if self._telemetry:
            async with self._telemetry.span("tool_execution", {"tool_name": call.tool_name}):
                return await self._executor.execute(call)
        else:
            return await self._executor.execute(call)

    async def execute_batch(self, batch: BatchToolCall) -> BatchToolResult:
        """
        Execute multiple tool calls.

        Args:
            batch: The batch of tool calls

        Returns:
            Results of all tool executions
        """
        await self._ensure_initialized()
        if self._config.auto_sync:
            await self.sync()
        
        if self._telemetry:
            async with self._telemetry.span("batch_execution", {"count": len(batch.calls)}):
                return await self._executor.execute_batch(batch)
        else:
            return await self._executor.execute_batch(batch)

    def serve_mcp(self, transport: str = "stdio") -> None:
        """
        Start serving as an MCP server.

        Args:
            transport: Transport type ("stdio" or "sse")
        """
        raise NotImplementedError("MCP server not yet implemented")

    def serve_a2a(self, host: str = "0.0.0.0", port: int = 8080) -> None:
        """
        Start serving as an A2A agent.

        Args:
            host: Host to bind to
            port: Port to listen on
        """
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
        await self._ensure_initialized()
        if self._config.auto_sync:
            await self.sync()
        return await self._vector_store.get_by_name(name, namespace)

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
        await self._ensure_initialized()
        if self._config.auto_sync:
            await self.sync()
        return await self._vector_store.list_all(namespace=namespace)

    async def delete_tool(self, name: str, namespace: str = "default") -> bool:
        """
        Delete a tool.

        Args:
            name: Tool name
            namespace: Tool namespace

        Returns:
            True if tool was deleted
        """
        await self._ensure_initialized()
        return await self._vector_store.delete(name, namespace)

    async def health_check(self) -> dict[str, bool]:
        """
        Check health of all components.

        Returns:
            Dictionary of component health status
        """
        import inspect

        await self._ensure_initialized()
        vector_store_ok = await self._vector_store.health_check()
        embedder_ok = await self._embedder.health_check()
        telemetry_ok = False
        if self._telemetry is not None:
            health = getattr(self._telemetry, "health_check", None)
            if callable(health):
                result = health()
                telemetry_ok = await result if inspect.isawaitable(result) else bool(result)
            else:
                telemetry_ok = True
        return {
            "vector_store": vector_store_ok,
            "embedder": embedder_ok,
            "telemetry": telemetry_ok,
        }

    def _tool_to_text(self, tool: ToolDefinition) -> str:
        """Flatten tool metadata into a text string for embedding."""
        tags = " ".join(tool.tags)
        return f"{tool.name} {tool.namespace} {tool.description} {tags} {' '.join(tool.examples)}"


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
        if param_type == int or param_type == "int":
            param_schema["type"] = "integer"
        elif param_type == float or param_type == "float":
            param_schema["type"] = "number"
        elif param_type == bool or param_type == "bool":
            param_schema["type"] = "boolean"
        elif param_type == str or param_type == "str":
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
