"""
Main AgentGantry facade.

Primary entry point for the Agent-Gantry library.
"""

from __future__ import annotations

import asyncio
import importlib
import logging
import uuid
from collections.abc import Callable, Sequence
from time import perf_counter
from typing import TYPE_CHECKING, Any

from agent_gantry.adapters.embedders.openai import AzureOpenAIEmbedder, OpenAIEmbedder
from agent_gantry.adapters.embedders.simple import SimpleEmbedder
from agent_gantry.adapters.rerankers.cohere import CohereReranker
from agent_gantry.adapters.vector_stores.memory import InMemoryVectorStore
from agent_gantry.adapters.vector_stores.remote import (
    ChromaVectorStore,
    PGVectorStore,
    QdrantVectorStore,
)
from agent_gantry.core.executor import ExecutionEngine
from agent_gantry.core.registry import ToolRegistry
from agent_gantry.core.router import RoutingWeights, SemanticRouter
from agent_gantry.core.security import SecurityPolicy
from agent_gantry.observability.console import ConsoleTelemetryAdapter, NoopTelemetryAdapter
from agent_gantry.observability.opentelemetry_adapter import (
    OpenTelemetryAdapter,
    PrometheusTelemetryAdapter,
)
from agent_gantry.schema.config import (
    A2AAgentConfig,
    AgentGantryConfig,
    EmbedderConfig,
    MCPServerConfig,
    RerankerConfig,
    TelemetryConfig,
    VectorStoreConfig,
)
from agent_gantry.schema.query import RetrievalResult, ScoredTool, ToolQuery
from agent_gantry.schema.tool import ToolCapability, ToolDefinition

if TYPE_CHECKING:
    from agent_gantry.adapters.embedders.base import EmbeddingAdapter
    from agent_gantry.adapters.rerankers.base import RerankerAdapter
    from agent_gantry.adapters.vector_stores.base import VectorStoreAdapter
    from agent_gantry.observability.telemetry import TelemetryAdapter
    from agent_gantry.schema.execution import BatchToolCall, BatchToolResult, ToolCall, ToolResult


logger = logging.getLogger(__name__)


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
        modules: Sequence[str] | None = None,
        module_attr: str = "tools",
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
        self._vector_store = vector_store or self._build_vector_store(self._config.vector_store)
        self._embedder = embedder or self._build_embedder(self._config.embedder)
        self._reranker = reranker or self._build_reranker(self._config.reranker)
        self._telemetry = telemetry or self._build_telemetry(self._config.telemetry)
        self._security_policy = security_policy or SecurityPolicy()
        self._registry = ToolRegistry()
        routing_weights = RoutingWeights(**self._config.routing.weights)
        self._router = SemanticRouter(
            vector_store=self._vector_store,
            embedder=self._embedder,
            reranker=self._reranker,
            weights=routing_weights,
        )
        self._executor = ExecutionEngine(
            registry=self._registry,
            default_timeout_ms=self._config.execution.default_timeout_ms,
            max_retries=self._config.execution.max_retries,
            circuit_breaker_threshold=self._config.execution.circuit_breaker_threshold,
            circuit_breaker_timeout_s=self._config.execution.circuit_breaker_timeout_s,
            security_policy=self._security_policy,
            telemetry=self._telemetry,
        )
        self._pending_tools: list[ToolDefinition] = []
        self._tool_handlers: dict[str, Callable[..., Any]] = {}
        self._initialized = False
        self._modules: Sequence[str] | None = None
        self._module_attr: str | None = None

        if modules:
            # Store modules configuration for explicit async initialization.
            # Users should call `collect_tools_from_modules` in an async context
            # or use `AgentGantry.from_modules(...)` if available.
            self._modules = modules
            self._module_attr = module_attr

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

    @classmethod
    async def from_modules(
        cls,
        modules: Sequence[str],
        *,
        attr: str = "tools",
        config: AgentGantryConfig | None = None,
        vector_store: VectorStoreAdapter | None = None,
        embedder: EmbeddingAdapter | None = None,
        reranker: RerankerAdapter | None = None,
        telemetry: TelemetryAdapter | None = None,
        security_policy: SecurityPolicy | None = None,
    ) -> AgentGantry:
        """
        Build a Gantry instance and populate it by importing tool-bearing modules.

        Args:
            modules: Iterable of module paths (dot-notation) to import.
            attr: Attribute on each module that holds an AgentGantry instance (default "tools").
            config/vector_store/embedder/reranker/telemetry/security_policy: Optional overrides
                for the constructed gantry instance.

        Returns:
            A populated AgentGantry instance.
        """

        gantry = cls(
            config=config,
            vector_store=vector_store,
            embedder=embedder,
            reranker=reranker,
            telemetry=telemetry,
            security_policy=security_policy,
        )
        await gantry.collect_tools_from_modules(modules, attr=attr)
        return gantry

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
        # If modules were provided in constructor but not yet loaded, load them now
        if self._modules is not None:
            await self.collect_tools_from_modules(self._modules, attr=self._module_attr)
            self._modules = None
            self._module_attr = None

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

    async def collect_tools_from_modules(
        self,
        modules: Sequence[str],
        attr: str = "tools",
    ) -> int:
        """
        Import AgentGantry instances from other modules and register their tools locally.

        This is useful when you split tools across multiple files (e.g., a tools/ package). The
        tools are re-embedded with this gantry's embedder and added to its vector store and
        registry so they can be retrieved and executed without sharing vector stores.

        Args:
            modules: Iterable of module paths (dot-notation) to import.
            attr: Attribute name on each module that holds an AgentGantry instance (default "tools").

        Returns:
            Number of tools imported into this gantry.
            
        Raises:
            ValueError: If a module doesn't expose an AgentGantry at the specified attribute.
        """

        imported = 0
        seen: set[str] = set()

        for module_path in modules:
            module = importlib.import_module(module_path)
            other = getattr(module, attr, None)
            if not isinstance(other, AgentGantry):
                raise ValueError(
                    f"Module '{module_path}' does not expose an AgentGantry instance at '{attr}'. "
                    f"Found: {type(other).__name__ if other else 'None'}"
                )

            # Collect tools from the source gantry:
            # 1. Already registered/synced tools from the registry
            source_tools = other._registry.list_tools()
            
            # 2. Pending tools that haven't been synced yet
            # Use explicit checks to validate API expectations
            pending_from_registry = []
            if hasattr(other._registry, "get_pending") and callable(other._registry.get_pending):
                try:
                    pending_from_registry = other._registry.get_pending() or []
                except Exception as e:
                    logger.warning(
                        f"Failed to get pending tools from registry in '{module_path}': {e}"
                    )
            
            pending_unsynced = getattr(other, "_pending_tools", []) or []

            all_tools = [*source_tools, *pending_from_registry, *pending_unsynced]
            
            for tool in all_tools:
                key = f"{tool.namespace}.{tool.name}"
                
                # Check for duplicates across modules
                if key in seen:
                    logger.warning(
                        f"Skipping duplicate tool '{key}' from module '{module_path}'. "
                        f"A tool with this name was already imported from another module."
                    )
                    continue
                
                # Get the tool handler from the source gantry
                handler = other._registry.get_handler(key)

                # Add the tool to this gantry (will be embedded and synced)
                await self.add_tool(tool)
                
                # Register the handler if available
                if handler:
                    self._registry.register_handler(key, handler)
                    self._tool_handlers[tool.name] = handler
                else:
                    logger.debug(f"No handler found for tool '{key}' in module '{module_path}'")

                seen.add(key)
                imported += 1
            
            logger.info(f"Imported {len(all_tools)} tools from module '{module_path}'")

        return imported

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
        if self._config.reranker.enabled and self._reranker is not None:
            if query.enable_reranking is None:
                query.enable_reranking = True
        # Use telemetry span if available, otherwise use a no-op async context manager
        class _AsyncNoopContext:
            async def __aenter__(self) -> _AsyncNoopContext:
                return self

            async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> bool:
                return False

        span_cm = (
            self._telemetry.span("tool_retrieval", {"query": query.context.query})
            if self._telemetry else _AsyncNoopContext()
        )
        async with span_cm:
            routing_result = await self._router.route(query)

        # routing_result.tools is a list of (tool, semantic_score) tuples
        scored = []
        for tool, semantic_score in routing_result.tools:
            scored.append(
                ScoredTool(
                    tool=tool,
                    semantic_score=semantic_score,
                    rerank_score=None,  # Rerank scores handled separately if needed
                )
            )

        total_time_ms = (perf_counter() - overall_start) * 1000
        retrieval = RetrievalResult(
            tools=scored,
            query_embedding_time_ms=routing_result.query_embedding_time_ms,
            vector_search_time_ms=routing_result.vector_search_time_ms,
            rerank_time_ms=routing_result.rerank_time_ms,
            total_time_ms=total_time_ms,
            candidate_count=routing_result.candidate_count,
            filtered_count=routing_result.filtered_count,
            trace_id=str(uuid.uuid4()),
        )
        if self._telemetry:
            await self._telemetry.record_retrieval(query, retrieval)
        return retrieval

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

    async def add_mcp_server(self, config: MCPServerConfig) -> int:
        """
        Add an MCP server to discover and register its tools.

        Args:
            config: Configuration for the MCP server

        Returns:
            Number of tools discovered and registered
        """
        from agent_gantry.adapters.executors.mcp_client import MCPClient

        await self._ensure_initialized()

        # Create MCP client
        client = MCPClient(config)

        # Discover tools from the server
        tools = await client.list_tools()

        # Add tools to the gantry
        for tool in tools:
            await self.add_tool(tool)

        return len(tools)

    async def serve_mcp(
        self, transport: str = "stdio", mode: str = "dynamic", name: str = "agent-gantry"
    ) -> None:
        """
        Start serving as an MCP server.

        Args:
            transport: Transport type ("stdio" or "sse")
            mode: Server mode ("dynamic", "static", or "hybrid")
            name: Server name for identification
        """
        from agent_gantry.servers.mcp_server import create_mcp_server

        await self._ensure_initialized()
        if self._config.auto_sync:
            await self.sync()

        server = create_mcp_server(self, mode=mode, name=name)

        if transport == "stdio":
            await server.run_stdio()
        elif transport == "sse":
            await server.run_sse()
        else:
            raise ValueError(f"Unsupported transport: {transport}")

    async def add_a2a_agent(self, config: A2AAgentConfig) -> int:
        """
        Add an A2A agent to discover and register its skills as tools.

        Args:
            config: Configuration for the A2A agent

        Returns:
            Number of skills discovered and registered as tools
        """
        from agent_gantry.providers.a2a_client import A2AClient

        await self._ensure_initialized()

        # Create A2A client
        client = A2AClient(config)

        # Discover agent and its skills
        await client.discover()
        tools = await client.list_tools()

        # Add tools to the gantry
        for tool in tools:
            await self.add_tool(tool)

        return len(tools)

    def serve_a2a(self, host: str = "0.0.0.0", port: int = 8080) -> None:
        """
        Start serving as an A2A agent.

        Args:
            host: Host to bind to
            port: Port to listen on

        Note:
            This method requires FastAPI and uvicorn to be installed.
            Install with: pip install fastapi uvicorn
        """
        try:
            import uvicorn  # type: ignore[import-not-found]
        except ImportError as e:
            raise ImportError(
                "uvicorn is required for A2A server. Install with: pip install fastapi uvicorn"
            ) from e

        from agent_gantry.servers.a2a_server import create_a2a_server

        # Create FastAPI app
        base_url = f"http://{host}:{port}"
        app = create_a2a_server(self, base_url=base_url)

        # Run server
        uvicorn.run(app, host=host, port=port)

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

    def _build_vector_store(self, config: VectorStoreConfig) -> VectorStoreAdapter:
        """Construct a vector store adapter from configuration."""
        if config.type == "qdrant":
            if not config.url:
                raise ValueError("Qdrant vector store requires a 'url' in the configuration.")
            return QdrantVectorStore(url=config.url, api_key=config.api_key)
        if config.type == "chroma":
            if not config.url:
                raise ValueError("Chroma vector store requires a 'url' in the configuration.")
            return ChromaVectorStore(url=config.url, api_key=config.api_key)
        if config.type == "pgvector":
            if not config.url:
                raise ValueError("PGVector vector store requires a 'url' in the configuration.")
            return PGVectorStore(url=config.url, api_key=config.api_key)
        if config.type == "lancedb":
            from agent_gantry.adapters.vector_stores.lancedb import LanceDBVectorStore

            return LanceDBVectorStore(
                db_path=config.db_path,
                tools_table=config.collection_name,
                dimension=config.dimension or 768,
            )
        return InMemoryVectorStore()

    def _build_embedder(self, config: EmbedderConfig) -> EmbeddingAdapter:
        """Construct an embedder from configuration."""
        if config.type == "openai" and config.api_key:
            return OpenAIEmbedder(config)
        if config.type == "azure" and config.api_key:
            return AzureOpenAIEmbedder(config)
        if config.type == "nomic":
            from agent_gantry.adapters.embedders.nomic import NomicEmbedder

            return NomicEmbedder(
                model=config.model or "nomic-ai/nomic-embed-text-v1.5",
                dimension=config.dimension,
                task_type=config.task_type or "search_document",
            )
        return SimpleEmbedder()

    def _build_reranker(self, config: RerankerConfig) -> RerankerAdapter | None:
        """Construct a reranker from configuration."""
        if not config.enabled:
            return None
        if config.type == "cohere":
            return CohereReranker(model=config.model)
        return None

    def _build_telemetry(self, config: TelemetryConfig) -> TelemetryAdapter:
        """Construct telemetry adapter from configuration."""
        if not config.enabled:
            return NoopTelemetryAdapter()
        if config.type == "opentelemetry":
            return OpenTelemetryAdapter(
                service_name=config.service_name,
                otlp_endpoint=config.otlp_endpoint,
            )
        if config.type == "prometheus":
            return PrometheusTelemetryAdapter(
                service_name=config.service_name,
                prometheus_port=config.prometheus_port,
            )
        return ConsoleTelemetryAdapter()

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
        if param_type is int or param_type == "int":
            param_schema["type"] = "integer"
        elif param_type is float or param_type == "float":
            param_schema["type"] = "number"
        elif param_type is bool or param_type == "bool":
            param_schema["type"] = "boolean"
        elif param_type is str or param_type == "str":
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
