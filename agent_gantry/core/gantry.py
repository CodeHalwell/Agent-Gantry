"""
Main AgentGantry facade.

Primary entry point for the Agent-Gantry library.
"""

from __future__ import annotations

import importlib
import logging
import uuid
from collections.abc import Callable, Sequence
from time import perf_counter
from typing import TYPE_CHECKING, Any

from agent_gantry.adapters.embedders.openai import AzureOpenAIEmbedder, OpenAIEmbedder
from agent_gantry.adapters.embedders.simple import SimpleEmbedder
from agent_gantry.adapters.vector_stores.memory import InMemoryVectorStore
from agent_gantry.core.executor import ExecutionEngine
from agent_gantry.core.registry import ToolRegistry
from agent_gantry.core.router import RoutingWeights, SemanticRouter
from agent_gantry.core.security import SecurityPolicy
from agent_gantry.observability.console import ConsoleTelemetryAdapter, NoopTelemetryAdapter
from agent_gantry.observability.opentelemetry_adapter import (
    OpenTelemetryAdapter,
    PrometheusTelemetryAdapter,
)
from agent_gantry.utils.fingerprint import compute_tool_fingerprint
from agent_gantry.schema.config import (
    A2AAgentConfig,
    AgentGantryConfig,
    EmbedderConfig,
    MCPServerConfig,
    RerankerConfig,
    TelemetryConfig,
    VectorStoreConfig,
)
from agent_gantry.schema.introspection import build_parameters_schema
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

        # Initialize LLM client for intent classification if enabled
        self._llm_client = None
        if self._config.routing.use_llm_for_intent:
            from agent_gantry.adapters.llm_client import LLMClient
            try:
                self._llm_client = LLMClient(self._config.routing.llm)
            except Exception as e:
                logger.warning(f"Failed to initialize LLM client for intent classification: {e}")

        routing_weights = RoutingWeights(**self._config.routing.weights)
        self._router = SemanticRouter(
            vector_store=self._vector_store,
            embedder=self._embedder,
            reranker=self._reranker,
            weights=routing_weights,
            llm_client=self._llm_client,
            use_llm_for_intent=self._config.routing.use_llm_for_intent,
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
        self._synced = False  # Track if we've done initial sync check
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
    async def quick_start(
        cls,
        embedder: str = "auto",
        dimension: int = 256,
        **kwargs: Any,
    ) -> AgentGantry:
        """
        Quick setup with sensible defaults for getting started.

        Automatically detects the best available embedder and sets up
        an in-memory vector store for immediate use.

        Args:
            embedder: Embedder type - "auto", "nomic", "openai", or "simple"
            dimension: Embedding dimension (for Nomic, default 256)
            **kwargs: Additional AgentGantry constructor arguments

        Returns:
            Ready-to-use AgentGantry instance

        Example:
            >>> gantry = await AgentGantry.quick_start()
            >>>
            >>> @gantry.register
            ... def my_tool(x: int) -> int:
            ...     '''Double a number.'''
            ...     return x * 2
            >>>
            >>> await gantry.sync()
            >>> tools = await gantry.retrieve_tools("double a number")
        """
        import warnings

        config = AgentGantryConfig()
        embedder_instance: EmbeddingAdapter

        if embedder == "auto":
            # Try Nomic first (best for local use)
            try:
                # Test that sentence-transformers is actually available
                import sentence_transformers  # noqa: F401

                from agent_gantry.adapters.embedders.nomic import NomicEmbedder
                embedder_instance = NomicEmbedder(dimension=dimension)
            except ImportError:
                warnings.warn(
                    "Nomic embedder not available. Using SimpleEmbedder (hash-based, low accuracy). "
                    "For better semantic search: pip install agent-gantry[nomic]",
                    UserWarning,
                    stacklevel=2,
                )
                embedder_instance = SimpleEmbedder()
        elif embedder == "nomic":
            try:
                from agent_gantry.adapters.embedders.nomic import NomicEmbedder
            except ImportError as exc:
                raise ImportError(
                    "Nomic embedder is not available. To enable it, install the optional "
                    "dependencies:\n"
                    "  pip install agent-gantry[nomic]"
                ) from exc

            try:
                import sentence_transformers  # noqa: F401
            except ImportError as exc:
                raise ImportError(
                    "sentence-transformers is required for the Nomic embedder. Install it with:\n"
                    "  pip install agent-gantry[nomic]"
                ) from exc

            embedder_instance = NomicEmbedder(dimension=dimension)
        elif embedder == "openai":
            api_key = kwargs.pop("openai_api_key", None)
            if not api_key:
                raise ValueError(
                    "OpenAI embedder requires a valid API key. "
                    "Pass openai_api_key=... to quick_start() or configure AgentGantryConfig."
                )
            embedder_config = EmbedderConfig(type="openai", api_key=api_key)
            try:
                embedder_instance = OpenAIEmbedder(embedder_config)
            except Exception as exc:
                raise RuntimeError(
                    "Failed to initialize OpenAI embedder. Ensure optional dependencies are "
                    'installed with "pip install agent-gantry[openai]" and that your OpenAI '
                    "API key is valid."
                ) from exc
        else:  # "simple" or unknown
            embedder_instance = SimpleEmbedder()

        return cls(config=config, embedder=embedder_instance, **kwargs)

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
        await gantry.collect_tools_from_modules(modules, module_attr=attr)
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
            parameters_schema = build_parameters_schema(fn)

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

            # Register both tool definition and handler in the registry
            key = f"{namespace}.{tool_name}"
            self._registry.register_tool(tool)
            self._registry.register_handler(key, fn)

            return fn

        if func is not None:
            return decorator(func)
        return decorator

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
        self._pending_tools.append(tool)
        if self._config.auto_sync:
            await self.sync()

    async def sync(self, batch_size: int = 100, force: bool = False) -> int:
        """
        Sync pending registrations to vector store with smart change detection.

        This method uses fingerprinting to detect which tools have actually changed
        and only re-embeds those tools. On subsequent runs with the same tools,
        this operation is nearly instant.

        Args:
            batch_size: Number of tools to embed and sync in each batch
            force: If True, re-embed all tools regardless of fingerprints

        Returns:
            Number of tools synced (0 if nothing changed)
        """
        # If modules were provided in constructor but not yet loaded, load them now
        if self._modules is not None:
            await self.collect_tools_from_modules(self._modules, module_attr=self._module_attr or "tools")
            self._modules = None
            self._module_attr = None

        await self._ensure_initialized()

        # Get all registered tools (pending + already registered)
        all_tools = self.export_tools()
        if not all_tools:
            self._synced = True
            return 0

        # Compute fingerprints for current tools
        current_fingerprints = {
            f"{t.namespace}.{t.name}": compute_tool_fingerprint(t)
            for t in all_tools
        }

        # Get stored fingerprints from vector store (if supported)
        stored_fingerprints: dict[str, str] = {}
        embedder_id = self._get_embedder_id()
        needs_full_resync = force

        stored_fingerprints = await self._vector_store.get_stored_fingerprints()

        # Check if embedder changed (requires full re-embed)
        stored_embedder = await self._vector_store.get_metadata("embedder_id")
        stored_dim = await self._vector_store.get_metadata("dimension")

        if stored_embedder and stored_embedder != embedder_id:
            logger.info(
                f"Embedder changed from '{stored_embedder}' to '{embedder_id}'. "
                "Full re-sync required."
            )
            needs_full_resync = True
        elif stored_dim and int(stored_dim) != self._vector_store.dimension:
            logger.info(
                f"Dimension changed from {stored_dim} to {self._vector_store.dimension}. "
                "Full re-sync required."
            )
            needs_full_resync = True

        # Determine which tools need syncing
        if needs_full_resync:
            tools_to_sync = all_tools
        else:
            tools_to_sync = []
            for tool in all_tools:
                tool_id = f"{tool.namespace}.{tool.name}"
                current_fp = current_fingerprints[tool_id]
                stored_fp = stored_fingerprints.get(tool_id, "")

                if current_fp != stored_fp:
                    tools_to_sync.append(tool)
                    if stored_fp:
                        logger.debug(f"Tool '{tool_id}' changed, will re-embed")
                    else:
                        logger.debug(f"Tool '{tool_id}' is new, will embed")

        # Nothing to sync
        if not tools_to_sync:
            logger.debug(f"All {len(all_tools)} tools up-to-date, skipping sync")
            self._synced = True

            # Ensure handlers are registered even if tools are already in DB
            for tool in all_tools:
                self._registry.register_tool(tool)

            return 0

        logger.info(f"Syncing {len(tools_to_sync)}/{len(all_tools)} tools to vector store...")

        # Clear pending tools since we're processing them
        self._pending_tools = []

        total_synced = 0
        for i in range(0, len(tools_to_sync), batch_size):
            batch = tools_to_sync[i : i + batch_size]
            texts = [t.to_searchable_text() for t in batch]
            embeddings = await self._embedder.embed_batch(texts)
            count = await self._vector_store.add_tools(batch, embeddings, upsert=True)

            # Register tools in registry
            for tool in batch:
                self._registry.register_tool(tool)

            total_synced += count

        # Update sync metadata (if supported)
        await self._vector_store.update_sync_metadata(
            embedder_id=embedder_id,
            dimension=self._vector_store.dimension,
        )

        # Ensure all tools are registered (even those not synced)
        for tool in all_tools:
            if tool not in tools_to_sync:
                self._registry.register_tool(tool)

        self._synced = True
        logger.info(f"Synced {total_synced} tools")
        return total_synced

    def _get_embedder_id(self) -> str:
        """
        Get a unique identifier for the current embedder configuration.

        Returns:
            String identifier combining embedder class and key params
        """
        embedder_class = self._embedder.__class__.__name__

        # Try to get dimension from embedder
        dimension = getattr(self._embedder, "dimension", None)
        if dimension is None:
            dimension = getattr(self._embedder, "_dimension", None)

        # Try to get model name
        model = getattr(self._embedder, "model", None)
        if model is None:
            model = getattr(self._embedder, "_model_name", None)

        parts = [embedder_class]
        if model:
            parts.append(str(model))
        if dimension:
            parts.append(f"dim{dimension}")

        return "-".join(parts)

    async def ensure_synced(self) -> None:
        """
        Ensure tools are synced to the vector store.

        This is called automatically before retrieval operations.
        Uses smart fingerprinting to avoid unnecessary re-embedding.
        """
        if not self._synced:
            await self.sync()

    async def collect_tools_from_modules(
        self,
        modules: Sequence[str],
        module_attr: str = "tools",
    ) -> int:
        """
        Import AgentGantry instances from other modules and register their tools locally.

        This is useful when you split tools across multiple files (e.g., a tools/ package). The
        tools are re-embedded with this gantry's embedder and added to its vector store and
        registry so they can be retrieved and executed without sharing vector stores.

        Args:
            modules: Iterable of module paths (dot-notation) to import.
            module_attr: Attribute name on each module that holds an AgentGantry instance (default "tools").

        Returns:
            Number of tools imported into this gantry.

        Raises:
            ValueError: If a module doesn't expose an AgentGantry at the specified attribute.
        """

        imported = 0
        seen: set[str] = set()
        tools_to_add: list[ToolDefinition] = []

        for module_path in modules:
            module = importlib.import_module(module_path)
            other = getattr(module, module_attr, None)
            if not isinstance(other, AgentGantry):
                raise ValueError(
                    f"Module '{module_path}' does not expose an AgentGantry instance at '{module_attr}'. "
                    f"Found: {type(other).__name__ if other else 'None'}"
                )

            # Collect tools from the source gantry using the public API
            all_tools = other.export_tools()

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

                # Add to batch for efficient processing
                tools_to_add.append(tool)

                # Register the handler if available
                if handler:
                    self._registry.register_handler(key, handler)
                    self._tool_handlers[tool.name] = handler
                else:
                    logger.debug(f"No handler found for tool '{key}' in module '{module_path}'")

                seen.add(key)
                imported += 1

            logger.info(f"Imported {len(all_tools)} tools from module '{module_path}'")

        if tools_to_add:
            await self._ensure_initialized()
            batch_size = 100
            for i in range(0, len(tools_to_add), batch_size):
                batch = tools_to_add[i : i + batch_size]
                texts = [t.to_searchable_text() for t in batch]
                embeddings = await self._embedder.embed_batch(texts)
                await self._vector_store.add_tools(batch, embeddings, upsert=True)
                for tool in batch:
                    self._registry.register_tool(tool)

        return imported

    async def retrieve(self, query: ToolQuery) -> RetrievalResult:
        """
        Core semantic routing function.

        Automatically ensures tools are synced before retrieval using smart
        fingerprint-based change detection.

        Args:
            query: The tool query with context and filters

        Returns:
            RetrievalResult with scored tools
        """
        await self._ensure_initialized()
        
        # Auto-sync with smart change detection
        await self.ensure_synced()

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
        dialect: str = "openai",
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        """
        Convenience wrapper: returns provider-specific tool schemas.

        Args:
            query: The natural language query
            limit: Maximum number of tools to return
            dialect: Target dialect/provider name (default: 'openai')
                Supported: 'openai', 'anthropic', 'gemini', 'mistral', 'groq', 'auto'
            **kwargs: Additional query parameters (e.g., score_threshold)

        Returns:
            List of provider-specific tool schemas
        """
        from agent_gantry.schema.query import ConversationContext, ToolQuery

        context = ConversationContext(query=query)
        tool_query = ToolQuery(context=context, limit=limit, **kwargs)
        result = await self.retrieve(tool_query)
        return result.to_dialect(dialect)

    async def execute(self, call: ToolCall) -> ToolResult:
        """
        Execute a tool call with full protections.

        Args:
            call: The tool call to execute

        Returns:
            Result of the tool execution
        """
        await self._ensure_initialized()
        
        # Auto-sync to ensure handlers are registered
        await self.ensure_synced()

        if self._telemetry:
            async with self._telemetry.span("tool_execution", {"tool_name": call.tool_name}):
                return await self._executor.execute(call)
        else:
            return await self._executor.execute(call)

    async def search_and_execute(
        self,
        query: str,
        arguments: dict[str, Any] | None = None,
        limit: int = 1,
        **kwargs: Any,
    ) -> ToolResult:
        """
        One-shot convenience: search for a tool and execute it.

        Combines retrieve_tools() and execute() into a single operation.
        Useful for simple scripting and quick tool invocation.

        Args:
            query: Natural language query to find the tool
            arguments: Arguments to pass to the tool (optional)
            limit: Number of tools to retrieve (default: 1, uses best match)
            **kwargs: Additional retrieval parameters (score_threshold, etc.)

        Returns:
            Result of executing the best matching tool

        Raises:
            ValueError: If no matching tools found

        Example:
            >>> result = await gantry.search_and_execute(
            ...     "calculate tax on 100",
            ...     arguments={"amount": 100.0}
            ... )
            >>> print(result.result)
            8.0
        """
        from agent_gantry.schema.execution import ToolCall
        from agent_gantry.schema.query import ConversationContext, ToolQuery

        # Retrieve best matching tool
        context = ConversationContext(query=query)
        tool_query = ToolQuery(context=context, limit=limit, **kwargs)
        result = await self.retrieve(tool_query)

        if not result.tools:
            raise ValueError(
                f"No tools found matching query: '{query}'. "
                f"Try a different query or check registered tools."
            )

        # Use the best scoring tool
        best_tool = result.tools[0].tool
        tool_name = best_tool.name

        # Use provided arguments or empty dict
        if arguments is None:
            arguments = {}

        # Execute the tool
        return await self.execute(
            ToolCall(tool_name=tool_name, arguments=arguments)
        )

    async def execute_batch(self, batch: BatchToolCall) -> BatchToolResult:
        """
        Execute multiple tool calls.

        Args:
            batch: The batch of tool calls

        Returns:
            Results of all tool executions
        """
        await self._ensure_initialized()
        
        # Auto-sync to ensure handlers are registered
        await self.ensure_synced()

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
        await self.ensure_synced()

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
            import uvicorn
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
        await self.ensure_synced()
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
        await self.ensure_synced()
        return await self._vector_store.list_all(namespace=namespace)

    def export_tools(self) -> list[ToolDefinition]:
        """
        Export all registered and pending tools.

        Useful for importing tools into another AgentGantry instance
        without accessing private attributes.

        Returns:
            List of all tool definitions (registered + pending)
        """
        registered = self._registry.list_tools()
        pending = self._pending_tools.copy()

        # Deduplicate by qualified name
        seen: set[str] = set()
        result: list[ToolDefinition] = []

        for tool in registered + pending:
            key = f"{tool.namespace}.{tool.name}"
            if key not in seen:
                seen.add(key)
                result.append(tool)

        return result

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
        import asyncio

        await self._ensure_initialized()

        results = {
            "vector_store": await self._vector_store.health_check(),
            "embedder": await self._embedder.health_check(),
        }

        if self._telemetry is not None:
            try:
                health_method = getattr(self._telemetry, "health_check", None)
                if health_method is not None:
                    if asyncio.iscoroutinefunction(health_method):
                        results["telemetry"] = await health_method()
                    else:
                        results["telemetry"] = bool(health_method())
                else:
                    results["telemetry"] = True
            except Exception:
                results["telemetry"] = False

        return results

    def _build_vector_store(self, config: VectorStoreConfig) -> VectorStoreAdapter:
        """Construct a vector store adapter from configuration."""
        if config.type == "qdrant":
            from agent_gantry.adapters.vector_stores.remote import QdrantVectorStore

            if not config.url:
                raise ValueError("Qdrant requires 'url' in configuration")
            return QdrantVectorStore(
                url=config.url,
                api_key=config.api_key,
                collection_name=config.collection_name,
                dimension=config.dimension or 1536,
            )
        if config.type == "chroma":
            from agent_gantry.adapters.vector_stores.remote import ChromaVectorStore

            return ChromaVectorStore(
                url=config.url,
                collection_name=config.collection_name,
                persist_directory=config.db_path,
            )
        if config.type == "pgvector":
            from agent_gantry.adapters.vector_stores.remote import PGVectorStore

            if not config.url:
                raise ValueError("PGVector requires 'url' (connection string) in configuration")
            return PGVectorStore(
                url=config.url,
                table_name=config.collection_name,
                dimension=config.dimension or 1536,
            )
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
            from agent_gantry.adapters.rerankers.cohere import CohereReranker

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


def create_default_gantry(dimension: int = 256) -> AgentGantry:
    """
    Factory function to create a pre-configured AgentGantry instance.

    This provides a convenient way to create an AgentGantry instance with
    sensible defaults, automatically selecting the best available embedder:
    - NomicEmbedder (if sentence-transformers is available)
    - SimpleEmbedder (fallback, hash-based)

    This avoids module-level instantiation which can cause issues with
    testing, cleanup, or if multiple instances are needed.

    Args:
        dimension: Embedding dimension for Nomic embedder (default: 256).
                   Ignored if NomicEmbedder is not available.

    Returns:
        A configured AgentGantry instance ready for tool registration.

    Example:
        >>> from agent_gantry import create_default_gantry
        >>>
        >>> tools = create_default_gantry()
        >>>
        >>> @tools.register(tags=["math"])
        ... def add(a: int, b: int) -> int:
        ...     '''Add two numbers.'''
        ...     return a + b

    Note:
        For better semantic search quality, install the Nomic dependencies:
        `pip install agent-gantry[nomic]`
    """
    import warnings

    embedder: EmbeddingAdapter

    # Try to use NomicEmbedder if available
    try:
        import sentence_transformers  # noqa: F401

        from agent_gantry.adapters.embedders.nomic import NomicEmbedder

        embedder = NomicEmbedder(dimension=dimension)
    except ImportError:
        warnings.warn(
            "Nomic embedder not available. Using SimpleEmbedder (hash-based, low accuracy). "
            "For better semantic search: pip install agent-gantry[nomic]",
            UserWarning,
            stacklevel=2,
        )
        embedder = SimpleEmbedder()

    return AgentGantry(embedder=embedder)
