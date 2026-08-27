"""
Main AgentGantry facade.

Primary entry point for the Agent-Gantry library.
"""

from __future__ import annotations

import asyncio
import importlib
import inspect
import logging
import uuid
from collections.abc import Callable, Sequence
from datetime import datetime, timezone
from time import perf_counter
from typing import TYPE_CHECKING, Any

from agent_gantry.adapters.embedders.openai import OpenAIEmbedder
from agent_gantry.adapters.embedders.simple import SimpleEmbedder
from agent_gantry.core.executor import ExecutionEngine
from agent_gantry.core.factories import (
    build_embedder,
    build_reranker,
    build_telemetry,
    build_vector_store,
)
from agent_gantry.core.registry import ToolRegistry
from agent_gantry.core.router import RoutingWeights, SemanticRouter
from agent_gantry.core.security import SecurityPolicy
from agent_gantry.schema.config import (
    A2AAgentConfig,
    AgentGantryConfig,
    EmbedderConfig,
    MCPServerConfig,
)
from agent_gantry.schema.introspection import build_parameters_schema
from agent_gantry.schema.mcp import MCPServerDefinition
from agent_gantry.schema.query import RetrievalResult, ScoredTool, ToolQuery
from agent_gantry.schema.skill import Skill, SkillSearchResult
from agent_gantry.schema.tool import ToolCapability, ToolDefinition
from agent_gantry.utils.fingerprint import compute_tool_fingerprint

if TYPE_CHECKING:
    from agent_gantry.adapters.embedders.base import EmbeddingAdapter
    from agent_gantry.adapters.rerankers.base import RerankerAdapter
    from agent_gantry.adapters.vector_stores.base import VectorStoreAdapter
    from agent_gantry.core.rate_limiter import RateLimiter
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
        self._vector_store = vector_store or build_vector_store(self._config.vector_store)
        self._embedder = embedder or build_embedder(self._config.embedder)
        self._reranker = reranker or build_reranker(self._config.reranker)
        self._telemetry = telemetry or build_telemetry(self._config.telemetry)
        self._security_policy = security_policy or SecurityPolicy()
        self._registry = ToolRegistry()

        # SyncManager has no optional-dependency requirements and the MCP
        # manager depends on it, so build it before wiring MCP support.
        from agent_gantry.core.sync_manager import SyncManager

        self._sync_manager = SyncManager(
            vector_store=self._vector_store,
            embedder=self._embedder,
            registry=self._registry,
        )

        self._create_mcp_components()
        self._llm_client = self._create_llm_client()
        self._router = self._create_router()
        self._rate_limiter = self._create_rate_limiter()
        self._executor = self._create_executor()
        self._init_runtime_state(modules, module_attr)

    def _create_mcp_components(self) -> None:
        """Wire the optional MCP registry/router/manager trio.

        MCP is an optional dependency: if the ``mcp`` package is unavailable the
        three attributes stay ``None`` and MCP features degrade gracefully. The
        manager depends on :attr:`_sync_manager`, so this runs after it exists.
        """
        self._mcp_registry = None
        self._mcp_router = None
        self._mcp_manager = None

        # Tier 1 — the optional 'mcp' package being absent is the expected, quiet
        # path: log at DEBUG and leave MCP disabled.
        try:
            from agent_gantry.core.mcp_manager import MCPManager
            from agent_gantry.core.mcp_registry import MCPRegistry
            from agent_gantry.core.mcp_router import MCPRouter
        except ImportError:
            logger.debug("MCP support not available (install 'mcp' package to enable)")
            return

        # Tier 2 — with the package importable, a construction failure is
        # unexpected (broken/partial install). Surface it at WARNING rather than
        # hiding it at DEBUG, but still degrade to no-MCP across the board so a
        # bad MCP stack never crashes AgentGantry() construction.
        try:
            self._mcp_registry = MCPRegistry()
            self._mcp_router = MCPRouter(
                vector_store=self._vector_store,
                embedder=self._embedder,
                registry=self._mcp_registry,
            )
            self._mcp_manager = MCPManager(
                vector_store=self._vector_store,
                embedder=self._embedder,
                registry=self._mcp_registry,
                router=self._mcp_router,
                get_embedder_id=self._sync_manager.get_embedder_id,
            )
        except Exception:
            logger.warning("MCP components failed to initialize; MCP disabled.", exc_info=True)
            self._mcp_registry = None
            self._mcp_router = None
            self._mcp_manager = None

    def _create_llm_client(self) -> Any:
        """Build the optional LLM client used for intent classification."""
        if not self._config.routing.use_llm_for_intent:
            return None
        from agent_gantry.adapters.llm_client import LLMClient

        try:
            return LLMClient(self._config.routing.llm)
        except Exception as e:
            logger.warning(f"Failed to initialize LLM client for intent classification: {e}")
            return None

    def _create_router(self) -> SemanticRouter:
        """Build the semantic router from routing config."""
        return SemanticRouter(
            vector_store=self._vector_store,
            embedder=self._embedder,
            reranker=self._reranker,
            weights=RoutingWeights(**self._config.routing.weights),
            llm_client=self._llm_client,
            use_llm_for_intent=self._config.routing.use_llm_for_intent,
        )

    def _create_rate_limiter(self) -> RateLimiter | None:
        """Build the optional rate limiter from execution config."""
        if not self._config.execution.rate_limit.enabled:
            return None
        from agent_gantry.core.rate_limiter import RateLimiter

        return RateLimiter(self._config.execution.rate_limit)

    def _create_executor(self) -> ExecutionEngine:
        """Build the execution engine from execution config."""
        return ExecutionEngine(
            registry=self._registry,
            default_timeout_ms=self._config.execution.default_timeout_ms,
            max_retries=self._config.execution.max_retries,
            circuit_breaker_threshold=self._config.execution.circuit_breaker_threshold,
            circuit_breaker_timeout_s=self._config.execution.circuit_breaker_timeout_s,
            security_policy=self._security_policy,
            telemetry=self._telemetry,
            rate_limiter=self._rate_limiter,
        )

    def _init_runtime_state(
        self, modules: Sequence[str] | None, module_attr: str
    ) -> None:
        """Initialise the mutable buffers, handler maps, callbacks, and flags.

        ``modules`` is stored for explicit async initialization later (via
        ``collect_tools_from_modules`` or ``AgentGantry.from_modules``); it is
        not loaded here because import + embedding is async.
        """
        self._pending_tools: list[ToolDefinition] = []
        # One-shot guard for _ensure_skill_vectors_current (embedder is fixed
        # for this instance's lifetime); lock created lazily on first use
        self._skill_vectors_checked = False
        self._skill_vectors_lock: asyncio.Lock | None = None
        self._pending_mcp_servers: list[MCPServerDefinition] = []
        self._tool_handlers: dict[str, Callable[..., Any]] = {}
        # MCP clients created by add_mcp_server (immediate discovery), kept so
        # their persistent connections can be reused by handlers and closed on
        # close(). Keyed by namespace-qualified config name.
        self._direct_mcp_clients: dict[str, Any] = {}
        # Framework-agnostic post-execution callbacks (see on_tool_call).
        self._tool_call_callbacks: list[Callable[..., Any]] = []
        self._initialized = False
        self._synced = False  # Track if we've done initial sync check
        self._mcp_synced = False  # Track if MCP servers are synced
        self._modules: Sequence[str] | None = modules or None
        self._module_attr: str | None = module_attr if modules else None

    async def close(self) -> None:
        """
        Release all resources held by this AgentGantry instance.

        Closes vector store connections, MCP clients, and any other resources.
        Safe to call multiple times.
        """
        # Close MCP clients (persistent server connections)
        for client in list(self._direct_mcp_clients.values()):
            try:
                await client.close()
            except Exception:
                logger.debug("Error closing MCP client", exc_info=True)
        self._direct_mcp_clients.clear()
        if self._mcp_registry is not None:
            close_clients = getattr(self._mcp_registry, "close_all_clients", None)
            if close_clients is not None:
                await close_clients()

        # Close the execution engine (cached A2A clients, etc.)
        await self._executor.close()

        # Close vector store if it has a close method
        close_method = getattr(self._vector_store, "close", None)
        if close_method:
            if asyncio.iscoroutinefunction(close_method):
                await close_method()
            else:
                close_method()

        # Close telemetry if it has a close method
        if self._telemetry:
            telemetry_close = getattr(self._telemetry, "close", None)
            if telemetry_close:
                if asyncio.iscoroutinefunction(telemetry_close):
                    await telemetry_close()
                else:
                    telemetry_close()

        self._initialized = False

    async def __aenter__(self) -> AgentGantry:
        """Enter async context manager."""
        return self

    async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> bool:
        """Exit async context manager, releasing resources."""
        await self.close()
        return False

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
    def quick_start(
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
            >>> gantry = AgentGantry.quick_start()
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
            # Accept agent_framework FunctionTool (or any tool-spec wrapper
            # exposing .name + .func). AF's @tool decorator produces an
            # object without ``__name__``, which would otherwise break the
            # legacy ``fn.__name__`` lookup. Unwrap to the bare callable
            # while remembering the original wrapper so the decorator's
            # return value still matches what the caller passed in.
            original = fn
            wrapper_name = getattr(fn, "name", None)
            wrapper_func = getattr(fn, "func", None)
            wrapper_desc = getattr(fn, "description", None)
            if wrapper_func is not None and callable(wrapper_func):
                underlying = wrapper_func
            else:
                underlying = fn

            tool_name = (
                name
                or wrapper_name
                or getattr(underlying, "__name__", None)
                or getattr(original, "__name__", None)
            )
            if not tool_name:
                raise TypeError(
                    "AgentGantry.register() could not determine a tool name. "
                    "Pass `name=...` explicitly, or supply a callable with a "
                    "__name__ (or an agent_framework FunctionTool with .name)."
                )

            tool_description = (
                (underlying.__doc__ if hasattr(underlying, "__doc__") else None)
                or wrapper_desc
                or f"Tool: {tool_name}"
            )

            # Build parameters schema from the underlying callable's signature.
            parameters_schema = build_parameters_schema(underlying)

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

            # Register both tool definition and handler in the registry.
            # _tool_handlers is keyed by the namespace-qualified name so
            # same-named tools in different namespaces never clobber each
            # other (its only consumer is the tool_count property).
            key = f"{namespace}.{tool_name}"
            self._tool_handlers[key] = underlying
            self._registry.register_tool(tool)
            self._registry.register_handler(key, underlying)

            return original

        if func is not None:
            return decorator(func)
        return decorator

    async def _ensure_initialized(self) -> None:
        """Initialize backing services once."""
        if not self._initialized:
            await self._vector_store.initialize()
            self._initialized = True

    async def add_tool(
        self, tool: ToolDefinition, handler: Callable[..., Any] | None = None
    ) -> None:
        """
        Add a tool definition directly, optionally wiring an execution handler.

        Args:
            tool: The tool definition to add
            handler: Optional callable that executes the tool. When provided,
                it is registered exactly like the ``register()`` decorator wires
                a decorated function's handler, so the tool is immediately
                executable via :meth:`execute` (security, retries, telemetry —
                the same path as ``@gantry.register``ed tools). Omit this for
                sources that dispatch through their own executor instead of a
                registry-held handler (MCP/A2A discovery already do this, and
                keep working unchanged since ``handler`` defaults to ``None``).
        """
        self._pending_tools.append(tool)
        if handler is not None:
            # Mirror register(): the definition goes into the registry right
            # away (not just _pending_tools) so the tool is executable before
            # the next sync() even with auto_sync=False, and the handler map
            # is keyed by the namespace-qualified name to avoid cross-namespace
            # clobbering.
            key = f"{tool.namespace}.{tool.name}"
            self._registry.register_tool(tool)
            self._registry.register_handler(key, handler)
            self._tool_handlers[key] = handler
        if self._config.auto_sync:
            await self.sync()

    async def _detect_changes(self, all_tools: list[ToolDefinition], force: bool) -> list[ToolDefinition]:
        """Detect which tools need to be synced. Delegates to SyncManager."""
        return await self._sync_manager.detect_changes(all_tools, force)

    async def _sync_batches(self, tools_to_sync: list[ToolDefinition], batch_size: int) -> int:
        """Embed and save tools in batches. Delegates to SyncManager."""
        return await self._sync_manager.sync_batches(tools_to_sync, batch_size)

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
            await self.collect_tools_from_modules(
                self._modules, module_attr=self._module_attr or "tools"
            )
            self._modules = None
            self._module_attr = None

        await self._ensure_initialized()

        # Get all registered tools (pending + already registered)
        all_tools = self.export_tools()
        if not all_tools:
            self._synced = True
            return 0

        # Determine which tools need syncing
        tools_to_sync = await self._detect_changes(all_tools, force)

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

        total_synced = await self._sync_batches(tools_to_sync, batch_size)

        # Update sync metadata (if supported)
        await self._sync_manager.update_metadata()

        # Ensure all tools are registered (even those not synced). Compare by
        # qualified name — `tool in tools_to_sync` would deep-compare Pydantic
        # models pairwise (O(N²) over full schemas).
        synced_keys = {f"{t.namespace}.{t.name}" for t in tools_to_sync}
        for tool in all_tools:
            if f"{tool.namespace}.{tool.name}" not in synced_keys:
                self._registry.register_tool(tool)

        self._synced = True
        logger.info(f"Synced {total_synced} tools")
        return total_synced

    def _get_embedder_id(self) -> str:
        """Get a unique identifier for the current embedder configuration."""
        return self._sync_manager.get_embedder_id()

    async def ensure_synced(self) -> None:
        """
        Ensure tools are synced to the vector store.

        This is called automatically before retrieval operations.
        Uses smart fingerprinting to avoid unnecessary re-embedding.
        """
        if not self._synced:
            await self.sync()

    async def _ensure_mcp_synced(self) -> None:
        """
        Ensure MCP servers are synced to the vector store.

        Similar to ensure_synced() but for MCP server metadata.
        """
        if not self._mcp_synced:
            await self.sync_mcp_servers()

    async def sync_mcp_servers(self, batch_size: int = 100, force: bool = False) -> int:
        """
        Sync pending MCP server registrations to vector store.

        Uses smart fingerprinting to avoid unnecessary re-embedding.

        Args:
            batch_size: Number of servers to embed and sync in each batch
            force: If True, re-embed all servers regardless of fingerprints

        Returns:
            Number of servers synced
        """
        await self._ensure_initialized()

        if self._mcp_registry is None:
            self._mcp_synced = True
            return 0

        # Get all registered servers
        all_servers = self._mcp_registry.list_servers()
        if not all_servers:
            self._mcp_synced = True
            return 0

        # Transform all MCP servers into pseudo-tools to calculate fingerprints
        pseudo_tools_map: dict[str, ToolDefinition] = {}
        for server in all_servers:
            pseudo_name = f"mcp_server_{server.namespace}_{server.name}".replace("-", "_")
            pseudo_tool = ToolDefinition(
                name=pseudo_name,
                namespace="__mcp_servers__",
                description=server.to_searchable_text(),
                parameters_schema={"type": "object", "properties": {}},
                metadata={
                    "entity_type": "mcp_server",
                    "server_name": server.name,
                    "server_namespace": server.namespace,
                    "server_tags": server.tags,
                    "server_capabilities": server.capabilities,
                    "server_command": server.command,
                },
            )
            pseudo_tools_map[f"{server.namespace}.{server.name}"] = pseudo_tool

        # Compute fingerprints for current servers (using their pseudo-tool representations)
        current_fingerprints = {
            f"{server.namespace}.{server.name}": compute_tool_fingerprint(pseudo_tool)
            for server in all_servers
            for pseudo_tool in [pseudo_tools_map[f"{server.namespace}.{server.name}"]]
        }

        # Get stored fingerprints from vector store
        embedder_id = self._get_embedder_id()
        needs_full_resync = force

        stored_fingerprints = await self._vector_store.get_stored_fingerprints()

        # Check if embedder changed (requires full re-embed)
        stored_embedder = await self._vector_store.get_metadata("embedder_id")
        stored_dim = await self._vector_store.get_metadata("dimension")

        if stored_embedder and stored_embedder != embedder_id:
            logger.info(
                f"Embedder changed from '{stored_embedder}' to '{embedder_id}'. "
                "Full re-sync required for MCP servers."
            )
            needs_full_resync = True
        elif stored_dim and int(stored_dim) != self._vector_store.dimension:
            logger.info(
                f"Dimension changed from {stored_dim} to {self._vector_store.dimension}. "
                "Full re-sync required for MCP servers."
            )
            needs_full_resync = True

        # Determine which servers need syncing
        if needs_full_resync:
            servers_to_sync = all_servers
        else:
            servers_to_sync = []
            for server in all_servers:
                server_id = f"{server.namespace}.{server.name}"
                pseudo_name = f"mcp_server_{server.namespace}_{server.name}".replace("-", "_")
                pseudo_tool_id = f"__mcp_servers__.{pseudo_name}"

                current_fp = current_fingerprints[server_id]
                # Stored fingerprint uses the pseudo_tool_id
                stored_fp = stored_fingerprints.get(pseudo_tool_id, "")

                if current_fp != stored_fp:
                    servers_to_sync.append(server)
                    if stored_fp:
                        logger.debug(f"MCP server '{server_id}' changed, will re-embed")
                    else:
                        logger.debug(f"MCP server '{server_id}' is new, will embed")

        # Clear pending servers since we're handling sync via fingerprints now
        self._mcp_registry.clear_pending()

        # Nothing to sync
        if not servers_to_sync:
            logger.debug(f"All {len(all_servers)} MCP servers up-to-date, skipping sync")
            self._mcp_synced = True
            return 0

        logger.info(f"Syncing {len(servers_to_sync)}/{len(all_servers)} MCP servers to vector store...")

        total_synced = 0
        for i in range(0, len(servers_to_sync), batch_size):
            batch = servers_to_sync[i : i + batch_size]
            texts = [s.to_searchable_text() for s in batch]
            embeddings = await self._embedder.embed_batch(texts)

            # Get the pre-created pseudo-tools for the batch
            pseudo_tools = [pseudo_tools_map[f"{server.namespace}.{server.name}"] for server in batch]

            # Use the existing add_tools method with upsert=True
            count = await self._vector_store.add_tools(pseudo_tools, embeddings, upsert=True)
            total_synced += count

        # Update sync metadata (if supported)
        await self._vector_store.update_sync_metadata(
            embedder_id=embedder_id,
            dimension=self._vector_store.dimension,
        )

        self._mcp_synced = True
        logger.info(f"Synced {total_synced} MCP servers")
        return total_synced

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
                    self._tool_handlers[key] = handler
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

        # SimpleEmbedder produces hash-based scores that cluster tightly
        # regardless of semantic relevance. Pairing it with a non-zero
        # score_threshold typically results in "0 tools surfaced" silently
        # (which integrators report as a frustrating first-day failure).
        # Warn loudly on the first such retrieval call.
        if isinstance(self._embedder, SimpleEmbedder) and not SimpleEmbedder._warned_about_threshold:
            threshold = getattr(query, "score_threshold", None) or 0.0
            if threshold > 0.0:
                SimpleEmbedder._warned_about_threshold = True
                import warnings as _warnings

                _warnings.warn(
                    "SimpleEmbedder produces hash-based similarity scores with "
                    "no semantic understanding; pairing it with "
                    f"score_threshold={threshold} will likely filter all "
                    "tools out. Set score_threshold=0.0 for SimpleEmbedder, "
                    "or install a real embedder: "
                    "pip install agent-gantry[nomic]",
                    UserWarning,
                    stacklevel=3,
                )

        overall_start = perf_counter()
        if (
            self._config.reranker.enabled
            and self._reranker is not None
            and query.enable_reranking is None
        ):
            # Copy rather than assign: this branch was unreachable while the
            # field defaulted to False, so making it live also made the
            # in-place mutation live, and a caller reusing one ToolQuery (a
            # cached template, say) would have the field flipped underneath it.
            query = query.model_copy(update={"enable_reranking": True})

        # Use telemetry span if available, otherwise use a no-op async context manager
        from agent_gantry.utils.async_utils import AsyncNoopContext

        span_cm = (
            self._telemetry.span("tool_retrieval", {"query": query.context.query})
            if self._telemetry else AsyncNoopContext()
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

    @property
    def telemetry(self) -> Any:
        """The configured telemetry adapter (``None`` when disabled).

        Exposed so the integration layers can report provider-reported token
        usage without reaching into a private attribute.
        """
        return self._telemetry

    async def retrieve_tools(
        self,
        query: str,
        limit: int = 5,
        dialect: str = "openai",
        score_threshold: float = 0.0,
        dialect_options: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        """
        Convenience wrapper: returns provider-specific tool schemas.

        Args:
            query: The natural language query
            limit: Maximum number of tools to return
            dialect: Target dialect/provider name (default: 'openai')
                Supported: 'openai', 'openai_responses', 'anthropic', 'gemini',
                'mistral', 'groq', 'agent_framework', 'auto'
            score_threshold: Minimum cosine score to keep a tool. Defaults to
                ``0.0`` (no filtering) — unlike the raw :class:`ToolQuery`
                default of ``0.5``. A 0.5 absolute cutoff silently drops correct
                tools for embedders whose scores sit below it (e.g. MiniLM), so
                the high-level convenience API opts out of filtering by default
                and lets ranking + ``limit`` do the work.
            dialect_options: Options forwarded to the dialect adapter rather
                than the query — e.g. ``{"strict": True}`` for OpenAI. Any
                keyword in ``**kwargs`` that is not a :class:`ToolQuery` field
                is routed here too, so ``retrieve_tools(..., strict=True)``
                works; this explicit dict wins on conflict.
            **kwargs: Additional query parameters. Keywords matching a
                ``ToolQuery`` field configure retrieval; the rest are treated
                as dialect options (see ``dialect_options``). Previously any
                non-``ToolQuery`` keyword was silently discarded, so ``strict``
                never reached the adapter.

        Returns:
            List of provider-specific tool schemas
        """
        from agent_gantry.schema.query import ConversationContext, ToolQuery

        # Split retrieval parameters from per-dialect adapter options so that
        # neither silently swallows the other's keywords.
        query_fields = set(ToolQuery.model_fields)
        query_kwargs = {k: v for k, v in kwargs.items() if k in query_fields}
        adapter_options = {k: v for k, v in kwargs.items() if k not in query_fields}
        if dialect_options:
            adapter_options.update(dialect_options)

        context = ConversationContext(query=query)
        tool_query = ToolQuery(
            context=context, limit=limit, score_threshold=score_threshold, **query_kwargs
        )
        result = await self.retrieve(tool_query)
        return result.to_dialect(dialect, **adapter_options)

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
                result = await self._executor.execute(call)
        else:
            result = await self._executor.execute(call)

        await self._emit_tool_call(call, result)
        return result

    def on_tool_call(self, callback: Callable[..., Any]) -> Callable[[], None]:
        """Register a callback fired after every tool execution.

        This is the framework-agnostic observability seam: every call routed
        through :meth:`execute` (and each call in :meth:`execute_batch`) invokes
        the registered callbacks with a single
        :class:`~agent_gantry.schema.execution.ToolCallEvent` once the call
        finishes — whether it succeeded, failed, timed out, or was denied.
        Because ``execute`` is the single choke point every framework adapter
        flows through, one ``on_tool_call`` registration yields logging/metrics
        across LangChain, CrewAI, Agent Framework, direct calls, and the rest —
        no per-framework middleware required.

        Callbacks may be sync or async and are invoked in registration order.
        Exceptions raised by a callback are caught and logged (never propagated
        into the tool run), so a broken listener cannot break execution.

        .. note::
           **Batch timing differs from single execute.** In :meth:`execute`
           the event fires the instant that call finishes. In
           :meth:`execute_batch` events are emitted *after the whole batch
           completes* — one per call, paired with its result by index — so for
           a ``parallel`` batch they all arrive together at the end rather than
           in completion order. The per-call ``ToolCallEvent.latency_ms`` is
           still accurate; only the *delivery* time is batched, which matters
           if you timestamp events for latency dashboards.

        Registering the same callable twice registers it twice (it will fire
        twice); the returned unsubscribe removes one registration.

        Args:
            callback: A callable accepting a single ``ToolCallEvent`` argument.

        Returns:
            An unsubscribe function; call it to remove the registration.
        """
        self._tool_call_callbacks.append(callback)

        def _unsubscribe() -> None:
            try:
                self._tool_call_callbacks.remove(callback)
            except ValueError:
                pass

        return _unsubscribe

    async def _emit_tool_call(self, call: ToolCall, result: ToolResult) -> None:
        """Build a ``ToolCallEvent`` and dispatch it to registered callbacks.

        Error-isolated: a callback raising must never affect the tool result or
        sibling callbacks. Awaitable return values are awaited so async
        listeners work transparently.
        """
        if not self._tool_call_callbacks:
            return
        # Local import: ToolCallEvent isn't imported at module top (the
        # execution-schema imports there are TYPE_CHECKING-only). Importing it
        # here — after the early-return guard above — also keeps it off the
        # path entirely when no listeners are registered.
        from agent_gantry.schema.execution import ToolCallEvent

        event = ToolCallEvent(call=call, result=result)
        for callback in list(self._tool_call_callbacks):
            try:
                outcome = callback(event)
                if inspect.isawaitable(outcome):
                    await outcome
            except Exception:
                logger.exception(
                    "on_tool_call callback %r raised; continuing.",
                    getattr(callback, "__name__", callback),
                )

    async def search_and_execute(
        self,
        query: str,
        arguments: dict[str, Any] | None = None,
        limit: int = 1,
        score_threshold: float = 0.0,
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
        tool_query = ToolQuery(
            context=context, limit=limit, score_threshold=score_threshold, **kwargs
        )
        result = await self.retrieve(tool_query)

        if not result.tools:
            raise ValueError(
                f"No tools found matching query: '{query}'. "
                f"Try a different query or check registered tools."
            )

        # Use the best scoring tool
        best_tool = result.tools[0].tool

        # Use provided arguments or empty dict
        if arguments is None:
            arguments = {}

        # Execute the tool we actually selected: passing the namespace keeps a
        # same-named tool in another namespace from being run instead.
        return await self.execute(
            ToolCall(
                tool_name=best_tool.name,
                namespace=best_tool.namespace,
                arguments=arguments,
            )
        )

    async def execute_tool_calls(
        self,
        response: Any,
        *,
        dialect: str = "openai",
        timeout_ms: int = 30000,
        parallel: bool = True,
    ) -> list[dict[str, Any]]:
        """Run every tool call in ``response`` and format the results to send back.

        Closes the loop that the dialect adapters could already describe but
        nothing drove: extract the calls (including parallel ones), execute
        each through the full protection stack, and format each result in the
        provider's own reply shape, ready to append to the conversation.

        Failures are reported to the model rather than raised: a tool that
        errors comes back as an error-flagged result so the model can react,
        which is what a tool-use loop needs. Genuine programming errors — an
        unknown dialect, a malformed response — still raise.

        Args:
            response: The provider response, an equivalent dict, or an already
                extracted list of ``ToolCallPayload`` (e.g. from
                :class:`~agent_gantry.adapters.tool_spec.round_trip.StreamingToolCallAccumulator`).
            dialect: Which provider shape to read and reply in.
            timeout_ms: Per-call execution timeout.
            parallel: Execute the calls concurrently (default). Set ``False``
                to run them in the order the model emitted, for tools that
                share mutable state.

        Returns:
            One provider-formatted tool result per call, in emission order.
            Empty when the model called no tools.

        Example:
            >>> response = await client.chat.completions.create(...)  # doctest: +SKIP
            >>> results = await gantry.execute_tool_calls(response)  # doctest: +SKIP
            >>> messages.extend(results)  # doctest: +SKIP
        """
        from agent_gantry.adapters.tool_spec.base import ToolCallPayload
        from agent_gantry.adapters.tool_spec.registry import get_adapter
        from agent_gantry.adapters.tool_spec.round_trip import extract_tool_calls
        from agent_gantry.schema.execution import ExecutionStatus

        adapter = get_adapter(dialect)
        if isinstance(response, list) and all(
            isinstance(item, ToolCallPayload) for item in response
        ):
            payloads = response
        else:
            payloads = extract_tool_calls(response, dialect)
        if not payloads:
            return []

        calls = [adapter.to_tool_call(p, timeout_ms=timeout_ms) for p in payloads]

        if parallel and len(calls) > 1:
            results = await asyncio.gather(*(self.execute(call) for call in calls))
        else:
            results = [await self.execute(call) for call in calls]

        formatted: list[dict[str, Any]] = []
        for payload, result in zip(payloads, results):
            is_error = result.status != ExecutionStatus.SUCCESS
            formatted.append(
                adapter.format_tool_result(
                    payload.tool_name,
                    f"Error: {result.error}" if is_error else result.result,
                    payload.tool_call_id,
                    is_error=is_error,
                )
            )
        return formatted

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
                batch_result = await self._executor.execute_batch(batch)
        else:
            batch_result = await self._executor.execute_batch(batch)

        if self._tool_call_callbacks:
            # Results align with calls by index for every strategy: parallel
            # gather preserves order, and sequential (incl. fail_fast early
            # break) yields an aligned prefix, so zip pairs them correctly.
            # Events are delivered here, after the whole batch finishes — but
            # each ToolResult.latency_ms was measured by the executor at that
            # call's own completion, so per-call latency stays accurate even
            # though delivery is batched. Don't timestamp events at this point.
            for call, result in zip(batch.calls, batch_result.results):
                await self._emit_tool_call(call, result)
        return batch_result

    async def add_mcp_server(self, config: MCPServerConfig) -> int:
        """
        Add an MCP server to discover and register its tools.

        Note: This method immediately discovers and registers ALL tools from the server.
        For dynamic server selection, use register_mcp_server() instead.

        Args:
            config: Configuration for the MCP server

        Returns:
            Number of tools discovered and registered
        """
        from agent_gantry.adapters.executors.mcp_client import MCPClient

        await self._ensure_initialized()

        # Reuse the client (and its persistent connection) across repeat
        # calls. Keyed by namespace-qualified name — the same identity the
        # MCP registry uses — and invalidated when the config itself changed
        # (same name, different command/env/etc.), so a re-add never keeps
        # executing against the old server process.
        client_key = f"{config.namespace}.{config.name}"
        existing = self._direct_mcp_clients.get(client_key)
        if existing is not None and existing.config == config:
            client = existing
            tools = await client.list_tools()
        else:
            # New or reconfigured server: verify the replacement can discover
            # BEFORE committing the swap. Closing/evicting the old client
            # first would, on discovery failure, leave registry handlers
            # closing over a client that gantry.close() can no longer reach
            # — still executing against the obsolete command.
            client = MCPClient(config)
            try:
                tools = await client.list_tools()
            except BaseException:
                try:
                    await client.close()
                except Exception:
                    logger.debug("Error closing failed MCP client", exc_info=True)
                raise
            if existing is not None:
                try:
                    await existing.close()
                except Exception:
                    logger.debug("Error closing reconfigured MCP client", exc_info=True)
            self._direct_mcp_clients[client_key] = client

        # Wire execution handlers so discovered tools run through
        # gantry.execute() (security, retries, telemetry) via the MCP client.
        self._register_mcp_tool_handlers(client, tools)
        await self._remove_stale_mcp_tools(client, tools)

        # Add all tools first, then sync once — per-tool add_tool() would
        # trigger a full sync (and a size-1 embedding batch) per tool when
        # auto_sync is enabled.
        self._pending_tools.extend(tools)
        if self._config.auto_sync:
            await self.sync()

        return len(tools)

    def _register_mcp_tool_handlers(self, client: Any, tools: list[ToolDefinition]) -> None:
        """Register execution handlers that proxy tool calls to an MCP client.

        Without a handler, MCP-discovered tools are retrievable but fail with
        "No handler found" when executed through the engine.
        """

        def make_handler(tool_name: str) -> Callable[..., Any]:
            async def mcp_tool_handler(**arguments: Any) -> Any:
                return await client.call_tool(tool_name, arguments)

            mcp_tool_handler.__name__ = f"mcp_{tool_name}"
            return mcp_tool_handler

        for tool in tools:
            key = f"{tool.namespace}.{tool.name}"
            # Qualified-name dedup (export_tools) keeps the first-seen
            # definition, so overwriting only the handler on a collision would
            # validate and authorize against one tool while dispatching to
            # another. Keep first-wins for definition AND handler: skip when
            # the name is already owned by a different source. Re-discovery
            # from the same MCP server stays idempotent.
            registered = self._registry.get_tool(tool.name, tool.namespace)
            existing = registered or next(
                (
                    t
                    for t in self._pending_tools
                    if t.name == tool.name and t.namespace == tool.namespace
                ),
                None,
            )
            if existing is not None and existing.metadata.get("mcp_server") != client.config.name:
                logger.warning(
                    f"MCP server '{client.config.name}' exposes tool '{key}', which is "
                    f"already registered by a different source; keeping the existing "
                    f"tool. Use a distinct namespace to register both."
                )
                continue
            if existing is not None:
                # Same-server re-discovery: the schema may have changed (e.g.
                # add_mcp_server re-called with a reconfigured server), and
                # first-wins dedup would otherwise keep the stale definition
                # while the handler dispatches to the new server. Drop stale
                # pending copies (the caller re-appends the fresh ones); the
                # register_tool below refreshes the registered copy.
                self._pending_tools[:] = [
                    t
                    for t in self._pending_tools
                    if not (t.name == tool.name and t.namespace == tool.namespace)
                ]
            # Register the definition immediately, mirroring add_tool(): the
            # tool must be executable before the next sync() even with
            # auto_sync=False — a handler alone isn't enough, execute() looks
            # the definition up in the registry.
            self._registry.register_tool(tool)
            handler = make_handler(tool.name)
            self._registry.register_handler(key, handler)
            self._tool_handlers[key] = handler

    async def _remove_stale_mcp_tools(self, client: Any, tools: list[ToolDefinition]) -> None:
        """Remove this server's previously registered tools that its latest
        discovery no longer exposes.

        A stale tool's handler closes over the replaced client, so executing
        it would reconnect using the old server command — retrieval could
        silently dispatch work to the obsolete process.
        """
        server_name = client.config.name
        namespace = client.config.namespace
        fresh = {tool.name for tool in tools}

        def is_stale(t: ToolDefinition) -> bool:
            return (
                t.namespace == namespace
                and t.metadata.get("mcp_server") == server_name
                and t.name not in fresh
            )

        stale = [t for t in self._registry.list_tools(namespace) if is_stale(t)]
        for t in stale:
            self._registry.delete_tool(t.name, t.namespace)
            self._tool_handlers.pop(f"{t.namespace}.{t.name}", None)
            try:
                await self._vector_store.delete(t.name, t.namespace)
            except Exception:
                logger.debug("Error removing stale MCP tool from vector store", exc_info=True)
        self._pending_tools[:] = [t for t in self._pending_tools if not is_stale(t)]
        if stale:
            logger.info(
                f"Removed {len(stale)} stale tool(s) no longer exposed by "
                f"MCP server '{namespace}.{server_name}'"
            )

    def register_mcp_server(
        self,
        name: str,
        command: list[str],
        *,
        description: str,
        namespace: str = "default",
        args: list[str] | None = None,
        env: dict[str, str] | None = None,
        tags: list[str] | None = None,
        examples: list[str] | None = None,
        capabilities: list[str] | None = None,
    ) -> None:
        """
        Register an MCP server for dynamic selection via semantic routing.

        Unlike add_mcp_server(), this method does NOT immediately connect to the server.
        Instead, it registers the server metadata for semantic search, and tools are
        discovered on-demand when the server is selected.

        Args:
            name: Unique name for the server
            command: Command to start the MCP server (e.g., ["npx", "@modelcontextprotocol/server-filesystem"])
            description: Description of what the server provides
            namespace: Namespace for organizing servers
            args: Additional command-line arguments
            env: Environment variables for the server process
            tags: Tags for categorizing the server
            examples: Example queries this server handles
            capabilities: Server capabilities (e.g., "read_files", "write_files")

        Example:
            >>> gantry.register_mcp_server(
            ...     name="filesystem",
            ...     command=["npx", "-y", "@modelcontextprotocol/server-filesystem"],
            ...     description="Provides tools for reading and writing files on the local filesystem",
            ...     args=["--path", "/tmp"],
            ...     tags=["filesystem", "files", "io"],
            ...     examples=["read a file", "write to a file", "list directory contents"],
            ...     capabilities=["read_files", "write_files", "list_directory"],
            ... )
        """
        from agent_gantry.schema.mcp import MCPServerDefinition

        server_def = MCPServerDefinition(
            name=name,
            namespace=namespace,
            description=description,
            command=command,
            args=args or [],
            env=env or {},
            tags=tags or [],
            examples=examples or [],
            capabilities=capabilities or [],
        )

        # Register in MCP registry and mark as pending for sync
        self._mcp_registry.register_server(server_def)
        self._mcp_registry.add_pending(server_def)

        logger.info(f"Registered MCP server: {server_def.qualified_name}")

    async def retrieve_mcp_servers(
        self,
        query: str,
        limit: int = 3,
        score_threshold: float | None = None,
        namespaces: list[str] | None = None,
    ) -> list[MCPServerDefinition]:
        """
        Retrieve relevant MCP servers based on a query using semantic search.

        Args:
            query: Natural language query describing needed functionality
            limit: Maximum number of servers to return
            score_threshold: Minimum similarity score (0-1)
            namespaces: Filter by server namespaces

        Returns:
            List of relevant MCP server definitions

        Example:
            >>> servers = await gantry.retrieve_mcp_servers(
            ...     "read and write files",
            ...     limit=2
            ... )
            >>> for server in servers:
            ...     print(f"Server: {server.name} - {server.description}")
        """
        await self._ensure_initialized()
        await self._ensure_mcp_synced()

        result = await self._mcp_router.route(
            query=query,
            limit=limit,
            score_threshold=score_threshold,
            namespaces=namespaces,
        )

        return [scored.server for scored in result.servers]

    async def discover_tools_from_server(
        self,
        server_name: str,
        namespace: str = "default",
        timeout: float = 30.0,
    ) -> int:
        """
        Dynamically discover and register tools from a previously registered MCP server.

        This method connects to the server, discovers its tools, and adds them to
        the gantry's tool registry.

        Args:
            server_name: Name of the registered MCP server
            namespace: Server namespace
            timeout: Connection timeout in seconds (default: 30.0)

        Returns:
            Number of tools discovered and registered

        Raises:
            ValueError: If the server is not registered
            TimeoutError: If the connection or tool discovery times out
            Exception: If tool discovery fails

        Example:
            >>> # First register the server metadata
            >>> gantry.register_mcp_server(...)
            >>> await gantry.sync()
            >>>
            >>> # Later, discover tools on-demand
            >>> count = await gantry.discover_tools_from_server("filesystem")
            >>> print(f"Discovered {count} tools")
        """
        await self._ensure_initialized()

        # Get the MCP client for this server
        client = self._mcp_registry.get_client(server_name, namespace)
        if not client:
            raise ValueError(
                f"MCP server '{namespace}.{server_name}' not found. "
                f"Register it first with register_mcp_server()."
            )

        try:
            # Discover tools from the server with timeout protection
            tools = await asyncio.wait_for(client.list_tools(), timeout=timeout)

            # Wire execution handlers (tools execute via the MCP client)
            self._register_mcp_tool_handlers(client, tools)
            await self._remove_stale_mcp_tools(client, tools)

            # Add all tools first, then sync once (avoids per-tool full syncs)
            self._pending_tools.extend(tools)
            if self._config.auto_sync:
                await self.sync()

            # Update server health
            self._mcp_registry.update_health(
                server_name,
                namespace,
                available=True,
                last_success=datetime.now(timezone.utc),
                consecutive_failures=0,
            )

            logger.info(f"Discovered {len(tools)} tools from MCP server: {namespace}.{server_name}")
            return len(tools)

        except asyncio.TimeoutError:
            # Update server health on timeout
            server = self._mcp_registry.get_server(server_name, namespace)
            consecutive_failures = server.health.consecutive_failures + 1 if server else 1

            self._mcp_registry.update_health(
                server_name,
                namespace,
                available=False,
                last_failure=datetime.now(timezone.utc),
                consecutive_failures=consecutive_failures,
            )

            logger.error(
                f"Timeout while discovering tools from MCP server {namespace}.{server_name} "
                f"(timeout: {timeout}s)"
            )
            raise TimeoutError(
                f"MCP server {namespace}.{server_name} did not respond within {timeout}s"
            )

        except Exception as e:
            # Update server health on failure
            server = self._mcp_registry.get_server(server_name, namespace)
            consecutive_failures = server.health.consecutive_failures + 1 if server else 1

            self._mcp_registry.update_health(
                server_name,
                namespace,
                available=False,
                last_failure=datetime.now(timezone.utc),
                consecutive_failures=consecutive_failures,
            )

            logger.error(f"Failed to discover tools from MCP server {namespace}.{server_name}: {e}")
            raise

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

        # Discovery-only client: close it when done so its persistent HTTP
        # connections are released deterministically (execution goes through
        # the engine's own A2AExecutor clients, not this one).
        client = A2AClient(config)
        try:
            await client.discover()
            tools = await client.list_tools()
        finally:
            await client.close()

        # Add all tools first, then sync once (avoids per-tool full syncs)
        self._pending_tools.extend(tools)
        if self._config.auto_sync:
            await self.sync()

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

    @property
    def embedder(self) -> EmbeddingAdapter:
        """Return the embedder this gantry is using.

        Public accessor exposed so sibling modules (CLI helpers,
        the registry linter, custom tooling) can perform their own
        embedding work without reaching into ``_embedder``.
        """
        return self._embedder

    async def get_tool(self, name: str, namespace: str = "default") -> ToolDefinition | None:
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

    def list_tools_sync(
        self,
        namespace: str | None = None,
    ) -> list[ToolDefinition]:
        """
        Synchronously list locally-registered tools.

        Reads from the in-memory registry only — does not hit the vector
        store and does not trigger sync. This is the natural API for
        read-only inspection (telemetry, debug printers, validators) where
        going through ``await`` is awkward and the in-memory view is
        sufficient.

        Args:
            namespace: Filter by namespace.

        Returns:
            List of tool definitions known to the registry.
        """
        registered = self._registry.list_tools()
        pending = self._pending_tools

        seen: set[str] = set()
        out: list[ToolDefinition] = []
        for tool in (*registered, *pending):
            key = f"{tool.namespace}.{tool.name}"
            if key in seen:
                continue
            if namespace is not None and tool.namespace != namespace:
                continue
            seen.add(key)
            out.append(tool)
        return out

    async def preview(
        self,
        query: str,
        *,
        limit: int = 10,
        score_threshold: float | None = None,
        **query_kwargs: Any,
    ) -> list[tuple[str, float]]:
        """
        Preview which tools would surface for a given query.

        Read-only helper for calibrating ``score_threshold`` and ``top_k``
        without spinning up the full agent. Returns ``(tool_name, score)``
        pairs sorted by descending score.

        Args:
            query: Natural-language query to retrieve against.
            limit: Maximum number of candidates to return (default 10).
            score_threshold: Override the score threshold. ``None`` uses
                the configured default of ``0.0`` so you can see the full
                ranking, including tools that would be filtered out.
            **query_kwargs: Forwarded to :class:`ToolQuery` /
                :class:`ConversationContext` (``namespaces``, etc.).

        Returns:
            List of ``(qualified_name, semantic_score)`` tuples sorted
            descending by score. Qualified name is ``namespace.name``.

        Example:
            >>> ranked = await gantry.preview(
            ...     "find boundaries in OCR text",
            ...     limit=10,
            ...     score_threshold=0.0,
            ... )
            >>> for name, score in ranked:
            ...     print(f"{score:.3f}  {name}")
        """
        from agent_gantry.schema.query import ConversationContext, ToolQuery

        threshold = 0.0 if score_threshold is None else score_threshold
        context = ConversationContext(query=query)
        tool_query = ToolQuery(
            context=context,
            limit=limit,
            score_threshold=threshold,
            **query_kwargs,
        )
        result = await self.retrieve(tool_query)
        return [
            (f"{st.tool.namespace}.{st.tool.name}", st.semantic_score)
            for st in result.tools
        ]

    async def analyze_registry(
        self,
        *,
        similarity_threshold: float = 0.85,
        tag_overlap_share: float = 0.5,
    ) -> Any:
        """Lint the registry for common tool-description mistakes.

        Convenience wrapper around
        :func:`agent_gantry.utils.registry_linter.analyze_registry`.
        Returns a :class:`RegistryAnalysis` listing tools whose
        descriptions name other registered tools (which pulls them
        toward the wrong queries via embedding similarity), pairs of
        tools whose searchable text is too similar to disambiguate,
        and tags that appear on so many tools they no longer carry
        discriminative value.
        """
        from agent_gantry.utils.registry_linter import (
            analyze_registry as _analyze,
        )

        return await _analyze(
            self,
            similarity_threshold=similarity_threshold,
            tag_overlap_share=tag_overlap_share,
        )

    async def pairwise_similarity(self, tool_a: str, tool_b: str) -> float:
        """Cosine similarity between two registered tools' searchable text."""
        from agent_gantry.utils.registry_linter import (
            pairwise_similarity as _sim,
        )

        return await _sim(self, tool_a, tool_b)

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

    # ------------------------------------------------------------------
    # Skills — semantic procedural memory alongside tools: registered
    # once, retrieved by meaning per prompt, and injected as context
    # (never executed). Same embedder and vector store as tools.
    # ------------------------------------------------------------------

    _SKILL_STORE_METHODS = (
        "add_skills",
        "search_skills",
        "get_skill_by_name",
        "delete_skill",
        "list_all_skills",
        "count_skills",
    )

    def _skills_store(self) -> Any:
        """Return the vector store if it supports skills, else raise.

        Checks the full skill API surface up front so a partially
        implemented adapter fails with this clear message instead of an
        AttributeError from deep inside a later call.
        """
        store = self._vector_store
        missing = [m for m in self._SKILL_STORE_METHODS if not hasattr(store, m)]
        if missing:
            raise NotImplementedError(
                f"{type(store).__name__} does not support skills "
                f"(missing: {', '.join(missing)}). "
                f"InMemoryVectorStore (default) and LanceDBVectorStore do."
            )
        return store

    async def add_skill(self, skill: Skill) -> None:
        """
        Add a single skill. See :meth:`add_skills`.

        Args:
            skill: The skill to add
        """
        await self.add_skills([skill])

    async def _ensure_skill_vectors_current(self, store: Any) -> None:
        """Re-embed persisted skills if they were made by a different embedder.

        Tools get this via SyncManager's fingerprint/embedder-id machinery;
        skills need the equivalent or reopening a persistent store with a new
        embedding model silently searches the old model's vectors. Runs once
        per gantry instance — the embedder is fixed for the instance's
        lifetime. Stores without the metadata API skip the check (nothing to
        compare against).

        Limitations: switching to an embedder of a *different dimension*
        cannot be migrated in place on fixed-schema stores (LanceDB tables
        have a fixed vector width) — recreate the store instead; the failure
        is surfaced with that guidance. Concurrent gantry instances using
        *different* embedders against one shared store are unsupported: each
        would re-migrate to its own vector space, thrashing the other's.
        And the lock here serializes tasks within this process only —
        multiple worker *processes* sharing one embedded store path follow
        the backend's multi-writer semantics (LanceDB's upsert is a
        non-atomic delete-then-add), so serialize the first post-model-switch
        access externally when running multiple workers, or let one worker
        warm the store before the others start.
        """
        if self._skill_vectors_checked:
            return
        # Serialize: concurrent first calls must not skip past an in-flight
        # migration and search stale vectors, and the flag is only set after
        # full success so a transient failure gets retried instead of
        # permanently bypassing migration.
        if self._skill_vectors_lock is None:
            self._skill_vectors_lock = asyncio.Lock()
        async with self._skill_vectors_lock:
            if self._skill_vectors_checked:
                return
            get_meta = getattr(store, "get_metadata", None)
            set_meta = getattr(store, "set_metadata", None)
            if get_meta is not None and set_meta is not None:
                # Scope the marker to the skills table where the store names
                # one: multiple configured skills tables can share a single
                # metadata table (LanceDB), and an unscoped key would let one
                # table's migration mark the others as migrated too. Adapter
                # contract: stores whose metadata table is shared across
                # differently-configured skills tables should expose
                # _skills_table_name; stores without it get a store-wide
                # marker, which is correct when metadata is per-store.
                table_id = getattr(store, "_skills_table_name", None)
                marker_key = (
                    f"skills_embedder_id:{table_id}" if table_id else "skills_embedder_id"
                )
                current = self._sync_manager.get_embedder_id()
                stored = await get_meta(marker_key)
                if stored != current:
                    # Unknown or different embedder: re-embed whatever is
                    # stored (covers both a model switch and pre-existing
                    # skills with no recorded id)
                    skills = await store.list_all_skills(limit=1_000_000)
                    if skills:
                        embeddings = await self._embedder.embed_batch(
                            [skill.to_embedding_text() for skill in skills]
                        )
                        try:
                            await store.add_skills(skills, embeddings, upsert=True)
                        except Exception as exc:
                            raise RuntimeError(
                                f"Failed to re-embed stored skills for the current "
                                f"embedder ({current!r}). If the new embedder has a "
                                f"different dimension, fixed-schema stores (e.g. "
                                f"LanceDB) cannot be migrated in place — recreate the "
                                f"store, or use a same-dimension model."
                            ) from exc
                        logger.info(
                            f"Re-embedded {len(skills)} skill(s): stored embedder id "
                            f"{stored!r} != current {current!r}"
                        )
                    await set_meta(marker_key, current)
            self._skill_vectors_checked = True

    async def add_skills(self, skills: list[Skill]) -> int:
        """
        Embed and store skills for semantic retrieval.

        Args:
            skills: Skills to add (upserted by ``namespace.name``)

        Returns:
            Number of skills stored
        """
        if not skills:
            return 0
        store = self._skills_store()
        await self._ensure_initialized()
        await self._ensure_skill_vectors_current(store)
        embeddings = await self._embedder.embed_batch(
            [skill.to_embedding_text() for skill in skills]
        )
        return int(await store.add_skills(skills, embeddings, upsert=True))

    async def retrieve_skills(
        self,
        query: str,
        limit: int = 3,
        namespace: str | list[str] | None = None,
        category: str | None = None,
        score_threshold: float | None = None,
    ) -> list[SkillSearchResult]:
        """
        Retrieve the skills most relevant to a query.

        Args:
            query: Natural-language query (typically the user prompt)
            limit: Maximum number of skills to return
            namespace: Optional namespace (or list of namespaces) filter
            category: Optional category filter as its string value, e.g.
                ``"how_to"`` or ``SkillCategory.PATTERN.value``
            score_threshold: Minimum similarity score (0-1)

        Returns:
            Scored skills, most relevant first
        """
        # A blank query has no semantic signal: it embeds to a zero vector,
        # every skill ties at 0.0, and with no threshold the first `limit`
        # skills would be returned (and prompt-injected) arbitrarily.
        if not query.strip():
            return []
        store = self._skills_store()
        await self._ensure_initialized()
        await self._ensure_skill_vectors_current(store)
        query_embedding = await self._embedder.embed_text(query)
        filters: dict[str, Any] = {}
        if namespace is not None:
            filters["namespace"] = namespace
        if category is not None:
            filters["category"] = category
        matches = await store.search_skills(
            query_vector=query_embedding,
            limit=limit,
            filters=filters or None,
            score_threshold=score_threshold,
        )
        # Clamp: float32 cosine can exceed 1.0 by rounding error, and the
        # schema bounds score to [0, 1].
        return [
            SkillSearchResult(skill=skill, score=min(1.0, max(0.0, float(score))))
            for skill, score in matches
        ]

    async def retrieve_skills_as_prompt(
        self,
        query: str,
        limit: int = 3,
        namespace: str | list[str] | None = None,
        category: str | None = None,
        score_threshold: float | None = None,
    ) -> str:
        """
        Retrieve relevant skills formatted for system-prompt injection.

        Returns an empty string when nothing matches, so the result can be
        appended to a prompt unconditionally.

        Skill content is injected into the prompt verbatim — register skills
        only from sources you trust, exactly as you would tool descriptions.
        """
        results = await self.retrieve_skills(
            query,
            limit=limit,
            namespace=namespace,
            category=category,
            score_threshold=score_threshold,
        )
        return "\n\n".join(result.skill.to_prompt_text() for result in results)

    async def delete_skill(self, name: str, namespace: str = "default") -> bool:
        """Delete a skill by name."""
        store = self._skills_store()
        await self._ensure_initialized()
        return bool(await store.delete_skill(name, namespace))

    async def list_skills(
        self, namespace: str | None = None, limit: int = 1000, offset: int = 0
    ) -> list[Skill]:
        """List stored skills."""
        store = self._skills_store()
        await self._ensure_initialized()
        return list(await store.list_all_skills(namespace=namespace, limit=limit, offset=offset))

    async def count_skills(self, namespace: str | None = None) -> int:
        """Count stored skills."""
        store = self._skills_store()
        await self._ensure_initialized()
        return int(await store.count_skills(namespace=namespace))

    async def delete_tool(self, name: str, namespace: str = "default") -> bool:
        """
        Delete a tool.

        Removes the tool everywhere it lives: the vector store (so retrieval
        stops returning it), the in-memory registry and its handler map (so
        ``execute``, ``list_tools_sync`` and ``required=[...]`` pin resolution
        stop seeing it), and any not-yet-synced pending entry.

        Args:
            name: Tool name
            namespace: Tool namespace

        Returns:
            True if the tool was deleted from any of those places
        """
        await self._ensure_initialized()
        deleted_from_store = await self._vector_store.delete(name, namespace)
        deleted_from_registry = self._registry.delete_tool(name, namespace)
        # The facade keeps its own handler map (``tool_count`` reads it), so
        # purge the qualified key here too — the registry only clears its own.
        removed_handler = self._tool_handlers.pop(f"{namespace}.{name}", None)
        deleted_handler = removed_handler is not None
        if removed_handler is not None:
            # The executor memoizes per-handler argument coercers keyed by the
            # callable itself, so a deleted tool's handler would stay reachable
            # from that cache until eviction. Bounded, but there is no reason
            # to hold it.
            from agent_gantry.core.executor import forget_handler

            forget_handler(removed_handler)
        before = len(self._pending_tools)
        self._pending_tools = [
            t
            for t in self._pending_tools
            if not (t.name == name and t.namespace == namespace)
        ]
        return (
            deleted_from_store
            or deleted_from_registry
            or deleted_handler
            or len(self._pending_tools) < before
        )

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
    return AgentGantry.quick_start(embedder="auto", dimension=dimension)
