"""
Microsoft Agent Framework bridge for Agent-Gantry.

Provides seamless integration between Agent-Gantry's semantic tool routing
and Microsoft Agent Framework (1.0 GA) agents. The bridge converts Gantry
tool definitions into Python callables that AF agents can invoke directly,
enabling dynamic tool selection that reduces token usage in multi-agent systems.

Key classes:
    - ``GantryToolBridge``: Main bridge that wraps Gantry tools for AF agents.

Usage:
    from agent_gantry import AgentGantry
    from agent_gantry.integrations.agent_framework_bridge import GantryToolBridge

    gantry = AgentGantry()

    @gantry.register
    def get_weather(city: str) -> str:
        '''Get current weather for a city.'''
        return f"Weather in {city}: Sunny, 22C"

    await gantry.sync()

    bridge = GantryToolBridge(gantry)
    tools = await bridge.get_tools("What's the weather?", limit=3)

    # Pass directly to any AF agent:
    agent = client.as_agent(name="Assistant", instructions="...", tools=tools)
    result = await agent.run("What's the weather in London?")
"""

from __future__ import annotations

import inspect
import json
import logging
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Annotated, Any

from pydantic import Field

from agent_gantry.schema.execution import ToolCall
from agent_gantry.schema.tool import ToolCapability

if TYPE_CHECKING:
    from agent_gantry.core.gantry import AgentGantry
    from agent_gantry.schema.query import RetrievalResult
    from agent_gantry.schema.tool import ToolDefinition

logger = logging.getLogger(__name__)


# Capabilities that indicate a tool is potentially destructive / requires
# explicit human approval in production. Mapped to AF's
# ``approval_mode="always_require"`` so that AF surfaces an approval event
# before the tool actually runs.
_APPROVAL_REQUIRED_CAPS: frozenset[ToolCapability] = frozenset(
    {
        ToolCapability.WRITE_DATA,
        ToolCapability.DELETE_DATA,
        ToolCapability.EXECUTE_CODE,
        ToolCapability.FINANCIAL,
        ToolCapability.PII_ACCESS,
    }
)


def _try_import_af_tool() -> Any | None:
    """Return ``agent_framework.tool`` if the package is installed, else None.

    The bridge degrades gracefully: when AF is not installed (unit tests,
    LangChain-only users, etc.) ``_build_callable_for_tool`` returns a bare
    typed Python callable, which AF 1.0 still auto-wraps into a FunctionTool
    when passed into ``Agent(tools=[...])``. When AF *is* installed we return
    a genuine ``FunctionTool`` so that ``approval_mode``, ``max_invocations``
    and other GA-only metadata flow through to the agent.
    """
    try:
        from agent_framework import tool as af_tool

        return af_tool
    except Exception:  # pragma: no cover - exercised in environments without AF
        return None


def _tool_approval_mode(tool_def: ToolDefinition) -> str | None:
    """Map Gantry ``ToolCapability`` set to AF ``approval_mode``.

    Destructive / sensitive capabilities elevate the tool to
    ``"always_require"`` so AF pauses for human approval before invocation.
    Everything else returns ``None`` (AF defaults to ``"never_require"``).
    """
    caps = set(tool_def.capabilities)
    if caps & _APPROVAL_REQUIRED_CAPS:
        return "always_require"
    return None


def _build_callable_for_tool(
    tool_def: ToolDefinition,
    gantry: AgentGantry,
    *,
    as_function_tool: bool | None = None,
) -> Any:
    """
    Build a Python callable wrapping a Gantry tool for Microsoft Agent Framework.

    The callable is created with proper type annotations (using ``Annotated``
    and Pydantic ``Field``) so that AF can auto-generate the correct function
    schema for the LLM. This avoids sending raw JSON schemas and lets AF's
    native function-tool infrastructure handle serialisation.

    When ``agent-framework`` is importable and ``as_function_tool`` is not
    ``False``, the callable is wrapped with ``@agent_framework.tool`` so AF
    receives a real ``FunctionTool`` with full GA metadata
    (``approval_mode`` derived from Gantry capabilities, description,
    name). When AF is not installed, a plain typed async callable is
    returned; AF 1.0 still auto-wraps those at agent construction time.

    Args:
        tool_def: The Gantry ToolDefinition to wrap.
        gantry: The AgentGantry instance for execution.
        as_function_tool: If ``True``, always wrap with ``@agent_framework.tool``
            (raises ImportError when AF isn't available). If ``False``, always
            return a bare callable. ``None`` (default) auto-detects.

    Returns:
        Either an ``agent_framework.FunctionTool`` or a bare async callable,
        both accepted by ``Agent(tools=[...])``.
    """
    tool_name = tool_def.name
    tool_desc = tool_def.description
    params_schema = tool_def.parameters_schema

    # Extract parameter info from the JSON Schema
    properties = params_schema.get("properties", {})
    required_params = set(params_schema.get("required", []))

    async def _execute(**kwargs: Any) -> str:
        result = await gantry.execute(
            ToolCall(tool_name=tool_name, arguments=kwargs)
        )
        if result.status.value == "success":
            val = result.result
            return val if isinstance(val, str) else json.dumps(val)
        return json.dumps({"error": result.error or "Tool execution failed"})

    # Build the wrapper with proper annotations for AF
    # AF inspects __name__, __doc__, and __annotations__ to generate schemas.
    # Two separate named functions avoid the mypy "conditional function variant"
    # error that arises when the same name is assigned in both if/else branches
    # with incompatible signatures.
    async def _wrapper_no_args() -> str:
        return await _execute()

    async def _wrapper_with_args(*args: Any, **kwargs: Any) -> str:
        param_names = list(properties.keys())
        if args:
            if len(args) > len(param_names):
                raise TypeError(
                    f"{tool_name}() takes at most {len(param_names)} positional "
                    f"arguments but {len(args)} were given"
                )
            for idx, value in enumerate(args):
                p_name = param_names[idx]
                if p_name not in kwargs:
                    kwargs[p_name] = value
        return await _execute(**kwargs)

    wrapper: Callable[..., Awaitable[str]]
    if len(properties) == 0:
        wrapper = _wrapper_no_args
    else:
        wrapper = _wrapper_with_args
        new_params = []
        for p_name, p_info in properties.items():
            p_desc = p_info.get("description", f"Parameter: {p_name}")
            p_type = _json_type_to_python(p_info.get("type", "string"))
            default = (
                inspect.Parameter.empty
                if p_name in required_params
                else p_info.get("default")
            )
            new_params.append(
                inspect.Parameter(
                    p_name,
                    inspect.Parameter.POSITIONAL_OR_KEYWORD,
                    annotation=Annotated[p_type, Field(description=p_desc)],
                    default=default,
                )
            )
        wrapper.__signature__ = inspect.Signature(parameters=new_params)  # type: ignore[attr-defined]

    wrapper.__name__ = tool_name
    wrapper.__qualname__ = tool_name
    wrapper.__doc__ = tool_desc

    # Optionally upgrade to a real AF FunctionTool so approval_mode and the
    # rest of the GA metadata flows through. AF 1.0 also accepts bare
    # callables (it auto-wraps them at Agent(tools=...) time), so the
    # fallback path remains fully functional for environments without AF.
    if as_function_tool is False:
        return wrapper

    af_tool = _try_import_af_tool()
    if af_tool is None:
        if as_function_tool is True:
            raise ImportError(
                "as_function_tool=True requires the 'agent-framework' package. "
                "Install with: pip install 'agent-gantry[agent-frameworks]'"
            )
        return wrapper

    approval_mode = _tool_approval_mode(tool_def)
    decorated = af_tool(
        wrapper,
        name=tool_name,
        description=tool_desc,
        approval_mode=approval_mode,
    )
    return decorated


def _json_type_to_python(json_type: str) -> type:
    """Map JSON Schema type strings to Python types."""
    mapping: dict[str, type] = {
        "string": str,
        "integer": int,
        "number": float,
        "boolean": bool,
        "array": list,
        "object": dict,
    }
    return mapping.get(json_type, str)


def _cache_key(tool_def: ToolDefinition) -> str:
    """Build a namespace-qualified cache key for a tool definition."""
    return f"{tool_def.namespace}:{tool_def.name}"


class GantryToolBridge:
    """
    Bridge between Agent-Gantry and Microsoft Agent Framework.

    Retrieves semantically relevant tools from Gantry and wraps them as
    Python callables that AF agents can use directly. This is the primary
    integration point for production multi-agent systems where token savings
    from semantic routing are critical.

    The bridge supports two usage patterns:

    1. **Query-time retrieval** (recommended for token savings):
       Retrieve only the relevant tools for each query, minimising the
       tool definitions sent to the LLM.

       .. code-block:: python

           bridge = GantryToolBridge(gantry)
           tools = await bridge.get_tools("book a flight", limit=3)
           agent = client.as_agent(tools=tools, ...)

    2. **Pre-built tool set**:
       Build tools once from specific Gantry tool definitions for agents
       that need a fixed tool set.

       .. code-block:: python

           bridge = GantryToolBridge(gantry)
           tools = bridge.wrap_tools(my_tool_definitions)
           agent = client.as_agent(tools=tools, ...)

    Args:
        gantry: The AgentGantry instance providing tool retrieval and execution.
        score_threshold: Minimum relevance score for tool selection (default: 0.3).
    """

    def __init__(
        self,
        gantry: AgentGantry,
        *,
        score_threshold: float = 0.3,
        as_function_tool: bool | None = None,
    ) -> None:
        """Initialize the bridge.

        Args:
            gantry: The AgentGantry instance providing tool retrieval and execution.
            score_threshold: Minimum relevance score for tool selection (default: 0.3).
            as_function_tool: Whether wrapped tools should be elevated to
                ``agent_framework.FunctionTool`` via the ``@tool`` decorator.
                ``None`` (default) = auto-detect (wrap if AF is importable);
                ``True`` = force wrapping (raise if AF is missing);
                ``False`` = always return bare callables. Defaults produce the
                most idiomatic AF behaviour without introducing a hard dep.
        """
        self._gantry = gantry
        self._score_threshold = score_threshold
        self._as_function_tool = as_function_tool
        self._tool_cache: dict[str, Any] = {}

    async def _retrieve(
        self,
        query: str,
        *,
        limit: int = 5,
        score_threshold: float | None = None,
        **query_kwargs: Any,
    ) -> RetrievalResult:
        """Shared retrieval logic for get_tools and get_tools_with_scores."""
        from agent_gantry.schema.query import ConversationContext, ToolQuery

        threshold = score_threshold if score_threshold is not None else self._score_threshold

        # Separate context-level kwargs from query-level kwargs
        context_fields = set(ConversationContext.model_fields.keys()) - {"query"}
        context_kwargs = {k: v for k, v in query_kwargs.items() if k in context_fields}
        tool_query_fields = set(ToolQuery.model_fields.keys()) - {"context", "limit", "score_threshold"}
        tq_kwargs = {k: v for k, v in query_kwargs.items() if k in tool_query_fields}

        return await self._gantry.retrieve(
            ToolQuery(
                context=ConversationContext(query=query, **context_kwargs),
                limit=limit,
                score_threshold=threshold,
                **tq_kwargs,
            )
        )

    def _get_or_build(
        self,
        tool_def: ToolDefinition,
        cache: bool,
    ) -> Any:
        """Look up a cached wrapper or build a new one."""
        key = _cache_key(tool_def)
        if cache and key in self._tool_cache:
            return self._tool_cache[key]
        wrapper = _build_callable_for_tool(
            tool_def,
            self._gantry,
            as_function_tool=self._as_function_tool,
        )
        if cache:
            self._tool_cache[key] = wrapper
        return wrapper

    async def get_tools(
        self,
        query: str,
        *,
        limit: int = 5,
        score_threshold: float | None = None,
        cache: bool = True,
        **query_kwargs: Any,
    ) -> list[Any]:
        """
        Retrieve semantically relevant tools as AF-compatible callables.

        This is the primary method for dynamic tool selection. It queries
        Gantry's semantic router, selects the top-k tools, and wraps each
        as a Python async callable with proper type annotations that
        Microsoft Agent Framework can introspect.

        Args:
            query: The user query or task description to match tools against.
            limit: Maximum number of tools to return (default: 5).
            score_threshold: Override the bridge's default score threshold.
            cache: Whether to reuse previously wrapped callables for the same
                   tool (avoids re-creating wrappers). Default: True.
            **query_kwargs: Additional keyword arguments passed through to
                ``ToolQuery`` (e.g. ``namespaces``, ``required_capabilities``,
                ``sources``, ``exclude_deprecated``, ``enable_reranking``) and
                to ``ConversationContext`` (e.g. ``user_capabilities``,
                ``conversation_summary``, ``recent_messages``).

        Returns:
            List of async callables suitable for AF agent ``tools=[...]``.
        """
        result = await self._retrieve(
            query, limit=limit, score_threshold=score_threshold, **query_kwargs
        )

        tools = [self._get_or_build(st.tool, cache) for st in result.tools]

        logger.debug(
            "GantryToolBridge: selected %d/%d tools for query '%s'",
            len(tools),
            result.candidate_count,
            query[:50],
        )
        return tools

    def wrap_tools(
        self,
        tool_definitions: list[ToolDefinition],
        *,
        cache: bool = True,
    ) -> list[Any]:
        """
        Wrap specific Gantry tool definitions as AF-compatible callables.

        Use this when you already have the tool definitions and want to
        create the callable wrappers without going through semantic retrieval.

        Args:
            tool_definitions: List of ToolDefinition objects to wrap.
            cache: Whether to cache/reuse wrappers (default: True).

        Returns:
            List of async callables suitable for AF agent ``tools=[...]``.
        """
        return [self._get_or_build(td, cache) for td in tool_definitions]

    def wrap_single(self, tool_def: ToolDefinition) -> Any:
        """
        Wrap a single Gantry tool definition as an AF-compatible callable.

        Args:
            tool_def: The ToolDefinition to wrap.

        Returns:
            An async callable suitable for AF agent ``tools=[...]``.
        """
        return self._get_or_build(tool_def, cache=True)

    def clear_cache(self) -> None:
        """Clear the cached tool wrappers."""
        self._tool_cache.clear()

    async def get_tools_with_scores(
        self,
        query: str,
        *,
        limit: int = 5,
        score_threshold: float | None = None,
        cache: bool = True,
        **query_kwargs: Any,
    ) -> list[tuple[Any, float]]:
        """
        Retrieve tools with their relevance scores for observability.

        Same as ``get_tools`` but also returns the semantic relevance score
        for each tool, useful for debugging and monitoring token savings.

        Args:
            query: The user query to match tools against.
            limit: Maximum number of tools to return.
            score_threshold: Override the bridge's default score threshold.
            cache: Whether to reuse previously wrapped callables for the same
                   tool (avoids re-creating wrappers). Default: True.
            **query_kwargs: Additional keyword arguments passed through to
                ``ToolQuery`` and ``ConversationContext``.

        Returns:
            List of (callable, score) tuples.
        """
        result = await self._retrieve(
            query, limit=limit, score_threshold=score_threshold, **query_kwargs
        )

        return [
            (self._get_or_build(st.tool, cache), st.final_score)
            for st in result.tools
        ]

    # ------------------------------------------------------------------
    # Agent construction helpers
    # ------------------------------------------------------------------

    async def build_agent(
        self,
        client: Any,
        query: str,
        *,
        name: str,
        instructions: str,
        limit: int = 5,
        score_threshold: float | None = None,
        middleware: Any = None,
        cache: bool = True,
        extra_tools: list[Any] | None = None,
        **query_kwargs: Any,
    ) -> Any:
        """Retrieve relevant tools and construct an AF ``Agent`` in one call.

        This is the idiomatic one-liner for single-agent flows where the
        tool set is determined by a single query (for example, the first
        user turn). For multi-turn conversations, use ``get_tools`` and
        keep the resulting agent across turns, or re-run this helper per
        turn if you want tools to adapt to the latest user message.

        Args:
            client: Any AF chat client exposing ``as_agent(...)`` (e.g.
                ``OpenAIChatClient``, ``AzureOpenAIChatClient``,
                ``AnthropicChatClient``, ``GeminiChatClient`` — new in AF 1.1.0).
            query: The user query whose tools will be retrieved.
            name: Name to give the constructed agent.
            instructions: System instructions for the agent.
            limit: Top-K tools to retrieve from Gantry.
            score_threshold: Override the bridge-level threshold.
            middleware: Optional AF middleware sequence forwarded to
                ``as_agent``. Accepts ``GantryApprovalMiddleware`` and any
                AF-native middleware; useful for layering approval gates
                or observability on top of semantically selected tools.
            cache: Whether to reuse cached wrappers.
            extra_tools: Additional static tools (AF ``FunctionTool``,
                ``MCPStreamableHTTPTool``, bare callables, …) to append
                after the Gantry-selected tools.
            **query_kwargs: Forwarded to ``ToolQuery`` / ``ConversationContext``.

        Returns:
            An AF agent constructed via ``client.as_agent(...)``.
        """
        tools = await self.get_tools(
            query,
            limit=limit,
            score_threshold=score_threshold,
            cache=cache,
            **query_kwargs,
        )
        if extra_tools:
            tools = tools + list(extra_tools)

        kwargs: dict[str, Any] = {
            "name": name,
            "instructions": instructions,
            "tools": tools,
        }
        if middleware is not None:
            kwargs["middleware"] = middleware
        return client.as_agent(**kwargs)

    async def as_agent(
        self,
        client: Any,
        query: str,
        *,
        name: str,
        instructions: str,
        limit: int = 5,
        score_threshold: float | None = None,
        middleware: Any = None,
        cache: bool = True,
        extra_tools: list[Any] | None = None,
        query_kwargs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> Any:
        """Retrieve relevant tools and construct a bare AF ``Agent(client, ...)`` directly.

        Unlike :meth:`build_agent` which uses ``client.as_agent()``, this
        constructs ``Agent`` via its constructor so the result is a first-class
        ``Agent`` object suitable for direct use with ``WorkflowBuilder``,
        ``WorkflowAgent``, and other multi-agent orchestration patterns.

        .. code-block:: python

            from agent_framework import WorkflowBuilder, WorkflowAgent
            from agent_framework.openai import OpenAIChatClient

            client = OpenAIChatClient()
            bridge = GantryToolBridge(gantry)

            triage  = await bridge.as_agent(client, "triage",   name="Triage",   instructions="Route the request.")
            billing = await bridge.as_agent(client, "billing",  name="Billing",  instructions="Handle billing questions.")
            support = await bridge.as_agent(client, "support",  name="Support",  instructions="Handle support tickets.")

            workflow = (
                WorkflowBuilder(start_executor=triage)
                .add_edge(triage, billing,  condition=lambda ctx: "invoice" in str(ctx).lower())
                .add_edge(triage, support)
                .build()
            )
            agent = WorkflowAgent(workflow, name="CustomerService")
            result = await agent.run("I need help with my invoice")

        Args:
            client: Any AF chat client (``OpenAIChatClient``, ``AzureOpenAIChatClient``,
                ``AnthropicChatClient``, ``GeminiChatClient`` — new in AF 1.1.0, …).
            query: The user query used to semantically select tools.
            name: Name for the agent.
            instructions: System instructions for the agent.
            limit: Top-K tools to retrieve from Gantry.
            score_threshold: Override the bridge-level threshold.
            middleware: Optional AF middleware sequence.
            cache: Whether to reuse cached tool wrappers.
            extra_tools: Additional static tools to append after Gantry-selected tools.
            query_kwargs: Extra keyword arguments forwarded to :meth:`get_tools`
                (e.g. ``conversation_context``, ``namespace``).
            **kwargs: Additional keyword arguments forwarded to ``Agent()``.

        Returns:
            A bare ``agent_framework.Agent`` instance.
        """
        try:
            from agent_framework import Agent
        except ImportError as exc:
            raise ImportError(
                "as_agent() requires the 'agent-framework' package. "
                "Install with: pip install 'agent-gantry[agent-frameworks]'"
            ) from exc

        tools = await self.get_tools(
            query,
            limit=limit,
            score_threshold=score_threshold,
            cache=cache,
            **(query_kwargs or {}),
        )
        if extra_tools:
            tools = tools + list(extra_tools)

        agent_kwargs: dict[str, Any] = {
            "name": name,
            "tools": tools,
            **kwargs,
        }
        if middleware is not None:
            agent_kwargs["middleware"] = middleware

        return Agent(client, instructions, **agent_kwargs)

    async def build_workflow(
        self,
        agent_specs: list[dict[str, Any]],
        *,
        edges: list[tuple[str, str]] | list[tuple[str, str, Any]] | None = None,
        chain: bool = False,
        workflow_name: str | None = None,
        cache: bool = True,
    ) -> Any:
        """Build a multi-agent ``WorkflowAgent`` from Gantry-equipped agent specs.

        Constructs each agent via :meth:`as_agent`, then wires them together
        with ``WorkflowBuilder`` using the supplied edge list or a linear chain.
        The result is a ``WorkflowAgent`` that can be run like any single agent.

        .. code-block:: python

            from agent_framework.openai import OpenAIChatClient

            client = OpenAIChatClient()
            bridge  = GantryToolBridge(gantry)

            # Fan-out: triage routes to billing or support based on a condition
            wa = await bridge.build_workflow(
                agent_specs=[
                    dict(client=client, query="triage customer",     name="Triage",   instructions="Route the customer."),
                    dict(client=client, query="billing invoices",     name="Billing",  instructions="Handle billing."),
                    dict(client=client, query="support tickets bugs", name="Support",  instructions="Handle support."),
                ],
                edges=[
                    ("Triage", "Billing",  lambda ctx: "invoice" in str(ctx).lower()),
                    ("Triage", "Support"),
                ],
            )
            result = await wa.run("My last invoice was wrong")

        Linear chain shortcut:

        .. code-block:: python

            wa = await bridge.build_workflow(
                agent_specs=[
                    dict(client=client, query="gather info",    name="Gather",    instructions="Collect user details."),
                    dict(client=client, query="resolve issue",  name="Resolver",  instructions="Resolve the issue."),
                    dict(client=client, query="send summary",   name="Summarise", instructions="Summarise the outcome."),
                ],
                chain=True,
            )

        Args:
            agent_specs: List of dicts passed as keyword arguments to :meth:`as_agent`.
                Required keys: ``client``, ``query``, ``name``, ``instructions``.
                Optional keys: ``limit``, ``score_threshold``, ``middleware``,
                ``extra_tools``, plus any extra ``Agent()`` kwargs.
            edges: List of ``(source_name, target_name)`` or
                ``(source_name, target_name, condition)`` tuples.
                ``condition`` is an optional callable
                ``(context: Any) -> bool | Awaitable[bool]``.
                Ignored when ``chain=True``.
            chain: If ``True``, wire all agents in the order given as a linear
                chain (``WorkflowBuilder.add_chain``). Overrides ``edges``.
            workflow_name: Optional name for the produced ``WorkflowAgent``.
            cache: Whether to reuse cached tool wrappers.

        Returns:
            A ``WorkflowAgent`` wrapping the constructed ``Workflow``.
        """
        try:
            from agent_framework import WorkflowAgent, WorkflowBuilder
        except ImportError as exc:
            raise ImportError(
                "build_workflow() requires the 'agent-framework' package. "
                "Install with: pip install 'agent-gantry[agent-frameworks]'"
            ) from exc

        # Build each agent and keep a name → agent mapping for edge resolution
        built: dict[str, Any] = {}
        ordered: list[Any] = []
        for spec in agent_specs:
            spec = dict(spec)
            agent_name: str = spec["name"]
            agent = await self.as_agent(cache=cache, **spec)
            built[agent_name] = agent
            ordered.append(agent)

        if not ordered:
            raise ValueError("agent_specs must contain at least one agent.")

        # Validate edge names up front so users get a clear error instead of KeyError.
        if not chain and edges:
            unknown = {
                name
                for edge in edges
                for name in (edge[0], edge[1])
                if name not in built
            }
            if unknown:
                raise ValueError(
                    f"build_workflow() edges reference unknown agent name(s): "
                    f"{sorted(unknown)}. Known names: {sorted(built)}"
                )

        builder = WorkflowBuilder(start_executor=ordered[0])

        if chain:
            builder.add_chain(ordered)
        else:
            for edge in (edges or []):
                source_name, target_name = edge[0], edge[1]
                condition = edge[2] if len(edge) > 2 else None
                source_agent = built[source_name]
                target_agent = built[target_name]
                if condition is not None:
                    builder.add_edge(source_agent, target_agent, condition=condition)
                else:
                    builder.add_edge(source_agent, target_agent)

        workflow = builder.build()
        wa_kwargs: dict[str, Any] = {}
        if workflow_name is not None:
            wa_kwargs["name"] = workflow_name
        return WorkflowAgent(workflow, **wa_kwargs)

    def as_tool_list(self, tool_defs: list[ToolDefinition]) -> list[Any]:
        """Alias for :meth:`wrap_tools` with a name that reads naturally in
        orchestration code where the returned list will be spread across
        multiple agents (e.g. ``Agent(tools=bridge.as_tool_list([...]))``).
        """
        return self.wrap_tools(tool_defs)
