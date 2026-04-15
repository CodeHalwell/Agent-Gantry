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
from typing import TYPE_CHECKING, Annotated, Any

from pydantic import Field

from agent_gantry.schema.execution import ToolCall

if TYPE_CHECKING:
    from agent_gantry.core.gantry import AgentGantry
    from agent_gantry.schema.query import RetrievalResult
    from agent_gantry.schema.tool import ToolDefinition

logger = logging.getLogger(__name__)


def _build_callable_for_tool(
    tool_def: ToolDefinition,
    gantry: AgentGantry,
) -> Any:
    """
    Build a Python callable wrapping a Gantry tool for Microsoft Agent Framework.

    The callable is created with proper type annotations (using ``Annotated``
    and Pydantic ``Field``) so that AF can auto-generate the correct function
    schema for the LLM. This avoids sending raw JSON schemas and lets AF's
    native function-tool infrastructure handle serialisation.

    Args:
        tool_def: The Gantry ToolDefinition to wrap.
        gantry: The AgentGantry instance for execution.

    Returns:
        An async callable suitable for passing to AF agent ``tools=[...]``.
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
    # AF inspects __name__, __doc__, and __annotations__ to generate schemas
    if len(properties) == 0:
        # No-arg tool
        async def wrapper() -> str:
            return await _execute()
    else:
        # Tool with one or more arguments: support both positional and keyword args
        param_names = list(properties.keys())

        async def wrapper(*args: Any, **kwargs: Any) -> str:
            # Map positional arguments to parameter names based on JSON schema order
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

    return wrapper


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
    ) -> None:
        self._gantry = gantry
        self._score_threshold = score_threshold
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
        wrapper = _build_callable_for_tool(tool_def, self._gantry)
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
