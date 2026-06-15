"""OpenAI Agents SDK adapter: export selected Gantry tools as ``FunctionTool``s.

The OpenAI Agents SDK (``openai-agents``, import name ``agents``) consumes
:class:`agents.FunctionTool` objects. This module wraps Gantry-selected
:class:`ToolSpec` handles into that native type, routing every call back through
``gantry.execute`` so retries, timeouts, circuit breakers, and the security
policy still apply.

The ``agents`` import is lazy (performed inside the builder) so that ``import
agent_gantry`` never requires the OpenAI Agents SDK to be installed.

Public entry point: :class:`OpenAIAgentsAdapter`.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from agent_gantry.integrations.frameworks.base import GantryToolset

if TYPE_CHECKING:
    from agent_gantry.core.gantry import AgentGantry
    from agent_gantry.integrations.frameworks.base import ToolSpec


def _strict_schema(params: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of ``params`` with ``additionalProperties: False`` set.

    OpenAI strict-mode function tools require ``additionalProperties`` to be
    ``False`` on the parameter schema.
    """
    schema = dict(params or {"type": "object", "properties": {}})
    schema["additionalProperties"] = False
    return schema


def _spec_to_openai_agents(spec: ToolSpec) -> Any:
    """Convert a :class:`ToolSpec` into an OpenAI Agents SDK ``FunctionTool``.

    Raises:
        ImportError: If ``openai-agents`` is not installed.
    """
    try:
        from agents import FunctionTool
    except ImportError as exc:  # pragma: no cover - exercised via fake module
        raise ImportError(
            "The OpenAI Agents SDK is required for _spec_to_openai_agents; "
            "install it with `pip install openai-agents`."
        ) from exc

    async def _on_invoke_tool(ctx: Any, args: Any) -> str:
        data = json.loads(args) if isinstance(args, str) else dict(args or {})
        return str(await spec.ainvoke(**data))

    return FunctionTool(
        name=spec.name,
        description=spec.description,
        params_json_schema=_strict_schema(spec.parameters),
        on_invoke_tool=_on_invoke_tool,
    )


async def _for_openai_agents(
    gantry: AgentGantry,
    query: str,
    *,
    limit: int = 3,
    **select_kwargs: Any,
) -> list:
    """Select tools for ``query`` and return them as OpenAI Agents ``FunctionTool``s."""
    specs = await GantryToolset(gantry).select(query, limit=limit, **select_kwargs)
    return [_spec_to_openai_agents(spec) for spec in specs]


class OpenAIAgentsAdapter:
    """Route Gantry-selected tools into the OpenAI Agents SDK.

    Static slice (``agents.FunctionTool`` objects) plus deep live re-selection as
    the conversation progresses. Every tool call routes through ``gantry.execute``.
    """

    def __init__(self, gantry: AgentGantry, *, default_limit: int = 3) -> None:
        self._gantry = gantry
        self._default_limit = default_limit

    @staticmethod
    def convert(spec: ToolSpec) -> Any:
        """Wrap a single :class:`ToolSpec` as an OpenAI Agents ``FunctionTool``."""
        return _spec_to_openai_agents(spec)

    async def select(
        self, query: str, *, limit: int | None = None, **select_kwargs: Any
    ) -> list[Any]:
        """Select tools for ``query`` as OpenAI Agents ``FunctionTool``s (static slice)."""
        return await _for_openai_agents(
            self._gantry,
            query,
            limit=self._default_limit if limit is None else limit,
            **select_kwargs,
        )

    async def run(
        self,
        agent: Any,
        run_input: Any,
        *,
        limit: int = 5,
        score_threshold: float = 0.0,
        namespaces: list[str] | None = None,
        **run_kwargs: Any,
    ) -> Any:
        """Re-select ``agent``'s tools for ``run_input`` and run it once via Gantry (one-shot live)."""
        from agent_gantry.integrations.frameworks.openai_agents_live import (
            _run_with_gantry,
        )

        return await _run_with_gantry(
            agent,
            self._gantry,
            run_input,
            limit=limit,
            score_threshold=score_threshold,
            namespaces=namespaces,
            **run_kwargs,
        )

    def session(
        self,
        agent: Any,
        *,
        limit: int = 5,
        score_threshold: float = 0.0,
        namespaces: list[str] | None = None,
    ) -> Any:
        """Return a live session that re-selects ``agent``'s tools each run."""
        from agent_gantry.integrations.frameworks.openai_agents_live import (
            GantryAgentSession,
        )

        return GantryAgentSession(
            agent,
            self._gantry,
            limit=limit,
            score_threshold=score_threshold,
            namespaces=namespaces,
        )

    def run_hooks(
        self,
        agent: Any,
        *,
        limit: int = 5,
        score_threshold: float = 0.0,
        namespaces: list[str] | None = None,
    ) -> Any:
        """Build ``agents.RunHooks`` that re-select ``agent.tools`` before each model call."""
        from agent_gantry.integrations.frameworks.openai_agents_live import (
            _gantry_run_hooks,
        )

        return _gantry_run_hooks(
            self._gantry,
            agent,
            limit=limit,
            score_threshold=score_threshold,
            namespaces=namespaces,
        )

    async def refresh(
        self,
        agent: Any,
        query_or_input: Any,
        *,
        limit: int = 5,
        score_threshold: float = 0.0,
        namespaces: list[str] | None = None,
    ) -> list[Any]:
        """Re-select tools and rewrite ``agent.tools`` in place; return the new tools."""
        from agent_gantry.integrations.frameworks.openai_agents_live import (
            _refresh_agent_tools,
        )

        return await _refresh_agent_tools(
            agent,
            self._gantry,
            query_or_input,
            limit=limit,
            score_threshold=score_threshold,
            namespaces=namespaces,
        )

    async def select_function_tools(
        self,
        query_or_input: Any,
        *,
        limit: int = 5,
        score_threshold: float = 0.0,
        namespaces: list[str] | None = None,
    ) -> list[Any]:
        """Re-select tools for a query string OR a run-input/message list; return ``FunctionTool``s."""
        from agent_gantry.integrations.frameworks.openai_agents_live import (
            _select_function_tools,
        )

        return await _select_function_tools(
            self._gantry,
            query_or_input,
            limit=limit,
            score_threshold=score_threshold,
            namespaces=namespaces,
        )
