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

import copy
import json
import logging
from typing import TYPE_CHECKING, Any

from agent_gantry.adapters.tool_spec.schema_utils import (
    strict_json_schema,
    unsupported_strict_paths,
)
from agent_gantry.integrations.frameworks.base import (
    DEFAULT_TOOL_LIMIT,
    BaseFrameworkAdapter,
    GantryToolset,
)

if TYPE_CHECKING:
    from agent_gantry.core.gantry import AgentGantry
    from agent_gantry.integrations.frameworks.base import ToolSpec

_logger = logging.getLogger(__name__)


def _strict_schema(params: dict[str, Any]) -> tuple[dict[str, Any], bool]:
    """Return ``params`` reshaped for OpenAI Agents strict-mode function tools.

    ``FunctionTool.strict_json_schema`` defaults to ``True``, and the SDK then
    runs ``ensure_strict_json_schema``, which rewrites ``required`` to list
    *every* property. Setting only a top-level ``additionalProperties: False``
    therefore silently promoted every optional Gantry parameter to mandatory
    (no ``null`` union was added), raised ``UserError`` on any nested
    ``additionalProperties: true``, and on SDK versions predating that
    transform sent a non-strict schema with ``strict=true`` — a 400.

    Delegating to the shared transform applies the constraints recursively and
    keeps optionality by widening those properties to admit ``null``. It is
    idempotent, so the SDK's own pass over the result is a no-op.

    Returns:
        A tuple of the schema to publish and whether strict mode is usable.
        An object with arbitrary keys (a ``dict[str, int]`` parameter, an
        untyped ``dict``) has no strict-mode representation, and
        ``strict_json_schema`` deliberately leaves it alone rather than
        forcing it closed — so the caller must turn ``strict_json_schema``
        *off* for that tool, or the SDK's own ``ensure_strict_json_schema``
        rejects it with ``UserError``.
    """
    unsupported = unsupported_strict_paths(params)
    if unsupported:
        _logger.warning(
            "Tool cannot use OpenAI Agents strict mode: %s describes an object "
            "with arbitrary keys, which strict mode cannot express. Emitting "
            "the tool with strict_json_schema=False so the SDK still accepts "
            "it — declare the object's properties explicitly to make it "
            "strict-compatible.",
            ", ".join(unsupported),
        )
        return copy.deepcopy(params), False
    return strict_json_schema(params), True


def _spec_to_openai_agents(spec: ToolSpec) -> Any:
    """Convert a :class:`ToolSpec` into an OpenAI Agents SDK ``FunctionTool``.

    Raises:
        ImportError: If ``openai-agents`` is not installed.
    """
    try:
        from agents import FunctionTool
    except ImportError as exc:  # pragma: no cover - exercised via fake module
        raise ImportError(
            "The OpenAI Agents SDK is required for OpenAIAgentsAdapter; "
            "install it with `pip install openai-agents`."
        ) from exc

    async def _on_invoke_tool(ctx: Any, args: Any) -> str:
        data = json.loads(args) if isinstance(args, str) else dict(args or {})
        result = await spec.ainvoke(**data)
        # ``str()`` on a dict yields Python repr (single quotes), which the
        # model then has to guess at. Serialize structured results as JSON,
        # matching the Agent Framework bridge.
        return result if isinstance(result, str) else json.dumps(result, default=str)

    params, use_strict = _strict_schema(spec.parameters)
    return FunctionTool(
        name=spec.name,
        description=spec.description,
        params_json_schema=params,
        strict_json_schema=use_strict,
        on_invoke_tool=_on_invoke_tool,
    )


async def _for_openai_agents(
    gantry: AgentGantry,
    query: str,
    *,
    limit: int = DEFAULT_TOOL_LIMIT,
    **select_kwargs: Any,
) -> list:
    """Select tools for ``query`` and return them as OpenAI Agents ``FunctionTool``s."""
    specs = await GantryToolset(gantry).select(query, limit=limit, **select_kwargs)
    return [_spec_to_openai_agents(spec) for spec in specs]


class OpenAIAgentsAdapter(BaseFrameworkAdapter):
    """Route Gantry-selected tools into the OpenAI Agents SDK.

    Static slice (``agents.FunctionTool`` objects) plus deep live re-selection as
    the conversation progresses. Every tool call routes through ``gantry.execute``.
    """

    live_tier = "per-turn"

    @staticmethod
    def convert(spec: ToolSpec) -> Any:
        """Wrap a single :class:`ToolSpec` as an OpenAI Agents ``FunctionTool``."""
        return _spec_to_openai_agents(spec)

    def live(
        self,
        *,
        limit: int | None = None,
        score_threshold: float = 0.0,
        namespaces: list[str] | None = None,
        required: list[str] | None = None,
        always_include: list[str] | None = None,
        **framework_kwargs: Any,
    ) -> Any:
        """Per-turn uniform entry point: delegates to :meth:`session`.

        Requires ``agent=<agents.Agent>`` in ``framework_kwargs`` — the SDK's
        tool-refresh hook rewrites a *specific* agent's ``.tools`` list, so
        there is no agent-agnostic standalone hook to return. Returns a
        :class:`~agent_gantry.integrations.frameworks.openai_agents_live.GantryAgentSession`;
        call ``await session.run(run_input)`` per conversational turn (it
        re-selects and applies tools before the run, and installs
        :meth:`run_hooks` for intra-run dynamism). ``required``/
        ``always_include`` are re-applied on every re-selection (see
        :meth:`~agent_gantry.integrations.frameworks.base.GantryToolset.select`).
        """
        agent = framework_kwargs.pop("agent")
        return self.session(
            agent,
            limit=limit,
            score_threshold=score_threshold,
            namespaces=namespaces,
            required=required,
            always_include=always_include,
        )

    async def run(
        self,
        agent: Any,
        run_input: Any,
        *,
        limit: int | None = None,
        score_threshold: float = 0.0,
        namespaces: list[str] | None = None,
        required: list[str] | None = None,
        always_include: list[str] | None = None,
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
            limit=self._default_limit if limit is None else limit,
            score_threshold=score_threshold,
            namespaces=namespaces,
            required=required,
            always_include=always_include,
            **run_kwargs,
        )

    def session(
        self,
        agent: Any,
        *,
        limit: int | None = None,
        score_threshold: float = 0.0,
        namespaces: list[str] | None = None,
        required: list[str] | None = None,
        always_include: list[str] | None = None,
    ) -> Any:
        """Return a live session that re-selects ``agent``'s tools each run."""
        from agent_gantry.integrations.frameworks.openai_agents_live import (
            GantryAgentSession,
        )

        return GantryAgentSession(
            agent,
            self._gantry,
            limit=self._default_limit if limit is None else limit,
            score_threshold=score_threshold,
            namespaces=namespaces,
            required=required,
            always_include=always_include,
        )

    def run_hooks(
        self,
        agent: Any,
        *,
        limit: int | None = None,
        score_threshold: float = 0.0,
        namespaces: list[str] | None = None,
        required: list[str] | None = None,
        always_include: list[str] | None = None,
    ) -> Any:
        """Build ``agents.RunHooks`` that re-select ``agent.tools`` before each model call."""
        from agent_gantry.integrations.frameworks.openai_agents_live import (
            _gantry_run_hooks,
        )

        return _gantry_run_hooks(
            self._gantry,
            agent,
            limit=self._default_limit if limit is None else limit,
            score_threshold=score_threshold,
            namespaces=namespaces,
            required=required,
            always_include=always_include,
        )

    async def refresh(
        self,
        agent: Any,
        query_or_input: Any,
        *,
        limit: int | None = None,
        score_threshold: float = 0.0,
        namespaces: list[str] | None = None,
        required: list[str] | None = None,
        always_include: list[str] | None = None,
    ) -> list[Any]:
        """Re-select tools and rewrite ``agent.tools`` in place; return the new tools."""
        from agent_gantry.integrations.frameworks.openai_agents_live import (
            _refresh_agent_tools,
        )

        return await _refresh_agent_tools(
            agent,
            self._gantry,
            query_or_input,
            limit=self._default_limit if limit is None else limit,
            score_threshold=score_threshold,
            namespaces=namespaces,
            required=required,
            always_include=always_include,
        )

    async def select_function_tools(
        self,
        query_or_input: Any,
        *,
        limit: int | None = None,
        score_threshold: float = 0.0,
        namespaces: list[str] | None = None,
        required: list[str] | None = None,
        always_include: list[str] | None = None,
    ) -> list[Any]:
        """Re-select tools for a query string OR a run-input/message list; return ``FunctionTool``s."""
        from agent_gantry.integrations.frameworks.openai_agents_live import (
            _select_function_tools,
        )

        return await _select_function_tools(
            self._gantry,
            query_or_input,
            limit=self._default_limit if limit is None else limit,
            score_threshold=score_threshold,
            namespaces=namespaces,
            required=required,
            always_include=always_include,
        )
