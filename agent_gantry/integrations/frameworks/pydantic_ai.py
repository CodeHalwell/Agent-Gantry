"""Pydantic AI native tool adapter for Agent-Gantry.

Selects a relevant slice of Gantry tools and wraps each as a Pydantic AI
``Tool`` — the native tool object a Pydantic AI ``Agent`` introspects (name /
description / JSON-schema parameters) and invokes. The ``pydantic_ai`` import is
lazy so ``import agent_gantry`` never requires Pydantic AI to be installed.

Public entry point: :class:`PydanticAIAdapter`.
"""

from __future__ import annotations

from typing import Any

from agent_gantry.integrations.frameworks.base import BaseFrameworkAdapter, ToolSpec

_INSTALL_HINT = (
    "Pydantic AI support requires `pydantic-ai`. "
    "Install it with `pip install pydantic-ai` (or `uv add pydantic-ai`)."
)


def _spec_to_pydantic_ai(spec: ToolSpec) -> Any:
    """Wrap a :class:`ToolSpec` as a Pydantic AI ``Tool``.

    Prefers the schema-explicit ``Tool.from_schema`` constructor so the JSON
    schema Gantry already produced is used verbatim. Older Pydantic AI versions
    without ``from_schema`` fall back to the function-based ``Tool`` constructor.

    The ``pydantic_ai`` import happens here, lazily, so callers without Pydantic
    AI installed only hit the error when they actually export a tool.
    """
    try:
        from pydantic_ai.tools import Tool
    except ImportError as exc:  # pragma: no cover - exercised via stub
        raise ImportError(_INSTALL_HINT) from exc

    function = spec.callable_for_signature()
    try:
        return Tool.from_schema(
            function=function,
            name=spec.name,
            description=spec.description,
            json_schema=spec.parameters,
        )
    except AttributeError:
        return Tool(
            function,
            name=spec.name,
            description=spec.description,
            takes_ctx=False,
        )


class PydanticAIAdapter(BaseFrameworkAdapter):
    """Route Gantry-selected tools into Pydantic AI.

    Static slice (``pydantic_ai.tools.Tool`` objects) plus a deep per-turn live
    toolset that re-selects tools on every run/step. Both route through
    ``gantry.execute``. :meth:`live` returns the :meth:`toolset` — plug it into
    ``Agent(model, toolsets=[<result>])``.
    """

    live_tier = "per-turn"
    _live_delegate = "toolset"

    @staticmethod
    def convert(spec: ToolSpec) -> Any:
        """Wrap a single :class:`ToolSpec` as a Pydantic AI ``Tool``."""
        return _spec_to_pydantic_ai(spec)

    def toolset(
        self,
        *,
        limit: int | None = None,
        score_threshold: float = 0.0,
        namespaces: list[str] | None = None,
        required: list[str] | None = None,
        always_include: list[str] | None = None,
    ) -> Any:
        """Build a live ``AbstractToolset`` for per-turn dynamic selection (``Agent(toolsets=[...])``)."""
        from agent_gantry.integrations.frameworks.pydantic_ai_live import (
            _gantry_toolset,
        )

        return _gantry_toolset(
            self._gantry,
            **self._selection_kwargs(limit, score_threshold, namespaces, required, always_include),
        )
