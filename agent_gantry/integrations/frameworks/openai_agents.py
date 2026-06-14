"""OpenAI Agents SDK adapter: export selected Gantry tools as ``FunctionTool``s.

The OpenAI Agents SDK (``openai-agents``, import name ``agents``) consumes
:class:`agents.FunctionTool` objects. This module wraps Gantry-selected
:class:`ToolSpec` handles into that native type, routing every call back through
``gantry.execute`` so retries, timeouts, circuit breakers, and the security
policy still apply.

The ``agents`` import is lazy (performed inside the builder) so that ``import
agent_gantry`` never requires the OpenAI Agents SDK to be installed.
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


def spec_to_openai_agents(spec: ToolSpec) -> Any:
    """Convert a :class:`ToolSpec` into an OpenAI Agents SDK ``FunctionTool``.

    Raises:
        ImportError: If ``openai-agents`` is not installed.
    """
    try:
        from agents import FunctionTool
    except ImportError as exc:  # pragma: no cover - exercised via fake module
        raise ImportError(
            "The OpenAI Agents SDK is required for spec_to_openai_agents; "
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


async def for_openai_agents(
    gantry: AgentGantry,
    query: str,
    *,
    limit: int = 3,
    **select_kwargs: Any,
) -> list:
    """Select tools for ``query`` and return them as OpenAI Agents ``FunctionTool``s."""
    specs = await GantryToolset(gantry).select(query, limit=limit, **select_kwargs)
    return [spec_to_openai_agents(spec) for spec in specs]
