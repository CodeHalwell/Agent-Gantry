"""Smolagents native tool adapter for Agent-Gantry.

Selects a relevant slice of Gantry tools and wraps each as a HuggingFace
``smolagents.Tool`` — the native tool object smolagents agents introspect
(``name`` / ``description`` / ``inputs`` / ``output_type``) and invoke via
``forward``. The ``smolagents`` import is lazy so ``import agent_gantry`` never
requires smolagents to be installed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from agent_gantry.integrations.frameworks.base import GantryToolset, ToolSpec

if TYPE_CHECKING:
    from agent_gantry.core.gantry import AgentGantry


# JSON-Schema type -> smolagents input type. Anything unmapped falls back to
# "string" (smolagents' most permissive accepted type).
_JSON_TO_SMOLAGENTS = {
    "string": "string",
    "integer": "integer",
    "number": "number",
    "boolean": "boolean",
    "array": "array",
    "object": "object",
}


def _build_inputs(parameters: dict[str, Any]) -> dict[str, dict[str, str]]:
    """Convert a JSON-Schema ``properties`` map into smolagents ``inputs``."""
    properties = parameters.get("properties", {}) or {}
    inputs: dict[str, dict[str, str]] = {}
    for argname, schema in properties.items():
        schema = schema or {}
        smol_type = _JSON_TO_SMOLAGENTS.get(schema.get("type"), "string")
        description = schema.get("description") or f"{argname} argument"
        inputs[argname] = {"type": smol_type, "description": description}
    return inputs


def spec_to_smolagents(spec: ToolSpec) -> Any:
    """Wrap a :class:`ToolSpec` as a smolagents ``Tool``.

    The ``smolagents`` import happens here, lazily, so callers without
    smolagents installed only hit the error when they actually export a tool. A
    subclass of ``Tool`` is built dynamically whose ``name`` / ``description`` /
    ``inputs`` / ``output_type`` come from the spec and whose ``forward`` routes
    through ``gantry.execute``.

    Raises:
        ImportError: If ``smolagents`` is not installed.
    """
    try:
        from smolagents import Tool
    except ImportError as exc:  # pragma: no cover - exercised via fake module
        raise ImportError(
            "Smolagents support requires `smolagents`. "
            "Install it with `pip install smolagents`."
        ) from exc

    def forward(self: Any, **kwargs: Any) -> Any:
        return spec.invoke(**kwargs)

    tool_cls = type(
        "GantrySmolagentsTool",
        (Tool,),
        {
            "name": spec.name,
            "description": spec.description,
            "inputs": _build_inputs(spec.parameters),
            "output_type": "string",
            "forward": forward,
        },
    )
    # Instantiating runs the inherited ``Tool.__init__`` (required setup such as
    # ``is_initialized``); ``forward`` is supplied so the abstract base is
    # concrete.
    return tool_cls()


async def for_smolagents(
    gantry: AgentGantry,
    query: str,
    *,
    limit: int = 3,
    **select_kwargs: Any,
) -> list[Any]:
    """Select tools for ``query`` and return them as smolagents ``Tool``s."""
    specs = await GantryToolset(gantry).select(query, limit=limit, **select_kwargs)
    return [spec_to_smolagents(s) for s in specs]
