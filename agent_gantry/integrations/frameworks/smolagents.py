"""Smolagents native tool adapter for Agent-Gantry.

Selects a relevant slice of Gantry tools and wraps each as a HuggingFace
``smolagents.Tool`` — the native tool object smolagents agents introspect
(``name`` / ``description`` / ``inputs`` / ``output_type``) and invoke via
``forward``. The ``smolagents`` import is lazy so ``import agent_gantry`` never
requires smolagents to be installed.

Public entry point: :class:`SmolagentsAdapter`.
"""

from __future__ import annotations

import inspect
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


def _build_inputs(parameters: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Convert a JSON-Schema ``properties`` map into smolagents ``inputs``.

    Optional parameters are marked ``nullable`` — smolagents requires this for
    any input that isn't always supplied, otherwise it rejects the tool.
    """
    properties = parameters.get("properties", {}) or {}
    required = set(parameters.get("required") or [])
    inputs: dict[str, dict[str, Any]] = {}
    for argname, schema in properties.items():
        schema = schema or {}
        smol_type = _JSON_TO_SMOLAGENTS.get(schema.get("type"), "string")
        description = schema.get("description") or f"{argname} argument"
        entry: dict[str, Any] = {"type": smol_type, "description": description}
        if argname not in required:
            entry["nullable"] = True
        inputs[argname] = entry
    return inputs


def _forward_signature(parameters: dict[str, Any]) -> inspect.Signature:
    """Build a ``forward`` signature (incl. ``self``) matching the inputs.

    smolagents validates that ``forward``'s parameters equal the declared
    ``inputs`` keys, so a bare ``**kwargs`` is rejected. We advertise the real
    named parameters here while the body still accepts ``**kwargs``.
    """
    params = [inspect.Parameter("self", inspect.Parameter.POSITIONAL_OR_KEYWORD)]
    required = set(parameters.get("required") or [])
    # Required params (no default) MUST precede optional ones (default=None),
    # otherwise inspect.Signature raises ValueError for POSITIONAL_OR_KEYWORD.
    for argname in _ordered_param_names(parameters):
        default = inspect.Parameter.empty if argname in required else None
        params.append(
            inspect.Parameter(argname, inspect.Parameter.POSITIONAL_OR_KEYWORD, default=default)
        )
    return inspect.Signature(params)


def _ordered_param_names(parameters: dict[str, Any]) -> list[str]:
    """Property names with required params first (stable within each group)."""
    properties = parameters.get("properties", {}) or {}
    required = set(parameters.get("required") or [])
    return sorted(properties, key=lambda name: name not in required)


def _spec_to_smolagents(spec: ToolSpec) -> Any:
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

    def forward(self: Any, *args: Any, **kwargs: Any) -> Any:
        # Map any positional args to their parameter names (same order the
        # advertised signature uses) so positional invocation isn't dropped.
        for name, value in zip(_ordered_param_names(spec.parameters), args):
            kwargs.setdefault(name, value)
        return spec.invoke(**kwargs)

    # Advertise the real parameter names so smolagents' forward/inputs
    # consistency check passes (it inspects forward's signature).
    forward.__signature__ = _forward_signature(spec.parameters)  # type: ignore[attr-defined]

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


async def _for_smolagents(
    gantry: AgentGantry,
    query: str,
    *,
    limit: int = 3,
    **select_kwargs: Any,
) -> list[Any]:
    """Select tools for ``query`` and return them as smolagents ``Tool``s."""
    specs = await GantryToolset(gantry).select(query, limit=limit, **select_kwargs)
    return [_spec_to_smolagents(s) for s in specs]


class SmolagentsAdapter:
    """Route Gantry-selected tools into smolagents.

    Static slice (``smolagents.Tool`` objects) plus a per-call live builder
    (smolagents fixes tools at construction, so it rebuilds the agent per call).
    Every call routes through ``gantry.execute``.
    """

    def __init__(self, gantry: AgentGantry, *, default_limit: int = 3) -> None:
        self._gantry = gantry
        self._default_limit = default_limit

    @staticmethod
    def convert(spec: ToolSpec) -> Any:
        """Wrap a single :class:`ToolSpec` as a smolagents ``Tool``."""
        return _spec_to_smolagents(spec)

    async def select(
        self, query: str, *, limit: int | None = None, **select_kwargs: Any
    ) -> list[Any]:
        """Select tools for ``query`` as smolagents ``Tool``s (static slice)."""
        return await _for_smolagents(
            self._gantry,
            query,
            limit=self._default_limit if limit is None else limit,
            **select_kwargs,
        )

    def agent_builder(
        self,
        *,
        limit: int | None = None,
        score_threshold: float = 0.0,
        **agent_kwargs: Any,
    ) -> Any:
        """Return a builder that rebuilds a fresh smolagents agent per call with re-selected tools.

        ``agent_kwargs`` (model/agent_cls/...) are forwarded. Call
        ``await builder.build(query)`` per run.
        """
        from agent_gantry.integrations.frameworks.live_wrappers import (
            GantryLiveSmolAgent,
        )

        return GantryLiveSmolAgent(
            self._gantry,
            limit=self._default_limit if limit is None else limit,
            score_threshold=score_threshold,
            **agent_kwargs,
        )
