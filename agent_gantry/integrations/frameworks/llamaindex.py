"""LlamaIndex adapter: export selected Gantry tools as ``FunctionTool`` objects.

LlamaIndex agents consume :class:`llama_index.core.tools.FunctionTool` objects.
This module wraps Gantry-selected :class:`ToolSpec` handles into that native
type, routing every call back through ``gantry.execute`` so retries, timeouts,
circuit breakers, and the security policy still apply.

The ``llama_index`` import is lazy (performed inside the builder) so that
``import agent_gantry`` never requires LlamaIndex to be installed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from agent_gantry.integrations.frameworks.base import GantryToolset

if TYPE_CHECKING:
    from agent_gantry.core.gantry import AgentGantry
    from agent_gantry.integrations.frameworks.base import ToolSpec


def spec_to_llamaindex(spec: ToolSpec) -> Any:
    """Convert a :class:`ToolSpec` into a LlamaIndex ``FunctionTool``.

    Raises:
        ImportError: If ``llama-index-core`` is not installed.
    """
    try:
        from llama_index.core.tools import FunctionTool
    except ImportError as exc:  # pragma: no cover - exercised via fake module
        raise ImportError(
            "LlamaIndex is required for spec_to_llamaindex; "
            "install it with `pip install llama-index-core`."
        ) from exc

    def _sync_fn(**kwargs: Any) -> Any:
        return spec.invoke(**kwargs)

    _sync_fn.__name__ = spec.name
    _sync_fn.__doc__ = spec.description

    return FunctionTool.from_defaults(
        fn=_sync_fn,
        async_fn=spec.callable_for_signature(),
        name=spec.name,
        description=spec.description,
    )


async def for_llamaindex(
    gantry: AgentGantry,
    query: str,
    *,
    limit: int = 3,
    **select_kwargs: Any,
) -> list:
    """Select tools for ``query`` and return them as LlamaIndex ``FunctionTool``s."""
    specs = await GantryToolset(gantry).select(query, limit=limit, **select_kwargs)
    return [spec_to_llamaindex(spec) for spec in specs]
