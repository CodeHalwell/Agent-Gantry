"""Semantic Kernel native tool adapter for Agent-Gantry.

Selects a relevant slice of Gantry tools and wraps each as a Semantic Kernel
``KernelFunction`` (via the ``@kernel_function`` decorator), the native unit SK
agents invoke. The ``semantic_kernel`` import is lazy so ``import agent_gantry``
never requires SK to be installed.

Use :func:`for_semantic_kernel` to get ``KernelFunction`` objects, or
:func:`gantry_plugin` to get a plugin dict (``{name: KernelFunction}``) ready
for ``kernel.add_plugin(plugin, plugin_name=...)``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from agent_gantry.integrations.frameworks.base import GantryToolset, ToolSpec

if TYPE_CHECKING:
    from agent_gantry.core.gantry import AgentGantry

_DEFAULT_PLUGIN = "gantry"


def spec_to_semantic_kernel(spec: ToolSpec, *, plugin_name: str = _DEFAULT_PLUGIN) -> Any:
    """Wrap a :class:`ToolSpec` as a Semantic Kernel ``KernelFunction``.

    The callable handed to SK carries a real ``__signature__`` (derived from the
    tool's JSON schema) and is decorated with ``@kernel_function`` so SK reads
    its name/description/parameters; every call routes through ``gantry.execute``.

    Raises:
        ImportError: If ``semantic-kernel`` is not installed.
    """
    try:
        from semantic_kernel.functions import KernelFunctionFromMethod, kernel_function
    except ImportError as exc:  # pragma: no cover - exercised via stub
        raise ImportError(
            "Semantic Kernel support requires `semantic-kernel`. "
            "Install it with `pip install semantic-kernel`."
        ) from exc

    fn = spec.callable_for_signature()
    # SK reads the return annotation to type the function result; default to str.
    fn.__annotations__.setdefault("return", str)
    decorated = kernel_function(name=spec.name, description=spec.description)(fn)
    return KernelFunctionFromMethod(method=decorated, plugin_name=plugin_name)


async def for_semantic_kernel(
    gantry: AgentGantry,
    query: str,
    *,
    limit: int = 3,
    plugin_name: str = _DEFAULT_PLUGIN,
    **select_kwargs: Any,
) -> list[Any]:
    """Select tools for ``query`` and return them as SK ``KernelFunction``s."""
    specs = await GantryToolset(gantry).select(query, limit=limit, **select_kwargs)
    return [spec_to_semantic_kernel(s, plugin_name=plugin_name) for s in specs]


async def gantry_plugin(
    gantry: AgentGantry,
    query: str,
    *,
    limit: int = 3,
    plugin_name: str = _DEFAULT_PLUGIN,
    **select_kwargs: Any,
) -> dict[str, Any]:
    """Return a ``{function_name: KernelFunction}`` plugin for the query.

    Ready to register with ``kernel.add_plugin(plugin, plugin_name=...)``.
    """
    functions = await for_semantic_kernel(
        gantry, query, limit=limit, plugin_name=plugin_name, **select_kwargs
    )
    return {f.name: f for f in functions}
