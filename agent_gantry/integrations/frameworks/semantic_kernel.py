"""Semantic Kernel native tool adapter for Agent-Gantry.

Selects a relevant slice of Gantry tools and wraps each as a Semantic Kernel
``KernelFunction`` (via the ``@kernel_function`` decorator), the native unit SK
agents invoke. The ``semantic_kernel`` import is lazy so ``import agent_gantry``
never requires SK to be installed.

Public entry point: :class:`SemanticKernelAdapter`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from agent_gantry.integrations.frameworks.base import (
    DEFAULT_TOOL_LIMIT,
    BaseFrameworkAdapter,
    GantryToolset,
    ToolSpec,
)

if TYPE_CHECKING:
    from agent_gantry.core.gantry import AgentGantry

_DEFAULT_PLUGIN = "gantry"


def _spec_to_semantic_kernel(spec: ToolSpec, *, plugin_name: str = _DEFAULT_PLUGIN) -> Any:
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

    # SK infers a parameter's required-ness from whether its annotation is
    # Optional (not from the default), so optional params must be `T | None`.
    fn = spec.callable_for_signature(union_optional=True)
    # SK reads the return annotation to type the function result; default to str.
    fn.__annotations__.setdefault("return", str)
    decorated = kernel_function(name=spec.name, description=spec.description)(fn)
    return KernelFunctionFromMethod(method=decorated, plugin_name=plugin_name)


async def _for_semantic_kernel(
    gantry: AgentGantry,
    query: str,
    *,
    limit: int = DEFAULT_TOOL_LIMIT,
    plugin_name: str = _DEFAULT_PLUGIN,
    **select_kwargs: Any,
) -> list[Any]:
    """Select tools for ``query`` and return them as SK ``KernelFunction``s."""
    specs = await GantryToolset(gantry).select(query, limit=limit, **select_kwargs)
    return [_spec_to_semantic_kernel(s, plugin_name=plugin_name) for s in specs]


async def _gantry_plugin(
    gantry: AgentGantry,
    query: str,
    *,
    limit: int = DEFAULT_TOOL_LIMIT,
    plugin_name: str = _DEFAULT_PLUGIN,
    **select_kwargs: Any,
) -> dict[str, Any]:
    """Return a ``{function_name: KernelFunction}`` mapping for the query.

    To register the functions, wrap them in a ``KernelPlugin`` (passing the dict
    straight to ``kernel.add_plugin`` registers an *empty* plugin)::

        from semantic_kernel.functions import KernelPlugin

        funcs = await _gantry_plugin(gantry, query)
        kernel.add_plugin(KernelPlugin(name="gantry", functions=list(funcs.values())))

    Or use :func:`~agent_gantry.integrations.frameworks.semantic_kernel_live._refresh_kernel_tools`
    / :class:`GantryFunctionProvider`, which do this wrapping for you each turn.
    """
    functions = await _for_semantic_kernel(
        gantry, query, limit=limit, plugin_name=plugin_name, **select_kwargs
    )
    return {f.name: f for f in functions}


class SemanticKernelAdapter(BaseFrameworkAdapter):
    """Route Gantry-selected tools into Semantic Kernel.

    Static slice (``KernelFunction`` objects / a plugin dict) plus a deep per-turn
    live plugin refresh. Every call routes through ``gantry.execute``.
    """

    live_tier = "per-turn"

    @staticmethod
    def convert(spec: ToolSpec, *, plugin_name: str = _DEFAULT_PLUGIN) -> Any:
        """Wrap a single :class:`ToolSpec` as a Semantic Kernel ``KernelFunction``."""
        return _spec_to_semantic_kernel(spec, plugin_name=plugin_name)

    def live(
        self,
        *,
        limit: int | None = None,
        score_threshold: float = 0.0,
        namespaces: list[str] | None = None,
        **framework_kwargs: Any,
    ) -> Any:
        """Per-turn uniform entry point: delegates to :meth:`function_provider`.

        Requires ``kernel=<semantic_kernel.Kernel>`` in ``framework_kwargs``
        — SK advertises functions from a specific kernel's registered
        plugins, so the live object is inherently kernel-bound. Returns a
        ``GantryFunctionProvider``; call ``await provider.refresh(history)``
        before each ``agent.get_response()`` / chat-completion invocation.
        Any other ``framework_kwargs`` (e.g. ``plugin_name``) are forwarded.
        """
        kernel = framework_kwargs.pop("kernel")
        return self.function_provider(
            kernel,
            limit=limit,
            score_threshold=score_threshold,
            namespaces=namespaces,
            **framework_kwargs,
        )

    async def select(
        self,
        query: str,
        *,
        limit: int | None = None,
        plugin_name: str = _DEFAULT_PLUGIN,
        score_threshold: float = 0.0,
        namespaces: list[str] | None = None,
        tools_already_used: list[str] | None = None,
    ) -> list[Any]:
        """Select tools for ``query`` as SK ``KernelFunction``s (static slice).

        Same explicit selection surface as :meth:`BaseFrameworkAdapter.select`
        (``score_threshold``, ``namespaces``, ``tools_already_used``), plus SK's
        own ``plugin_name`` (the plugin every returned ``KernelFunction`` is
        registered under).
        """
        return await _for_semantic_kernel(
            self._gantry,
            query,
            limit=self._default_limit if limit is None else limit,
            plugin_name=plugin_name,
            score_threshold=score_threshold,
            namespaces=namespaces,
            tools_already_used=tools_already_used,
        )

    async def plugin(
        self,
        query: str,
        *,
        limit: int | None = None,
        plugin_name: str = _DEFAULT_PLUGIN,
        score_threshold: float = 0.0,
        **select_kwargs: Any,
    ) -> dict[str, Any]:
        """Return a ``{function_name: KernelFunction}`` mapping for ``query``."""
        return await _gantry_plugin(
            self._gantry,
            query,
            limit=self._default_limit if limit is None else limit,
            plugin_name=plugin_name,
            score_threshold=score_threshold,
            **select_kwargs,
        )

    def function_provider(
        self,
        kernel: Any,
        *,
        plugin_name: str = _DEFAULT_PLUGIN,
        limit: int | None = None,
        score_threshold: float = 0.0,
        namespaces: list[str] | None = None,
    ) -> Any:
        """Build a live ``GantryFunctionProvider`` whose ``refresh(history)`` re-selects functions per turn."""
        from agent_gantry.integrations.frameworks.semantic_kernel_live import (
            GantryFunctionProvider,
        )

        return GantryFunctionProvider(
            self._gantry,
            kernel,
            plugin_name=plugin_name,
            limit=self._default_limit if limit is None else limit,
            score_threshold=score_threshold,
            namespaces=namespaces,
        )

    async def refresh(
        self,
        kernel: Any,
        query: Any,
        *,
        plugin_name: str = _DEFAULT_PLUGIN,
        limit: int | None = None,
        score_threshold: float = 0.0,
        namespaces: list[str] | None = None,
    ) -> dict[str, Any]:
        """Re-select tools for ``query`` and rebuild ``kernel``'s gantry plugin once."""
        from agent_gantry.integrations.frameworks.semantic_kernel_live import (
            _refresh_kernel_tools,
        )

        return await _refresh_kernel_tools(
            self._gantry,
            kernel,
            query,
            plugin_name=plugin_name,
            limit=self._default_limit if limit is None else limit,
            score_threshold=score_threshold,
            namespaces=namespaces,
        )
