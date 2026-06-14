"""DEEP per-turn dynamic-tool provider for Semantic Kernel.

Where :mod:`agent_gantry.integrations.frameworks.semantic_kernel` exposes the
*static* helpers (``for_semantic_kernel`` / ``gantry_plugin``) — you select a
slice of tools once and register them on a :class:`~semantic_kernel.Kernel` —
this module wires Agent-Gantry into Semantic Kernel as a **live, per-turn**
tool source, matching the depth of the Microsoft Agent Framework
``GantryContextProvider``: the set of functions the model can call is
**re-selected by Gantry on every invocation**.

The native hook
---------------
Semantic Kernel advertises callable functions to the model from the kernel's
registered plugins (``kernel.plugins``), gated by the chosen
``FunctionChoiceBehavior``. There is no per-round event callback on the kernel
itself, so the cleanest deep mechanism is **plugin refresh**: maintain a single
``"gantry"`` plugin whose :class:`~semantic_kernel.functions.KernelFunction`
members are re-selected by Gantry before each agent/chat invocation. Concretely,
each :meth:`GantryFunctionProvider.refresh` call:

1. derives a retrieval query from the latest activity in the supplied messages
   (or uses the string directly),
2. runs Gantry semantic selection to pick the top-k relevant tools,
3. **removes** the old ``gantry`` plugin from ``kernel.plugins`` and **adds** a
   fresh :class:`~semantic_kernel.functions.KernelPlugin` containing *only* the
   newly selected functions.

Because SK reads the function surface from ``kernel.plugins`` at call time, the
model on the *next* invocation sees exactly the functions Gantry chose for the
*current* turn — stale selections never accumulate. ``kernel.plugins`` is a
plain dict in the installed SK (1.x), so removal is a ``dict.pop`` and addition
is ``kernel.add_plugin(KernelPlugin(...))``.

Usage
-----
.. code-block:: python

    from semantic_kernel import Kernel
    from agent_gantry import AgentGantry
    from agent_gantry.integrations.frameworks.semantic_kernel_live import (
        GantryFunctionProvider,
    )

    kernel = Kernel()
    # ... add a chat-completion service, configure FunctionChoiceBehavior.Auto ...
    provider = GantryFunctionProvider(gantry, kernel, limit=5)

    # Before EACH turn, re-select the callable functions from the chat history:
    await provider.refresh(history)        # history: messages or a query string
    response = await agent.get_response(messages=history)

The free function :func:`refresh_kernel_tools` does the same once, for callers
that prefer not to hold a provider instance.

The ``semantic_kernel`` import is lazy so ``import agent_gantry`` never requires
SK to be installed; the helpful ``pip install semantic-kernel`` hint is raised
only when the live provider is actually used.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from agent_gantry.integrations.frameworks.semantic_kernel import gantry_plugin
from agent_gantry.query import latest_activity

if TYPE_CHECKING:
    from agent_gantry.core.gantry import AgentGantry

_DEFAULT_PLUGIN = "gantry"


def _import_kernel_plugin() -> Any:
    """Lazily import :class:`semantic_kernel.functions.KernelPlugin`.

    Deferred to call time so the module stays importable without SK installed.

    Raises:
        ImportError: If ``semantic-kernel`` is not installed.
    """
    try:
        from semantic_kernel.functions import KernelPlugin
    except ImportError as exc:  # pragma: no cover - exercised via stub
        raise ImportError(
            "Semantic Kernel support requires `semantic-kernel`. "
            "Install it with `pip install semantic-kernel`."
        ) from exc
    return KernelPlugin


def _query_from(query_or_messages: Any) -> str:
    """Coerce a query string or a message list into a retrieval query.

    A plain ``str`` is used verbatim; anything else is treated as a
    conversation history and run through :func:`latest_activity`, which derives
    the driving text from the most recent user/tool message.
    """
    if isinstance(query_or_messages, str):
        return query_or_messages
    return latest_activity(query_or_messages) or ""


def _set_plugin_functions(
    kernel: Any, plugin_name: str, functions: dict[str, Any]
) -> Any:
    """Replace the ``plugin_name`` plugin on ``kernel`` with ``functions``.

    Removes any existing plugin under ``plugin_name`` (``kernel.plugins`` is a
    plain dict in SK 1.x) and registers a fresh :class:`KernelPlugin` holding
    exactly the supplied functions — so only the freshly selected tools are
    advertised to the model. Returns the registered plugin.
    """
    KernelPlugin = _import_kernel_plugin()  # noqa: N806
    # Drop the previous selection so stale functions don't linger.
    plugins = getattr(kernel, "plugins", None)
    if isinstance(plugins, dict):
        plugins.pop(plugin_name, None)
    else:  # pragma: no cover - defensive for non-dict plugin collections
        remover = getattr(kernel, "remove_plugin", None)
        if callable(remover):
            try:
                remover(plugin_name)
            except (KeyError, ValueError):
                pass
    plugin = KernelPlugin(name=plugin_name, functions=list(functions.values()))
    kernel.add_plugin(plugin)
    return plugin


class GantryFunctionProvider:
    """Live, per-turn Gantry tool source for a Semantic Kernel ``Kernel``.

    Holds a reference to a ``gantry`` and a ``kernel`` and exposes
    :meth:`refresh`, the per-turn hook the integrator calls before each
    ``agent.get_response()`` / chat-completion invocation. Each refresh
    re-selects the relevant tools from Gantry and rebuilds the kernel's
    ``plugin_name`` plugin to contain *only* those functions — so the function
    surface advertised to the model tracks the conversation turn by turn.

    Args:
        gantry: The :class:`~agent_gantry.core.gantry.AgentGantry` providing
            semantic retrieval and execution.
        kernel: The :class:`semantic_kernel.Kernel` whose plugin set is
            refreshed in place.
        plugin_name: Name of the plugin maintained for Gantry tools. Defaults
            to ``"gantry"``.
        limit: Maximum number of functions re-selected per turn. Defaults to
            ``5``.
        score_threshold: Minimum semantic relevance score for selected tools.
            Defaults to ``0.0`` (no filtering).
    """

    def __init__(
        self,
        gantry: AgentGantry,
        kernel: Any,
        *,
        plugin_name: str = _DEFAULT_PLUGIN,
        limit: int = 5,
        score_threshold: float = 0.0,
    ) -> None:
        self._gantry = gantry
        self._kernel = kernel
        self._plugin_name = plugin_name
        self._limit = limit
        self._score_threshold = score_threshold

    @property
    def plugin_name(self) -> str:
        return self._plugin_name

    @property
    def kernel(self) -> Any:
        return self._kernel

    @property
    def gantry(self) -> AgentGantry:
        return self._gantry

    async def refresh(self, query_or_messages: Any) -> dict[str, Any]:
        """Re-select tools and rebuild the kernel's gantry plugin.

        Call this before every agent/chat invocation. ``query_or_messages`` may
        be a query string (used verbatim) or a conversation history (the
        retrieval query is derived from the latest activity via
        :func:`~agent_gantry.query.latest_activity`).

        Returns the ``{function_name: KernelFunction}`` mapping now registered
        under :attr:`plugin_name` (empty dict if nothing was selected — the
        plugin is still refreshed to an empty set so stale functions clear).
        """
        query = _query_from(query_or_messages)
        functions = await gantry_plugin(
            self._gantry,
            query,
            limit=self._limit,
            plugin_name=self._plugin_name,
            score_threshold=self._score_threshold,
        )
        _set_plugin_functions(self._kernel, self._plugin_name, functions)
        return functions


async def refresh_kernel_tools(
    gantry: AgentGantry,
    kernel: Any,
    query: str,
    *,
    plugin_name: str = _DEFAULT_PLUGIN,
    limit: int = 5,
    score_threshold: float = 0.0,
) -> dict[str, Any]:
    """Re-select tools for ``query`` and rebuild ``kernel``'s gantry plugin once.

    Convenience equivalent of constructing a :class:`GantryFunctionProvider`
    and calling :meth:`~GantryFunctionProvider.refresh` a single time. The
    ``query`` may be a string or a conversation history.

    Returns the ``{function_name: KernelFunction}`` mapping now registered
    under ``plugin_name``.
    """
    resolved = _query_from(query)
    functions = await gantry_plugin(
        gantry,
        resolved,
        limit=limit,
        plugin_name=plugin_name,
        score_threshold=score_threshold,
    )
    _set_plugin_functions(kernel, plugin_name, functions)
    return functions


__all__ = ["GantryFunctionProvider", "refresh_kernel_tools"]
