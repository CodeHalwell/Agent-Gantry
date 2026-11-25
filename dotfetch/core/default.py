"""
The default instance behind the module-level API.

``import dotfetch as dot`` gives a short form for scripts, notebooks and small
apps that need one registry::

    import dotfetch as dot

    @dot.register(tags=["weather"])
    def get_weather(city: str) -> str:
        '''Get the current weather for a city.'''
        ...

    tools = await dot.fetch("Is it raining in Leeds?")
    results = await dot.execute_tool_calls(response)

Each function calls the same method on one shared :class:`DotFetch`, created
on first use with the class's own defaults. :func:`configure` replaces it
before any tools are registered, and :func:`default_fetcher` returns it for
everything the shortcuts don't cover. Anything larger than one registry per
process (tests, several tenants) is better served by explicit instances.

An instance given to :func:`~dotfetch.set_default_fetcher` takes precedence in
the context it was set in, so ``@with_semantic_tools`` and the module-level
functions always agree on which instance they mean.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from dotfetch.core.fetcher import DotFetch

_lock = threading.Lock()
_process_default: DotFetch | None = None


def _context_default() -> DotFetch | None:
    from dotfetch.integrations.semantic_tools import _fetcher_context

    return _fetcher_context.get()


def current_default() -> DotFetch | None:
    """The default instance if one exists, without creating it."""
    return _context_default() or _process_default


def default_fetcher() -> DotFetch:
    """Return the instance the module-level functions use, creating it if needed.

    That is the instance given to :func:`~dotfetch.set_default_fetcher` in the
    current context, if any; otherwise one shared instance for the process,
    built by :func:`configure` or, on first use, as ``DotFetch()``.
    """
    global _process_default
    fetcher = current_default()
    if fetcher is not None:
        return fetcher
    with _lock:
        if _process_default is None:
            from dotfetch.core.fetcher import DotFetch

            _process_default = DotFetch()
        return _process_default


def configure(fetcher: DotFetch | None = None, **kwargs: Any) -> DotFetch:
    """Set up the shared instance: pass one, or the arguments to build one.

    Call it before registering anything, since the instance it replaces is
    dropped with whatever was registered on it::

        dot.configure(embedder=OpenAIEmbedder(...), vector_store=...)

    Args:
        fetcher: A ready instance to use. Mutually exclusive with ``kwargs``.
        **kwargs: Arguments for ``DotFetch(...)``.

    Returns:
        The instance now in use.

    Raises:
        RuntimeError: The current shared instance already has tools registered.
        TypeError: Both ``fetcher`` and keyword arguments were given.
    """
    global _process_default
    if fetcher is not None and kwargs:
        raise TypeError("configure() takes a DotFetch instance or its arguments, not both")
    with _lock:
        previous = _process_default
        registered = len(previous.list_tools_sync()) if previous is not None else 0
        if registered and previous is not fetcher:
            raise RuntimeError(
                f"configure() would drop the {registered} tool(s) already registered "
                "on the default instance. Call it before registering, or pass the "
                "configured instance around explicitly."
            )
        if fetcher is None:
            from dotfetch.core.fetcher import DotFetch

            fetcher = DotFetch(**kwargs)
        _process_default = fetcher
        return fetcher


def register(func: Callable[..., Any] | None = None, **kwargs: Any) -> Callable[..., Any]:
    """Register a function as a tool on the default instance.

    Works bare (``@dot.register``) or with the options of
    :meth:`DotFetch.register` (``@dot.register(tags=[...])``).
    """
    return default_fetcher().register(func, **kwargs)


async def fetch(query: str, limit: int = 5, **kwargs: Any) -> list[dict[str, Any]]:
    """Fetch the tools relevant to ``query`` from the default instance.

    Accepts the arguments of :meth:`DotFetch.fetch` (``dialect``,
    ``score_threshold``, ...). Registered tools are synced first if they
    have changed.
    """
    return await default_fetcher().fetch(query, limit=limit, **kwargs)


async def execute_tool_calls(response: Any, **kwargs: Any) -> list[dict[str, Any]]:
    """Run the tool calls in a model ``response`` on the default instance.

    Accepts the arguments of :meth:`DotFetch.execute_tool_calls` (``dialect``,
    ``parallel``, ...) and returns results ready to send back to the model.
    """
    return await default_fetcher().execute_tool_calls(response, **kwargs)


def _reset() -> None:
    """Forget the shared instance. For tests; it is not closed."""
    global _process_default
    with _lock:
        _process_default = None
