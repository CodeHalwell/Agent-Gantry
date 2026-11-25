"""dotfetch × Pydantic AI.

Clean per-framework import::

    from dotfetch.pydantic_ai import PydanticAIAdapter, current_run_context

    adapter = PydanticAIAdapter(fetcher)
    agent = Agent(model, capabilities=[adapter.capability()])          # per-turn
    agent = Agent(model, capabilities=[adapter.capability(mode="search")])  # tool search

Importing this module never requires Pydantic AI to be installed; the framework
is imported lazily when you build a tool, toolset or capability. The toolset and
capability classes (``DotFetchToolset``, ``DotFetchDeferredToolset``,
``DotFetchCapability``) subclass Pydantic AI's own base classes, so they are
resolved on first attribute access.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from dotfetch.integrations.frameworks.pydantic_ai import (
    ErrorFormatter,
    ErrorMode,
    PydanticAIAdapter,
    current_run_context,
)

if TYPE_CHECKING:  # resolved lazily by ``__getattr__`` below
    from dotfetch.integrations.frameworks.pydantic_ai_capability import (
        DotFetchCapability,
        fetcher_search_strategy,
    )
    from dotfetch.integrations.frameworks.pydantic_ai_live import (
        DotFetchDeferredToolset,
        DotFetchToolset,
    )

__all__ = [
    "ErrorFormatter",
    "ErrorMode",
    "DotFetchCapability",
    "DotFetchDeferredToolset",
    "DotFetchToolset",
    "PydanticAIAdapter",
    "current_run_context",
    "fetcher_search_strategy",
]


def __getattr__(name: str) -> Any:
    if name in ("DotFetchToolset", "DotFetchDeferredToolset"):
        from dotfetch.integrations.frameworks import pydantic_ai_live

        return getattr(pydantic_ai_live, name)
    if name in ("DotFetchCapability", "fetcher_search_strategy"):
        from dotfetch.integrations.frameworks import pydantic_ai_capability

        return getattr(pydantic_ai_capability, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
