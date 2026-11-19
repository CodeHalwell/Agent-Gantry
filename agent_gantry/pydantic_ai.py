"""Agent-Gantry × Pydantic AI.

Clean per-framework import::

    from agent_gantry.pydantic_ai import PydanticAIAdapter, current_run_context

    adapter = PydanticAIAdapter(gantry)
    agent = Agent(model, capabilities=[adapter.capability()])          # per-turn
    agent = Agent(model, capabilities=[adapter.capability(mode="search")])  # tool search

Importing this module never requires Pydantic AI to be installed; the framework
is imported lazily when you build a tool, toolset or capability. The toolset and
capability classes (``GantryToolset``, ``GantryDeferredToolset``,
``GantryCapability``) subclass Pydantic AI's own base classes, so they are
resolved on first attribute access.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from agent_gantry.integrations.frameworks.pydantic_ai import (
    ErrorMode,
    PydanticAIAdapter,
    current_run_context,
)

if TYPE_CHECKING:  # resolved lazily by ``__getattr__`` below
    from agent_gantry.integrations.frameworks.pydantic_ai_capability import (
        GantryCapability,
        gantry_search_strategy,
    )
    from agent_gantry.integrations.frameworks.pydantic_ai_live import (
        GantryDeferredToolset,
        GantryToolset,
    )

__all__ = [
    "ErrorMode",
    "GantryCapability",
    "GantryDeferredToolset",
    "GantryToolset",
    "PydanticAIAdapter",
    "current_run_context",
    "gantry_search_strategy",
]


def __getattr__(name: str) -> Any:
    if name in ("GantryToolset", "GantryDeferredToolset"):
        from agent_gantry.integrations.frameworks import pydantic_ai_live

        return getattr(pydantic_ai_live, name)
    if name in ("GantryCapability", "gantry_search_strategy"):
        from agent_gantry.integrations.frameworks import pydantic_ai_capability

        return getattr(pydantic_ai_capability, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
