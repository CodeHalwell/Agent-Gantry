"""
Selector adapters for Agent-Gantry.

A selector chooses catalogue entries by asking a decision model directly,
instead of embedding the query and searching a vector store. The base
:class:`SelectorAdapter` protocol is imported eagerly; concrete implementations
that pull in optional dependencies (typesafe-sdk) are lazy-loaded via
``__getattr__`` so importing this module does not require optional installs.
"""

from __future__ import annotations

from agent_gantry.adapters.selectors.base import SelectorAdapter, SelectorFallbackError

__all__ = [
    "SelectorAdapter",
    "SelectorFallbackError",
    "SystemOneSelector",
]


def __getattr__(name: str) -> type:
    if name == "SystemOneSelector":
        from agent_gantry.adapters.selectors.system_one import SystemOneSelector

        return SystemOneSelector
    if name == "JevSelector":  # renamed in 0.23.0
        from agent_gantry._deprecation import warn_renamed
        from agent_gantry.adapters.selectors.system_one import SystemOneSelector

        warn_renamed(f"{__name__}.JevSelector", f"{__name__}.SystemOneSelector")
        return SystemOneSelector
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(__all__)
