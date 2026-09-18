"""
Selector adapters for Agent-Gantry.

A selector chooses catalogue entries by asking a decision model directly,
instead of embedding the query and searching a vector store. The base
:class:`SelectorAdapter` protocol is imported eagerly; concrete implementations
that pull in optional dependencies (typesafe-sdk) are lazy-loaded via
``__getattr__`` so importing this module does not require optional installs.
"""

from agent_gantry.adapters.selectors.base import SelectorAdapter

__all__ = [
    "JevSelector",
    "SelectorAdapter",
]


def __getattr__(name: str) -> type:
    if name == "JevSelector":
        from agent_gantry.adapters.selectors.jev import JevSelector

        return JevSelector
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(__all__)
