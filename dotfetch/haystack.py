"""dotfetch × Haystack.

Clean per-framework import::

    from dotfetch.haystack import HaystackAdapter

Importing this module never requires Haystack to be installed; it is imported
lazily when you call an adapter method.
"""

from __future__ import annotations

from dotfetch.integrations.frameworks.haystack import (
    DotFetchConfirmationHook,
    HaystackAdapter,
    HaystackToolError,
    current_agent_state,
)

__all__ = [
    "DotFetchConfirmationHook",
    "HaystackAdapter",
    "HaystackToolError",
    "current_agent_state",
]
