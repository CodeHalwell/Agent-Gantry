"""Agent-Gantry × Haystack.

Clean per-framework import::

    from agent_gantry.haystack import HaystackAdapter

Importing this module never requires Haystack to be installed; it is imported
lazily when you call an adapter method.
"""

from __future__ import annotations

from agent_gantry.integrations.frameworks.haystack import HaystackAdapter

__all__ = ["HaystackAdapter"]
