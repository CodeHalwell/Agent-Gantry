"""Agent-Gantry × Smolagents.

Clean per-framework import::

    from agent_gantry.smolagents import SmolagentsAdapter

Importing this module never requires smolagents to be installed; it is imported
lazily when you call an adapter method.
"""

from __future__ import annotations

from agent_gantry.integrations.frameworks.smolagents import SmolagentsAdapter

__all__ = ["SmolagentsAdapter"]
