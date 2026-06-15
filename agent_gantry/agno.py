"""Agent-Gantry × Agno.

Clean per-framework import::

    from agent_gantry.agno import AgnoAdapter

Importing this module never requires Agno to be installed; it is imported lazily
when you call an adapter method.
"""

from __future__ import annotations

from agent_gantry.integrations.frameworks.agno import AgnoAdapter

__all__ = ["AgnoAdapter"]
