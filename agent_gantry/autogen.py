"""Agent-Gantry × AutoGen / AG2.

Clean per-framework import::

    from agent_gantry.autogen import AutoGenAdapter

Importing this module never requires AutoGen to be installed; it is imported
lazily when you call an adapter method.
"""

from __future__ import annotations

from agent_gantry.integrations.frameworks.autogen import AutoGenAdapter

__all__ = ["AutoGenAdapter"]
