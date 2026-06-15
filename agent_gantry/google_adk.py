"""Agent-Gantry × Google ADK.

Clean per-framework import::

    from agent_gantry.google_adk import GoogleADKAdapter

Importing this module never requires Google ADK to be installed; it is imported
lazily when you call an adapter method.
"""

from __future__ import annotations

from agent_gantry.integrations.frameworks.google_adk import GoogleADKAdapter

__all__ = ["GoogleADKAdapter"]
