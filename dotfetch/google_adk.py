"""dotfetch × Google ADK.

Clean per-framework import::

    from dotfetch.google_adk import GoogleADKAdapter

Importing this module never requires Google ADK to be installed; it is imported
lazily when you call an adapter method.
"""

from __future__ import annotations

from dotfetch.integrations.frameworks.google_adk import GoogleADKAdapter, current_tool_context
from dotfetch.integrations.frameworks.native import ErrorFormatter, ErrorMode

__all__ = ["ErrorFormatter", "ErrorMode", "GoogleADKAdapter", "current_tool_context"]
