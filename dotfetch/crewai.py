"""dotfetch × CrewAI.

Clean per-framework import::

    from dotfetch.crewai import CrewAIAdapter

Importing this module never requires CrewAI to be installed; the framework is
imported lazily when you call an adapter method.
"""

from __future__ import annotations

from dotfetch.integrations.frameworks.crewai import (
    ApprovalRequest,
    Approver,
    CrewAIAdapter,
    current_crew_context,
)
from dotfetch.integrations.frameworks.native import ErrorFormatter, ErrorMode

__all__ = [
    "ApprovalRequest",
    "Approver",
    "CrewAIAdapter",
    "ErrorFormatter",
    "ErrorMode",
    "current_crew_context",
]
