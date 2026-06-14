"""Agent-Gantry × CrewAI.

Clean per-framework imports::

    from agent_gantry.crewai import for_crewai, gantry_crew_tools

Re-exports CrewAI's static adapter (select + convert) and the deep per-turn "live" provider. Importing
this module does not require CrewAI until you actually call into it.
"""

from __future__ import annotations

from agent_gantry._framework_ns import make_lazy_getattr
from agent_gantry.integrations.frameworks.crewai import (
    for_crewai,
    spec_to_crewai,
)

__getattr__ = make_lazy_getattr({'gantry_crew_tools': 'live_wrappers', 'GantryLiveCrewAgent': 'live_wrappers'})

__all__ = [
    "for_crewai",
    "spec_to_crewai",
    "gantry_crew_tools",  # noqa: F822 (lazy via __getattr__)
    "GantryLiveCrewAgent",  # noqa: F822 (lazy via __getattr__)
]
