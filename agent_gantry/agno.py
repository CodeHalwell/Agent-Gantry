"""Agent-Gantry × Agno.

Clean per-framework imports::

    from agent_gantry.agno import for_agno, GantryLiveAgnoAgent

Re-exports Agno's static adapter (select + convert) and the deep per-turn "live" provider. Importing
this module does not require Agno until you actually call into it.
"""

from __future__ import annotations

from agent_gantry._framework_ns import make_lazy_getattr
from agent_gantry.integrations.frameworks.agno import (
    for_agno,
    spec_to_agno,
)

__getattr__ = make_lazy_getattr({'GantryLiveAgnoAgent': 'live_wrappers'})

__all__ = [
    "for_agno",
    "spec_to_agno",
    "GantryLiveAgnoAgent",  # noqa: F822 (lazy via __getattr__)
]
