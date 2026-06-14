"""Agent-Gantry × Smolagents.

Clean per-framework imports::

    from agent_gantry.smolagents import for_smolagents, GantryLiveSmolAgent

Re-exports Smolagents's static adapter (select + convert) and the deep per-turn "live" provider. Importing
this module does not require Smolagents until you actually call into it.
"""

from __future__ import annotations

from agent_gantry._framework_ns import make_lazy_getattr
from agent_gantry.integrations.frameworks.smolagents import (
    for_smolagents,
    spec_to_smolagents,
)

__getattr__ = make_lazy_getattr({'GantryLiveSmolAgent': 'live_wrappers'})

__all__ = [
    "for_smolagents",
    "spec_to_smolagents",
    "GantryLiveSmolAgent",  # noqa: F822 (lazy via __getattr__)
]
