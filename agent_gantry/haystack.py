"""Agent-Gantry × Haystack.

Clean per-framework imports::

    from agent_gantry.haystack import for_haystack, gantry_haystack_tools

Re-exports Haystack's static adapter (select + convert) and the deep per-turn "live" provider. Importing
this module does not require Haystack until you actually call into it.
"""

from __future__ import annotations

from agent_gantry._framework_ns import make_lazy_getattr
from agent_gantry.integrations.frameworks.haystack import (
    for_haystack,
    spec_to_haystack,
)

__getattr__ = make_lazy_getattr({'gantry_haystack_tools': 'live_wrappers', 'GantryLiveHaystackToolInvoker': 'live_wrappers'})

__all__ = [
    "for_haystack",
    "spec_to_haystack",
    "gantry_haystack_tools",  # noqa: F822 (lazy via __getattr__)
    "GantryLiveHaystackToolInvoker",  # noqa: F822 (lazy via __getattr__)
]
