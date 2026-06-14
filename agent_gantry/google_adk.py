"""Agent-Gantry × Google ADK.

Clean per-framework imports::

    from agent_gantry.google_adk import for_google_adk, gantry_before_model_callback

Re-exports Google ADK's static adapter (select + convert) and the deep per-turn "live" provider. Importing
this module does not require Google ADK until you actually call into it.
"""

from __future__ import annotations

from agent_gantry._framework_ns import make_lazy_getattr
from agent_gantry.integrations.frameworks.google_adk import (
    for_google_adk,
    spec_to_google_adk,
)

__getattr__ = make_lazy_getattr({'gantry_before_model_callback': 'google_adk_live', 'gantry_adk_agent': 'google_adk_live'})

__all__ = [
    "for_google_adk",
    "spec_to_google_adk",
    "gantry_before_model_callback",  # noqa: F822 (lazy via __getattr__)
    "gantry_adk_agent",  # noqa: F822 (lazy via __getattr__)
]
