"""Agent-Gantry × the OpenAI Agents SDK.

Clean per-framework imports::

    from agent_gantry.openai_agents import for_openai_agents, gantry_run_hooks

Re-exports the OpenAI Agents SDK's static adapter (select + convert) and the deep per-turn "live" provider. Importing
this module does not require the OpenAI Agents SDK until you actually call into it.
"""

from __future__ import annotations

from agent_gantry._framework_ns import make_lazy_getattr
from agent_gantry.integrations.frameworks.openai_agents import (
    for_openai_agents,
    spec_to_openai_agents,
)

__getattr__ = make_lazy_getattr({'gantry_run_hooks': 'openai_agents_live', 'run_with_gantry': 'openai_agents_live', 'GantryAgentSession': 'openai_agents_live', 'refresh_agent_tools': 'openai_agents_live', 'select_function_tools': 'openai_agents_live'})

__all__ = [
    "for_openai_agents",
    "spec_to_openai_agents",
    "gantry_run_hooks",  # noqa: F822 (lazy via __getattr__)
    "run_with_gantry",  # noqa: F822 (lazy via __getattr__)
    "GantryAgentSession",  # noqa: F822 (lazy via __getattr__)
    "refresh_agent_tools",  # noqa: F822 (lazy via __getattr__)
    "select_function_tools",  # noqa: F822 (lazy via __getattr__)
]
