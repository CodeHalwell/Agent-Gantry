"""Agent-Gantry × Pydantic AI.

Clean per-framework imports::

    from agent_gantry.pydantic_ai import for_pydantic_ai, gantry_toolset

Re-exports Pydantic AI's static adapter (select + convert) and the deep per-turn "live" provider. Importing
this module does not require Pydantic AI until you actually call into it.
"""

from __future__ import annotations

from agent_gantry._framework_ns import make_lazy_getattr
from agent_gantry.integrations.frameworks.pydantic_ai import (
    for_pydantic_ai,
    spec_to_pydantic_ai,
)

__getattr__ = make_lazy_getattr({'gantry_toolset': 'pydantic_ai_live'})

__all__ = [
    "for_pydantic_ai",
    "spec_to_pydantic_ai",
    "gantry_toolset",  # noqa: F822 (lazy via __getattr__)
]
