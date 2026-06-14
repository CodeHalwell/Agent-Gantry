"""Agent-Gantry × AutoGen.

Clean per-framework imports::

    from agent_gantry.autogen import for_autogen, gantry_workbench

Re-exports AutoGen's static adapter (select + convert) and the deep per-turn "live" provider. Importing
this module does not require AutoGen until you actually call into it.
"""

from __future__ import annotations

from agent_gantry._framework_ns import make_lazy_getattr
from agent_gantry.integrations.frameworks.autogen import (
    for_autogen,
    register_with_autogen,
    spec_to_autogen,
)

__getattr__ = make_lazy_getattr({'gantry_workbench': 'autogen_live'})

__all__ = [
    "for_autogen",
    "spec_to_autogen",
    "register_with_autogen",
    "gantry_workbench",  # noqa: F822 (lazy via __getattr__)
]
