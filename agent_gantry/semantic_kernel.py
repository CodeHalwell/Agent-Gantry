"""Agent-Gantry × Semantic Kernel.

Clean per-framework imports::

    from agent_gantry.semantic_kernel import for_semantic_kernel, GantryFunctionProvider

Re-exports Semantic Kernel's static adapter (select + convert) and the deep per-turn "live" provider. Importing
this module does not require Semantic Kernel until you actually call into it.
"""

from __future__ import annotations

from agent_gantry._framework_ns import make_lazy_getattr
from agent_gantry.integrations.frameworks.semantic_kernel import (
    for_semantic_kernel,
    gantry_plugin,
    spec_to_semantic_kernel,
)

__getattr__ = make_lazy_getattr({'GantryFunctionProvider': 'semantic_kernel_live', 'refresh_kernel_tools': 'semantic_kernel_live'})

__all__ = [
    "for_semantic_kernel",
    "spec_to_semantic_kernel",
    "gantry_plugin",
    "GantryFunctionProvider",  # noqa: F822 (lazy via __getattr__)
    "refresh_kernel_tools",  # noqa: F822 (lazy via __getattr__)
]
