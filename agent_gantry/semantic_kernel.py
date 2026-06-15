"""Agent-Gantry × Semantic Kernel.

Clean per-framework import::

    from agent_gantry.semantic_kernel import SemanticKernelAdapter

Importing this module never requires Semantic Kernel to be installed; it is
imported lazily when you call an adapter method.
"""

from __future__ import annotations

from agent_gantry.integrations.frameworks.semantic_kernel import SemanticKernelAdapter

__all__ = ["SemanticKernelAdapter"]
