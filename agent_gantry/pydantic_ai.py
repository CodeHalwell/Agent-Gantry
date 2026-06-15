"""Agent-Gantry × Pydantic AI.

Clean per-framework import::

    from agent_gantry.pydantic_ai import PydanticAIAdapter

Importing this module never requires Pydantic AI to be installed; the framework
is imported lazily when you call an adapter method.
"""

from __future__ import annotations

from agent_gantry.integrations.frameworks.pydantic_ai import PydanticAIAdapter

__all__ = ["PydanticAIAdapter"]
