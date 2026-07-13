"""Agent-Gantry × AWS Strands Agents.

Clean per-framework import::

    from agent_gantry.strands import StrandsAdapter

Importing this module never requires Strands Agents to be installed; it is
imported lazily when you call an adapter method.
"""

from __future__ import annotations

from agent_gantry.integrations.frameworks.strands import StrandsAdapter

__all__ = ["StrandsAdapter"]
