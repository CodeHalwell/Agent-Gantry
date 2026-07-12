"""Agent-Gantry × DSPy.

Clean per-framework import::

    from agent_gantry.dspy import DSPyAdapter

This module's name shadows the third-party ``dspy`` package only within the
``agent_gantry`` namespace (``agent_gantry.dspy`` vs. top-level ``dspy``) — an
absolute ``import dspy`` (as used inside
``agent_gantry.integrations.frameworks.dspy``) always resolves to the
installed ``dspy`` package on ``sys.path``, never to this shim, the same way
``agent_gantry/openai.py`` never shadows the ``openai`` SDK or
``agent_gantry/langgraph.py`` never shadows ``langgraph``. Importing this
module never requires DSPy to be installed; it is imported lazily when you
call an adapter method.
"""

from __future__ import annotations

from agent_gantry.integrations.frameworks.dspy import DSPyAdapter

__all__ = ["DSPyAdapter"]
