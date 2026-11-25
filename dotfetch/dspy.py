"""dotfetch × DSPy.

Clean per-framework import::

    from dotfetch.dspy import DSPyAdapter

This module's name shadows the third-party ``dspy`` package only within the
``dotfetch`` namespace (``dotfetch.dspy`` vs. top-level ``dspy``) — an
absolute ``import dspy`` (as used inside
``dotfetch.integrations.frameworks.dspy``) always resolves to the
installed ``dspy`` package on ``sys.path``, never to this shim, the same way
``dotfetch/openai.py`` never shadows the ``openai`` SDK or
``dotfetch/langgraph.py`` never shadows ``langgraph``. Importing this
module never requires DSPy to be installed; it is imported lazily when you
call an adapter method.
"""

from __future__ import annotations

from dotfetch.integrations.frameworks.dspy import (
    ApprovalRequest,
    Approver,
    DSPyAdapter,
    DSPyToolError,
    current_dspy_module,
)

__all__ = ["ApprovalRequest", "Approver", "DSPyAdapter", "DSPyToolError", "current_dspy_module"]
