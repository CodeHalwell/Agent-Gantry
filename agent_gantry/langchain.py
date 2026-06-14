"""Agent-Gantry × LangChain.

Clean per-framework imports::

    from agent_gantry.langchain import for_langchain

Re-exports LangChain's static adapter (select + convert). Importing this module
does not require LangChain until you actually call into it.
"""

from __future__ import annotations

from agent_gantry.integrations.frameworks.langchain import (
    for_langchain,
    spec_to_langchain,
)

__all__ = [
    "for_langchain",
    "spec_to_langchain",
]
