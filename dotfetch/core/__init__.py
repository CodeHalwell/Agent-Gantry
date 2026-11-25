"""
Core modules for dotfetch.

Contains the main facade, registry, router, executor, and context management.
"""

from dotfetch.core.context import ConversationContextManager
from dotfetch.core.executor import ExecutionEngine
from dotfetch.core.fetcher import DotFetch
from dotfetch.core.registry import ToolRegistry
from dotfetch.core.router import SemanticRouter

__all__ = [
    "DotFetch",
    "ConversationContextManager",
    "ExecutionEngine",
    "SemanticRouter",
    "ToolRegistry",
]
