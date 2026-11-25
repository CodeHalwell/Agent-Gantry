"""
Lightweight metrics helpers for dotfetch.

This package keeps optional observability math (e.g., token savings) separate
from the core runtime to avoid additional runtime dependencies.
"""

from dotfetch.metrics.token_usage import ProviderUsage, TokenSavings, calculate_token_savings

__all__ = ["ProviderUsage", "TokenSavings", "calculate_token_savings"]
