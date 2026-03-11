"""
Async utility classes and helpers.
"""

from __future__ import annotations

from typing import Any


class AsyncNoopContext:
    """Async context manager that does nothing. Used as a fallback when telemetry is disabled."""

    async def __aenter__(self) -> AsyncNoopContext:
        return self

    async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> bool:
        return False
