"""Coroutine detection compatible with both legacy and current decorators."""

from __future__ import annotations

import asyncio
import inspect
from typing import Any

# asgiref and older async decorators use asyncio's identity marker on Python
# 3.10/3.11. Inspect alone does not recognize it, while asyncio's detection API
# is deprecated on Python 3.14. A unique fallback avoids matching unmarked values
# if a future Python removes the legacy marker altogether.
_LEGACY_COROUTINE_MARKER = getattr(asyncio.coroutines, "_is_coroutine", object())


def is_coroutine_function(value: Any) -> bool:
    """Recognize async functions without calling asyncio's deprecated detector."""
    return inspect.iscoroutinefunction(value) or (
        getattr(value, "_is_coroutine", None) is _LEGACY_COROUTINE_MARKER
    )
