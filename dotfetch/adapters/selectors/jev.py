"""Deprecated: use :mod:`dotfetch.adapters.selectors.system_one`.

``JevSelector`` is now :class:`~dotfetch.adapters.selectors.system_one.SystemOneSelector`:
Jev is the default model, not the only one it can ask.
"""

from __future__ import annotations

from typing import Any

from dotfetch._deprecation import warn_renamed
from dotfetch.adapters.selectors import system_one as _new

_RENAMED = {"JevSelector": "SystemOneSelector", "JevClient": "SystemOneClient"}


def __getattr__(name: str) -> Any:
    target = _RENAMED.get(name)
    if target is not None:
        warn_renamed(f"{__name__}.{name}", f"{_new.__name__}.{target}")
        return getattr(_new, target)
    if not name.startswith("__") and hasattr(_new, name):
        warn_renamed(f"{__name__}.{name}", f"{_new.__name__}.{name}")
        return getattr(_new, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
