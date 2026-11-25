"""Deprecated: use :mod:`dotfetch.adapters.system_one_client`.

The client was named after TypeSafe's Jev, the endpoint's default model, but it
serves any System One decision model. The old names resolve to the new classes.
"""

from __future__ import annotations

from typing import Any

from dotfetch._deprecation import warn_renamed
from dotfetch.adapters import system_one_client as _new

_RENAMED = {
    "JevClient": "SystemOneClient",
    "JevVerdict": "SystemOneVerdict",
    "JevIncompleteResponseError": "SystemOneIncompleteResponseError",
}


def __getattr__(name: str) -> Any:
    target = _RENAMED.get(name)
    if target is not None:
        warn_renamed(f"{__name__}.{name}", f"{_new.__name__}.{target}")
        return getattr(_new, target)
    if not name.startswith("__") and hasattr(_new, name):
        warn_renamed(f"{__name__}.{name}", f"{_new.__name__}.{name}")
        return getattr(_new, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
