"""Internal helper for the per-framework public namespaces.

Each ``agent_gantry.<framework>`` module re-exports its framework's static
adapter eagerly (those are import-safe — the third-party framework is only
needed when you *call* them) and exposes the deep per-turn "live" providers
lazily via a module-level ``__getattr__``. The live provider modules build
framework subclasses at import time, so deferring them keeps
``import agent_gantry.<framework>`` cheap and only pulls in the framework when a
live symbol is actually accessed.
"""

from __future__ import annotations

import importlib
from collections.abc import Callable
from typing import Any

_PKG = "agent_gantry.integrations.frameworks"


def make_lazy_getattr(live_map: dict[str, str]) -> Callable[[str], Any]:
    """Return a module ``__getattr__`` resolving ``live_map`` names lazily.

    ``live_map`` maps an exported symbol name to the submodule of
    ``agent_gantry.integrations.frameworks`` that defines it.
    """

    def _module_getattr(name: str) -> Any:  # bound as module __getattr__ (PEP 562)
        module = live_map.get(name)
        if module is None:
            raise AttributeError(f"module has no attribute {name!r}")
        return getattr(importlib.import_module(f"{_PKG}.{module}"), name)

    return _module_getattr
