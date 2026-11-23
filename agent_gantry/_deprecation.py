"""Helpers for names that moved: the old spelling keeps working, with a warning."""

from __future__ import annotations

import warnings


def warn_renamed(old: str, new: str, *, stacklevel: int = 3) -> None:
    """Emit the ``DeprecationWarning`` for a renamed public name.

    Args:
        old: The deprecated spelling, as the caller wrote it.
        new: What to use instead.
        stacklevel: Frames between the warning and the caller's own code. The
            default suits a module-level ``__getattr__``.
    """
    warnings.warn(
        f"{old} is deprecated and will be removed in a future release; use {new} instead.",
        DeprecationWarning,
        stacklevel=stacklevel,
    )
