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


class RenamedValue(str):
    """A string value that still compares equal to the spelling it replaced.

    For a public *value* that was renamed (``live_tier == "per-call"`` is now
    ``"per-run"``), as :func:`warn_renamed` is for a renamed name. The object
    is the new spelling in every respect (``str()``, ``repr()``, hashing,
    JSON) except that ``==`` and ``!=`` against the old spelling still answer
    as they did, with a ``DeprecationWarning``, so code branching on the old
    value keeps working and says what to change.

    Hashing cannot be bridged: a ``dict`` or ``set`` keyed by the old spelling
    will not find this value. Copies and pickles are plain ``str``.

    Args:
        value: The current spelling.
        old: The spelling it replaced.
        what: What the value is, for the warning (``"live_tier"``).
    """

    __slots__ = ("_old", "_what")

    def __new__(cls, value: str, *, old: str, what: str) -> RenamedValue:
        self = super().__new__(cls, value)
        self._old = old
        self._what = what
        return self

    def _is_old(self, other: object) -> bool:
        if not isinstance(other, str) or str.__ne__(other, self._old):
            return False
        warnings.warn(
            f"{self._what} {self._old!r} is now {str(self)!r}; comparing against "
            f"{self._old!r} is deprecated and will stop matching in a future release.",
            DeprecationWarning,
            stacklevel=3,
        )
        return True

    def __eq__(self, other: object) -> bool:
        return self._is_old(other) or str.__eq__(self, other) is True

    def __ne__(self, other: object) -> bool:
        return not self.__eq__(other)

    __hash__ = str.__hash__

    def __reduce__(self) -> tuple[type, tuple[str]]:
        return (str, (str.__str__(self),))
