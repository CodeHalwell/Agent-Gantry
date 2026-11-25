"""Shared error types for the framework-adapter selection layer.

Split out from :mod:`dotfetch.integrations.frameworks.base` (and, prior to
the shared ``required``/``always_include`` support landing there, from
:mod:`dotfetch.integrations.agent_framework_provider`, the only place that
previously implemented pinned-tool resolution) so both the shared
:class:`~dotfetch.integrations.frameworks.base.DotFetchToolset` and the
Microsoft Agent Framework ``DotFetchContextProvider`` raise (and callers can
catch) the *same* exception type.

``dotfetch.integrations.agent_framework_provider`` re-exports
:class:`MissingRequiredToolError` from here for backward compatibility — the
historical import path (``from
dotfetch.integrations.agent_framework_provider import
MissingRequiredToolError``), and the top-level ``dotfetch`` /
``dotfetch.integrations`` / ``dotfetch.agent_framework`` re-exports
that build on it, all continue to resolve to this same class.
"""

from __future__ import annotations


class MissingRequiredToolError(LookupError):
    """Raised when a tool listed in ``required=[...]`` is not present in the fetcher.

    Raised by both :meth:`~dotfetch.integrations.frameworks.base.DotFetchToolset.select`
    (and ``select_or_empty``) and
    :class:`~dotfetch.integrations.agent_framework_provider.DotFetchContextProvider`
    when a name passed as ``required`` cannot be resolved against the fetcher's
    registry (by bare name or by ``namespace.name`` qualified name).
    """


__all__ = ["MissingRequiredToolError"]
