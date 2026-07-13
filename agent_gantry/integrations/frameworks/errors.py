"""Shared error types for the framework-adapter selection layer.

Split out from :mod:`agent_gantry.integrations.frameworks.base` (and, prior to
the shared ``required``/``always_include`` support landing there, from
:mod:`agent_gantry.integrations.agent_framework_provider`, the only place that
previously implemented pinned-tool resolution) so both the shared
:class:`~agent_gantry.integrations.frameworks.base.GantryToolset` and the
Microsoft Agent Framework ``GantryContextProvider`` raise (and callers can
catch) the *same* exception type.

``agent_gantry.integrations.agent_framework_provider`` re-exports
:class:`MissingRequiredToolError` from here for backward compatibility — the
historical import path (``from
agent_gantry.integrations.agent_framework_provider import
MissingRequiredToolError``), and the top-level ``agent_gantry`` /
``agent_gantry.integrations`` / ``agent_gantry.agent_framework`` re-exports
that build on it, all continue to resolve to this same class.
"""

from __future__ import annotations


class MissingRequiredToolError(LookupError):
    """Raised when a tool listed in ``required=[...]`` is not present in the gantry.

    Raised by both :meth:`~agent_gantry.integrations.frameworks.base.GantryToolset.select`
    (and ``select_or_empty``) and
    :class:`~agent_gantry.integrations.agent_framework_provider.GantryContextProvider`
    when a name passed as ``required`` cannot be resolved against the gantry's
    registry (by bare name or by ``namespace.name`` qualified name).
    """


__all__ = ["MissingRequiredToolError"]
