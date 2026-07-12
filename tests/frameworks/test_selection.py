"""Tests for ``required`` / ``always_include`` pinning on the shared selection core.

Ports the Microsoft Agent Framework's ``GantryContextProvider(required=...,
always_include=...)`` feature into ``GantryToolset.select`` / ``select_or_empty``
and ``BaseFrameworkAdapter.select`` so every framework adapter — not just AF —
gets guaranteed-present tools and always-on pins. See
``agent_gantry/integrations/frameworks/base.py`` (``_resolve_pins`` /
``_pin_specs`` / ``_resolve_tool_names``) for the implementation and
``agent_gantry/integrations/frameworks/errors.py`` for the shared
``MissingRequiredToolError``.
"""

from __future__ import annotations

import logging

import pytest

from agent_gantry import AgentGantry
from agent_gantry.adapters.embedders.simple import SimpleEmbedder
from agent_gantry.integrations.frameworks.base import (
    BaseFrameworkAdapter,
    GantryToolset,
    MissingRequiredToolError,
    ToolSpec,
)


@pytest.fixture
async def gantry():
    g = AgentGantry(embedder=SimpleEmbedder(dimension=64))

    @g.register(tags=["email"])
    def send_email(to: str, body: str = "") -> str:
        "Send an email message to a recipient."
        return f"sent:{to}"

    @g.register(tags=["math"])
    def add(a: int, b: int) -> int:
        "Add two integers together and return the sum."
        return a + b

    @g.register(tags=["util"])
    def ping() -> str:
        "Ping a remote host to check whether it is reachable."
        return "pong"

    @g.register(namespace="ops", tags=["util"])
    def restart_service(name: str) -> str:
        "Restart a named backend service."
        return f"restarted:{name}"

    await g.sync()
    return g


EMAIL_QUERY = "send an email to my boss about the quarterly report"


# --------------------------------------------------------------------------- #
# GantryToolset.select — required
# --------------------------------------------------------------------------- #


async def test_required_tool_already_in_semantic_selection(gantry) -> None:
    """A required tool the semantic slice already picked isn't duplicated."""
    toolset = GantryToolset(gantry)
    specs = await toolset.select(EMAIL_QUERY, limit=3, required=["send_email"])
    names = [s.name for s in specs]
    assert names.count("send_email") == 1
    assert "send_email" in names


async def test_required_tool_missing_from_selection_is_appended(gantry) -> None:
    """A required tool the semantic slice would NOT have picked is appended."""
    toolset = GantryToolset(gantry)
    # limit=1 → only the top semantic match (send_email) is dynamically
    # selected; "ping" is unrelated to the query and must still be pinned.
    specs = await toolset.select(EMAIL_QUERY, limit=1, required=["ping"])
    names = [s.name for s in specs]
    assert "ping" in names
    assert names[-1] == "ping", "required tools are appended after the semantic slice"


async def test_required_tool_not_in_registry_raises(gantry) -> None:
    """An unresolvable required tool raises MissingRequiredToolError, not a silent drop."""
    toolset = GantryToolset(gantry)
    with pytest.raises(MissingRequiredToolError, match="no_such_tool"):
        await toolset.select(EMAIL_QUERY, limit=2, required=["no_such_tool"])


async def test_required_resolves_qualified_name(gantry) -> None:
    """A required tool may be given as a bare name or a ``namespace.name`` qualified name."""
    toolset = GantryToolset(gantry)
    specs = await toolset.select(EMAIL_QUERY, limit=1, required=["ops.restart_service"])
    names = [s.name for s in specs]
    assert "restart_service" in names


async def test_required_multiple_missing_lists_all_in_error(gantry) -> None:
    toolset = GantryToolset(gantry)
    with pytest.raises(MissingRequiredToolError) as excinfo:
        await toolset.select(EMAIL_QUERY, limit=1, required=["nope_one", "nope_two"])
    assert "nope_one" in str(excinfo.value)
    assert "nope_two" in str(excinfo.value)


# --------------------------------------------------------------------------- #
# GantryToolset.select — always_include
# --------------------------------------------------------------------------- #


async def test_always_include_present_tool_is_appended(gantry) -> None:
    toolset = GantryToolset(gantry)
    specs = await toolset.select(EMAIL_QUERY, limit=1, always_include=["add"])
    names = [s.name for s in specs]
    assert "add" in names


async def test_always_include_missing_tool_is_skipped_with_warning(gantry, caplog) -> None:
    """A missing always_include tool logs a WARNING and is silently skipped (no raise)."""
    toolset = GantryToolset(gantry)
    with caplog.at_level(logging.WARNING, logger="agent_gantry.integrations.frameworks.base"):
        specs = await toolset.select(
            EMAIL_QUERY, limit=1, always_include=["no_such_tool", "add"]
        )
    names = [s.name for s in specs]
    assert "add" in names
    assert "no_such_tool" not in names
    assert any("no_such_tool" in record.message for record in caplog.records)
    assert any(record.levelno == logging.WARNING for record in caplog.records)


async def test_always_include_does_not_raise(gantry) -> None:
    toolset = GantryToolset(gantry)
    # Should not raise even though the name doesn't exist anywhere.
    specs = await toolset.select(EMAIL_QUERY, limit=1, always_include=["totally_missing"])
    assert all(s.name != "totally_missing" for s in specs)


# --------------------------------------------------------------------------- #
# Dedup + ordering
# --------------------------------------------------------------------------- #


async def test_dedup_between_required_and_always_include(gantry) -> None:
    """A name in both ``required`` and ``always_include`` is pinned exactly once."""
    toolset = GantryToolset(gantry)
    specs = await toolset.select(
        EMAIL_QUERY, limit=1, required=["ping"], always_include=["ping"]
    )
    names = [s.name for s in specs]
    assert names.count("ping") == 1


async def test_dedup_against_semantic_selection(gantry) -> None:
    """A required/always_include tool already in the semantic slice isn't re-appended."""
    toolset = GantryToolset(gantry)
    specs = await toolset.select(
        EMAIL_QUERY,
        limit=3,
        required=["send_email"],
        always_include=["send_email"],
    )
    names = [s.name for s in specs]
    assert names.count("send_email") == 1


async def test_ordering_semantic_then_required_then_always_include(gantry) -> None:
    """Semantic slice first, then required (in order), then always_include (in order)."""
    toolset = GantryToolset(gantry)
    specs = await toolset.select(
        EMAIL_QUERY,
        limit=1,
        required=["ping"],
        always_include=["add"],
    )
    names = [s.name for s in specs]
    # send_email is the semantic top-1 for the email query.
    assert names == ["send_email", "ping", "add"]


async def test_required_order_preserved_for_multiple_names(gantry) -> None:
    toolset = GantryToolset(gantry)
    specs = await toolset.select(
        EMAIL_QUERY, limit=1, required=["ping", "add", "ops.restart_service"]
    )
    names = [s.name for s in specs]
    # send_email is the semantic top-1 for the email query; the three
    # required tools must follow, in the order they were listed.
    assert names[0] == "send_email"
    assert names[1:] == ["ping", "add", "restart_service"]


# --------------------------------------------------------------------------- #
# limit interaction
# --------------------------------------------------------------------------- #


async def test_required_not_dropped_by_small_limit(gantry) -> None:
    """Pinned tools are never counted against ``limit`` — they're always additional."""
    toolset = GantryToolset(gantry)
    specs = await toolset.select(EMAIL_QUERY, limit=1, required=["ping", "add"])
    names = [s.name for s in specs]
    assert "ping" in names
    assert "add" in names
    # limit=1 still bounds the semantic slice to at most one dynamic pick.
    assert len(names) == 3  # 1 semantic + 2 required


async def test_no_pins_leaves_limit_semantics_unchanged(gantry) -> None:
    """Without required/always_include, behaviour is identical to before this feature."""
    toolset = GantryToolset(gantry)
    specs = await toolset.select(EMAIL_QUERY, limit=2)
    assert len(specs) <= 2


# --------------------------------------------------------------------------- #
# select_or_empty
# --------------------------------------------------------------------------- #


async def test_select_or_empty_blank_query_no_pins_returns_empty(gantry) -> None:
    toolset = GantryToolset(gantry)
    specs = await toolset.select_or_empty("")
    assert specs == []


async def test_select_or_empty_blank_query_still_resolves_pins(gantry) -> None:
    """Pins don't depend on the query's retrieval signal — they still resolve blank."""
    toolset = GantryToolset(gantry)
    specs = await toolset.select_or_empty("", required=["ping"], always_include=["add"])
    names = [s.name for s in specs]
    assert names == ["ping", "add"]


async def test_select_or_empty_blank_query_missing_required_still_raises(gantry) -> None:
    toolset = GantryToolset(gantry)
    with pytest.raises(MissingRequiredToolError):
        await toolset.select_or_empty("", required=["no_such_tool"])


async def test_select_or_empty_nonblank_query_behaves_like_select(gantry) -> None:
    toolset = GantryToolset(gantry)
    specs = await toolset.select_or_empty(EMAIL_QUERY, limit=1, required=["ping"])
    names = [s.name for s in specs]
    assert "send_email" in names
    assert "ping" in names


# --------------------------------------------------------------------------- #
# BaseFrameworkAdapter.select
# --------------------------------------------------------------------------- #


class _EchoAdapter(BaseFrameworkAdapter):
    """Minimal concrete adapter for exercising the shared ``select()`` plumbing."""

    live_tier = "per-call"

    @staticmethod
    def convert(spec: ToolSpec) -> ToolSpec:
        return spec


async def test_base_framework_adapter_select_threads_required(gantry) -> None:
    adapter = _EchoAdapter(gantry)
    specs = await adapter.select(EMAIL_QUERY, limit=1, required=["ping"])
    names = [s.name for s in specs]
    assert "send_email" in names
    assert "ping" in names


async def test_base_framework_adapter_select_threads_always_include(gantry) -> None:
    adapter = _EchoAdapter(gantry)
    specs = await adapter.select(EMAIL_QUERY, limit=1, always_include=["add"])
    names = [s.name for s in specs]
    assert "add" in names


async def test_base_framework_adapter_select_required_missing_raises(gantry) -> None:
    adapter = _EchoAdapter(gantry)
    with pytest.raises(MissingRequiredToolError):
        await adapter.select(EMAIL_QUERY, limit=1, required=["no_such_tool"])


# --------------------------------------------------------------------------- #
# Shared error type identity (backward-compat import paths)
# --------------------------------------------------------------------------- #


def test_missing_required_tool_error_import_paths_are_identical() -> None:
    from agent_gantry import MissingRequiredToolError as FromTopLevel
    from agent_gantry.integrations import MissingRequiredToolError as FromIntegrations
    from agent_gantry.integrations.agent_framework_provider import (
        MissingRequiredToolError as FromProvider,
    )
    from agent_gantry.integrations.frameworks import (
        MissingRequiredToolError as FromFrameworks,
    )
    from agent_gantry.integrations.frameworks.errors import (
        MissingRequiredToolError as FromErrors,
    )

    assert (
        FromTopLevel
        is FromIntegrations
        is FromProvider
        is FromFrameworks
        is FromErrors
        is MissingRequiredToolError
    )
    assert issubclass(MissingRequiredToolError, LookupError)
