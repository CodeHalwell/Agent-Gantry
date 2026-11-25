"""Tests for ``required`` / ``always_include`` pinning on the shared selection core.

Ports the Microsoft Agent Framework's ``DotFetchContextProvider(required=...,
always_include=...)`` feature into ``DotFetchToolset.select`` / ``select_or_empty``
and ``BaseFrameworkAdapter.select`` so every framework adapter — not just AF —
gets guaranteed-present tools and always-on pins. See
``dotfetch/integrations/frameworks/base.py`` (``_resolve_pins`` /
``_pin_specs`` / ``_resolve_tool_names``) for the implementation and
``dotfetch/integrations/frameworks/errors.py`` for the shared
``MissingRequiredToolError``.
"""

from __future__ import annotations

import logging

import pytest

from dotfetch import DotFetch
from dotfetch.adapters.embedders.simple import SimpleEmbedder
from dotfetch.integrations.frameworks.base import (
    BaseFrameworkAdapter,
    DotFetchToolset,
    MissingRequiredToolError,
    ToolSpec,
)


@pytest.fixture
async def fetcher():
    g = DotFetch(embedder=SimpleEmbedder(dimension=64))

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
# DotFetchToolset.select — required
# --------------------------------------------------------------------------- #


async def test_required_tool_already_in_semantic_selection(fetcher) -> None:
    """A required tool the semantic slice already picked isn't duplicated."""
    toolset = DotFetchToolset(fetcher)
    specs = await toolset.select(EMAIL_QUERY, limit=3, required=["send_email"])
    names = [s.name for s in specs]
    assert names.count("send_email") == 1
    assert "send_email" in names


async def test_required_tool_missing_from_selection_is_appended(fetcher) -> None:
    """A required tool the semantic slice would NOT have picked is appended."""
    toolset = DotFetchToolset(fetcher)
    # limit=1 → only the top semantic match (send_email) is dynamically
    # selected; "ping" is unrelated to the query and must still be pinned.
    specs = await toolset.select(EMAIL_QUERY, limit=1, required=["ping"])
    names = [s.name for s in specs]
    assert "ping" in names
    assert names[-1] == "ping", "required tools are appended after the semantic slice"


async def test_required_tool_not_in_registry_raises(fetcher) -> None:
    """An unresolvable required tool raises MissingRequiredToolError, not a silent drop."""
    toolset = DotFetchToolset(fetcher)
    with pytest.raises(MissingRequiredToolError, match="no_such_tool"):
        await toolset.select(EMAIL_QUERY, limit=2, required=["no_such_tool"])


async def test_required_resolves_qualified_name(fetcher) -> None:
    """A required tool may be given as a bare name or a ``namespace.name`` qualified name."""
    toolset = DotFetchToolset(fetcher)
    specs = await toolset.select(EMAIL_QUERY, limit=1, required=["ops.restart_service"])
    names = [s.name for s in specs]
    assert "restart_service" in names


async def test_qualified_pin_not_satisfied_by_same_named_tool_elsewhere(fetcher) -> None:
    """A qualified pin is satisfied only by that exact tool.

    Regression test for a dedup bug where pinning ``other.foo`` was silently
    dropped whenever the semantic slice already contained *any* tool with the
    bare name ``foo`` — even one from a different namespace.
    """

    @fetcher.register(namespace="ops", tags=["email"])
    def send_email(to: str) -> str:
        "Send an email through the ops relay."
        return f"ops-sent:{to}"

    await fetcher.sync()
    toolset = DotFetchToolset(fetcher)
    # The semantic slice picks the default-namespace send_email; ops.send_email
    # must still be pinned because the pin names that exact tool.
    specs = await toolset.select(EMAIL_QUERY, limit=1, required=["ops.send_email"])
    pairs = [(s._namespace, s.name) for s in specs]
    assert ("default", "send_email") in pairs
    assert ("ops", "send_email") in pairs


async def test_required_multiple_missing_lists_all_in_error(fetcher) -> None:
    toolset = DotFetchToolset(fetcher)
    with pytest.raises(MissingRequiredToolError) as excinfo:
        await toolset.select(EMAIL_QUERY, limit=1, required=["nope_one", "nope_two"])
    assert "nope_one" in str(excinfo.value)
    assert "nope_two" in str(excinfo.value)


# --------------------------------------------------------------------------- #
# DotFetchToolset.select — always_include
# --------------------------------------------------------------------------- #


async def test_always_include_present_tool_is_appended(fetcher) -> None:
    toolset = DotFetchToolset(fetcher)
    specs = await toolset.select(EMAIL_QUERY, limit=1, always_include=["add"])
    names = [s.name for s in specs]
    assert "add" in names


async def test_always_include_missing_tool_is_skipped_with_warning(fetcher, caplog) -> None:
    """A missing always_include tool logs a WARNING and is silently skipped (no raise)."""
    toolset = DotFetchToolset(fetcher)
    with caplog.at_level(logging.WARNING, logger="dotfetch.integrations.frameworks.base"):
        specs = await toolset.select(
            EMAIL_QUERY, limit=1, always_include=["no_such_tool", "add"]
        )
    names = [s.name for s in specs]
    assert "add" in names
    assert "no_such_tool" not in names
    assert any("no_such_tool" in record.message for record in caplog.records)
    assert any(record.levelno == logging.WARNING for record in caplog.records)


async def test_always_include_does_not_raise(fetcher) -> None:
    toolset = DotFetchToolset(fetcher)
    # Should not raise even though the name doesn't exist anywhere.
    specs = await toolset.select(EMAIL_QUERY, limit=1, always_include=["totally_missing"])
    assert all(s.name != "totally_missing" for s in specs)


# --------------------------------------------------------------------------- #
# Dedup + ordering
# --------------------------------------------------------------------------- #


async def test_dedup_between_required_and_always_include(fetcher) -> None:
    """A name in both ``required`` and ``always_include`` is pinned exactly once."""
    toolset = DotFetchToolset(fetcher)
    specs = await toolset.select(
        EMAIL_QUERY, limit=1, required=["ping"], always_include=["ping"]
    )
    names = [s.name for s in specs]
    assert names.count("ping") == 1


async def test_dedup_against_semantic_selection(fetcher) -> None:
    """A required/always_include tool already in the semantic slice isn't re-appended."""
    toolset = DotFetchToolset(fetcher)
    specs = await toolset.select(
        EMAIL_QUERY,
        limit=3,
        required=["send_email"],
        always_include=["send_email"],
    )
    names = [s.name for s in specs]
    assert names.count("send_email") == 1


async def test_ordering_semantic_then_required_then_always_include(fetcher) -> None:
    """Semantic slice first, then required (in order), then always_include (in order)."""
    toolset = DotFetchToolset(fetcher)
    specs = await toolset.select(
        EMAIL_QUERY,
        limit=1,
        required=["ping"],
        always_include=["add"],
    )
    names = [s.name for s in specs]
    # send_email is the semantic top-1 for the email query.
    assert names == ["send_email", "ping", "add"]


async def test_required_order_preserved_for_multiple_names(fetcher) -> None:
    toolset = DotFetchToolset(fetcher)
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


async def test_required_not_dropped_by_small_limit(fetcher) -> None:
    """Pinned tools are never counted against ``limit`` — they're always additional."""
    toolset = DotFetchToolset(fetcher)
    specs = await toolset.select(EMAIL_QUERY, limit=1, required=["ping", "add"])
    names = [s.name for s in specs]
    assert "ping" in names
    assert "add" in names
    # limit=1 still bounds the semantic slice to at most one dynamic pick.
    assert len(names) == 3  # 1 semantic + 2 required


async def test_no_pins_leaves_limit_semantics_unchanged(fetcher) -> None:
    """Without required/always_include, behaviour is identical to before this feature."""
    toolset = DotFetchToolset(fetcher)
    specs = await toolset.select(EMAIL_QUERY, limit=2)
    assert len(specs) <= 2


# --------------------------------------------------------------------------- #
# select_or_empty
# --------------------------------------------------------------------------- #


async def test_select_or_empty_blank_query_no_pins_returns_empty(fetcher) -> None:
    toolset = DotFetchToolset(fetcher)
    specs = await toolset.select_or_empty("")
    assert specs == []


async def test_select_or_empty_blank_query_still_resolves_pins(fetcher) -> None:
    """Pins don't depend on the query's retrieval signal — they still resolve blank."""
    toolset = DotFetchToolset(fetcher)
    specs = await toolset.select_or_empty("", required=["ping"], always_include=["add"])
    names = [s.name for s in specs]
    assert names == ["ping", "add"]


async def test_select_or_empty_blank_query_missing_required_still_raises(fetcher) -> None:
    toolset = DotFetchToolset(fetcher)
    with pytest.raises(MissingRequiredToolError):
        await toolset.select_or_empty("", required=["no_such_tool"])


async def test_select_or_empty_nonblank_query_behaves_like_select(fetcher) -> None:
    toolset = DotFetchToolset(fetcher)
    specs = await toolset.select_or_empty(EMAIL_QUERY, limit=1, required=["ping"])
    names = [s.name for s in specs]
    assert "send_email" in names
    assert "ping" in names


# --------------------------------------------------------------------------- #
# BaseFrameworkAdapter.select
# --------------------------------------------------------------------------- #


class _EchoAdapter(BaseFrameworkAdapter):
    """Minimal concrete adapter for exercising the shared ``select()`` plumbing."""

    live_tier = "per-run"

    @staticmethod
    def convert(spec: ToolSpec) -> ToolSpec:
        return spec


async def test_base_framework_adapter_select_threads_required(fetcher) -> None:
    adapter = _EchoAdapter(fetcher)
    specs = await adapter.select(EMAIL_QUERY, limit=1, required=["ping"])
    names = [s.name for s in specs]
    assert "send_email" in names
    assert "ping" in names


async def test_base_framework_adapter_select_threads_always_include(fetcher) -> None:
    adapter = _EchoAdapter(fetcher)
    specs = await adapter.select(EMAIL_QUERY, limit=1, always_include=["add"])
    names = [s.name for s in specs]
    assert "add" in names


async def test_base_framework_adapter_select_required_missing_raises(fetcher) -> None:
    adapter = _EchoAdapter(fetcher)
    with pytest.raises(MissingRequiredToolError):
        await adapter.select(EMAIL_QUERY, limit=1, required=["no_such_tool"])


# --------------------------------------------------------------------------- #
# Shared error type identity (backward-compat import paths)
# --------------------------------------------------------------------------- #


def test_missing_required_tool_error_import_paths_are_identical() -> None:
    from dotfetch import MissingRequiredToolError as FromTopLevel
    from dotfetch.integrations import MissingRequiredToolError as FromIntegrations
    from dotfetch.integrations.agent_framework_provider import (
        MissingRequiredToolError as FromProvider,
    )
    from dotfetch.integrations.frameworks import (
        MissingRequiredToolError as FromFrameworks,
    )
    from dotfetch.integrations.frameworks.errors import (
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


async def test_confirmation_gated_tool_raises_typed_error():
    """A confirmation-gated tool surfaces ToolConfirmationRequiredError (a
    ToolExecutionError subclass) with the executor's explanation — frameworks
    can catch it to drive an approval flow instead of string-matching."""
    from dotfetch import DotFetch
    from dotfetch.adapters.embedders.simple import SimpleEmbedder
    from dotfetch.integrations.frameworks import (
        DotFetchToolset,
        ToolConfirmationRequiredError,
        ToolExecutionError,
    )

    fetcher = DotFetch(embedder=SimpleEmbedder(dimension=64))

    # Name deliberately outside the default SecurityPolicy patterns
    # (delete_*/payment_*/drop_*/refund_*) so this exercises the tool-flag
    # path (`requires_confirmation=True`), whose result carries the
    # re-issue hint. The policy-pattern path is covered separately below.
    @fetcher.register(tags=["danger"], requires_confirmation=True)
    def wipe_target(target: str) -> str:
        "Wipe a target permanently."
        return f"wiped:{target}"

    await fetcher.sync()
    specs = await DotFetchToolset(fetcher).select("wipe a target permanently", limit=1)
    assert specs and specs[0].requires_confirmation

    with pytest.raises(ToolConfirmationRequiredError) as excinfo:
        await specs[0].ainvoke(target="prod")
    assert isinstance(excinfo.value, ToolExecutionError)
    assert "confirmation" in str(excinfo.value).lower()
    assert "require_confirmation=False" in str(excinfo.value)


async def test_policy_pattern_confirmation_raises_typed_error():
    """The default SecurityPolicy's confirmation patterns (delete_* etc.)
    surface the same typed error, relaying the policy's own reason."""
    from dotfetch import DotFetch
    from dotfetch.adapters.embedders.simple import SimpleEmbedder
    from dotfetch.integrations.frameworks import (
        DotFetchToolset,
        ToolConfirmationRequiredError,
    )

    fetcher = DotFetch(embedder=SimpleEmbedder(dimension=64))

    @fetcher.register(tags=["danger"])
    def delete_everything(target: str) -> str:
        "Delete a target permanently."
        return f"deleted:{target}"

    await fetcher.sync()
    specs = await DotFetchToolset(fetcher).select("delete a target permanently", limit=1)

    with pytest.raises(ToolConfirmationRequiredError) as excinfo:
        await specs[0].ainvoke(target="prod")
    assert "delete_everything" in str(excinfo.value)
    assert "approval" in str(excinfo.value).lower()


async def test_policy_pattern_confirmation_approvable_via_require_confirmation_false():
    """ToolCall(require_confirmation=False) — the documented approval signal —
    now clears the SecurityPolicy pattern gate too, so both confirmation
    gates share one approve/resume mechanism."""
    from dotfetch import DotFetch, ToolCall
    from dotfetch.adapters.embedders.simple import SimpleEmbedder
    from dotfetch.schema.execution import ExecutionStatus

    fetcher = DotFetch(embedder=SimpleEmbedder(dimension=64))

    @fetcher.register(tags=["danger"])
    def delete_everything(target: str) -> str:
        "Delete a target permanently."
        return f"deleted:{target}"

    await fetcher.sync()

    gated = await fetcher.execute(
        ToolCall(tool_name="delete_everything", arguments={"target": "prod"})
    )
    assert gated.status == ExecutionStatus.PENDING_CONFIRMATION

    approved = await fetcher.execute(
        ToolCall(
            tool_name="delete_everything",
            arguments={"target": "prod"},
            require_confirmation=False,
        )
    )
    assert approved.status == ExecutionStatus.SUCCESS
    assert approved.result == "deleted:prod"
