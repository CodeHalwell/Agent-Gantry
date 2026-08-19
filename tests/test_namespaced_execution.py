"""Execution must target the namespace that selection resolved.

Selection is namespace-aware throughout: ``ToolSpec`` carries ``_namespace``
and ``qualified_name``, pinning distinguishes ``"other.foo"`` from ``"foo"``,
and the Agent Framework bridge caches per namespace. Execution was not:
``ToolCall`` had only a bare ``tool_name``, and the registry's bare-name lookup
prefers ``default.<name>`` before falling back to a first-match index. Two MCP
servers each exposing ``search`` is a supported configuration, so the selected
``other.search`` could silently run ``default.search``.
"""

from __future__ import annotations

import logging

import pytest

from agent_gantry import AgentGantry
from agent_gantry.schema.execution import ExecutionStatus, ToolCall
from agent_gantry.schema.tool import ToolDefinition

_QUERY_SCHEMA = {
    "type": "object",
    "properties": {"q": {"type": "string"}},
    "required": ["q"],
}


@pytest.fixture
async def two_namespaces() -> AgentGantry:
    """Register the same tool name in ``default`` and in ``billing``."""
    gantry = AgentGantry()

    async def default_search(q: str) -> str:
        return "default-result"

    async def billing_search(q: str) -> str:
        return "billing-result"

    await gantry.add_tool(
        ToolDefinition(
            name="search",
            description="Search the default corpus for a query.",
            parameters_schema=_QUERY_SCHEMA,
        ),
        handler=default_search,
    )
    await gantry.add_tool(
        ToolDefinition(
            name="search",
            namespace="billing",
            description="Search billing records for a query.",
            parameters_schema=_QUERY_SCHEMA,
        ),
        handler=billing_search,
    )
    return gantry


async def test_namespace_selects_the_right_tool(two_namespaces: AgentGantry) -> None:
    """An explicit namespace must win over the default-namespace preference."""
    result = await two_namespaces.execute(
        ToolCall(tool_name="search", namespace="billing", arguments={"q": "x"})
    )
    assert result.status == ExecutionStatus.SUCCESS
    assert result.result == "billing-result"


async def test_default_namespace_still_reachable(two_namespaces: AgentGantry) -> None:
    result = await two_namespaces.execute(
        ToolCall(tool_name="search", namespace="default", arguments={"q": "x"})
    )
    assert result.result == "default-result"


async def test_qualified_tool_name_is_honoured(two_namespaces: AgentGantry) -> None:
    """Tool names cannot contain a dot, so a dotted name is unambiguous."""
    result = await two_namespaces.execute(
        ToolCall(tool_name="billing.search", arguments={"q": "x"})
    )
    assert result.status == ExecutionStatus.SUCCESS
    assert result.result == "billing-result"


async def test_bare_name_warns_when_ambiguous(
    two_namespaces: AgentGantry, caplog: pytest.LogCaptureFixture
) -> None:
    """A bare name that exists twice resolves silently — say so."""
    with caplog.at_level(logging.WARNING, logger="agent_gantry.core.executor"):
        await two_namespaces.execute(ToolCall(tool_name="search", arguments={"q": "x"}))

    assert any("registered in 2 namespaces" in r.getMessage() for r in caplog.records)


async def test_unknown_namespace_fails_cleanly(two_namespaces: AgentGantry) -> None:
    """Naming a namespace that does not have the tool must not fall back."""
    result = await two_namespaces.execute(
        ToolCall(tool_name="search", namespace="nope", arguments={"q": "x"})
    )
    assert result.status == ExecutionStatus.FAILURE
    assert result.error_type == "ToolNotFound"


async def test_single_namespace_needs_no_warning(caplog: pytest.LogCaptureFixture) -> None:
    """The common case stays quiet and keeps working by bare name."""
    gantry = AgentGantry()

    async def only_search(q: str) -> str:
        return "only"

    await gantry.add_tool(
        ToolDefinition(
            name="search",
            description="Search the one and only corpus.",
            parameters_schema=_QUERY_SCHEMA,
        ),
        handler=only_search,
    )

    with caplog.at_level(logging.WARNING, logger="agent_gantry.core.executor"):
        result = await gantry.execute(ToolCall(tool_name="search", arguments={"q": "x"}))

    assert result.result == "only"
    assert not [r for r in caplog.records if "namespaces" in r.getMessage()]


async def test_tool_spec_invokes_its_own_namespace(two_namespaces: AgentGantry) -> None:
    """The framework adapter path must execute the spec it selected.

    This is the regression that matters in practice: every adapter goes through
    ``ToolSpec.ainvoke``, which used to execute by bare name.
    """
    from agent_gantry.integrations.frameworks.base import GantryToolset

    toolset = GantryToolset(two_namespaces)
    specs = await toolset.select("search billing records", limit=5)
    billing = [s for s in specs if s._namespace == "billing"]
    assert billing, f"expected billing.search among {[s.qualified_name for s in specs]}"

    assert await billing[0].ainvoke(q="x") == "billing-result"


async def test_select_and_execute_uses_the_selected_tool(two_namespaces: AgentGantry) -> None:
    """``search_and_execute`` picks a tool namespace-aware; it must run that one."""
    result = await two_namespaces.search_and_execute(
        "search billing records", arguments={"q": "x"}
    )
    assert result.status == ExecutionStatus.SUCCESS
