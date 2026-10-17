"""Native framework indexes must retain tools sharing a name across namespaces."""

from __future__ import annotations

import re
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from agent_gantry import AgentGantry
from agent_gantry.integrations.frameworks.base import GantryToolset, _native_tool_names
from agent_gantry.schema.tool import ToolDefinition


@pytest.fixture
def duplicate_gantry(request):
    gantry = AgentGantry()
    for namespace in ("billing", "inventory"):

        def make_lookup(label):
            def lookup(item: str) -> str:
                """Look up an item in the requested service."""
                return f"{label}:{item}"

            return lookup

        gantry.register(
            make_lookup(namespace), name=getattr(request, "param", "lookup"), namespace=namespace
        )
    return gantry


async def test_duplicate_names_keep_both_execution_targets(duplicate_gantry):
    specs = await GantryToolset(duplicate_gantry).select("look up an item", limit=2)
    assert {s.name for s in specs} == {"lookup"}
    assert len({s.native_name for s in specs}) == 2
    assert {await s.ainvoke(item="123") for s in specs} == {"billing:123", "inventory:123"}
    assert {t.name for t in duplicate_gantry.list_tools_sync()} == {"lookup"}


@pytest.mark.parametrize("duplicate_gantry", ["lookup", "x" * 64, "x" * 128], indirect=True)
async def test_aliases_stay_stable_across_selection_subsets_and_pins(duplicate_gantry):
    toolset = GantryToolset(duplicate_gantry)
    all_specs = await toolset.select("look up an item", limit=2)
    expected = {s._namespace: s.native_name for s in all_specs}
    bare_name = all_specs[0].name
    if len(bare_name) >= 64:
        assert {len(s.native_name) for s in all_specs} == {64}
        assert len(set(expected.values())) == 2
    one = await toolset.select("look up an item", namespaces=["billing"], limit=1)
    pinned = await toolset.select_or_empty(
        "", required=[f"inventory.{bare_name}", f"billing.{bare_name}"]
    )
    assert one[0].native_name == expected["billing"]
    assert {s._namespace: s.native_name for s in pinned} == expected


async def test_usage_tracking_resolves_native_aliases(duplicate_gantry, monkeypatch):
    toolset = GantryToolset(duplicate_gantry)
    specs = await toolset.select("look up an item", limit=2)
    retrieve = AsyncMock(return_value=SimpleNamespace(tools=[]))
    monkeypatch.setattr(duplicate_gantry, "retrieve", retrieve)
    await toolset.select("look up an item", tools_already_used=[specs[0].native_name])
    assert retrieve.call_args.args[0].context.tools_already_used == [
        f"{specs[0]._namespace}.lookup"
    ]


@pytest.mark.parametrize("name", ["lookup", "billing.lookup"])
def test_used_and_failed_signals_respect_qualified_names(duplicate_gantry, name):
    from agent_gantry.core.router import TaskIntent, _QueryContextCache

    context = _QueryContextCache(
        summary_lower="", messages_lower=[], tools_used={name}, tools_failed={name}
    )
    for tool in duplicate_gantry.list_tools_sync():
        signals = duplicate_gantry._router._compute_signals(
            tool, semantic_score=0.5, intent=TaskIntent.UNKNOWN, ctx=context
        )
        affected = name == "lookup" or tool.namespace == "billing"
        assert signals.conversation_relevance == (0.5 if affected else 0.0)
        assert signals.already_used_penalty == (0.1 if affected else 0.0)
        assert signals.already_failed_penalty == (0.3 if affected else 0.0)


def _definition(namespace, name="lookup"):
    return ToolDefinition(
        name=name, namespace=namespace, description="Look up an item.", parameters_schema={}
    )


def test_native_names_are_valid_distinct_and_independent_of_registry_order():
    definitions = [
        _definition("team-a"),
        _definition("team_a"),
        _definition("😀"),
        _definition("ops", "a" * 128),
    ]
    names = _native_tool_names(definitions)
    assert names == _native_tool_names(list(reversed(definitions)))
    assert len(set(names.values())) == len(definitions)
    assert all(re.fullmatch(r"[a-z][a-z0-9_]{0,63}", name) for name in names.values())


def test_generated_names_do_not_shadow_authored_names():
    definitions = [_definition("billing"), _definition("inventory")]
    generated = _native_tool_names(definitions)["billing.lookup"]
    definitions.append(_definition("default", generated))
    names = _native_tool_names(definitions)
    assert names[f"default.{generated}"] == generated
    assert names["billing.lookup"] != generated
    assert len(set(names.values())) == len(definitions)


async def test_langgraph_superset_matches_per_turn_selection(duplicate_gantry):
    pytest.importorskip("langchain_core")
    from agent_gantry.integrations.frameworks.langgraph_live import _all_tools

    superset = await _all_tools(duplicate_gantry)
    selected = await duplicate_gantry.framework("langgraph").select(
        "look up an item", namespaces=["billing"], limit=1
    )
    by_name = {tool.name: tool for tool in superset}
    assert len(by_name) == 2
    assert selected[0].name in by_name
    assert await by_name[selected[0].name].ainvoke({"item": "123"}) == "billing:123"


async def test_pydantic_live_toolset_keeps_duplicate_names(duplicate_gantry):
    pytest.importorskip("pydantic_ai")
    from types import SimpleNamespace

    toolset = duplicate_gantry.framework("pydantic_ai").live(limit=2).set_query("look up an item")
    tools = await toolset.get_tools(SimpleNamespace(max_retries=1))
    assert len(tools) == 2
    results = {
        await toolset.call_tool(name, {"item": "123"}, None, tool) for name, tool in tools.items()
    }
    assert results == {"billing:123", "inventory:123"}


async def test_bridge_cache_updates_names_after_namespace_collision():
    from agent_gantry.integrations.agent_framework_bridge import GantryToolBridge

    gantry = AgentGantry()

    @gantry.register(namespace="billing", name="lookup")
    def billing(item: str) -> str:
        """Look up an item in the requested service."""
        return f"billing:{item}"

    bridge = GantryToolBridge(gantry, as_function_tool=False)
    original = (await bridge.get_tools("look up an item", limit=1))[0]
    assert original.__name__ == "lookup"

    @gantry.register(namespace="inventory", name="lookup")
    def inventory(item: str) -> str:
        """Look up an item in the requested service."""
        return f"inventory:{item}"

    tools = await bridge.get_tools("look up an item", limit=2)
    assert len({t.__name__ for t in tools}) == 2
    assert all(t is not original for t in tools)
    assert {await t(item="123") for t in tools} == {"billing:123", "inventory:123"}
    assert {t.__name__: t for t in bridge.wrap_tools(gantry.list_tools_sync())} == {
        t.__name__: t for t in tools
    }


async def test_aliases_keep_confirmation_gates_on_original_target(duplicate_gantry):
    from agent_gantry.integrations.frameworks.base import ToolConfirmationRequiredError

    @duplicate_gantry.register(namespace="billing", name="lookup", requires_confirmation=True)
    def sensitive(item: str) -> str:
        """Look up a sensitive item after confirmation."""
        pytest.fail("confirmation-gated handler must not run")

    specs = await GantryToolset(duplicate_gantry).select_or_empty("", required=["billing.lookup"])
    assert specs[0].native_name != "lookup"
    with pytest.raises(ToolConfirmationRequiredError):
        await specs[0].ainvoke(item="123")


async def test_agent_framework_live_pins_deduplicate_against_native_aliases(duplicate_gantry):
    pytest.importorskip("agent_framework")
    provider = duplicate_gantry.framework("agent_framework").live(
        limit=1,
        required=["billing.lookup", "inventory.lookup"],
        always_include=["lookup"],
    )
    tools = await provider._collect_tools(
        [{"role": "user", "content": "look up an item"}],
        include_dynamic=True,
    )
    assert len(tools) == len({tool.name for tool in tools}) == 2
    assert {await tool.invoke(item="123", skip_parsing=True) for tool in tools} == {
        "billing:123",
        "inventory:123",
    }
