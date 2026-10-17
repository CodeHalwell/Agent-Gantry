"""Bridge caches follow tool revisions without retaining unbounded old wrappers."""

from __future__ import annotations

import inspect
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

from agent_gantry import AgentGantry
from agent_gantry.integrations.agent_framework_bridge import GantryToolBridge, _tool_approval_mode
from agent_gantry.schema.tool import ToolCapability


@pytest.fixture
def registered():
    gantry = AgentGantry()

    @gantry.register
    def lookup(value: int) -> str:
        """Look up an item using its numeric identifier."""
        return str(value)

    return gantry, gantry.list_tools_sync()[0]


async def test_replacement_refreshes_the_cached_signature_and_description(registered):
    gantry, _ = registered
    bridge = GantryToolBridge(gantry, as_function_tool=False)
    before = (await bridge.get_tools("look up an item", limit=1))[0]

    @gantry.register(name="lookup")
    def replacement(text: str) -> str:
        """Look up an item using its text identifier."""
        return text

    after = (await bridge.get_tools("look up an item", limit=1))[0]
    assert after is not before
    assert list(inspect.signature(after).parameters) == ["text"]
    assert "text identifier" in after.__doc__
    assert await after(text="abc") == "abc"


def test_nested_schema_edits_refresh_cached_wrappers(registered):
    gantry, definition = registered
    bridge = GantryToolBridge(gantry, as_function_tool=False)
    before = bridge.wrap_single(definition)
    definition.parameters_schema["properties"]["value"]["description"] = "An updated parameter"
    after = bridge.wrap_single(definition)
    assert after is not before
    assert "An updated parameter" in str(inspect.signature(after))


@pytest.mark.parametrize("change", ["confirmation", "capabilities"])
def test_approval_metadata_changes_refresh_cached_wrappers(registered, change):
    gantry, definition = registered
    bridge = GantryToolBridge(gantry, as_function_tool=False)
    before = bridge.wrap_single(definition)
    if change == "confirmation":
        definition.requires_confirmation = True
    else:
        definition.capabilities.append(ToolCapability.DELETE_DATA)
    assert bridge.wrap_single(definition) is not before
    assert _tool_approval_mode(definition) == "always_require"


def test_health_and_routing_metadata_do_not_rebuild_native_wrappers(registered):
    gantry, definition = registered
    bridge = GantryToolBridge(gantry, as_function_tool=False)
    before = bridge.wrap_single(definition)
    definition.health.total_calls += 1
    definition.tags.append("routing-only")
    definition.metadata["last_reviewed"] = "today"
    assert bridge.wrap_single(definition) is before


def test_property_reordering_refreshes_the_positional_signature(registered):
    gantry, definition = registered
    definition.parameters_schema = {
        "type": "object",
        "properties": {"first": {"type": "string"}, "second": {"type": "string"}},
        "required": ["first", "second"],
    }
    bridge = GantryToolBridge(gantry, as_function_tool=False)
    before = bridge.wrap_single(definition)
    definition.parameters_schema["properties"] = {
        "second": {"type": "string"},
        "first": {"type": "string"},
    }
    after = bridge.wrap_single(definition)
    assert after is not before
    assert list(inspect.signature(after).parameters) == ["second", "first"]


def test_cache_evicts_least_recently_used_wrapper(registered):
    gantry, first = registered
    second = first.model_copy(update={"name": "second_lookup"})
    third = first.model_copy(update={"name": "third_lookup"})
    bridge = GantryToolBridge(gantry, as_function_tool=False, cache_size=2)
    first_wrapper = bridge.wrap_single(first)
    second_wrapper = bridge.wrap_single(second)
    assert bridge.wrap_single(first) is first_wrapper
    bridge.wrap_single(third)
    assert bridge.wrap_single(first) is first_wrapper
    assert bridge.wrap_single(second) is not second_wrapper


def test_zero_sized_cache_disables_reuse(registered):
    gantry, definition = registered
    bridge = GantryToolBridge(gantry, as_function_tool=False, cache_size=0)
    assert bridge.wrap_single(definition) is not bridge.wrap_single(definition)


@pytest.mark.parametrize("size", [-1, 1.5, "2"])
def test_bad_cache_size_fails_at_construction(registered, size):
    with pytest.raises(ValueError, match="cache_size"):
        GantryToolBridge(registered[0], cache_size=size)


def test_threads_share_one_wrapper_build(registered, monkeypatch):
    from agent_gantry.integrations import agent_framework_bridge as module

    gantry, definition = registered
    bridge = GantryToolBridge(gantry, as_function_tool=False)
    original = module._build_callable_for_tool
    builds = []

    def build(*args, **kwargs):
        builds.append(1)
        time.sleep(0.01)
        return original(*args, **kwargs)

    monkeypatch.setattr(module, "_build_callable_for_tool", build)
    with ThreadPoolExecutor(max_workers=8) as pool:
        wrappers = list(pool.map(lambda _: bridge.wrap_single(definition), range(16)))
    assert len(builds) == 1
    assert all(wrapper is wrappers[0] for wrapper in wrappers)
    bridge.clear_cache()
    assert bridge.wrap_single(definition) is not wrappers[0]


async def test_real_af_refreshes_native_schema_and_confirmation(registered):
    pytest.importorskip("agent_framework")
    gantry, definition = registered
    bridge = GantryToolBridge(gantry, as_function_tool=True)
    before = bridge.wrap_single(definition)
    definition.requires_confirmation = True
    definition.description = "Look up an item after a human confirms the request."
    after = bridge.wrap_single(definition)
    assert after is not before
    assert after.approval_mode == "always_require"
    assert after.description == definition.description
