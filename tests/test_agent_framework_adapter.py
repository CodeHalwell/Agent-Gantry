"""Surface tests for the unified Microsoft Agent Framework adapter.

``AgentFrameworkAdapter`` is a thin, gantry-bound entry point whose methods
construct the underlying AF primitives (``GantryContextProvider`` /
``GantryToolBridge`` / the middleware), each of which has its own tests. These
checks lock the adapter's structure and confirm it imports without
``agent-framework`` installed — the actual AF-object construction is exercised by
the agent-frameworks CI matrix (where AF is present) and the provider/bridge
test suites.
"""

from __future__ import annotations

from agent_gantry.agent_framework import AgentFrameworkAdapter


async def test_common_select_surface_preserves_default_budget_and_pins(gantry, monkeypatch):
    @gantry.register
    def first() -> str:
        """Read the first available record."""
        return "first"

    @gantry.register
    def second() -> str:
        """Read the second available record."""
        return "second"

    monkeypatch.setattr(AgentFrameworkAdapter, "convert", staticmethod(lambda spec: spec))
    adapter = AgentFrameworkAdapter(gantry, default_top_k=1)
    selected = await adapter.select("read a record", required=["first", "second"])
    assert len(selected) == 2
    assert {spec.name for spec in selected} == {"first", "second"}


async def test_common_select_has_actionable_missing_sdk_error(gantry, monkeypatch):
    import pytest

    from agent_gantry.integrations import agent_framework_bridge

    @gantry.register
    def lookup() -> str:
        """Look up an available record."""
        return "ok"

    monkeypatch.setattr(agent_framework_bridge, "_try_import_af_tool", lambda: None)
    with pytest.raises(ImportError, match="pip install agent-framework-core"):
        await gantry.framework("agent_framework").select("look up a record")


def test_adapter_constructs_and_stores_gantry() -> None:
    sentinel = object()
    adapter = AgentFrameworkAdapter(sentinel, default_top_k=7)
    assert adapter._gantry is sentinel
    assert adapter._default_top_k == 7


def test_adapter_exposes_expected_methods() -> None:
    adapter = AgentFrameworkAdapter(object())
    for name in (
        "select",
        "convert",
        "context_provider",
        "tool_bridge",
        "approval_middleware",
        "observability_middleware",
        "tool_choice_middleware",
    ):
        assert callable(getattr(adapter, name)), name
