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


def test_adapter_constructs_and_stores_gantry() -> None:
    sentinel = object()
    adapter = AgentFrameworkAdapter(sentinel, default_top_k=7)
    assert adapter._gantry is sentinel
    assert adapter._default_top_k == 7


def test_adapter_exposes_expected_methods() -> None:
    adapter = AgentFrameworkAdapter(object())
    for name in (
        "context_provider",
        "tool_bridge",
        "approval_middleware",
        "observability_middleware",
        "tool_choice_middleware",
    ):
        assert callable(getattr(adapter, name)), name


def test_tool_bridge_returns_concrete_gantry_tool_bridge() -> None:
    """tool_bridge() builds a concrete GantryToolBridge — no agent-framework needed."""
    from agent_gantry.agent_framework import GantryToolBridge

    bridge = AgentFrameworkAdapter(object()).tool_bridge()
    assert isinstance(bridge, GantryToolBridge)
