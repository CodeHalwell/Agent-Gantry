"""Tests for the AutoGen / AG2 native tool adapter.

AutoGen (pyautogen) is not installed in this environment, so for the
registration path a minimal fake ``autogen`` module is injected into
``sys.modules`` (cleaned up by ``monkeypatch.setitem``) to resolve the
adapter's lazy import. Its ``register_function`` records each call so tests can
assert what was registered. The schema-only helpers need no framework at all.
"""

from __future__ import annotations

import sys
import types

import pytest

from agent_gantry import AgentGantry
from agent_gantry.adapters.embedders.simple import SimpleEmbedder


@pytest.fixture
async def gantry():
    g = AgentGantry(embedder=SimpleEmbedder(dimension=64))

    @g.register(tags=["email"])
    def send_email(to: str, body: str = "") -> str:
        "Send an email message to a recipient."
        return f"sent:{to}"

    @g.register(tags=["math"])
    def add(a: int, b: int) -> int:
        "Add two integers together."
        return a + b

    await g.sync()
    return g


@pytest.fixture
def fake_autogen(monkeypatch):
    """Inject a fake ``autogen`` module whose ``register_function`` records calls."""
    records: list[dict] = []

    def register_function(func, *, caller, executor, name, description):
        records.append(
            {
                "func": func,
                "caller": caller,
                "executor": executor,
                "name": name,
                "description": description,
            }
        )

    pkg = types.ModuleType("autogen")
    pkg.register_function = register_function
    pkg.records = records
    monkeypatch.setitem(sys.modules, "autogen", pkg)
    return pkg


async def test_for_autogen_returns_mappings(gantry):
    from agent_gantry.autogen import AutoGenAdapter

    tools = await AutoGenAdapter(gantry).select("send an email", limit=2)

    assert isinstance(tools, list)
    assert len(tools) >= 1
    names = {t["name"] for t in tools}
    assert "send_email" in names

    spec = next(t for t in tools if t["name"] == "send_email")
    assert spec["description"] == "Send an email message to a recipient."
    assert callable(spec["callable"])
    # The callable routes through gantry.execute and returns the result.
    assert await spec["callable"](to="boss@x.com") == "sent:boss@x.com"


async def test_spec_to_autogen_shape(gantry):
    from agent_gantry.autogen import AutoGenAdapter
    from agent_gantry.integrations.frameworks.base import GantryToolset

    specs = await GantryToolset(gantry).select("send an email", limit=1)
    mapping = AutoGenAdapter.convert(specs[0])

    assert set(mapping) == {"name", "description", "callable"}
    assert mapping["name"] == "send_email"
    assert mapping["description"] == "Send an email message to a recipient."
    assert mapping["callable"].__name__ == "send_email"
    assert await mapping["callable"](to="boss@x.com") == "sent:boss@x.com"


async def test_register_with_autogen_records_names(fake_autogen, gantry):
    from agent_gantry.autogen import AutoGenAdapter

    caller = object()
    executor = object()

    names = await AutoGenAdapter(gantry).register(
        "send an email", caller=caller, executor=executor, limit=2
    )

    assert isinstance(names, list)
    assert "send_email" in names
    assert len(fake_autogen.records) == len(names)

    record = next(r for r in fake_autogen.records if r["name"] == "send_email")
    assert record["caller"] is caller
    assert record["executor"] is executor
    assert record["description"] == "Send an email message to a recipient."
    # The captured func executes through gantry.
    assert await record["func"](to="boss@x.com") == "sent:boss@x.com"


async def test_missing_autogen_raises_helpful_error(monkeypatch, gantry):
    from agent_gantry.autogen import AutoGenAdapter

    # Ensure the lazy import fails even if a real package is somehow present.
    monkeypatch.setitem(sys.modules, "autogen", None)

    with pytest.raises(ImportError, match="pip install pyautogen"):
        await AutoGenAdapter(gantry).register(
            "send an email", caller=object(), executor=object()
        )
