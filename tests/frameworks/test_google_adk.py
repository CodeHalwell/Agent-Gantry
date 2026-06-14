"""Unit tests for the Google ADK adapter using a stubbed ``google.adk``."""

from __future__ import annotations

import sys
import types

import pytest

from agent_gantry import AgentGantry
from agent_gantry.adapters.embedders.simple import SimpleEmbedder
from agent_gantry.integrations.frameworks.base import GantryToolset
from agent_gantry.integrations.frameworks.google_adk import (
    for_google_adk,
    spec_to_google_adk,
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
        "Add two integers together."
        return a + b

    await g.sync()
    return g


@pytest.fixture
def fake_adk(monkeypatch):
    class FunctionTool:
        def __init__(self, func, *, require_confirmation=False):
            self.func = func
            self.require_confirmation = require_confirmation
            self.name = getattr(func, "__name__", "tool")

    pkg = types.ModuleType("google")
    adk = types.ModuleType("google.adk")
    tools = types.ModuleType("google.adk.tools")
    tools.FunctionTool = FunctionTool
    monkeypatch.setitem(sys.modules, "google", pkg)
    monkeypatch.setitem(sys.modules, "google.adk", adk)
    monkeypatch.setitem(sys.modules, "google.adk.tools", tools)
    return FunctionTool


async def test_spec_to_google_adk_builds_and_routes(gantry, fake_adk):
    tool = (await for_google_adk(gantry, "send an email", limit=1))[0]
    assert tool.name == "send_email"
    # The wrapped func carries a real signature derived from the JSON schema.
    import inspect

    params = inspect.signature(tool.func).parameters
    assert "to" in params and "body" in params
    assert await tool.func(to="boss@x.com") == "sent:boss@x.com"


async def test_for_google_adk_maps_all(gantry, fake_adk):
    tools = await for_google_adk(gantry, "math add numbers", limit=2)
    assert len(tools) == 2


async def test_missing_google_adk_raises_helpful_error(monkeypatch, gantry):
    monkeypatch.setitem(sys.modules, "google.adk", None)
    monkeypatch.setitem(sys.modules, "google.adk.tools", None)
    specs = await GantryToolset(gantry).select("send an email", limit=1)
    with pytest.raises(ImportError, match="pip install google-adk"):
        spec_to_google_adk(specs[0])
