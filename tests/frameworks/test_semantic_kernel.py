"""Unit tests for the Semantic Kernel adapter using a stubbed ``semantic_kernel``."""

from __future__ import annotations

import sys
import types

import pytest

from agent_gantry import AgentGantry
from agent_gantry.adapters.embedders.simple import SimpleEmbedder
from agent_gantry.integrations.frameworks.base import GantryToolset
from agent_gantry.integrations.frameworks.semantic_kernel import (
    SemanticKernelAdapter,
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
def fake_sk(monkeypatch):
    """Stub semantic_kernel.functions with kernel_function + KernelFunctionFromMethod."""

    def kernel_function(func=None, name=None, description=None):
        def deco(f):
            f.__sk_name__ = name
            f.__sk_description__ = description
            return f

        return deco(func) if func is not None else deco

    class KernelFunctionFromMethod:
        def __init__(self, method, plugin_name=None):
            self.method = method
            self.plugin_name = plugin_name
            self.name = getattr(method, "__sk_name__", None) or method.__name__

        @classmethod
        def from_method(cls, method, plugin_name=None, stream_method=None):
            return cls(method, plugin_name)

    sk = types.ModuleType("semantic_kernel")
    functions = types.ModuleType("semantic_kernel.functions")
    functions.kernel_function = kernel_function
    functions.KernelFunctionFromMethod = KernelFunctionFromMethod
    monkeypatch.setitem(sys.modules, "semantic_kernel", sk)
    monkeypatch.setitem(sys.modules, "semantic_kernel.functions", functions)
    return KernelFunctionFromMethod


async def test_spec_to_semantic_kernel_builds_and_routes(gantry, fake_sk):
    kf = (await SemanticKernelAdapter(gantry).select("send an email", limit=1))[0]
    assert kf.name == "send_email"
    assert kf.plugin_name == "gantry"
    # the decorated method carries a real signature + return annotation
    assert kf.method.__annotations__.get("return") is str
    assert await kf.method(to="boss@x.com") == "sent:boss@x.com"


async def test_gantry_plugin_returns_name_keyed_dict(gantry, fake_sk):
    plugin = await SemanticKernelAdapter(gantry).plugin(
        "send an email", limit=1, plugin_name="mail"
    )
    assert "send_email" in plugin
    assert plugin["send_email"].plugin_name == "mail"


async def test_missing_semantic_kernel_raises_helpful_error(monkeypatch, gantry):
    monkeypatch.setitem(sys.modules, "semantic_kernel", None)
    monkeypatch.setitem(sys.modules, "semantic_kernel.functions", None)
    specs = await GantryToolset(gantry).select("send an email", limit=1)
    with pytest.raises(ImportError, match="pip install semantic-kernel"):
        SemanticKernelAdapter.convert(specs[0])
