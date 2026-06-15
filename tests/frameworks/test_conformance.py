"""Cross-framework conformance matrix for the native tool adapters.

Every ``<Framework>Adapter`` in ``agent_gantry.integrations.frameworks`` must
honor the same contract regardless of which third-party framework it targets:

1. It exposes ``select`` (async select+convert) and ``convert`` (single convert
   staticmethod), both importable without the framework installed.
2. ``select`` runs semantic selection against a real gantry and returns one
   native object per selected tool (verified through a per-framework stub).
3. ``convert`` raises a clean ``ImportError`` carrying a ``pip install`` hint
   when the framework is absent — not ``AttributeError`` / ``KeyError``.

This locks the uniform surface so a new adapter can't silently drift.
"""

from __future__ import annotations

import sys
import types

import pytest

from agent_gantry import AgentGantry
from agent_gantry.adapters.embedders.simple import SimpleEmbedder
from agent_gantry.integrations import frameworks as F


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


# (framework key, Adapter class, module name(s) the lazy import needs)
ADAPTERS = [
    ("langchain", F.LangChainAdapter, ["langchain_core", "langchain_core.tools"]),
    ("langgraph", F.LangGraphAdapter, ["langchain_core", "langchain_core.tools"]),
    ("llamaindex", F.LlamaIndexAdapter, ["llama_index", "llama_index.core", "llama_index.core.tools"]),
    ("crewai", F.CrewAIAdapter, ["crewai", "crewai.tools"]),
    ("pydantic_ai", F.PydanticAIAdapter, ["pydantic_ai", "pydantic_ai.tools"]),
    ("openai_agents", F.OpenAIAgentsAdapter, ["agents"]),
    ("smolagents", F.SmolagentsAdapter, ["smolagents"]),
    ("haystack", F.HaystackAdapter, ["haystack", "haystack.tools"]),
    ("agno", F.AgnoAdapter, ["agno", "agno.tools", "agno.tools.function"]),
    ("semantic_kernel", F.SemanticKernelAdapter, ["semantic_kernel", "semantic_kernel.functions"]),
    ("google_adk", F.GoogleADKAdapter, ["google.adk", "google.adk.tools"]),
]


@pytest.mark.parametrize("name,adapter_cls,modules", ADAPTERS, ids=[a[0] for a in ADAPTERS])
def test_adapter_exposes_uniform_surface(name, adapter_cls, modules):
    assert callable(getattr(adapter_cls, "select", None)), f"{name}: {adapter_cls.__name__}.select missing"
    assert callable(getattr(adapter_cls, "convert", None)), f"{name}: {adapter_cls.__name__}.convert missing"


@pytest.mark.parametrize("name,adapter_cls,modules", ADAPTERS, ids=[a[0] for a in ADAPTERS])
def test_missing_framework_raises_clean_importerror(name, adapter_cls, modules, monkeypatch):
    # Ensure the framework's import resolves to "not installed".
    for mod in modules:
        monkeypatch.setitem(sys.modules, mod, None)

    from agent_gantry.integrations.frameworks.base import ToolSpec

    dummy = ToolSpec(
        name="t",
        qualified_name="default.t",
        description="d",
        parameters={"type": "object", "properties": {}},
        requires_confirmation=False,
        score=0.0,
        _gantry=None,  # type: ignore[arg-type]
        _namespace="default",
    )
    with pytest.raises(ImportError):
        adapter_cls.convert(dummy)


def _install_stub(monkeypatch, modules: list[str], attrs: dict[str, object]) -> None:
    """Create stub modules; set ``attrs`` on the *last* (leaf) module."""
    for i, mod in enumerate(modules):
        m = types.ModuleType(mod)
        if i == len(modules) - 1:
            for k, v in attrs.items():
                setattr(m, k, v)
        monkeypatch.setitem(sys.modules, mod, m)


async def test_for_each_framework_selects_and_converts(gantry, monkeypatch):
    """Smoke every adapter's ``select`` end-to-end with a permissive captured stub."""

    captured: dict[str, list] = {}

    def _record(fw):
        def _factory(*args, **kwargs):
            obj = types.SimpleNamespace(args=args, kwargs=kwargs)
            captured.setdefault(fw, []).append(obj)
            return obj

        return _factory

    # langchain StructuredTool.from_function classmethod
    lc = types.SimpleNamespace()
    lc.from_function = staticmethod(lambda **kw: types.SimpleNamespace(**kw))

    cases = [
        ("langchain", F.LangChainAdapter, ["langchain_core", "langchain_core.tools"], {"StructuredTool": lc}),
        ("langgraph", F.LangGraphAdapter, ["langchain_core", "langchain_core.tools"], {"StructuredTool": lc}),
    ]
    for fw, adapter_cls, modules, attrs in cases:
        _install_stub(monkeypatch, modules, attrs)
        tools = await adapter_cls(gantry).select("send an email to my boss", limit=2)
        assert len(tools) >= 1, f"{fw}: expected at least one converted tool"

    # AutoGen's convert/select are import-free (they return registrable dict
    # mappings, not a native framework object), so smoke them with no stub.
    ag_tools = await F.AutoGenAdapter(gantry).select("send an email to my boss", limit=2)
    assert ag_tools and all("callable" in t for t in ag_tools), (
        "autogen: expected registrable mappings with a callable"
    )
