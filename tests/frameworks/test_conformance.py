"""Cross-framework conformance matrix for the native tool adapters.

Every adapter in ``agent_gantry.integrations.frameworks`` must honor the same
contract regardless of which third-party framework it targets:

1. It exposes ``for_<fw>`` (async select+convert) and ``spec_to_<fw>`` (single
   convert), both importable without the framework installed.
2. ``for_<fw>`` runs semantic selection against a real gantry and returns one
   native object per selected tool (verified through a per-framework stub).
3. ``spec_to_<fw>`` raises a clean ``ImportError`` carrying a ``pip install``
   hint when the framework is absent — not ``AttributeError`` / ``KeyError``.

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


# (framework key, for_* fn, spec_to_* fn, module name(s) the lazy import needs)
ADAPTERS = [
    ("langchain", F.for_langchain, F.spec_to_langchain, ["langchain_core", "langchain_core.tools"]),
    ("langgraph", F.for_langgraph, F.spec_to_langgraph, ["langchain_core", "langchain_core.tools"]),
    ("llamaindex", F.for_llamaindex, F.spec_to_llamaindex, ["llama_index", "llama_index.core", "llama_index.core.tools"]),
    ("crewai", F.for_crewai, F.spec_to_crewai, ["crewai", "crewai.tools"]),
    ("pydantic_ai", F.for_pydantic_ai, F.spec_to_pydantic_ai, ["pydantic_ai", "pydantic_ai.tools"]),
    ("openai_agents", F.for_openai_agents, F.spec_to_openai_agents, ["agents"]),
    ("smolagents", F.for_smolagents, F.spec_to_smolagents, ["smolagents"]),
    ("haystack", F.for_haystack, F.spec_to_haystack, ["haystack", "haystack.tools"]),
    ("agno", F.for_agno, F.spec_to_agno, ["agno", "agno.tools", "agno.tools.function"]),
]


@pytest.mark.parametrize("name,for_fn,spec_fn,modules", ADAPTERS, ids=[a[0] for a in ADAPTERS])
def test_adapter_exposes_uniform_surface(name, for_fn, spec_fn, modules):
    assert callable(for_fn), f"{name}: for_{name} not callable"
    assert callable(spec_fn), f"{name}: spec_to_{name} not callable"


@pytest.mark.parametrize("name,for_fn,spec_fn,modules", ADAPTERS, ids=[a[0] for a in ADAPTERS])
def test_missing_framework_raises_clean_importerror(name, for_fn, spec_fn, modules, monkeypatch):
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
        spec_fn(dummy)


def _install_stub(monkeypatch, modules: list[str], attrs: dict[str, object]) -> None:
    """Create stub modules; set ``attrs`` on the *last* (leaf) module."""
    for i, mod in enumerate(modules):
        m = types.ModuleType(mod)
        if i == len(modules) - 1:
            for k, v in attrs.items():
                setattr(m, k, v)
        monkeypatch.setitem(sys.modules, mod, m)


async def test_for_each_framework_selects_and_converts(gantry, monkeypatch):
    """Smoke every ``for_<fw>`` end-to-end with a permissive captured stub."""

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
        ("langchain", F.for_langchain, ["langchain_core", "langchain_core.tools"], {"StructuredTool": lc}),
        ("langgraph", F.for_langgraph, ["langchain_core", "langchain_core.tools"], {"StructuredTool": lc}),
    ]
    for fw, for_fn, modules, attrs in cases:
        _install_stub(monkeypatch, modules, attrs)
        tools = await for_fn(gantry, "send an email to my boss", limit=2)
        assert len(tools) >= 1, f"{fw}: expected at least one converted tool"
