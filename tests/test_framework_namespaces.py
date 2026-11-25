"""The per-framework public namespaces: ``from dotfetch.<framework> import …``.

Each integration gets a top-level ``dotfetch.<module>`` shim re-exporting a
single ``<Framework>Adapter`` class (LLM SDK shims do the same). Importing
``dotfetch`` must NOT pull any of them in (the dependency-free guarantee),
and the adapter class must import even when the third-party framework is absent
(the third-party import is lazy, deferred to ``select``/``convert``).
"""

from __future__ import annotations

import importlib
import sys

import pytest

# (namespace module, adapter class name, third-party modules to mask). Each shim
# exports exactly one ``<Framework>Adapter`` that must import lazily — i.e. with
# the third-party framework forced to "not installed".
FRAMEWORK_ADAPTERS = {
    "dotfetch.langchain": ("LangChainAdapter", ["langchain_core", "langchain_core.tools"]),
    "dotfetch.langgraph": ("LangGraphAdapter", ["langchain_core", "langchain_core.tools", "langgraph"]),
    "dotfetch.llamaindex": (
        "LlamaIndexAdapter",
        ["llama_index", "llama_index.core", "llama_index.core.tools"],
    ),
    "dotfetch.crewai": ("CrewAIAdapter", ["crewai", "crewai.tools"]),
    "dotfetch.pydantic_ai": ("PydanticAIAdapter", ["pydantic_ai", "pydantic_ai.tools"]),
    "dotfetch.openai_agents": ("OpenAIAgentsAdapter", ["agents"]),
    "dotfetch.haystack": ("HaystackAdapter", ["haystack", "haystack.tools"]),
    "dotfetch.agno": ("AgnoAdapter", ["agno", "agno.tools", "agno.tools.function"]),
    "dotfetch.google_adk": ("GoogleADKAdapter", ["google.adk", "google.adk.tools"]),
    "dotfetch.strands": ("StrandsAdapter", ["strands", "strands.hooks"]),
    "dotfetch.dspy": (
        "DSPyAdapter",
        ["dspy", "dspy.adapters", "dspy.adapters.types", "dspy.adapters.types.tool"],
    ),
    # Microsoft Agent Framework: unified entry point.
    "dotfetch.agent_framework": ("AgentFrameworkAdapter", ["agent_framework"]),
}

# LLM SDK shims expose one ``<Provider>Adapter`` each; same lazy-import contract.
LLM_ADAPTERS = {
    "dotfetch.openai": ("OpenAIAdapter", ["openai"]),
    "dotfetch.anthropic": ("AnthropicAdapter", ["anthropic"]),
    "dotfetch.gemini": ("GeminiAdapter", ["google.generativeai", "google.genai"]),
    "dotfetch.groq": ("GroqAdapter", ["groq"]),
    "dotfetch.vertexai": ("VertexAIAdapter", ["vertexai"]),
    "dotfetch.mistral": ("MistralAdapter", ["mistralai"]),
}

ALL_ADAPTERS = {**FRAMEWORK_ADAPTERS, **LLM_ADAPTERS}


@pytest.mark.parametrize(
    "module,cls_name", [(m, c) for m, (c, _mods) in ALL_ADAPTERS.items()]
)
def test_namespace_exposes_adapter_class(module, cls_name):
    """``from dotfetch.<fw> import <Framework>Adapter`` resolves to a class."""
    mod = importlib.import_module(module)
    adapter = getattr(mod, cls_name, None)
    assert adapter is not None, f"{module} missing {cls_name}"
    assert isinstance(adapter, type), f"{module}.{cls_name} is not a class"
    # The single adapter class is advertised in __all__.
    assert cls_name in mod.__all__, f"{module} __all__ missing {cls_name}"


@pytest.mark.parametrize(
    "module,cls_name,mods", [(m, c, mods) for m, (c, mods) in ALL_ADAPTERS.items()]
)
def test_adapter_imports_without_framework(module, cls_name, mods, monkeypatch):
    """The adapter class imports even when its third-party framework is absent.

    The third-party import is lazy (deferred to ``select``/``convert``), so the
    namespace shim must bind the class without requiring the package installed.
    """
    for mod_name in mods:
        monkeypatch.setitem(sys.modules, mod_name, None)
    # Force a fresh import of the shim with the framework masked out. Use
    # monkeypatch.delitem so the eviction is rolled back after the test (a bare
    # sys.modules.pop would leak the masked module into later tests).
    monkeypatch.delitem(sys.modules, module, raising=False)
    mod = importlib.import_module(module)
    adapter = getattr(mod, cls_name)
    assert isinstance(adapter, type)


def test_importing_dotfetch_does_not_load_framework_namespaces():
    """`import dotfetch` must not eagerly import any framework/LLM namespace.

    Run in a fresh subprocess so the assertion is not contaminated by other
    tests in this session that may have imported the shims explicitly.
    """
    import subprocess

    shim_leaves = sorted(m.rsplit(".", 1)[1] for m in ALL_ADAPTERS)
    code = (
        "import sys, dotfetch; "
        f"shims = set({shim_leaves!r}); "
        "loaded = [m for m in sys.modules "
        "if m.startswith('dotfetch.') and m.count('.') == 1 "
        "and m.rsplit('.', 1)[1] in shims]; "
        "assert not loaded, loaded; print('clean')"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "clean" in result.stdout


def test_importing_dotfetch_does_not_load_third_party_frameworks():
    """`import dotfetch` must not eagerly import any third-party framework/SDK.

    The Microsoft Agent Framework primitives (``DotFetchContextProvider``,
    ``RetrievalDecision``) are imported at the top level of ``dotfetch``, so
    this guards that they — and every adapter — keep their third-party imports
    lazy. Runs in a fresh subprocess (frameworks may be installed in CI).
    """
    import subprocess

    third_party = [
        # agent frameworks
        "agent_framework", "langchain_core", "langgraph", "llama_index",
        "crewai", "pydantic_ai", "agents", "haystack", "agno",
        "google.adk", "strands", "dspy",
        # LLM provider SDKs — the LLM adapters never import them (they only emit
        # schemas via fetcher.retrieve_tools), so none should load on import.
        "openai", "anthropic", "groq", "mistralai", "google.generativeai",
    ]
    code = (
        "import sys, dotfetch; "
        f"third = {third_party!r}; "
        "loaded = [m for m in third if m in sys.modules]; "
        "assert not loaded, loaded; print('clean')"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "clean" in result.stdout
