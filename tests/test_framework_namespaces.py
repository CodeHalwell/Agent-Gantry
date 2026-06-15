"""The per-framework public namespaces: ``from agent_gantry.<framework> import …``.

Each integration gets a top-level ``agent_gantry.<module>`` shim re-exporting a
single ``<Framework>Adapter`` class (LLM SDK shims do the same). Importing
``agent_gantry`` must NOT pull any of them in (the dependency-free guarantee),
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
    "agent_gantry.langchain": ("LangChainAdapter", ["langchain_core", "langchain_core.tools"]),
    "agent_gantry.langgraph": ("LangGraphAdapter", ["langchain_core", "langchain_core.tools", "langgraph"]),
    "agent_gantry.llamaindex": (
        "LlamaIndexAdapter",
        ["llama_index", "llama_index.core", "llama_index.core.tools"],
    ),
    "agent_gantry.crewai": ("CrewAIAdapter", ["crewai", "crewai.tools"]),
    "agent_gantry.pydantic_ai": ("PydanticAIAdapter", ["pydantic_ai", "pydantic_ai.tools"]),
    "agent_gantry.openai_agents": ("OpenAIAgentsAdapter", ["agents"]),
    "agent_gantry.smolagents": ("SmolagentsAdapter", ["smolagents"]),
    "agent_gantry.haystack": ("HaystackAdapter", ["haystack", "haystack.tools"]),
    "agent_gantry.agno": ("AgnoAdapter", ["agno", "agno.tools", "agno.tools.function"]),
    "agent_gantry.autogen": ("AutoGenAdapter", ["autogen", "autogen_core"]),
    "agent_gantry.semantic_kernel": (
        "SemanticKernelAdapter",
        ["semantic_kernel", "semantic_kernel.functions"],
    ),
    "agent_gantry.google_adk": ("GoogleADKAdapter", ["google.adk", "google.adk.tools"]),
    # Microsoft Agent Framework: unified entry point.
    "agent_gantry.agent_framework": ("AgentFrameworkAdapter", ["agent_framework"]),
}

# LLM SDK shims expose one ``<Provider>Adapter`` each; same lazy-import contract.
LLM_ADAPTERS = {
    "agent_gantry.openai": ("OpenAIAdapter", ["openai"]),
    "agent_gantry.anthropic": ("AnthropicAdapter", ["anthropic"]),
    "agent_gantry.gemini": ("GeminiAdapter", ["google.generativeai", "google.genai"]),
    "agent_gantry.groq": ("GroqAdapter", ["groq"]),
    "agent_gantry.vertexai": ("VertexAIAdapter", ["vertexai"]),
    "agent_gantry.mistral": ("MistralAdapter", ["mistralai"]),
}

ALL_ADAPTERS = {**FRAMEWORK_ADAPTERS, **LLM_ADAPTERS}


@pytest.mark.parametrize(
    "module,cls_name", [(m, c) for m, (c, _mods) in ALL_ADAPTERS.items()]
)
def test_namespace_exposes_adapter_class(module, cls_name):
    """``from agent_gantry.<fw> import <Framework>Adapter`` resolves to a class."""
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
    # Force a fresh import of the shim with the framework masked out.
    sys.modules.pop(module, None)
    mod = importlib.import_module(module)
    adapter = getattr(mod, cls_name)
    assert isinstance(adapter, type)


def test_importing_agent_gantry_does_not_load_framework_namespaces():
    """`import agent_gantry` must not eagerly import any framework/LLM namespace.

    Run in a fresh subprocess so the assertion is not contaminated by other
    tests in this session that may have imported the shims explicitly.
    """
    import subprocess

    shim_leaves = sorted(m.rsplit(".", 1)[1] for m in ALL_ADAPTERS)
    code = (
        "import sys, agent_gantry; "
        f"shims = set({shim_leaves!r}); "
        "loaded = [m for m in sys.modules "
        "if m.startswith('agent_gantry.') and m.count('.') == 1 "
        "and m.rsplit('.', 1)[1] in shims]; "
        "assert not loaded, loaded; print('clean')"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "clean" in result.stdout


def test_importing_agent_gantry_does_not_load_third_party_frameworks():
    """`import agent_gantry` must not eagerly import any third-party framework/SDK.

    The Microsoft Agent Framework primitives (``GantryContextProvider``,
    ``RetrievalDecision``) are imported at the top level of ``agent_gantry``, so
    this guards that they — and every adapter — keep their third-party imports
    lazy. Runs in a fresh subprocess (frameworks may be installed in CI).
    """
    import subprocess

    third_party = [
        "agent_framework", "langchain_core", "langgraph", "llama_index",
        "crewai", "pydantic_ai", "agents", "smolagents", "haystack", "agno",
        "semantic_kernel", "google.adk",
    ]
    code = (
        "import sys, agent_gantry; "
        f"third = {third_party!r}; "
        "loaded = [m for m in third if m in sys.modules]; "
        "assert not loaded, loaded; print('clean')"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "clean" in result.stdout
