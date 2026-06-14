"""The per-framework public namespaces: ``from agent_gantry.<framework> import …``.

Each framework gets a top-level module re-exporting its static adapter and deep
live provider. Importing ``agent_gantry`` must NOT pull any of them in (the
dependency-free guarantee), and the static names must import even when the
third-party framework is absent.
"""

from __future__ import annotations

import importlib
import sys

import pytest

# (namespace module, static names that must import without the framework, all names)
NAMESPACES = {
    "agent_gantry.langchain": (["for_langchain", "spec_to_langchain"], []),
    "agent_gantry.langgraph": (
        ["for_langgraph", "spec_to_langgraph"],
        ["create_gantry_react_agent", "select_tools_for_state"],
    ),
    "agent_gantry.llamaindex": (
        ["for_llamaindex", "spec_to_llamaindex"],
        ["gantry_tool_retriever", "gantry_function_agent"],
    ),
    "agent_gantry.crewai": (
        ["for_crewai", "spec_to_crewai"],
        ["gantry_crew_tools", "GantryLiveCrewAgent"],
    ),
    "agent_gantry.pydantic_ai": (["for_pydantic_ai", "spec_to_pydantic_ai"], ["gantry_toolset"]),
    "agent_gantry.openai_agents": (
        ["for_openai_agents", "spec_to_openai_agents"],
        ["gantry_run_hooks", "run_with_gantry", "GantryAgentSession"],
    ),
    "agent_gantry.smolagents": (["for_smolagents", "spec_to_smolagents"], ["GantryLiveSmolAgent"]),
    "agent_gantry.haystack": (
        ["for_haystack", "spec_to_haystack"],
        ["gantry_haystack_tools", "GantryLiveHaystackToolInvoker"],
    ),
    "agent_gantry.agno": (["for_agno", "spec_to_agno"], ["GantryLiveAgnoAgent"]),
    "agent_gantry.autogen": (
        ["for_autogen", "spec_to_autogen", "register_with_autogen"],
        ["gantry_workbench"],
    ),
    "agent_gantry.semantic_kernel": (
        ["for_semantic_kernel", "spec_to_semantic_kernel", "gantry_plugin"],
        ["GantryFunctionProvider", "refresh_kernel_tools"],
    ),
    "agent_gantry.google_adk": (
        ["for_google_adk", "spec_to_google_adk"],
        ["gantry_before_model_callback", "gantry_adk_agent"],
    ),
    "agent_gantry.agent_framework": (
        ["GantryContextProvider", "GantryToolBridge", "GantryApprovalMiddleware"],
        [],
    ),
}


@pytest.mark.parametrize("module,static,live", [(m, s, l) for m, (s, l) in NAMESPACES.items()])
def test_namespace_exposes_all_names(module, static, live):
    mod = importlib.import_module(module)
    # Static names are eagerly bound and import-safe.
    for name in static:
        assert hasattr(mod, name), f"{module} missing {name}"
    # Live names are advertised in __all__ (resolved lazily on access).
    for name in live:
        assert name in mod.__all__, f"{module} __all__ missing {name}"


def test_importing_agent_gantry_does_not_load_framework_namespaces():
    """`import agent_gantry` must not eagerly import any framework namespace.

    Run in a fresh subprocess so the assertion is not contaminated by other
    tests in this session that may have imported the shims explicitly.
    """
    import subprocess

    code = (
        "import sys, agent_gantry; "
        "loaded = [m for m in sys.modules "
        "if m.startswith('agent_gantry.') and m.count('.') == 1 "
        "and m.rsplit('.', 1)[1] in "
        "{'langchain','langgraph','llamaindex','crewai','pydantic_ai',"
        "'openai_agents','smolagents','haystack','agno','autogen',"
        "'semantic_kernel','google_adk','agent_framework'}]; "
        "assert not loaded, loaded; print('clean')"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "clean" in result.stdout


def test_static_namespace_imports_without_framework(monkeypatch):
    """`from agent_gantry.langchain import for_langchain` works w/o langchain."""
    monkeypatch.setitem(sys.modules, "langchain_core", None)
    monkeypatch.setitem(sys.modules, "langchain_core.tools", None)
    sys.modules.pop("agent_gantry.langchain", None)
    mod = importlib.import_module("agent_gantry.langchain")
    assert callable(mod.for_langchain)
