"""The facade finds every adapter while keeping third-party imports lazy."""

from __future__ import annotations

import importlib
import subprocess
import sys

import pytest

from dotfetch import DotFetch, supported_frameworks
from dotfetch.integrations.frameworks.base import DEFAULT_TOOL_LIMIT, QueryBoundsError
from tests.test_framework_namespaces import FRAMEWORK_ADAPTERS


@pytest.mark.parametrize("module,case", FRAMEWORK_ADAPTERS.items())
def test_framework_factory_finds_every_public_adapter(module, case, fetcher):
    name = module.rsplit(".", 1)[1]
    adapter = fetcher.framework(name, default_limit=2)
    assert isinstance(adapter, getattr(importlib.import_module(module), case[0]))
    assert adapter._fetcher is fetcher
    default = adapter._default_top_k if name == "agent_framework" else adapter._default_limit
    assert default == 2


def test_framework_discovery_matches_public_namespaces():
    assert supported_frameworks() == tuple(sorted(m.rsplit(".", 1)[1] for m in FRAMEWORK_ADAPTERS))


@pytest.mark.parametrize(
    "alias,canonical",
    [
        (" Pydantic-AI ", "pydantic_ai"),
        ("openai-agents", "openai_agents"),
        ("google-adk", "google_adk"),
        ("agent-framework", "agent_framework"),
        ("crew_ai", "crewai"),
        ("llama-index", "llamaindex"),
        ("LangChain", "langchain"),
    ],
)
def test_framework_names_accept_common_spellings(alias, canonical, fetcher):
    assert type(fetcher.framework(alias)) is type(fetcher.framework(canonical))


def test_unknown_framework_has_a_suggestion_and_supported_names(fetcher):
    with pytest.raises(ValueError, match="Did you mean 'langchain'.*Supported frameworks:"):
        fetcher.framework("langchan")


@pytest.mark.parametrize("limit", [0, -1, 51])
def test_bad_default_budget_fails_before_loading_framework(fetcher, limit):
    with pytest.raises(QueryBoundsError, match="DotFetch.framework.*limit"):
        fetcher.framework("pydantic_ai", default_limit=limit)


def test_factory_default_matches_shared_adapter_budget(fetcher):
    assert fetcher.framework("langchain")._default_limit == DEFAULT_TOOL_LIMIT
    assert fetcher.framework("agent_framework")._default_top_k == DEFAULT_TOOL_LIMIT


def test_discovery_and_factory_do_not_import_optional_sdks():
    # A fresh process makes this meaningful even when every SDK is installed.
    modules = sorted({m for _, imports in FRAMEWORK_ADAPTERS.values() for m in imports})
    code = f"""
import sys
from dotfetch import DotFetch, supported_frameworks
g = DotFetch()
for name in supported_frameworks():
    g.framework(name)
loaded = set({modules!r}) & sys.modules.keys()
assert not loaded, loaded
"""
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr


async def test_factory_selects_and_executes_through_real_langchain():
    pytest.importorskip("langchain_core")
    async with DotFetch() as fetcher:

        @fetcher.register
        def greet(name: str) -> str:
            """Greet a person by their given name."""
            return f"Hello {name}"

        tools = await fetcher.framework("langchain", default_limit=1).select("greet someone")
        assert len(tools) == 1
        assert await tools[0].ainvoke({"name": "Ada"}) == "Hello Ada"


async def test_missing_sdk_still_has_install_instructions(fetcher, monkeypatch):
    monkeypatch.setitem(sys.modules, "langchain_core.tools", None)

    @fetcher.register
    def greet(name: str) -> str:
        """Greet a person by their given name."""
        return name

    with pytest.raises(ImportError, match=r"dotfetch\[langchain\]"):
        await fetcher.framework("langchain").select("greet someone")
