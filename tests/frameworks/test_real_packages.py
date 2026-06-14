"""Real-package smoke tests for the native framework adapters.

The per-framework unit tests stub the third-party framework so they run
everywhere. Those stubs *cannot* catch real-API drift — e.g. CrewAI's
``BaseTool`` being a Pydantic v2 model, or smolagents validating ``forward``'s
signature against the declared inputs. This module builds **and invokes** each
adapter against the *actual* installed package, skipping any framework that
isn't present (``pytest.importorskip``). In CI (where the ``agent-frameworks``
extra installs them) this is the guard against silent adapter breakage.
"""

from __future__ import annotations

import inspect
import json
import os

import pytest

# Keep frameworks from making outbound calls during the smoke test (CrewAI ships
# opt-out telemetry that otherwise blocks for ~30s when the network is firewalled).
os.environ.setdefault("CREWAI_DISABLE_TELEMETRY", "true")
os.environ.setdefault("OTEL_SDK_DISABLED", "true")

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

    await g.sync()
    return g


# Each entry: (test id, importable module, adapter coroutine, invoke callable).
# ``invoke`` receives the built native tool and must run it with to="boss@x.com",
# returning a result that stringifies to contain "sent:boss@x.com".
def _invoke_langchain(tool):
    return tool.invoke({"to": "boss@x.com"})


def _invoke_llamaindex(tool):
    return tool.call(to="boss@x.com").raw_output


def _invoke_crewai(tool):
    return tool.run(to="boss@x.com")


def _invoke_smolagents(tool):
    return tool.forward(to="boss@x.com")


def _invoke_haystack(tool):
    return tool.function(to="boss@x.com")


def _invoke_agno(tool):
    return tool.entrypoint(to="boss@x.com")


async def _invoke_openai_agents(tool):
    return await tool.on_invoke_tool(None, json.dumps({"to": "boss@x.com"}))


async def _invoke_google_adk(tool):
    return await tool.run_async(args={"to": "boss@x.com"}, tool_context=None)


async def _invoke_semantic_kernel(tool):
    from semantic_kernel import Kernel

    return await tool.invoke(Kernel(), to="boss@x.com")


REAL_ADAPTERS = [
    ("langchain", "langchain_core", F.for_langchain, _invoke_langchain),
    ("langgraph", "langgraph", F.for_langgraph, _invoke_langchain),
    ("llamaindex", "llama_index.core", F.for_llamaindex, _invoke_llamaindex),
    ("crewai", "crewai", F.for_crewai, _invoke_crewai),
    ("smolagents", "smolagents", F.for_smolagents, _invoke_smolagents),
    ("haystack", "haystack", F.for_haystack, _invoke_haystack),
    ("agno", "agno", F.for_agno, _invoke_agno),
    ("openai_agents", "agents", F.for_openai_agents, _invoke_openai_agents),
    ("google_adk", "google.adk", F.for_google_adk, _invoke_google_adk),
    ("semantic_kernel", "semantic_kernel", F.for_semantic_kernel, _invoke_semantic_kernel),
]


@pytest.mark.parametrize(
    "name,module,adapter,invoke", REAL_ADAPTERS, ids=[a[0] for a in REAL_ADAPTERS]
)
async def test_real_adapter_builds_and_invokes(name, module, adapter, invoke, gantry):
    pytest.importorskip(module, reason=f"{name} not installed")

    tools = await adapter(gantry, "send an email to my boss", limit=1)
    assert tools, f"{name}: adapter returned no tools"

    result = invoke(tools[0])
    if inspect.isawaitable(result):
        result = await result
    assert "sent:boss@x.com" in str(result), f"{name}: invocation did not route through gantry"


async def test_pydantic_ai_builds(gantry):
    """Pydantic AI tool build (its run path needs an agent context, so just build)."""
    pytest.importorskip("pydantic_ai", reason="pydantic-ai not installed")
    tools = await F.for_pydantic_ai(gantry, "send an email", limit=1)
    assert tools
    assert tools[0].name == "send_email"
