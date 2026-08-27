"""Real-package smoke tests for the native framework adapters.

The per-framework unit tests stub the third-party framework so they run
everywhere. Those stubs *cannot* catch real-API drift — e.g. CrewAI's
``BaseTool`` being a Pydantic v2 model, or a framework validating a wrapper's
signature against the declared inputs. This module builds **and invokes** each
adapter against the *actual* installed package, skipping any framework that
isn't present (``pytest.importorskip``). In CI (where the ``agent-frameworks``
extra installs them) this is the guard against silent adapter breakage.

``test_agent_framework_builds_and_invokes`` covers the Microsoft Agent
Framework separately, below the ``REAL_ADAPTERS`` table: ``AgentFrameworkAdapter``
has no ``select``/``convert`` staticmethods (unlike the ``BaseFrameworkAdapter``
family that populates ``REAL_ADAPTERS``) — its select-equivalent is
``tool_bridge().get_tools(...)``, so it doesn't fit the shared parametrization.
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
from agent_gantry.integrations import frameworks as F  # noqa: N812


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


def _invoke_haystack(tool):
    return tool.function(to="boss@x.com")


def _invoke_agno(tool):
    return tool.entrypoint(to="boss@x.com")


async def _invoke_openai_agents(tool):
    return await tool.on_invoke_tool(None, json.dumps({"to": "boss@x.com"}))


async def _invoke_google_adk(tool):
    return await tool.run_async(args={"to": "boss@x.com"}, tool_context=None)


def _invoke_dspy(tool):
    # dspy.Tool.__call__ — sync, matching ReAct.forward's invoke path; the
    # wrapped function is a sync bridge over ``ToolSpec.invoke``.
    return tool(to="boss@x.com")


async def _invoke_strands(tool):
    # ``DecoratedFunctionTool`` keeps the wrapped function callable directly;
    # calling it returns the coroutine from the async wrapper, which routes
    # through ``gantry.execute`` like every other adapter's invoke path.
    return await tool(to="boss@x.com")


def _invoke_pydantic_ai(tool):
    # ``Tool``/``Tool.from_schema`` both stash the wrapped callable as the
    # public ``function`` attribute (see pydantic_ai.tools.Tool.__init__).
    # Calling it directly exercises the exact function a live agent run would
    # invoke, without needing a full ``RunContext`` — it's the same async
    # `ToolSpec.callable_for_signature()` wrapper every other adapter uses,
    # which already routes through ``gantry.execute``.
    return tool.function(to="boss@x.com")


# (id, importable module, Adapter class, invoke).
REAL_ADAPTERS = [
    ("langchain", "langchain_core", F.LangChainAdapter, _invoke_langchain),
    ("langgraph", "langgraph", F.LangGraphAdapter, _invoke_langchain),
    ("llamaindex", "llama_index.core", F.LlamaIndexAdapter, _invoke_llamaindex),
    ("crewai", "crewai", F.CrewAIAdapter, _invoke_crewai),
    ("pydantic_ai", "pydantic_ai", F.PydanticAIAdapter, _invoke_pydantic_ai),
    ("haystack", "haystack", F.HaystackAdapter, _invoke_haystack),
    ("agno", "agno", F.AgnoAdapter, _invoke_agno),
    ("openai_agents", "agents", F.OpenAIAgentsAdapter, _invoke_openai_agents),
    ("google_adk", "google.adk", F.GoogleADKAdapter, _invoke_google_adk),
    ("strands", "strands", F.StrandsAdapter, _invoke_strands),
    ("dspy", "dspy", F.DSPyAdapter, _invoke_dspy),
]


@pytest.mark.parametrize(
    "name,module,adapter_cls,invoke", REAL_ADAPTERS, ids=[a[0] for a in REAL_ADAPTERS]
)
async def test_real_adapter_builds_and_invokes(name, module, adapter_cls, invoke, gantry):
    pytest.importorskip(module, reason=f"{name} not installed")

    tools = await adapter_cls(gantry).select("send an email to my boss", limit=1)
    assert tools, f"{name}: adapter returned no tools"

    result = invoke(tools[0])
    if inspect.isawaitable(result):
        result = await result
    assert "sent:boss@x.com" in str(result), f"{name}: invocation did not route through gantry"


async def test_agent_framework_builds_and_invokes(gantry):
    """AgentFrameworkAdapter's ``tool_bridge().get_tools(...)`` is its
    select+convert equivalent — it has no ``select``/``convert`` staticmethods
    (see ``agent_gantry.agent_framework.AgentFrameworkAdapter``), so it isn't in
    ``REAL_ADAPTERS`` above. With ``agent-framework`` installed, ``get_tools()``
    upgrades the bare callable to a real ``agent_framework.FunctionTool`` (see
    ``GantryToolBridge._maybe_wrap_as_function_tool``); invoke it through its
    native ``.invoke()`` and confirm the call still routes through gantry.
    """
    pytest.importorskip("agent_framework", reason="agent-framework not installed")
    from agent_gantry.agent_framework import AgentFrameworkAdapter

    bridge = AgentFrameworkAdapter(gantry).tool_bridge()
    tools = await bridge.get_tools("send an email to my boss", limit=1)
    assert tools, "agent_framework: adapter returned no tools"

    tool = tools[0]
    assert type(tool).__name__ == "FunctionTool", (
        f"agent_framework: expected a real FunctionTool with agent-framework "
        f"installed, got {type(tool)!r}"
    )
    # `skip_parsing=True` returns the wrapped function's raw string result
    # instead of wrapping it in `list[Content]` — see `FunctionTool.invoke`.
    result = await tool.invoke(to="boss@x.com", skip_parsing=True)
    assert "sent:boss@x.com" in str(result), (
        "agent_framework: invocation did not route through gantry"
    )
