"""Real-package smoke tests for the native framework adapters.

The per-framework unit tests stub the third-party framework so they run
everywhere. Those stubs *cannot* catch real-API drift — e.g. CrewAI's
``BaseTool`` being a Pydantic v2 model, or a framework validating a wrapper's
signature against the declared inputs. This module builds **and invokes** each
adapter against the *actual* installed package, skipping any framework that
isn't present (``pytest.importorskip``). In CI (where the ``agent-frameworks``
extra installs them) this is the guard against silent adapter breakage.

All twelve frameworks share the select/convert matrix. Additional Agent
Framework tests retain coverage of its lower-level bridge API.
"""

from __future__ import annotations

import importlib
import inspect
import json
import os
from typing import Literal

import pytest

# Keep frameworks from making outbound calls during the smoke test (CrewAI ships
# opt-out telemetry that otherwise blocks for ~30s when the network is firewalled).
os.environ.setdefault("CREWAI_DISABLE_TELEMETRY", "true")
os.environ.setdefault("OTEL_SDK_DISABLED", "true")

from agent_gantry import AgentGantry
from agent_gantry.adapters.embedders.simple import SimpleEmbedder
from agent_gantry.agent_framework import AgentFrameworkAdapter
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
def _invoke_langchain(tool, arguments):
    return tool.invoke(arguments)


def _invoke_llamaindex(tool, arguments):
    return tool.call(**arguments).raw_output


def _invoke_crewai(tool, arguments):
    return tool.run(**arguments)


def _invoke_haystack(tool, arguments):
    return tool.function(**arguments)


def _invoke_agno(tool, arguments):
    return tool.entrypoint(**arguments)


async def _invoke_openai_agents(tool, arguments):
    return await tool.on_invoke_tool(None, json.dumps(arguments))


async def _invoke_google_adk(tool, arguments):
    return await tool.run_async(args=arguments, tool_context=None)


def _invoke_dspy(tool, arguments):
    # dspy.Tool.__call__ — sync, matching ReAct.forward's invoke path; the
    # wrapped function is a sync bridge over ``ToolSpec.invoke``.
    return tool(**arguments)


async def _invoke_strands(tool, arguments):
    # ``DecoratedFunctionTool`` keeps the wrapped function callable directly;
    # calling it returns the coroutine from the async wrapper, which routes
    # through ``gantry.execute`` like every other adapter's invoke path.
    return await tool(**arguments)


def _invoke_pydantic_ai(tool, arguments):
    # ``Tool``/``Tool.from_schema`` both stash the wrapped callable as the
    # public ``function`` attribute (see pydantic_ai.tools.Tool.__init__).
    # Calling it directly exercises the exact function a live agent run would
    # invoke, without needing a full ``RunContext`` — it's the same async
    # `ToolSpec.callable_for_signature()` wrapper every other adapter uses,
    # which already routes through ``gantry.execute``.
    return tool.function(**arguments)


async def _invoke_agent_framework(tool, arguments):
    return await tool.invoke(**arguments, skip_parsing=True)


# (id, importable module, Adapter class, invoke).
REAL_ADAPTERS = [
    ("agent_framework", "agent_framework", AgentFrameworkAdapter, _invoke_agent_framework),
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


@pytest.fixture(scope="module", autouse=True)
def require_installed_frameworks():
    """CI must exercise the SDKs it installs instead of silently skipping them.

    Local minimal installs keep the usual importorskip behavior. CI declares
    its expected SDKs with GANTRY_REQUIRED_FRAMEWORKS (canonical comma-separated
    names); missing packages and broken transitive imports then fail the job.
    """
    required = {
        name.strip()
        for name in os.environ.get("GANTRY_REQUIRED_FRAMEWORKS", "").split(",")
        if name.strip()
    }
    modules = {name: module for name, module, _, _ in REAL_ADAPTERS}
    unknown = required - modules.keys()
    if unknown:
        pytest.fail(f"Unknown GANTRY_REQUIRED_FRAMEWORKS entries: {sorted(unknown)}", pytrace=False)
    for name in sorted(required):
        try:
            importlib.import_module(modules[name])
        except ImportError as exc:
            pytest.fail(f"Required framework {name!r} could not import: {exc}", pytrace=False)


@pytest.mark.parametrize(
    "name,module,adapter_cls,invoke", REAL_ADAPTERS, ids=[a[0] for a in REAL_ADAPTERS]
)
async def test_real_adapter_builds_and_invokes(name, module, adapter_cls, invoke, gantry):
    pytest.importorskip(module, reason=f"{name} not installed")

    tools = await adapter_cls(gantry).select("send an email to my boss", limit=1)
    assert tools, f"{name}: adapter returned no tools"

    result = invoke(tools[0], {"to": "boss@x.com"})
    if inspect.isawaitable(result):
        result = await result
    assert "sent:boss@x.com" in str(result), f"{name}: invocation did not route through gantry"


@pytest.mark.parametrize(
    "name,module,adapter_cls,invoke", REAL_ADAPTERS, ids=[a[0] for a in REAL_ADAPTERS]
)
async def test_real_adapter_preserves_schema_and_namespace(
    name, module, adapter_cls, invoke, gantry
):
    """Exercise arrays, enums, optional defaults, explicit null, and namespaces.

    The fixture already has default.send_email; this namespaced tool must be
    invoked instead, even though both advertise the same native tool name.
    """
    pytest.importorskip(module, reason=f"{name} not installed")

    @gantry.register(name="send_email", namespace="reports")
    def send_report(
        to: str,
        amounts: list[int],
        mode: Literal["sum", "max"] = "sum",
        note: str | None = "ready",
    ) -> str:
        """Send a report with aggregated amounts to a recipient."""
        amount = sum(amounts) if mode == "sum" else max(amounts)
        return f"report:{to}:{amount}:{note}"

    tools = await gantry.framework(name).select("send a report", namespaces=["reports"], limit=1)
    assert len(tools) == 1
    result = invoke(tools[0], {"to": "boss@x.com", "amounts": [1, 2, 3], "note": None})
    if inspect.isawaitable(result):
        result = await result
    assert "report:boss@x.com:6:None" in str(result), f"{name}: schema or namespace changed"


@pytest.mark.parametrize(
    "name,module,adapter_cls,invoke", REAL_ADAPTERS, ids=[a[0] for a in REAL_ADAPTERS]
)
async def test_real_adapter_keeps_same_named_tools(name, module, adapter_cls, invoke, gantry):
    pytest.importorskip(module, reason=f"{name} not installed")

    @gantry.register(name="send_email", namespace="billing")
    def billing_email(to: str) -> str:
        """Send an email from the billing service."""
        return f"billing:{to}"

    @gantry.register(name="send_email", namespace="inventory")
    def inventory_email(to: str) -> str:
        """Send an email from the inventory service."""
        return f"inventory:{to}"

    tools = await adapter_cls(gantry).select(
        "send an email", namespaces=["billing", "inventory"], limit=2
    )
    if name == "llamaindex":
        names = [t.metadata.name for t in tools]
    elif name == "strands":
        names = [t.tool_spec["name"] for t in tools]
    else:
        names = [t.name for t in tools]
    assert len(set(names)) == 2, f"{name}: one tool shadows the other"
    results = []
    for tool in tools:
        result = invoke(tool, {"to": "boss@x.com"})
        results.append(await result if inspect.isawaitable(result) else result)
    assert set(results) == {"billing:boss@x.com", "inventory:boss@x.com"}


@pytest.mark.parametrize(
    "name,module,adapter_cls,invoke", REAL_ADAPTERS, ids=[a[0] for a in REAL_ADAPTERS]
)
async def test_real_adapter_executes_async_callable(name, module, adapter_cls, invoke, gantry):
    pytest.importorskip(module, reason=f"{name} not installed")

    class AsyncEmail:
        async def __call__(self, to: str) -> str:
            """Send an email through an async callable object."""
            return f"callable:{to}"

    gantry.register(AsyncEmail(), name="callable_email", namespace="callables")
    tools = await adapter_cls(gantry).select("send an email", namespaces=["callables"], limit=1)
    result = invoke(tools[0], {"to": "boss@x.com"})
    if inspect.isawaitable(result):
        result = await result
    assert result == "callable:boss@x.com"


async def test_agent_framework_same_names_and_async_callables(gantry):
    pytest.importorskip("agent_framework")

    class AsyncEmail:
        def __init__(self, service):
            self.service = service

        async def __call__(self, to: str) -> str:
            """Send an email through an async callable object."""
            return f"{self.service}:{to}"

    for namespace in ("billing", "inventory"):
        gantry.register(AsyncEmail(namespace), name="send_email", namespace=namespace)
    tools = (
        await gantry.framework("agent_framework")
        .tool_bridge()
        .get_tools(
            "send an email",
            namespaces=["billing", "inventory"],
            limit=2,
        )
    )
    assert len({tool.name for tool in tools}) == 2
    results = {await tool.invoke(to="boss@x.com", skip_parsing=True) for tool in tools}
    assert results == {"billing:boss@x.com", "inventory:boss@x.com"}


async def test_agent_framework_builds_and_invokes(gantry):
    """The existing bridge API also builds and invokes real native FunctionTools."""
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
