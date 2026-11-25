from typing import Any

import pytest

pytestmark = pytest.mark.filterwarnings(
    "ignore:Support for google-cloud-storage < 3.0.0:FutureWarning"
)


class _Result:
    def __init__(self, result: Any):
        self.status = "success"
        self.result = result
        self.error = None


@pytest.mark.asyncio
async def test_agent_framework_example_runs_with_fakes(monkeypatch, request):
    import sys
    from types import ModuleType

    # Define early so stub classes below can reference it via closure.
    captured_tools: list[Any] = []

    # Stub the agent_framework package so the example can import without it installed
    af_mod = ModuleType("agent_framework")
    af_openai_mod = ModuleType("agent_framework.openai")

    class StubOpenAIChatClient:
        pass

    af_openai_mod.OpenAIChatClient = StubOpenAIChatClient
    af_mod.openai = af_openai_mod

    # The example now also imports middleware primitives from agent_framework
    # at runtime (via DotFetchApprovalMiddleware/DotFetchObservabilityMiddleware).
    # Provide minimal stubs so the example module is fully exercisable.
    class _StubMiddlewareTerminationError(Exception):
        pass

    class _StubFunctionMiddleware:  # noqa: D401
        async def process(self, context, call_next):  # pragma: no cover - stub
            await call_next()

    def _stub_tool(func=None, **_kw):  # noqa: ANN001
        """Stand-in for agent_framework.tool decorator."""
        if func is None:
            return lambda f: f
        return func

    # The updated example directly imports Agent / WorkflowAgent / WorkflowBuilder.
    class _StubAgent:
        def __init__(self, client: Any, instructions: str, *, name: str = "", tools: Any = None, middleware: Any = None, **kwargs: Any) -> None:
            self.name = name
            self.tools = list(tools or [])
            self.middleware = middleware
            self.default_options = {"tools": self.tools}
            captured_tools.extend(self.tools)

        async def run(self, query: str, **kwargs: Any) -> Any:
            if self.tools:
                return await self.tools[0](user_id="abc123")
            return "no tools"

    class _StubWorkflow:
        pass

    class _StubWorkflowBuilder:
        def __init__(self, *, start_executor: Any = None, **kwargs: Any) -> None:
            pass

        def add_chain(self, agents: Any) -> "_StubWorkflowBuilder":
            return self

        def add_edge(self, source: Any, target: Any, condition: Any = None) -> "_StubWorkflowBuilder":
            return self

        def build(self) -> _StubWorkflow:
            return _StubWorkflow()

    class _StubWorkflowAgent:
        def __init__(self, workflow: Any, *, name: str | None = None, **kwargs: Any) -> None:
            self.name = name or "WorkflowAgent"

        async def run(self, query: str, **kwargs: Any) -> str:
            return "Customer is on the pro plan; invoice query routed to billing."

    class _StubAgentExecutor:
        """Stub for agent_framework.AgentExecutor (wraps Agent for WorkflowBuilder)."""

        def __init__(self, agent: Any, *, id: str = "") -> None:
            self._agent = agent
            self.id = id

    class _StubSequentialBuilder:
        def __init__(self, *, participants: Any = None, **kwargs: Any) -> None:
            self._participants = participants or []

        def build(self) -> "_StubWorkflow":
            return _StubWorkflow()

    af_orchestrations_mod = ModuleType("agent_framework.orchestrations")
    af_orchestrations_mod.SequentialBuilder = _StubSequentialBuilder
    af_orchestrations_mod.ConcurrentBuilder = _StubSequentialBuilder
    af_orchestrations_mod.HandoffBuilder = _StubSequentialBuilder
    af_orchestrations_mod.GroupChatBuilder = _StubSequentialBuilder

    af_mod.FunctionMiddleware = _StubFunctionMiddleware
    af_mod.ChatMiddlewareLayer = _StubFunctionMiddleware
    af_mod.MiddlewareTermination = _StubMiddlewareTerminationError
    af_mod.tool = _stub_tool
    af_mod.Agent = _StubAgent
    af_mod.AgentExecutor = _StubAgentExecutor
    af_mod.WorkflowAgent = _StubWorkflowAgent
    af_mod.WorkflowBuilder = _StubWorkflowBuilder
    af_mod.orchestrations = af_orchestrations_mod

    monkeypatch.setitem(sys.modules, "agent_framework", af_mod)
    monkeypatch.setitem(sys.modules, "agent_framework.openai", af_openai_mod)
    monkeypatch.setitem(sys.modules, "agent_framework.orchestrations", af_orchestrations_mod)

    # Rebuild the SDK-dependent middleware classes against the stubs, then
    # discard them at teardown. Evicting dotfetch modules from sys.modules leaves
    # stale class/module references behind and breaks tests in a different order.
    from dotfetch.integrations import agent_framework_middleware

    agent_framework_middleware._build_middleware_classes.cache_clear()
    request.addfinalizer(agent_framework_middleware._build_middleware_classes.cache_clear)

    from examples.agent_frameworks import agent_framework_example as mod

    class FakeOpenAIChatClient:
        def __init__(self, *_, **__):
            pass

        def as_agent(self, *, name, instructions, tools, middleware=None):
            return FakeAgent(name=name, tools=tools, middleware=middleware)

    class FakeAgent:
        def __init__(self, *, name="", tools, middleware=None):
            self.name = name
            self.tools = tools
            self.middleware = middleware
            # Mirror AF's public shape so example code that reads
            # ``agent.default_options["tools"]`` keeps working.
            self.default_options = {"tools": tools}
            captured_tools.extend(tools)

        async def run(self, query: str) -> str:
            if not self.tools:
                return "no tools"
            # Call the first tool to verify wiring works
            return await self.tools[0](user_id="abc123")

    monkeypatch.setattr(mod, "OpenAIChatClient", FakeOpenAIChatClient)

    resp = await mod.main()
    assert captured_tools, "tool wrapping was not performed"
    assert "pro" in str(resp)


@pytest.mark.asyncio
async def test_google_adk_example_runs_offline(capsys):
    """The ADK example drives a real ADK runner with a scripted model."""
    pytest.importorskip("google.adk.runners")
    from examples.agent_frameworks import google_adk_example as mod

    await mod.main()
    out = capsys.readouterr().out
    assert "tools offered at turn 2: ['lookup_customer', 'send_email']" in out
    assert "{'error': 'ZeroDivisionError: division by zero'}" in out
    assert "confirmed -> {'result': 'Email sent to jo@example.com.'}" in out
    assert "You are signed in as Dan." in out
