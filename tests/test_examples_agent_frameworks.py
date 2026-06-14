from types import SimpleNamespace
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
async def test_agent_framework_example_runs_with_fakes(monkeypatch):
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
    # at runtime (via GantryApprovalMiddleware/GantryObservabilityMiddleware).
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

    # The example module imports the middleware integration at top level,
    # and the middleware module caches its AF-subclass construction via
    # lru_cache. Force a fresh import so those modules pick up the stubs
    # registered above instead of any previously imported agent_framework
    # objects (and so the lru_cache starts empty for this test).
    for m in [
        "agent_gantry.integrations.agent_framework_middleware",
        "agent_gantry.integrations.agent_framework_bridge",
        "agent_gantry.integrations",
    ]:
        if m in sys.modules:
            del sys.modules[m]

    # Force re-import of the example module
    mod_name = "examples.agent_frameworks.agent_framework_example"
    if mod_name in sys.modules:
        del sys.modules[mod_name]

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
async def test_google_adk_example_runs_with_fakes(monkeypatch):
    import sys
    from types import ModuleType

    # Stub google.adk modules so the example can import without them installed
    # Always override with stub modules to avoid mutating real installed packages
    for mod_name in [
        "google", "google.adk", "google.adk.agents", "google.adk.runners",
        "google.adk.sessions", "google.adk.tools", "google.genai",
        "google.genai.types",
    ]:
        stub_module = ModuleType(mod_name)
        monkeypatch.setitem(sys.modules, mod_name, stub_module)

    google_mod = sys.modules["google"]
    adk_mod = sys.modules["google.adk"]
    # Use monkeypatch.setattr for attribute injection to guarantee cleanup
    monkeypatch.setattr(google_mod, "adk", adk_mod, raising=False)  # type: ignore[attr-defined]
    monkeypatch.setattr(adk_mod, "agents", sys.modules["google.adk.agents"], raising=False)  # type: ignore[attr-defined]
    monkeypatch.setattr(adk_mod, "runners", sys.modules["google.adk.runners"], raising=False)  # type: ignore[attr-defined]
    monkeypatch.setattr(adk_mod, "sessions", sys.modules["google.adk.sessions"], raising=False)  # type: ignore[attr-defined]
    monkeypatch.setattr(adk_mod, "tools", sys.modules["google.adk.tools"], raising=False)  # type: ignore[attr-defined]

    # Provide stub classes the example expects to import
    class StubAgent:
        pass
    class StubRunner:
        pass
    class StubInMemorySessionService:
        pass
    class StubFunctionTool:
        # for_google_adk calls FunctionTool(func=callable); store it so the
        # fake runner can invoke the tool through Gantry.
        def __init__(self, func, **kwargs):
            self.func = func
            self.name = getattr(func, "__name__", "tool")
    class StubContent:
        def __init__(self, *args, **kwargs):
            pass
    class StubPart:
        def __init__(self, *args, **kwargs):
            pass

    monkeypatch.setattr(sys.modules["google.adk.agents"], "Agent", StubAgent, raising=False)  # type: ignore[attr-defined]
    monkeypatch.setattr(sys.modules["google.adk.runners"], "Runner", StubRunner, raising=False)  # type: ignore[attr-defined]
    monkeypatch.setattr(sys.modules["google.adk.sessions"], "InMemorySessionService", StubInMemorySessionService, raising=False)  # type: ignore[attr-defined]
    monkeypatch.setattr(sys.modules["google.adk.tools"], "FunctionTool", StubFunctionTool, raising=False)  # type: ignore[attr-defined]
    monkeypatch.setattr(sys.modules["google.genai"], "types", sys.modules["google.genai.types"], raising=False)  # type: ignore[attr-defined]
    monkeypatch.setattr(sys.modules["google.genai.types"], "Content", StubContent, raising=False)  # type: ignore[attr-defined]
    monkeypatch.setattr(sys.modules["google.genai.types"], "Part", StubPart, raising=False)  # type: ignore[attr-defined]

    # Force reimport
    example_mod_name = "examples.agent_frameworks.google_adk_example"
    if example_mod_name in sys.modules:
        del sys.modules[example_mod_name]

    from examples.agent_frameworks import google_adk_example as mod

    class FakeEvent:
        def __init__(self, text: str):
            self.content = SimpleNamespace(parts=[SimpleNamespace(text=text)])

        def is_final_response(self) -> bool:
            return True

    class FakeRunner:
        def __init__(self, *, agent, app_name, session_service):
            self.agent = agent
            self.app_name = app_name
            self.session_service = session_service

        def run_async(self, *, user_id: str, session_id: str, new_message: Any) -> Any:
            async def _aiter():
                # Execute the first tool to simulate ADK calling it. The Gantry
                # adapter's callable uses keyword-only params (from the JSON
                # schema), so call by name.
                tool = self.agent.tools[0]
                text = await tool.func(order_id="123")
                yield FakeEvent(text)

            return _aiter()

    class FakeAgent:
        def __init__(self, *, model, name, instruction, tools):
            self.model = model
            self.name = name
            self.instruction = instruction
            self.tools = tools

    class FakeSessionService:
        async def create_session(
            self, *, app_name: str, user_id: str, session_id: str
        ) -> dict[str, str]:
            return {"app_name": app_name, "user_id": user_id, "session_id": session_id}

    async def fake_execute(self, tool_call):
        return _Result({"order_id": tool_call.arguments["order_id"], "status": "shipped"})

    # The migrated example builds tools via for_google_adk -> the (stubbed)
    # google.adk.tools.FunctionTool, so no module-level FunctionTool to patch.
    monkeypatch.setattr(mod, "Agent", FakeAgent)
    monkeypatch.setattr(mod, "Runner", FakeRunner)
    monkeypatch.setattr(mod, "InMemorySessionService", FakeSessionService)
    monkeypatch.setattr(mod.AgentGantry, "execute", fake_execute, raising=False)

    resp = await mod.run_query("status please")
    # The adapter returns the tool's raw result (here a dict); the fake runner
    # surfaces it directly as the final response.
    assert "shipped" in str(resp)
