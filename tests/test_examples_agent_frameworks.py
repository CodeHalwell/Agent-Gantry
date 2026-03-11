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

    # Stub the agent_framework package so the example can import without it installed
    af_mod = ModuleType("agent_framework")
    af_openai_mod = ModuleType("agent_framework.openai")

    class StubOpenAIResponsesClient:
        pass

    af_openai_mod.OpenAIResponsesClient = StubOpenAIResponsesClient
    af_mod.openai = af_openai_mod

    monkeypatch.setitem(sys.modules, "agent_framework", af_mod)
    monkeypatch.setitem(sys.modules, "agent_framework.openai", af_openai_mod)

    # Force re-import of the example module
    mod_name = "examples.agent_frameworks.agent_framework_example"
    if mod_name in sys.modules:
        del sys.modules[mod_name]

    from examples.agent_frameworks import agent_framework_example as mod

    captured_tools: list[Any] = []

    class FakeOpenAIResponsesClient:
        def __init__(self, *_, **__):
            pass

        def as_agent(self, *, name, instructions, tools):
            return FakeAgent(tools=tools)

    class FakeAgent:
        def __init__(self, *, tools):
            self.tools = tools
            captured_tools.extend(tools)

        async def run(self, query: str) -> str:
            if not self.tools:
                return "no tools"
            # Call the first tool to verify wiring works
            return await self.tools[0](user_id="abc123")

    monkeypatch.setattr(mod, "OpenAIResponsesClient", FakeOpenAIResponsesClient)

    resp = await mod.main()
    assert captured_tools, "tool wrapping was not performed"
    assert "pro" in str(resp)


@pytest.mark.asyncio
async def test_google_adk_example_runs_with_fakes(monkeypatch):
    import sys
    from types import ModuleType

    # Stub google.adk modules so the example can import without them installed
    for mod_name in [
        "google", "google.adk", "google.adk.agents", "google.adk.runners",
        "google.adk.sessions", "google.adk.tools", "google.genai",
        "google.genai.types",
    ]:
        if mod_name not in sys.modules:
            monkeypatch.setitem(sys.modules, mod_name, ModuleType(mod_name))

    google_mod = sys.modules["google"]
    adk_mod = sys.modules["google.adk"]
    google_mod.adk = adk_mod  # type: ignore[attr-defined]
    adk_mod.agents = sys.modules["google.adk.agents"]  # type: ignore[attr-defined]
    adk_mod.runners = sys.modules["google.adk.runners"]  # type: ignore[attr-defined]
    adk_mod.sessions = sys.modules["google.adk.sessions"]  # type: ignore[attr-defined]
    adk_mod.tools = sys.modules["google.adk.tools"]  # type: ignore[attr-defined]

    # Provide stub classes the example expects to import
    class StubAgent:
        pass
    class StubRunner:
        pass
    class StubInMemorySessionService:
        pass
    class StubFunctionTool:
        pass
    class StubContent:
        def __init__(self, *args, **kwargs):
            pass
    class StubPart:
        def __init__(self, *args, **kwargs):
            pass

    sys.modules["google.adk.agents"].Agent = StubAgent  # type: ignore[attr-defined]
    sys.modules["google.adk.runners"].Runner = StubRunner  # type: ignore[attr-defined]
    sys.modules["google.adk.sessions"].InMemorySessionService = StubInMemorySessionService  # type: ignore[attr-defined]
    sys.modules["google.adk.tools"].FunctionTool = StubFunctionTool  # type: ignore[attr-defined]
    sys.modules["google.genai"].types = sys.modules["google.genai.types"]  # type: ignore[attr-defined]
    sys.modules["google.genai.types"].Content = StubContent  # type: ignore[attr-defined]
    sys.modules["google.genai.types"].Part = StubPart  # type: ignore[attr-defined]

    # Force reimport
    example_mod_name = "examples.agent_frameworks.google_adk_example"
    if example_mod_name in sys.modules:
        del sys.modules[example_mod_name]

    from examples.agent_frameworks import google_adk_example as mod

    class FakeFunctionTool:
        def __init__(self, func):
            self.func = func

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

        def run_async(
            self, *, user_id: str, session_id: str, new_message: Any
        ) -> Any:
            async def _aiter():
                # Execute the first tool to simulate ADK calling it
                tool = self.agent.tools[0]
                text = await tool.func("123")
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

    monkeypatch.setattr(mod, "FunctionTool", FakeFunctionTool)
    monkeypatch.setattr(mod, "Agent", FakeAgent)
    monkeypatch.setattr(mod, "Runner", FakeRunner)
    monkeypatch.setattr(mod, "InMemorySessionService", FakeSessionService)
    monkeypatch.setattr(mod.AgentGantry, "execute", fake_execute, raising=False)

    resp = await mod.run_query("status please")
    assert "shipped" in resp
