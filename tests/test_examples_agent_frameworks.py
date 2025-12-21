import asyncio
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
    from examples.agent_frameworks import agent_framework_example as mod

    captured_tools: list[Any] = []

    class FakeOpenAIChatClient:
        def __init__(self, *_, **__):
            pass

    class FakeChatAgent:
        def __init__(self, *, chat_client, instructions, tools):
            self.chat_client = chat_client
            self.instructions = instructions
            self.tools = tools
            captured_tools.extend(tools)

        async def run(self, query: str) -> str:
            # call the first tool to ensure wiring works
            if not self.tools:
                return "no tools"
            return await self.tools[0]("abc123")

    async def fake_execute(self, tool_call):
        return _Result({"user_id": tool_call.arguments["user_id"], "plan": "pro"})

    monkeypatch.setattr(mod, "OpenAIChatClient", FakeOpenAIChatClient)
    monkeypatch.setattr(mod, "ChatAgent", FakeChatAgent)
    monkeypatch.setattr(mod.AgentGantry, "execute", fake_execute, raising=False)

    resp = await mod.main()
    assert captured_tools, "tool wrapping was not performed"
    assert "pro" in str(resp)


@pytest.mark.asyncio
async def test_google_adk_example_runs_with_fakes(monkeypatch):
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

        def run_async(self, *, user_id, session_id, new_message):  # noqa: ANN001
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
        async def create_session(self, *, app_name, user_id, session_id):  # noqa: ANN001
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
