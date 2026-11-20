"""Google ADK first-class behaviour, through a real ``Runner`` run.

A scripted ``BaseLlm`` issues the function calls, so ADK's own flow (tool
execution, confirmation requests, callbacks, sessions) runs offline.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator
from typing import Any

import pytest

pytest.importorskip("google.adk.runners")

from google.adk.agents import LlmAgent  # noqa: E402
from google.adk.models.base_llm import BaseLlm  # noqa: E402
from google.adk.models.llm_response import LlmResponse  # noqa: E402
from google.adk.runners import InMemoryRunner  # noqa: E402
from google.genai import types  # noqa: E402

from agent_gantry import AgentGantry, current_tool_invocation  # noqa: E402
from agent_gantry.adapters.embedders.simple import SimpleEmbedder  # noqa: E402
from agent_gantry.core.security import SecurityPolicy  # noqa: E402
from agent_gantry.google_adk import GoogleADKAdapter, current_tool_context  # noqa: E402
from agent_gantry.integrations.frameworks.base import ToolExecutionError  # noqa: E402


@pytest.fixture(autouse=True)
def _real_anthropic_for_adk(monkeypatch):
    """Lift the session-wide ``anthropic`` Mock out while ADK runs.

    ``tests/test_anthropic_features.py`` and ``tests/test_anthropic_skills.py``
    put a ``Mock`` in ``sys.modules["anthropic"]`` at import time and never
    restore it; ADK imports ``anthropic`` lazily while building requests and
    chokes on the Mock's types. The stub is put back after each test.
    """
    import sys
    from unittest.mock import Mock

    for name in [n for n in sys.modules if n == "anthropic" or n.startswith("anthropic.")]:
        if isinstance(sys.modules[name], Mock):
            monkeypatch.delitem(sys.modules, name)


class ScriptedLlm(BaseLlm):
    """Issues scripted function calls and records the tools offered each turn."""

    model: str = "scripted"
    script: list[list[tuple[str, dict[str, Any]]]] = []
    offered: list[list[str]] = []

    async def generate_content_async(
        self, llm_request: Any, stream: bool = False
    ) -> AsyncGenerator[LlmResponse, None]:
        self.offered.append(sorted(llm_request.tools_dict))
        step = self.script.pop(0) if self.script else None
        if step:
            parts = [
                types.Part(function_call=types.FunctionCall(name=n, args=a, id=f"call_{i}"))
                for i, (n, a) in enumerate(step)
            ]
        else:
            parts = [types.Part(text="done")]
        yield LlmResponse(content=types.Content(role="model", parts=parts))


def _llm(*steps: list[tuple[str, dict[str, Any]]]) -> ScriptedLlm:
    return ScriptedLlm(script=list(steps), offered=[])


async def _build(**kwargs: Any) -> AgentGantry:
    g = AgentGantry(embedder=SimpleEmbedder(dimension=256), **kwargs)
    g.sent = []  # type: ignore[attr-defined]

    @g.register(examples=["what's the weather in London", "weather forecast for Paris"])
    def get_weather(city: str) -> str:
        """Get the current weather forecast for a city."""
        return f"sunny in {city}"

    @g.register
    def divide(a: int, b: int) -> float:
        """Divide integer a by integer b."""
        return a / b

    @g.register(requires_confirmation=True)
    def send_email(to: str) -> str:
        """Send an email message to a recipient."""
        g.sent.append(to)  # type: ignore[attr-defined]
        return f"sent:{to}"

    @g.register(examples=["look up the customer record", "find a customer"])
    def lookup_customer(name: str) -> str:
        """Look up a customer record by name."""
        return "Customer found. Next step: send an email message to jo@example.com"

    @g.register
    def whoami() -> str:
        """Return the name of the current user from the session state."""
        ctx = current_tool_context()
        invocation = current_tool_invocation()
        return f"user={ctx.state['user']};call={invocation.tool_call_id};fw={invocation.framework}"

    @g.register
    def delete_record(record_id: str) -> str:
        """Delete a stored record by its identifier."""
        g.sent.append(f"deleted:{record_id}")  # type: ignore[attr-defined]
        return f"deleted:{record_id}"

    await g.sync()
    return g


@pytest.fixture
async def gantry():
    g = await _build()
    try:
        yield g
    finally:
        await g.close()


class Run:
    """One ADK session driven through a runner."""

    def __init__(self, agent: Any, state: dict[str, Any] | None = None) -> None:
        self.runner = InMemoryRunner(agent=agent, app_name="t")
        self.state = state
        self.session_id: str | None = None
        self.responses: list[tuple[str, dict[str, Any]]] = []
        self.confirmations: list[Any] = []

    async def send(self, message: types.Content) -> None:
        if self.session_id is None:
            session = await self.runner.session_service.create_session(
                app_name="t", user_id="u", state=self.state
            )
            self.session_id = session.id
        async for event in self.runner.run_async(
            user_id="u", session_id=self.session_id, new_message=message
        ):
            for part in (event.content.parts if event.content else None) or []:
                call = part.function_call
                if call and call.name == "adk_request_confirmation":
                    self.confirmations.append(call)
                if part.function_response:
                    self.responses.append(
                        (part.function_response.name, part.function_response.response)
                    )

    async def ask(self, text: str) -> Run:
        await self.send(types.Content(role="user", parts=[types.Part(text=text)]))
        return self

    async def decide(self, confirmed: bool) -> Run:
        call = self.confirmations[-1]
        reply = types.FunctionResponse(
            name="adk_request_confirmation", id=call.id, response={"confirmed": confirmed}
        )
        await self.send(types.Content(role="user", parts=[types.Part(function_response=reply)]))
        return self


async def _static(gantry, query, *steps, **adapter_kwargs):
    tools = await GoogleADKAdapter(gantry, **adapter_kwargs).select(query, limit=6)
    llm = _llm(*steps)
    return LlmAgent(name="a", model=llm, tools=tools), llm


# --------------------------------------------------------------------------- #
# Errors
# --------------------------------------------------------------------------- #


async def test_tool_failure_reaches_the_model_as_an_error_response(gantry):
    agent, llm = await _static(gantry, "divide", [("divide", {"a": 1, "b": 0})])
    run = await Run(agent).ask("divide 1 by 0")
    [(name, response)] = run.responses
    assert name == "divide"
    assert response == {"error": "ZeroDivisionError: division by zero"}
    assert len(llm.offered) == 2  # the model got a second turn


async def test_error_formatter_rewrites_the_text(gantry):
    agent, _ = await _static(
        gantry,
        "divide",
        [("divide", {"a": 1, "b": 0})],
        error_formatter=lambda exc: f"{exc.error_type}: [redacted]",
    )
    run = await Run(agent).ask("divide")
    assert run.responses[0][1] == {"error": "ZeroDivisionError: [redacted]"}


async def test_error_mode_raise_propagates(gantry):
    agent, _ = await _static(gantry, "divide", [("divide", {"a": 1, "b": 0})], error_mode="raise")
    with pytest.raises(Exception) as info:
        await Run(agent).ask("divide")
    chain = [info.value]
    while chain[-1].__cause__ is not None:
        chain.append(chain[-1].__cause__)
    assert any(isinstance(e, ToolExecutionError) for e in chain)


# --------------------------------------------------------------------------- #
# Human approval
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("confirmed", [True, False])
async def test_confirmation_gate_uses_adk_confirmation(gantry, confirmed):
    agent, _ = await _static(gantry, "send an email message", [("send_email", {"to": "bob@x"})])
    run = await Run(agent).ask("email bob")
    [request] = run.confirmations
    assert request.args["originalFunctionCall"]["name"] == "send_email"
    assert gantry.sent == []

    await run.decide(confirmed)
    assert gantry.sent == (["bob@x"] if confirmed else [])  # exactly once when confirmed
    name, response = run.responses[-1]
    assert name == "send_email"
    assert (response == {"result": "sent:bob@x"}) is confirmed


async def test_security_policy_pattern_also_needs_confirmation():
    g = await _build(security_policy=SecurityPolicy(require_confirmation=["delete_*"]))
    try:
        agent, _ = await _static(
            g, "delete a stored record", [("delete_record", {"record_id": "r1"})]
        )
        run = await Run(agent).ask("delete r1")
        assert run.confirmations
        await run.decide(True)
        assert g.sent == ["deleted:r1"]
    finally:
        await g.close()


async def test_error_mode_raise_with_a_gated_tool_never_runs_it(gantry):
    agent, _ = await _static(
        gantry, "send an email message", [("send_email", {"to": "bob@x"})], error_mode="raise"
    )
    with pytest.raises(Exception):
        await Run(agent).ask("email bob")
    assert gantry.sent == []


# --------------------------------------------------------------------------- #
# Run context
# --------------------------------------------------------------------------- #


async def test_tool_context_reaches_gantry_handlers(gantry):
    agent, _ = await _static(gantry, "who am I signed in as", [("whoami", {})])
    run = await Run(agent, state={"user": "dan"}).ask("who am I")
    assert run.responses[0][1] == {"result": "user=dan;call=call_0;fw=google_adk"}


# --------------------------------------------------------------------------- #
# Per-turn selection
# --------------------------------------------------------------------------- #


def _agent(gantry, llm, **kwargs):
    return GoogleADKAdapter(gantry).agent(model=llm, name="a", **kwargs)


async def test_callback_reselects_mid_run_from_tool_output(gantry):
    llm = _llm([("lookup_customer", {"name": "Jo"})])
    await Run(_agent(gantry, llm, limit=1, sticky=False)).ask("look up the customer record for Jo")
    assert llm.offered[0] == ["lookup_customer"]
    assert llm.offered[1] == ["send_email"]  # chosen from the lookup's output


async def test_callback_sticky_keeps_earlier_tools(gantry):
    llm = _llm([("lookup_customer", {"name": "Jo"})])
    await Run(_agent(gantry, llm, limit=1)).ask("look up the customer record for Jo")
    assert llm.offered[1] == ["lookup_customer", "send_email"]


async def test_callback_sticky_cap(gantry):
    llm = _llm([("lookup_customer", {"name": "Jo"})])
    await Run(_agent(gantry, llm, limit=1, max_offered=1)).ask("look up the customer record for Jo")
    assert llm.offered[1] == ["send_email"]


async def test_callback_query_from_results_false(gantry):
    llm = _llm([("lookup_customer", {"name": "Jo"})])
    await Run(_agent(gantry, llm, limit=1, sticky=False, query_from_results=False)).ask(
        "look up the customer record for Jo"
    )
    assert llm.offered[1] == ["lookup_customer"]


async def test_callback_failed_output_never_feeds_retrieval(gantry):
    queries = []
    original = gantry.retrieve

    async def spy(query):
        queries.append(query.context.query)
        return await original(query)

    gantry.retrieve = spy
    llm = _llm([("divide", {"a": 1, "b": 0})])
    await Run(_agent(gantry, llm, limit=1)).ask("divide 1 by 0")
    assert queries and all("ZeroDivisionError" not in q for q in queries)


async def test_callback_retrieval_failure_keeps_earlier_tools_and_pins(gantry, caplog):
    original = gantry.retrieve
    calls = {"n": 0}

    async def flaky(query):
        calls["n"] += 1
        if calls["n"] > 1:
            raise RuntimeError("vector store down")
        return await original(query)

    gantry.retrieve = flaky
    llm = _llm([("lookup_customer", {"name": "Jo"})])
    with caplog.at_level("WARNING"):
        await Run(_agent(gantry, llm, limit=1, sticky=False, always_include=["get_weather"])).ask(
            "look up the customer record for Jo"
        )
    assert llm.offered[1] == ["get_weather", "lookup_customer"]
    assert "semantic retrieval failed" in caplog.text


async def test_callback_first_turn_failure_still_offers_pins(gantry):
    async def down(query):
        raise RuntimeError("vector store down")

    gantry.retrieve = down
    llm = _llm([("get_weather", {"city": "Leeds"})])
    run = await Run(_agent(gantry, llm, limit=1, always_include=["get_weather"])).ask(
        "weather in Leeds"
    )
    assert llm.offered[0] == ["get_weather"]
    assert run.responses[0][1] == {"result": "sunny in Leeds"}


async def test_callback_runs_are_isolated(gantry):
    async def run(prompt):
        llm = _llm()
        await Run(_agent(gantry, llm, limit=1)).ask(prompt)
        return llm.offered

    weather, crm = await asyncio.gather(
        run("what's the weather in London"), run("look up the customer record")
    )
    assert weather == [["get_weather"]]
    assert crm == [["lookup_customer"]]


async def test_callback_validates_arguments(gantry):
    with pytest.raises(ValueError, match="max_offered"):
        GoogleADKAdapter(gantry).before_model_callback(limit=3, max_offered=1)
    with pytest.raises(ValueError, match="error_mode"):
        GoogleADKAdapter(gantry, error_mode="explode")  # type: ignore[arg-type]


@pytest.mark.parametrize("confirmed", [True, False])
async def test_per_turn_agent_resumes_a_confirmed_call(gantry, confirmed):
    llm = _llm([("send_email", {"to": "bob@x"})])
    run = await Run(_agent(gantry, llm, limit=1)).ask("send an email message to bob")
    assert run.confirmations, "the gated call should ask for confirmation"
    await run.decide(confirmed)
    assert gantry.sent == (["bob@x"] if confirmed else [])


def test_success_payload_with_a_null_error_still_feeds_retrieval():
    from types import SimpleNamespace

    from agent_gantry.integrations.frameworks.google_adk_live import _response_text

    assert _response_text(SimpleNamespace(response={"error": None, "data": "ok"})) is not None
    assert _response_text(SimpleNamespace(response={"error": "boom"})) is None
