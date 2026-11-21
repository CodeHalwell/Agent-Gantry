"""LlamaIndex first-class behaviour, through a real ``FunctionAgent`` run.

A scripted function-calling LLM issues the tool calls, so the real agent
workflow, ``ToolOutput`` error handling, ``Context`` injection and
human-in-the-loop events run offline.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

pytest.importorskip("llama_index.core.agent.workflow")

from llama_index.core.agent.workflow import FunctionAgent  # noqa: E402
from llama_index.core.llms import ChatMessage, ChatResponse, LLMMetadata, MessageRole  # noqa: E402
from llama_index.core.llms.function_calling import FunctionCallingLLM  # noqa: E402
from llama_index.core.llms.llm import ToolSelection  # noqa: E402
from llama_index.core.workflow import (  # noqa: E402
    Context,
    HumanResponseEvent,
    InputRequiredEvent,
)

from agent_gantry import AgentGantry, current_tool_invocation  # noqa: E402
from agent_gantry.adapters.embedders.simple import SimpleEmbedder  # noqa: E402
from agent_gantry.core.security import SecurityPolicy  # noqa: E402
from agent_gantry.integrations.frameworks.base import ToolExecutionError  # noqa: E402
from agent_gantry.llamaindex import (  # noqa: E402
    LlamaIndexAdapter,
    LlamaIndexToolError,
    current_workflow_context,
)


class ScriptedLLM(FunctionCallingLLM):
    """Issues scripted tool calls and records the tools offered each step."""

    script: list[list[tuple[str, dict[str, Any]]]] = []
    offered: list[list[str]] = []

    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(is_function_calling_model=True, model_name="scripted")

    def _prepare_chat_with_tools(
        self, tools: Any, user_msg: Any = None, chat_history: Any = None, **kwargs: Any
    ) -> dict[str, Any]:  # noqa: E501
        self.offered.append(sorted(t.metadata.name for t in tools))
        return {"messages": list(chat_history or [])}

    def _reply(self) -> ChatResponse:
        step = self.script.pop(0) if self.script else None
        if step:
            calls = [{"id": f"call_{i}", "name": n, "args": a} for i, (n, a) in enumerate(step)]
            message = ChatMessage(
                role=MessageRole.ASSISTANT, content="", additional_kwargs={"tool_calls": calls}
            )
        else:
            message = ChatMessage(role=MessageRole.ASSISTANT, content="done")
        return ChatResponse(message=message, delta=message.content)

    def chat(self, messages: Any, **kwargs: Any) -> ChatResponse:
        return self._reply()

    async def achat(self, messages: Any, **kwargs: Any) -> ChatResponse:
        return self._reply()

    def stream_chat(self, messages: Any, **kwargs: Any) -> Any:
        yield self._reply()

    async def astream_chat(self, messages: Any, **kwargs: Any) -> Any:
        reply = self._reply()

        async def gen() -> Any:
            yield reply

        return gen()

    def complete(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError

    async def acomplete(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError

    def stream_complete(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError

    async def astream_complete(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError

    def get_tool_calls_from_response(
        self, response: Any, error_on_no_tool_call: bool = True, **kwargs: Any
    ) -> list[ToolSelection]:  # noqa: E501
        calls = response.message.additional_kwargs.get("tool_calls", [])
        return [
            ToolSelection(tool_id=c["id"], tool_name=c["name"], tool_kwargs=c["args"])
            for c in calls
        ]


def _llm(*steps: list[tuple[str, dict[str, Any]]]) -> ScriptedLLM:
    return ScriptedLLM(script=list(steps), offered=[])


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

    @g.register
    async def whoami() -> str:
        """Return the user stored in the workflow state."""
        ctx = current_workflow_context()
        invocation = current_tool_invocation()
        user = await ctx.store.get("user")
        return f"user={user};fw={invocation.framework}"

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


async def _run(agent: Any, prompt: str = "go", *, respond: Any = None, ctx: Any = None) -> Any:
    """Run ``agent``; answer approval requests with ``respond``; return the tool results."""
    handler = agent.run(user_msg=prompt, ctx=ctx)
    results = []
    requests = []
    async for event in handler.stream_events():
        if type(event).__name__ == "ToolCallResult":
            results.append(event.tool_output)
        if isinstance(event, InputRequiredEvent):
            requests.append(event)
            answer = respond(event) if callable(respond) else respond
            handler.ctx.send_event(
                HumanResponseEvent(response=answer, gantry_approval_id=event.gantry_approval_id)
            )
    await handler
    return results, requests


async def _static(gantry, query, *steps, **adapter_kwargs):
    tools = await LlamaIndexAdapter(gantry, **adapter_kwargs).select(query, limit=6)
    llm = _llm(*steps)
    return FunctionAgent(tools=tools, llm=llm), llm


async def test_tool_failure_reaches_the_model_as_an_error_output(gantry):
    agent, llm = await _static(gantry, "divide", [("divide", {"a": 1, "b": 0})])
    [output], _ = await _run(agent)
    assert output.is_error
    assert output.content == "ZeroDivisionError: division by zero"
    assert len(llm.offered) == 2  # the model got another step


async def test_error_formatter_rewrites_the_text(gantry):
    agent, _ = await _static(
        gantry,
        "divide",
        [("divide", {"a": 1, "b": 0})],
        error_formatter=lambda exc: f"{exc.error_type}: [redacted]",
    )
    [output], _ = await _run(agent)
    assert output.content == "ZeroDivisionError: [redacted]"


async def test_failures_are_tool_execution_errors(gantry):
    [tool] = await LlamaIndexAdapter(gantry).select("divide", limit=1)
    with pytest.raises(LlamaIndexToolError) as info:
        await tool.async_fn(a=1, b=0)
    assert isinstance(info.value, ToolExecutionError)
    assert str(info.value) == "ZeroDivisionError: division by zero"


@pytest.mark.parametrize("respond", ["yes", "no"])
async def test_gated_call_waits_for_a_human_response(gantry, respond):
    agent, _ = await _static(gantry, "send an email message", [("send_email", {"to": "bob@x"})])
    [output], [request] = await _run(agent, respond=respond)
    assert request.gantry_tool == "send_email"
    assert request.gantry_arguments == {"to": "bob@x"}
    approved = respond == "yes"
    assert gantry.sent == (["bob@x"] if approved else [])  # exactly once when approved
    assert (output.content == "sent:bob@x") is approved
    assert output.is_error is not approved


async def test_security_policy_pattern_also_waits_for_approval():
    g = await _build(security_policy=SecurityPolicy(require_confirmation=["delete_*"]))
    try:
        agent, _ = await _static(
            g, "delete a stored record", [("delete_record", {"record_id": "r1"})]
        )
        _, [request] = await _run(agent, respond="yes")
        assert request.gantry_tool == "delete_record"
        assert g.sent == ["deleted:r1"]
    finally:
        await g.close()


async def test_direct_call_of_a_gated_tool_is_reported_not_run(gantry):
    [tool] = await LlamaIndexAdapter(gantry).select("send an email message", limit=1)
    with pytest.raises(LlamaIndexToolError, match="approval"):
        await tool.async_fn(to="bob@x")
    assert gantry.sent == []


async def test_error_mode_raise_with_a_gated_tool_never_runs_it(gantry):
    agent, _ = await _static(
        gantry, "send an email message", [("send_email", {"to": "bob@x"})], error_mode="raise"
    )
    [output], requests = await _run(agent, respond="yes")
    assert requests == []  # no approval flow in raise mode
    assert output.is_error and gantry.sent == []


async def test_workflow_context_reaches_gantry_handlers(gantry):
    agent, _ = await _static(gantry, "who am I", [("whoami", {})])
    ctx = Context(agent)
    await ctx.store.set("user", "dan")
    [output], _ = await _run(agent, ctx=ctx)
    assert output.content == "user=dan;fw=llamaindex"


async def test_tools_still_work_when_called_directly(gantry):
    [tool] = await LlamaIndexAdapter(gantry).select("weather forecast for Paris", limit=1)
    assert (await tool.acall(city="Paris")).content == "sunny in Paris"
    assert "ctx" not in tool.metadata.get_parameters_dict()["properties"]


async def test_two_identical_gated_calls_each_wait_for_their_own_approval(gantry):
    llm = _llm([("send_email", {"to": "bob@x"}), ("send_email", {"to": "bob@x"})])
    agent = LlamaIndexAdapter(gantry).function_agent(llm, limit=1)
    outputs, requests = await asyncio.wait_for(
        _run(agent, "send an email message to bob", respond="yes"), timeout=30
    )
    assert len(requests) == 2  # one approval request per call, no shared waiter
    assert gantry.sent == ["bob@x", "bob@x"]  # each approved call ran exactly once
    # LlamaIndex replays a waiting step when a response arrives and may re-emit
    # its result; the replay joins the call that already ran.
    assert {o.content for o in outputs} == {"sent:bob@x"}


async def test_each_response_only_decides_its_own_call():
    """With two gated calls pending, an answer for one never decides the other."""
    g = await _build(security_policy=SecurityPolicy(require_confirmation=["delete_*"]))
    try:
        llm = _llm([("send_email", {"to": "bob@x"}), ("delete_record", {"record_id": "r1"})])
        agent = LlamaIndexAdapter(g).function_agent(llm, limit=1, always_include=["delete_record"])
        handler = agent.run(user_msg="send an email message to bob")
        pending = []
        async for event in handler.stream_events():
            if isinstance(event, InputRequiredEvent):
                pending.append(event)
                if len(pending) == 2:
                    # Reject the deletion first, then approve the email.
                    for request in sorted(pending, key=lambda r: r.gantry_tool):
                        answer = "no" if request.gantry_tool == "delete_record" else "yes"
                        handler.ctx.send_event(
                            HumanResponseEvent(
                                response=answer, gantry_approval_id=request.gantry_approval_id
                            )
                        )
        await asyncio.wait_for(handler, timeout=30)
        assert g.sent == ["bob@x"]  # the email ran; the rejected deletion never did
    finally:
        await g.close()


async def test_a_later_identical_call_asks_and_runs_again(gantry):
    """A cached approval is only reused by replays of the same call."""
    agent, _ = await _static(
        gantry,
        "send an email message",
        [("send_email", {"to": "bob@x"})],
        [("send_email", {"to": "bob@x"})],
    )
    outputs, requests = await asyncio.wait_for(_run(agent, respond="yes"), timeout=30)
    assert len(requests) == 2
    assert gantry.sent == ["bob@x", "bob@x"]


async def test_gantry_function_agent_takes_static_tools(gantry):
    from agent_gantry.llamaindex import GantryFunctionAgent

    tools = await LlamaIndexAdapter(gantry).select("send an email message", limit=6)
    llm = _llm([("send_email", {"to": "bob@x"}), ("send_email", {"to": "bob@x"})])
    agent = GantryFunctionAgent(tools=tools, llm=llm)
    _, requests = await asyncio.wait_for(_run(agent, respond="yes"), timeout=30)
    assert len(requests) == 2
    assert gantry.sent == ["bob@x", "bob@x"]


async def test_an_unexpected_failure_after_approval_never_hangs_replays(gantry, monkeypatch):
    from agent_gantry.integrations.frameworks import llamaindex as module

    async def broken(*args, **kwargs):
        raise RuntimeError("tool runner broke")

    monkeypatch.setattr(module, "execute_spec", broken)
    llm = _llm([("send_email", {"to": "bob@x"}), ("send_email", {"to": "carol@x"})])
    agent = LlamaIndexAdapter(gantry).function_agent(llm, limit=1)
    outputs, requests = await asyncio.wait_for(
        _run(agent, "send an email message", respond="yes"), timeout=30
    )
    assert len(requests) == 2
    assert outputs and all(o.is_error for o in outputs)


async def test_a_tool_argument_named_ctx_is_passed_through():
    g = AgentGantry(embedder=SimpleEmbedder(dimension=256))

    @g.register
    def echo(ctx: str) -> str:
        """Echo the given context string."""
        return f"echo:{ctx}"

    await g.sync()
    try:
        [tool] = await LlamaIndexAdapter(g).select("echo the context string", limit=1)
        assert (await tool.acall(ctx="hello")).content == "echo:hello"
        assert tool.call(ctx="hi").content == "echo:hi"
        agent = FunctionAgent(tools=[tool], llm=_llm([("echo", {"ctx": "from-model"})]))
        [output], _ = await _run(agent)
        assert output.content == "echo:from-model"
    finally:
        await g.close()


async def test_tool_retriever_offers_pins_when_retrieval_fails(gantry, caplog):
    retriever = LlamaIndexAdapter(gantry).tool_retriever(limit=1, always_include=["get_weather"])

    async def down(query):
        raise RuntimeError("vector store down")

    gantry.retrieve = down
    with caplog.at_level("WARNING"):
        tools = await retriever.aretrieve("weather in Leeds")
    assert [t.metadata.name for t in tools] == ["get_weather"]
    assert "semantic retrieval failed" in caplog.text


async def test_function_agent_uses_the_adapter_settings(gantry):
    adapter = LlamaIndexAdapter(gantry, error_formatter=lambda exc: "[redacted]")
    llm = _llm([("divide", {"a": 1, "b": 0})])
    agent = adapter.function_agent(llm, limit=3)
    [output], _ = await _run(agent, "divide two numbers")
    assert output.content == "[redacted]"


@pytest.mark.parametrize("respond", ["yes", "no"])
async def test_per_step_agent_resumes_an_approved_call(gantry, respond):
    llm = _llm([("send_email", {"to": "bob@x"})])
    agent = LlamaIndexAdapter(gantry).function_agent(llm, limit=1)
    [output], [request] = await _run(agent, "send an email message to bob", respond=respond)
    assert request.gantry_tool == "send_email"
    assert gantry.sent == (["bob@x"] if respond == "yes" else [])
