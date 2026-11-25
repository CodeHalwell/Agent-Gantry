"""Haystack first-class behaviour, through a real ``Agent`` run (haystack-ai >= 3.0).

A scripted chat generator issues the tool calls, so Haystack's own agent loop,
tool execution, hooks, ``ConfirmationHook`` and ``State`` injection run
offline.
"""

from __future__ import annotations

from typing import Any

import pytest

pytest.importorskip("haystack.hooks.human_in_the_loop")

from haystack import component  # noqa: E402
from haystack.dataclasses import ChatMessage, ToolCall  # noqa: E402
from haystack.hooks.human_in_the_loop.dataclasses import ConfirmationUIResult  # noqa: E402

from dotfetch import DotFetch, current_tool_invocation  # noqa: E402
from dotfetch.adapters.embedders.simple import SimpleEmbedder  # noqa: E402
from dotfetch.core.security import SecurityPolicy  # noqa: E402
from dotfetch.haystack import (  # noqa: E402
    HaystackAdapter,
    HaystackToolError,
    current_agent_state,
)
from dotfetch.integrations.frameworks.base import ToolExecutionError  # noqa: E402


@component
class ScriptedGenerator:
    """Issues scripted tool calls, then a final answer; records the tools offered."""

    def __init__(self, *steps: list[tuple[str, dict[str, Any]]]) -> None:
        self.steps = list(steps)
        self.offered: list[list[str]] = []

    @component.output_types(replies=list[ChatMessage])
    def run(self, messages: list[ChatMessage], tools: Any = None, **kwargs: Any) -> dict:
        self.offered.append(sorted(tool.name for tool in tools or []))
        step = self.steps.pop(0) if self.steps else None
        if not step:
            return {"replies": [ChatMessage.from_assistant("done")]}
        calls = [
            ToolCall(tool_name=name, arguments=args, id=f"call_{len(self.offered)}_{i}")
            for i, (name, args) in enumerate(step)
        ]
        return {"replies": [ChatMessage.from_assistant(tool_calls=calls)]}


class ScriptedUI:
    """A ConfirmationUI that answers with a fixed action and records what it was asked."""

    def __init__(self, action: str = "confirm", new_params: dict[str, Any] | None = None) -> None:
        self.action = action
        self.new_params = new_params
        self.asked: list[tuple[str, dict[str, Any]]] = []

    def get_user_confirmation(self, tool_name, tool_description, tool_params):
        self.asked.append((tool_name, dict(tool_params)))
        return ConfirmationUIResult(action=self.action, new_tool_params=self.new_params)

    def to_dict(self) -> dict[str, Any]:
        return {}


async def _build(**kwargs: Any) -> DotFetch:
    g = DotFetch(embedder=SimpleEmbedder(dimension=256), **kwargs)
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
    def whoami() -> str:
        """Return the user stored in the agent state."""
        state = current_agent_state()
        invocation = current_tool_invocation()
        return f"user={state.get('user')};deps_user={invocation.deps['user']};fw={invocation.framework}"

    @g.register
    def delete_record(record_id: str) -> str:
        """Delete a stored record by its identifier."""
        g.sent.append(f"deleted:{record_id}")  # type: ignore[attr-defined]
        return f"deleted:{record_id}"

    @g.register(examples=["look up the customer record", "find customer details"])
    def lookup_customer(name: str) -> str:
        """Look up a customer's record by name."""
        return "customer Jo prefers to be contacted by email message"

    await g.sync()
    return g


@pytest.fixture
async def fetcher():
    g = await _build()
    try:
        yield g
    finally:
        await g.close()


def _results(result: dict) -> list[Any]:
    return [r for message in result["messages"] for r in (message.tool_call_results or [])]


async def _static(fetcher, query, *steps, ui=None, **adapter_kwargs):
    from haystack.components.agents import Agent

    adapter = HaystackAdapter(fetcher, **adapter_kwargs)
    tools = await adapter.select(query, limit=6)
    generator = ScriptedGenerator(*steps)
    hooks = {"before_tool": [adapter.confirmation_hook(ui)]} if ui is not None else None
    return Agent(chat_generator=generator, tools=tools, hooks=hooks), generator


def _ask(text: str) -> list[ChatMessage]:
    return [ChatMessage.from_user(text)]


async def test_tool_failure_is_an_error_tool_result(fetcher):
    agent, generator = await _static(fetcher, "divide", [("divide", {"a": 1, "b": 0})])
    result = agent.run(messages=_ask("divide 1 by 0"))
    [tool_result] = _results(result)
    assert tool_result.error is True
    assert tool_result.result == "ZeroDivisionError: division by zero"  # no argument echo
    assert len(generator.offered) == 2  # the model got another step


async def test_error_formatter_rewrites_the_text(fetcher):
    agent, _ = await _static(
        fetcher,
        "divide",
        [("divide", {"a": 1, "b": 0})],
        error_formatter=lambda exc: f"{exc.error_type}: [redacted]",
    )
    [tool_result] = _results(agent.run(messages=_ask("divide")))
    assert tool_result.result == "ZeroDivisionError: [redacted]"


async def test_failures_are_tool_execution_errors(fetcher):
    [tool] = await HaystackAdapter(fetcher).select("divide", limit=1)
    with pytest.raises(HaystackToolError) as info:
        tool.function(a=1, b=0)
    assert isinstance(info.value, ToolExecutionError)
    assert str(info.value) == "ZeroDivisionError: division by zero"


async def test_error_mode_raise_leaves_the_plain_error(fetcher):
    agent, _ = await _static(fetcher, "divide", [("divide", {"a": 1, "b": 0})], error_mode="raise")
    [tool_result] = _results(agent.run(messages=_ask("divide")))
    assert tool_result.error is True
    assert "failed (status=failure)" in tool_result.result


@pytest.mark.parametrize("action", ["confirm", "reject"])
async def test_confirmation_hook_asks_about_gated_calls(fetcher, action):
    ui = ScriptedUI(action)
    agent, _ = await _static(
        fetcher,
        "send an email message",
        [("send_email", {"to": "bob@x"}), ("get_weather", {"city": "Paris"})],
        ui=ui,
    )
    result = agent.run(messages=_ask("email bob"))
    assert ui.asked == [("send_email", {"to": "bob@x"})]  # not asked about get_weather
    assert fetcher.sent == (["bob@x"] if action == "confirm" else [])  # exactly once
    texts = [r.result for r in _results(result)]
    assert ("sent:bob@x" in texts) is (action == "confirm")


async def test_modified_call_runs_with_the_new_arguments(fetcher):
    ui = ScriptedUI("modify", new_params={"to": "carol@x"})
    agent, _ = await _static(
        fetcher, "send an email message", [("send_email", {"to": "bob@x"})], ui=ui
    )
    agent.run(messages=_ask("email bob"))
    assert fetcher.sent == ["carol@x"]


async def test_security_policy_pattern_also_asks():
    g = await _build(security_policy=SecurityPolicy(require_confirmation=["delete_*"]))
    try:
        ui = ScriptedUI()
        agent, _ = await _static(
            g, "delete a stored record", [("delete_record", {"record_id": "r1"})], ui=ui
        )
        agent.run(messages=_ask("delete r1"))
        assert [name for name, _ in ui.asked] == ["delete_record"]
        assert g.sent == ["deleted:r1"]
    finally:
        await g.close()


async def test_without_a_confirmation_ui_the_gated_call_is_reported_not_run(fetcher):
    agent, _ = await _static(fetcher, "send an email message", [("send_email", {"to": "bob@x"})])
    [tool_result] = _results(agent.run(messages=_ask("email bob")))
    assert tool_result.error is True and "approval" in tool_result.result
    assert fetcher.sent == []


async def test_direct_call_of_a_gated_tool_is_reported_not_run(fetcher):
    [tool] = await HaystackAdapter(fetcher).select("send an email message", limit=1)
    with pytest.raises(HaystackToolError, match="approval"):
        tool.function(to="bob@x")
    assert fetcher.sent == []


async def test_async_run_uses_the_native_async_function(fetcher):
    ui = ScriptedUI()
    agent, _ = await _static(
        fetcher, "send an email message", [("send_email", {"to": "bob@x"})], ui=ui
    )
    result = await agent.run_async(messages=_ask("email bob"))
    assert fetcher.sent == ["bob@x"]
    assert "sent:bob@x" in [r.result for r in _results(result)]


async def test_agent_state_reaches_fetcher_handlers(fetcher):
    from haystack.components.agents import Agent

    tools = await HaystackAdapter(fetcher).select("who am I", limit=6)
    agent = Agent(
        chat_generator=ScriptedGenerator([("whoami", {})]),
        tools=tools,
        state_schema={"user": {"type": str}},
    )
    result = agent.run(messages=_ask("who am I"), user="dan")
    assert "user=dan;deps_user=dan;fw=haystack" in [r.result for r in _results(result)]


async def test_injected_state_stays_out_of_the_schema(fetcher):
    [tool] = await HaystackAdapter(fetcher).select("weather forecast for Paris", limit=1)
    assert set(tool.parameters["properties"]) == {"city"}
    assert tool.function(city="Paris") == "sunny in Paris"


async def test_per_step_agent_reselects_after_a_tool_result(fetcher):
    generator = ScriptedGenerator([("lookup_customer", {"name": "Jo"})])
    agent = HaystackAdapter(fetcher).agent(chat_generator=generator, limit=1, sticky=False)
    agent.run(messages=_ask("look up the customer record for Jo"))
    assert generator.offered[0] == ["lookup_customer"]
    assert generator.offered[1] == ["send_email"]  # chosen from the tool result


async def test_per_step_agent_is_sticky_by_default(fetcher):
    generator = ScriptedGenerator([("lookup_customer", {"name": "Jo"})])
    agent = HaystackAdapter(fetcher).agent(chat_generator=generator, limit=1)
    agent.run(messages=_ask("look up the customer record for Jo"))
    assert generator.offered[1] == ["lookup_customer", "send_email"]


async def test_per_step_agent_reselects_for_each_run(fetcher):
    generator = ScriptedGenerator()
    agent = HaystackAdapter(fetcher).agent(chat_generator=generator, limit=1)
    agent.run(messages=_ask("what's the weather forecast for Paris"))
    await agent.run_async(messages=_ask("divide integer a by integer b"))
    assert generator.offered == [["get_weather"], ["divide"]]


@pytest.mark.parametrize("action", ["confirm", "reject"])
async def test_per_step_agent_asks_before_a_gated_call(fetcher, action):
    ui = ScriptedUI(action)
    generator = ScriptedGenerator([("send_email", {"to": "bob@x"})])
    agent = HaystackAdapter(fetcher).agent(chat_generator=generator, limit=1, confirmation_ui=ui)
    agent.run(messages=_ask("send an email message to bob"))
    assert ui.asked == [("send_email", {"to": "bob@x"})]
    assert fetcher.sent == (["bob@x"] if action == "confirm" else [])


async def test_a_tool_argument_named_state_keeps_the_approval_path(fetcher):
    @fetcher.register(requires_confirmation=True, examples=["register a voter in a US state"])
    def register_voter(name: str, state: str) -> str:
        """Register a voter in a US state."""
        fetcher.sent.append(f"{name}@{state}")
        return f"registered {name} in {state}"

    await fetcher.sync()
    ui = ScriptedUI("confirm")
    generator = ScriptedGenerator([("register_voter", {"name": "Jo", "state": "CA"})])
    agent = HaystackAdapter(fetcher).agent(chat_generator=generator, limit=1, confirmation_ui=ui)
    agent.run(messages=_ask("register a voter in a US state"))
    assert ui.asked == [("register_voter", {"name": "Jo", "state": "CA"})]
    assert fetcher.sent == ["Jo@CA"]


async def test_raise_mode_warns_that_the_confirmation_ui_is_not_asked(fetcher, caplog):
    adapter = HaystackAdapter(fetcher, error_mode="raise")
    with caplog.at_level("WARNING"):
        adapter.agent(chat_generator=ScriptedGenerator([]), limit=1, confirmation_ui=ScriptedUI())
        adapter.confirmation_hook(ScriptedUI())
    assert caplog.text.count("confirmation UI is never asked") == 2


async def test_model_mode_with_a_confirmation_ui_does_not_warn(fetcher, caplog):
    with caplog.at_level("WARNING"):
        HaystackAdapter(fetcher).agent(
            chat_generator=ScriptedGenerator([]), limit=1, confirmation_ui=ScriptedUI()
        )
    assert "confirmation UI is never asked" not in caplog.text


async def test_per_step_agent_uses_the_adapter_settings(fetcher):
    generator = ScriptedGenerator([("divide", {"a": 1, "b": 0})])
    agent = HaystackAdapter(fetcher, error_formatter=lambda exc: "[redacted]").agent(
        chat_generator=generator, limit=1
    )
    [tool_result] = _results(agent.run(messages=_ask("divide integer a by integer b")))
    assert tool_result.result == "[redacted]"


async def test_per_step_agent_keeps_user_hooks(fetcher):
    seen = []

    class Recorder:
        def run(self, state):
            seen.append(len(state.data["messages"]))

        def to_dict(self):
            return {}

    agent = HaystackAdapter(fetcher).agent(
        chat_generator=ScriptedGenerator(), limit=1, hooks={"before_run": [Recorder()]}
    )
    agent.run(messages=_ask("weather in Paris"))
    assert seen == [1]


async def test_per_step_agent_keeps_pins_when_retrieval_fails(fetcher, caplog):
    generator = ScriptedGenerator()
    agent = HaystackAdapter(fetcher).agent(
        chat_generator=generator, limit=1, always_include=["get_weather"]
    )

    async def down(*args, **kwargs):
        raise RuntimeError("vector store down")

    fetcher.retrieve = down
    with caplog.at_level("WARNING"):
        agent.run(messages=_ask("weather in Leeds"))
    assert generator.offered == [["get_weather"]]
    assert "semantic retrieval failed" in caplog.text


async def test_per_step_agent_rejects_static_tools(fetcher):
    with pytest.raises(TypeError, match="tools"):
        HaystackAdapter(fetcher).agent(chat_generator=ScriptedGenerator(), tools=[])


async def test_max_offered_below_limit_is_rejected(fetcher):
    with pytest.raises(ValueError, match="max_offered"):
        HaystackAdapter(fetcher).agent(chat_generator=ScriptedGenerator(), limit=3, max_offered=2)
