"""CrewAI first-class behaviour, through a real ``Crew.kickoff``.

A scripted ``BaseLLM`` answers in CrewAI's ReAct text format, so the real
agent executor, tool usage, hooks and ``ToolFailure`` recording run offline.
"""

from __future__ import annotations

import json
import os
from typing import Any

import pytest

pytest.importorskip("crewai.hooks")

os.environ.setdefault("CREWAI_DISABLE_TELEMETRY", "true")
os.environ.setdefault("CREWAI_TRACING_ENABLED", "false")

from crewai import Agent, Crew, Task  # noqa: E402
from crewai.llms.base_llm import BaseLLM  # noqa: E402

from dotfetch import DotFetch, current_tool_invocation  # noqa: E402
from dotfetch.adapters.embedders.simple import SimpleEmbedder  # noqa: E402
from dotfetch.core.security import SecurityPolicy  # noqa: E402
from dotfetch.crewai import ApprovalRequest, CrewAIAdapter, current_crew_context  # noqa: E402


class ScriptedLLM(BaseLLM):
    """Replies with scripted ReAct steps, then a final answer."""

    script: list[str] = []
    prompts: list[str] = []

    def call(self, messages: Any, *args: Any, **kwargs: Any) -> str:
        self.prompts.append(json.dumps(messages, default=str))
        return self.script.pop(0) if self.script else "Thought: done\nFinal Answer: done"

    def supports_function_calling(self) -> bool:
        return False

    def supports_stop_words(self) -> bool:
        return True

    def get_context_window_size(self) -> int:
        return 8000


def _llm(*calls: tuple[str, dict[str, Any]]) -> ScriptedLLM:
    script = [
        f"Thought: use {name}\nAction: {name}\nAction Input: {json.dumps(args)}"
        for name, args in calls
    ]
    return ScriptedLLM(model="scripted", script=script, prompts=[])


async def _build(**kwargs: Any) -> DotFetch:
    g = DotFetch(embedder=SimpleEmbedder(dimension=256), **kwargs)
    g.sent = []  # type: ignore[attr-defined]

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
        """Return the role of the agent making this call."""
        ctx = current_crew_context()
        invocation = current_tool_invocation()
        return f"role={ctx.agent.role};fw={invocation.framework};task={invocation.metadata.get('task_name')}"

    @g.register
    def delete_record(record_id: str) -> str:
        """Delete a stored record by its identifier."""
        g.sent.append(f"deleted:{record_id}")  # type: ignore[attr-defined]
        return f"deleted:{record_id}"

    await g.sync()
    return g


@pytest.fixture
async def fetcher():
    g = await _build()
    try:
        yield g
    finally:
        await g.close()


async def _kickoff(fetcher, query, *calls, **adapter_kwargs):
    tools = await CrewAIAdapter(fetcher, **adapter_kwargs).select(query, limit=6)
    llm = _llm(*calls)
    agent = Agent(role="clerk", goal="help", backstory="b", llm=llm, tools=tools, max_iter=5)
    task = Task(name="chores", description=query, expected_output="done", agent=agent)
    output = await Crew(agents=[agent], tasks=[task]).kickoff_async()
    return output, llm


def _observations(llm: ScriptedLLM) -> str:
    return "\n".join(llm.prompts[1:])


async def test_tool_failure_is_a_crewai_tool_failure(fetcher):
    output, llm = await _kickoff(fetcher, "divide", ("divide", {"a": 1, "b": 0}))
    [record] = output.tasks_output[0].tool_failures
    assert record.tool_name == "divide"
    assert record.failure.message == "ZeroDivisionError: division by zero"
    assert record.failure.code == "ZeroDivisionError"
    assert record.failure.retryable is False
    assert "ZeroDivisionError: division by zero" in _observations(llm)  # the agent saw it
    assert output.raw == "done"


async def test_error_formatter_rewrites_the_failure(fetcher):
    output, _ = await _kickoff(
        fetcher,
        "divide",
        ("divide", {"a": 1, "b": 0}),
        error_formatter=lambda exc: f"{exc.error_type}: [redacted]",
    )
    assert (
        output.tasks_output[0].tool_failures[0].failure.message == "ZeroDivisionError: [redacted]"
    )


@pytest.mark.parametrize("decision", [True, False])
async def test_approver_decides_gated_calls(fetcher, decision):
    seen: list[ApprovalRequest] = []

    def approve(request: ApprovalRequest) -> bool:
        seen.append(request)
        return decision

    await _kickoff(
        fetcher, "send an email message", ("send_email", {"to": "bob@x"}), approve=approve
    )
    [request] = seen
    assert request.tool_name == "send_email"
    assert request.arguments == {"to": "bob@x"}
    assert request.agent.role == "clerk"
    assert fetcher.sent == (["bob@x"] if decision else [])  # exactly once when approved


async def test_console_approver_uses_request_human_input(fetcher, monkeypatch):
    from crewai.hooks import ToolCallHookContext

    monkeypatch.setattr(ToolCallHookContext, "request_human_input", lambda self, prompt: "yes")
    await _kickoff(
        fetcher, "send an email message", ("send_email", {"to": "bob@x"}), approve="console"
    )
    assert fetcher.sent == ["bob@x"]


async def test_without_an_approver_the_gated_call_is_reported_not_run(fetcher):
    output, _ = await _kickoff(fetcher, "send an email message", ("send_email", {"to": "bob@x"}))
    assert fetcher.sent == []
    [record] = output.tasks_output[0].tool_failures
    assert "approval" in record.failure.message


async def test_security_policy_pattern_also_needs_approval():
    g = await _build(security_policy=SecurityPolicy(require_confirmation=["delete_*"]))
    try:
        await _kickoff(
            g,
            "delete a stored record",
            ("delete_record", {"record_id": "r1"}),
            approve=lambda r: True,
        )
        assert g.sent == ["deleted:r1"]
    finally:
        await g.close()


async def test_crew_context_reaches_fetcher_handlers(fetcher):
    _, llm = await _kickoff(fetcher, "who is making this call", ("whoami", {}))
    assert "role=clerk;fw=crewai;task=chores" in _observations(llm)


async def test_error_mode_raise_lets_crewai_handle_the_exception(fetcher):
    output, llm = await _kickoff(fetcher, "divide", ("divide", {"a": 1, "b": 0}), error_mode="raise")
    assert all(r.failure.reason.value == "exception" for r in output.tasks_output[0].tool_failures)


async def test_agent_builder_uses_the_adapter_settings(fetcher):
    adapter = CrewAIAdapter(fetcher, error_formatter=lambda exc: "[redacted]")
    builder = adapter.agent_builder(limit=3, llm=_llm(("divide", {"a": 1, "b": 0})), role="clerk")
    agent = await builder.build("divide two numbers")
    task = Task(description="divide", expected_output="done", agent=agent)
    output = await Crew(agents=[agent], tasks=[task]).kickoff_async()
    assert output.tasks_output[0].tool_failures[0].failure.message == "[redacted]"


async def test_each_adapter_keeps_its_own_approver(fetcher):
    """Two adapters exposing the same tool name never decide each other's calls."""
    refusing = await CrewAIAdapter(fetcher, approve=lambda r: False).select(
        "send an email message", limit=6
    )
    await CrewAIAdapter(fetcher, approve=lambda r: True).select("send an email message", limit=6)

    async def kickoff(tools):
        llm = _llm(("send_email", {"to": "bob@x"}))
        agent = Agent(role="clerk", goal="help", backstory="b", llm=llm, tools=tools, max_iter=5)
        task = Task(description="email bob", expected_output="done", agent=agent)
        await Crew(agents=[agent], tasks=[task]).kickoff_async()

    await kickoff(refusing)  # built first; the approving adapter's tools came later
    assert fetcher.sent == []


async def test_crew_context_reaches_handlers_in_raise_mode_too(fetcher):
    _, llm = await _kickoff(fetcher, "who is making this call", ("whoami", {}), error_mode="raise")
    assert "role=clerk;fw=crewai;task=chores" in _observations(llm)


def test_approve_is_validated(fetcher):
    with pytest.raises(ValueError, match="approve"):
        CrewAIAdapter(fetcher, approve="always")  # type: ignore[arg-type]
