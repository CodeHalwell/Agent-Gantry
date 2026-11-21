"""CrewAI + Agent-Gantry, through CrewAI's own mechanisms.

1. **Selected tools** -- ``adapter.select(query)`` hands an agent only the
   relevant slice of the catalogue, as native ``BaseTool``s.
2. **Readable errors** -- a failing tool returns CrewAI's ``ToolFailure``: the
   agent reads it and CrewAI records it on ``TaskOutput.tool_failures``.
3. **Approval** -- a ``requires_confirmation`` tool goes through the adapter's
   approver (CrewAI ``before_tool_call`` hook); a refused call is blocked.
4. **Run context** -- a Gantry tool reads the CrewAI agent and task through
   ``current_crew_context()``.

Everything runs **offline**: a scripted LLM answers in CrewAI's ReAct format,
so no API key is needed. Pass ``llm="openai/gpt-5.5"`` (with ``OPENAI_API_KEY``)
to use a real model.

Install and run (Python 3.10-3.13)::

    pip install 'agent-gantry[crewai]'
    python examples/agent_frameworks/crewai_example.py
"""

from __future__ import annotations

import asyncio
import json
import os
import re
from typing import Any

from agent_gantry import AgentGantry
from agent_gantry.adapters.embedders.simple import SimpleEmbedder

os.environ.setdefault("CREWAI_DISABLE_TELEMETRY", "true")
os.environ.setdefault("CREWAI_TRACING_ENABLED", "false")


def build_gantry() -> AgentGantry:
    gantry = AgentGantry(embedder=SimpleEmbedder(dimension=256))

    @gantry.register(tags=["math"], examples=["divide 10 by 2"])
    def divide(a: int, b: int) -> float:
        """Divide integer a by integer b."""
        return a / b

    @gantry.register(tags=["email"], examples=["email the team"], requires_confirmation=True)
    def send_email(to: str) -> str:
        """Send an email message to a recipient."""
        return f"Email sent to {to}."

    @gantry.register(tags=["identity"], examples=["who is doing this task"])
    def whoami() -> str:
        """Report which agent and task are making this call."""
        from agent_gantry.crewai import current_crew_context

        ctx = current_crew_context()
        return f"Agent {ctx.agent.role!r} on task {ctx.task.name!r}." if ctx else "Unknown."

    return gantry


def scripted(*calls: tuple[str, dict[str, Any]]) -> Any:
    """A CrewAI LLM that makes ``calls`` (ReAct format), then answers."""
    from crewai.llms.base_llm import BaseLLM

    class Scripted(BaseLLM):
        script: list[str] = []
        seen: list[str] = []

        def call(self, messages: Any, *args: Any, **kwargs: Any) -> str:
            self.seen.append(json.dumps(messages, default=str))
            return self.script.pop(0) if self.script else "Thought: done\nFinal Answer: done"

        def supports_function_calling(self) -> bool:
            return False

        def supports_stop_words(self) -> bool:
            return True

        def get_context_window_size(self) -> int:
            return 8000

    script = [
        f"Thought: use {name}\nAction: {name}\nAction Input: {json.dumps(args)}"
        for name, args in calls
    ]
    return Scripted(model="scripted", script=script, seen=[])


async def run(adapter: Any, query: str, *calls: tuple[str, dict[str, Any]]) -> tuple[Any, Any]:
    from crewai import Agent, Crew, Task

    llm = scripted(*calls)
    tools = await adapter.select(query, limit=1)
    agent = Agent(role="clerk", goal="help", backstory="Uses Gantry tools.", llm=llm, tools=tools)
    task = Task(name="chores", description=query, expected_output="done", agent=agent)
    output = await Crew(agents=[agent], tasks=[task]).kickoff_async()
    return output, llm


async def main() -> None:
    gantry = build_gantry()
    try:
        await gantry.sync()
        try:
            from agent_gantry.crewai import CrewAIAdapter
        except ImportError as exc:  # pragma: no cover
            print(f"Agent-Gantry import failed: {exc}")
            return
        try:
            import crewai  # noqa: F401
        except ImportError as exc:
            print(
                f"The CrewAI demo needs crewai ({exc}).\n"
                "Install it with `pip install 'agent-gantry[crewai]'` (Python 3.10-3.13)."
            )
            return

        approvals: list[str] = []

        def approve(request: Any) -> bool:
            approvals.append(f"{request.tool_name}({request.arguments})")
            return True  # a real app would ask a person here

        adapter = CrewAIAdapter(gantry, approve=approve)

        # 1 + 2. Selected tools and readable errors ------------------------- #
        output, _ = await run(adapter, "divide 1 by 0", ("divide", {"a": 1, "b": 0}))
        failure = output.tasks_output[0].tool_failures[0].failure
        print(f"[errors] ToolFailure code={failure.code!r} message={failure.message!r}\n")

        # 3. Approval ------------------------------------------------------ #
        await run(adapter, "send an email message to Jo", ("send_email", {"to": "jo@example.com"}))
        print(f"[approval] approver was asked about: {approvals}\n")

        # 4. Run context --------------------------------------------------- #
        _, llm = await run(adapter, "who is doing this task", ("whoami", {}))
        seen = re.search(r"Agent .clerk. on task .chores.\.", llm.seen[-1])
        print(f"[context] the tool replied: {seen.group(0) if seen else '(not found)'}")

        print(
            "\nEvery tool call above went through gantry.execute, so the security "
            "policy, retries, timeouts and circuit breakers all applied."
        )
    finally:
        await gantry.close()


if __name__ == "__main__":
    asyncio.run(main())
