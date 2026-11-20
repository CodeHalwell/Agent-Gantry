"""LangGraph + Agent-Gantry, through LangGraph's own mechanisms.

1. **Per-turn selection** -- ``adapter.middleware()`` is a ``create_agent``
   middleware that re-selects the Gantry tools offered on every model turn.
   After the CRM lookup returns, the email tool appears because the next
   selection reads the lookup's output.
2. **Readable errors** -- a failing tool comes back as an error ``ToolMessage``
   the model can react to, instead of ending the run.
3. **Approval** -- a ``requires_confirmation`` tool pauses the graph with
   ``interrupt()``; ``Command(resume=True)`` runs it exactly once.
4. **Run context** -- a Gantry tool reads the graph's ``context=`` through
   ``current_tool_runtime()``.

Everything runs **offline**: a scripted chat model stands in for a real one, so
no API key is needed. Swap in ``init_chat_model("openai:gpt-5.5")`` (or any
LangChain chat model) to use it for real.

Install and run::

    pip install 'agent-gantry[langgraph]'
    python examples/agent_frameworks/langgraph_example.py
"""

from __future__ import annotations

import asyncio
from typing import Any

from agent_gantry import AgentGantry
from agent_gantry.adapters.embedders.simple import SimpleEmbedder


def build_gantry() -> AgentGantry:
    gantry = AgentGantry(embedder=SimpleEmbedder(dimension=256))

    @gantry.register(tags=["crm"], examples=["find the customer record for Jo"])
    def lookup_customer(name: str) -> str:
        """Look up a customer record by name."""
        return f"Customer {name} found. Next step: send an email message to jo@example.com"

    @gantry.register(tags=["email"], examples=["email the team"], requires_confirmation=True)
    def send_email(to: str) -> str:
        """Send an email message to a recipient."""
        return f"Email sent to {to}."

    @gantry.register(tags=["math"], examples=["divide 10 by 2"])
    def divide(a: int, b: int) -> float:
        """Divide integer a by integer b."""
        return a / b

    @gantry.register(tags=["identity"], examples=["who am I signed in as"])
    def whoami() -> str:
        """Return the name of the signed-in user."""
        from agent_gantry.langgraph import current_tool_runtime

        runtime = current_tool_runtime()
        return f"You are signed in as {runtime.context['user']}." if runtime else "Unknown user."

    return gantry


def scripted(*steps: list[tuple[str, dict[str, Any]]]) -> Any:
    """A fake chat model that issues ``steps`` of tool calls, then answers."""
    from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
    from langchain_core.messages import AIMessage

    class Scripted(GenericFakeChatModel):
        offered: list[list[str]] = []

        def bind_tools(self, tools: Any, **kwargs: Any) -> Any:
            self.offered.append(sorted(t.name for t in tools))
            return self

    replies = [
        AIMessage(
            content="",
            tool_calls=[
                {"name": n, "args": a, "id": f"call_{i}_{j}"} for j, (n, a) in enumerate(step)
            ],
        )
        for i, step in enumerate(steps)
    ]
    model = Scripted(messages=iter([*replies, AIMessage(content="done")]))
    model.offered = []
    return model


def tool_outputs(result: dict[str, Any]) -> list[str]:
    return [m.content for m in result["messages"] if m.type == "tool"]


async def main() -> None:
    gantry = build_gantry()
    try:
        await gantry.sync()
        try:
            from langchain.agents import create_agent
            from langgraph.checkpoint.memory import InMemorySaver
            from langgraph.types import Command
        except ImportError as exc:
            print(
                f"The LangGraph demo needs langchain and langgraph ({exc}).\n"
                "Install them with `pip install 'agent-gantry[langgraph]'`."
            )
            return

        from agent_gantry.langgraph import LangGraphAdapter

        adapter = LangGraphAdapter(gantry)

        # 1. Per-turn selection ------------------------------------------- #
        model = scripted([("lookup_customer", {"name": "Jo"})])
        agent = create_agent(model, middleware=[await adapter.middleware(limit=1)])
        await agent.ainvoke({"messages": [("user", "find the customer record for Jo")]})
        print(f"[per-turn] tools offered at turn 1: {model.offered[0]}")
        print(f"[per-turn] tools offered at turn 2: {model.offered[1]}  <- after the lookup\n")

        # 2. Readable errors ---------------------------------------------- #
        agent = create_agent(
            scripted([("divide", {"a": 1, "b": 0})]),
            middleware=[await adapter.middleware(limit=1)],
        )
        result = await agent.ainvoke({"messages": [("user", "divide 1 by 0")]})
        print(f"[errors] the model sees: {tool_outputs(result)[0]!r}\n")

        # 3. Approval ----------------------------------------------------- #
        agent = create_agent(
            scripted([("send_email", {"to": "jo@example.com"})]),
            tools=await adapter.select("send an email message", limit=1),
            checkpointer=InMemorySaver(),
        )
        config = {"configurable": {"thread_id": "demo"}}
        paused = await agent.ainvoke({"messages": [("user", "email Jo")]}, config)
        request = paused["__interrupt__"][0].value
        print(f"[approval] paused for {request['tool']} {request['arguments']}")
        resumed = await agent.ainvoke(Command(resume=True), config)
        print(f"[approval] approved -> {tool_outputs(resumed)[-1]}\n")

        # 4. Run context -------------------------------------------------- #
        agent = create_agent(
            scripted([("whoami", {})]),
            tools=await adapter.select("who am I signed in as", limit=1),
            context_schema=dict,
        )
        result = await agent.ainvoke({"messages": [("user", "who am I?")]}, context={"user": "Dan"})
        print(f"[context] {tool_outputs(result)[0]}")

        print(
            "\nEvery tool call above went through gantry.execute, so the security "
            "policy, retries, timeouts and circuit breakers all applied."
        )
    finally:
        await gantry.close()


if __name__ == "__main__":
    asyncio.run(main())
