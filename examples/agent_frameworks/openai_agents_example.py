"""OpenAI Agents SDK + Agent-Gantry integration example.

Walks through each way Gantry plugs into the OpenAI Agents SDK:

1. **Static tools** -- ``adapter.select(query)`` selects once and returns native
   ``agents.FunctionTool`` objects for ``Agent(tools=[...])``.
2. **Per-turn selection** -- ``adapter.dynamic_tools()`` gives the agent the
   catalogue gated by ``FunctionTool.is_enabled``. After the CRM lookup returns,
   the email tool appears mid-run because the next selection reads the output.
3. **Readable errors** -- a failing tool returns ``Error: ...`` to the model
   instead of ending the run.
4. **Approval** -- a ``requires_confirmation`` tool pauses the run with
   ``result.interruptions``; approving it executes it exactly once.
5. **Run context** -- a Gantry tool reads ``Runner.run(context=...)`` through
   ``current_run_context()``.

Everything runs **offline**: the SDK's own ``agents.testing.ScriptedModel``
scripts the model's tool calls, so no API key is needed. Swap in a real model
(drop ``model=...``) to use it for real.

Install (in its own environment; it cannot share one with CrewAI):

    pip install 'agent-gantry[openai-agents]'

Run:

    python examples/agent_frameworks/openai_agents_example.py
"""

from __future__ import annotations

import asyncio
import os

from agent_gantry import AgentGantry
from agent_gantry.adapters.embedders.simple import SimpleEmbedder

os.environ.setdefault("OPENAI_AGENTS_DISABLE_TRACING", "1")


def build_gantry() -> AgentGantry:
    gantry = AgentGantry(embedder=SimpleEmbedder(dimension=256))

    @gantry.register(tags=["weather"], examples=["what's the weather in London"])
    def get_weather(city: str) -> str:
        """Get the current weather forecast for a city."""
        return f"It is 21C and sunny in {city}."

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
        from agent_gantry.openai_agents import current_run_context

        ctx = current_run_context()
        return f"You are signed in as {ctx.context['user']}." if ctx else "Unknown user."

    return gantry


def scripted(*steps):
    """A ScriptedModel issuing ``steps`` of tool calls, then a final answer."""
    from agents.testing import ScriptedModel, assistant_message, function_call

    turns = [
        [function_call(name, args, call_id=f"call_{i}_{j}") for j, (name, args) in enumerate(step)]
        for i, step in enumerate(steps)
    ]
    return ScriptedModel([*turns, [assistant_message("done")]])


def outputs(result) -> list[str]:
    return [i.output for i in result.new_items if type(i).__name__ == "ToolCallOutputItem"]


async def main() -> None:
    gantry = build_gantry()
    try:
        await gantry.sync()
        try:
            from agents import Agent, Runner
            from agents.testing import ScriptedModel  # noqa: F401  (SDK >= 0.21)
        except ImportError as exc:
            print(
                f"The OpenAI Agents SDK tools demo needs openai-agents>=0.23.1 ({exc}).\n"
                "Install it with `pip install 'agent-gantry[openai-agents]'`."
            )
            return

        adapter = gantry.framework("openai_agents")

        # 1. Static tools ---------------------------------------------------- #
        tools = await adapter.select("what's the weather in Paris?", limit=1)
        print(f"[static] selected tool(s): {[t.name for t in tools]}")
        result = await Runner.run(
            Agent(name="a", model=scripted([("get_weather", {"city": "Paris"})]), tools=tools),
            "what's the weather in Paris?",
        )
        print(f"[static] {outputs(result)[0]}\n")

        # 2. Per-turn selection ---------------------------------------------- #
        dynamic = await adapter.dynamic_tools(limit=1)
        model = scripted([("lookup_customer", {"name": "Jo"})])
        await Runner.run(
            Agent(name="a", model=model, tools=dynamic), "find the customer record for Jo"
        )
        offered = [[t.name for t in call.tools] for call in model.calls]
        print(f"[per-turn] tools offered at turn 1: {offered[0]}")
        print(f"[per-turn] tools offered at turn 2: {offered[1]}  <- after the lookup output\n")

        # 3. Readable errors --------------------------------------------------- #
        model = scripted([("divide", {"a": 1, "b": 0})])
        result = await Runner.run(Agent(name="a", model=model, tools=dynamic), "divide 1 by 0")
        print(f"[errors] the model sees: {outputs(result)[0]!r}\n")

        # 4. Approval ---------------------------------------------------------- #
        agent = Agent(
            name="a",
            model=scripted([("send_email", {"to": "jo@example.com"})]),
            tools=await adapter.select("send an email message", limit=1),
        )
        paused = await Runner.run(agent, "send an email message to Jo")
        print(f"[approval] run paused for: {[i.name for i in paused.interruptions]}")
        state = paused.to_state()
        state.approve(paused.interruptions[0])
        resumed = await Runner.run(agent, state)
        print(f"[approval] approved -> {outputs(resumed)[-1]}\n")

        # 5. Run context ------------------------------------------------------- #
        tools = await adapter.select("who am I signed in as", limit=1)
        result = await Runner.run(
            Agent(name="a", model=scripted([("whoami", {})]), tools=tools),
            "who am I?",
            context={"user": "Dan"},
        )
        print(f"[context] {outputs(result)[0]}")

        print(
            "\nEvery tool call above went through gantry.execute, so the security "
            "policy, retries, timeouts and circuit breakers all applied."
        )
    finally:
        await gantry.close()


if __name__ == "__main__":
    asyncio.run(main())
