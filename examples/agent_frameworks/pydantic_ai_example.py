"""Pydantic AI + dotfetch integration example.

Walks through each way dotfetch plugs into Pydantic AI:

1. **Static tools** -- ``adapter.select(query)`` selects once and returns native
   ``pydantic_ai.Tool`` objects for ``Agent(model, tools=[...])``.
2. **Per-turn capability** -- ``adapter.capability()`` re-selects tools on every
   model step. After the CRM lookup returns, the email tool appears mid-run
   because the next selection reads the tool result.
3. **Approval** -- a ``requires_confirmation`` tool pauses the run with Pydantic
   AI's ``DeferredToolRequests``; approving it via ``DeferredToolResults``
   executes it exactly once through ``fetcher.execute``.
4. **Run context** -- a dotfetch tool reads ``RunContext.deps`` with
   ``current_run_context()``.
5. **Tool search** -- ``adapter.capability(mode="search")`` hides the catalogue
   behind Pydantic AI's ``search_tools``, backed by dotfetch's semantic router.

Everything runs **offline**: ``FunctionModel`` scripts the model's tool calls,
so no API key is needed. Swap in a real model string (``"openai:gpt-5.5"``,
``"anthropic:claude-sonnet-4-6"``, ...) and drop the scripts to use it for real.

Install:

    pip install 'dotfetch[pydantic-ai]'

Run:

    python examples/agent_frameworks/pydantic_ai_example.py
"""

from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass

from dotfetch import DotFetch
from dotfetch.adapters.embedders.simple import SimpleEmbedder

os.environ.setdefault("PYDANTIC_AI_NO_BANNER", "1")


@dataclass
class Deps:
    """Dependencies Pydantic AI injects into a run (``Agent(deps_type=Deps)``)."""

    user: str


def build_fetcher() -> DotFetch:
    fetcher = DotFetch(embedder=SimpleEmbedder(dimension=256))

    @fetcher.register(tags=["weather"], examples=["what's the weather in London"])
    def get_weather(city: str) -> str:
        """Get the current weather forecast for a city."""
        return f"It is 21C and sunny in {city}."

    @fetcher.register(tags=["crm"], examples=["find the customer record for Jo"])
    def lookup_customer(name: str) -> str:
        """Look up a customer record by name."""
        return f"Customer {name} found. Next step: send an email message to jo@example.com"

    @fetcher.register(
        tags=["email"],
        examples=["email the team about the delay"],
        requires_confirmation=True,
    )
    def send_email(to: str, body: str = "") -> str:
        """Send an email message to a recipient."""
        return f"Email sent to {to}."

    @fetcher.register(tags=["identity"], examples=["who am I signed in as"])
    def whoami() -> str:
        """Return the name of the signed-in user."""
        from dotfetch.pydantic_ai import current_run_context

        ctx = current_run_context()
        return f"You are signed in as {ctx.deps.user}." if ctx else "Unknown user."

    return fetcher


def scripted(*steps):
    """A FunctionModel that issues ``steps`` of tool calls, then echoes the last result.

    ``model.seen`` records which tools the model was offered at each step.
    """
    from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart
    from pydantic_ai.models.function import FunctionModel

    queue = list(steps)

    def respond(messages, info):
        model.seen.append(sorted(t.name for t in info.function_tools if not t.defer_loading))
        if queue:
            return ModelResponse(parts=[ToolCallPart(n, a) for n, a in queue.pop(0)])
        last = messages[-1].parts[0]
        return ModelResponse(parts=[TextPart(str(getattr(last, "content", last)))])

    model = FunctionModel(respond)
    model.seen = []
    return model


async def main() -> None:
    fetcher = build_fetcher()
    try:
        await fetcher.sync()

        try:
            from pydantic_ai import Agent, DeferredToolRequests, DeferredToolResults
        except ImportError as exc:
            print(
                f"Pydantic AI is not installed ({exc}).\n"
                "Install it with `pip install 'dotfetch[pydantic-ai]'` to run this example."
            )
            return

        adapter = fetcher.framework("pydantic_ai")

        # 1. Static tools: select once ------------------------------------- #
        tools = await adapter.select("what's the weather in Paris?", limit=1)
        print(f"[static] selected tool(s): {[t.name for t in tools]}")
        result = await Agent(scripted([("get_weather", {"city": "Paris"})]), tools=tools).run(
            "what's the weather in Paris?"
        )
        print(f"[static] {result.output}\n")

        # 2. Per-turn capability: the tool surface follows the run ---------- #
        model = scripted([("lookup_customer", {"name": "Jo"})])
        agent = Agent(model, capabilities=[adapter.capability(limit=1)])
        await agent.run("find the customer record for Jo")
        print(f"[per-turn] tools offered at step 1: {model.seen[0]}")
        print(f"[per-turn] tools offered at step 2: {model.seen[1]}  <- after the lookup result\n")

        # 3. Approval: dotfetch's confirmation gate -> DeferredToolRequests ---- #
        agent = Agent(
            scripted([("send_email", {"to": "jo@example.com"})]),
            output_type=[str, DeferredToolRequests],
            toolsets=[adapter.toolset(limit=1)],
        )
        paused = await agent.run("send an email message to Jo")
        assert isinstance(paused.output, DeferredToolRequests)
        pending = paused.output.approvals[0]
        print(f"[approval] run paused; awaiting approval for tool {pending.tool_name!r}")
        resumed = await agent.run(
            message_history=paused.all_messages(),
            deferred_tool_results=DeferredToolResults(approvals={pending.tool_call_id: True}),
        )
        print(f"[approval] approved -> {resumed.output}\n")

        # 4. Run context: deps reach the dotfetch handler -------------------- #
        agent = Agent(
            scripted([("whoami", {})]),
            deps_type=Deps,
            toolsets=[adapter.toolset(limit=1)],
        )
        result = await agent.run("who am I signed in as?", deps=Deps(user="Dan"))
        print(f"[deps] {result.output}\n")

        # 5. Tool search: dotfetch ranks what search_tools returns ------------ #
        model = scripted(
            [("search_tools", {"queries": ["weather forecast"]})],
            [("get_weather", {"city": "Leeds"})],
        )
        agent = Agent(model, capabilities=[adapter.capability(mode="search", limit=1)])
        result = await agent.run("what's the weather in Leeds?")
        found = [
            part.content
            for message in result.all_messages()
            for part in message.parts
            if getattr(part, "tool_name", None) == "search_tools"
            and getattr(part, "part_kind", None) == "tool-return"
        ]
        print(f"[search] tools visible at step 1: {model.seen[0]}")
        print(f"[search] search_tools returned: {found[0] if found else None}")
        print(f"[search] {result.output}")

        print(
            "\nEvery tool call above went through fetcher.execute, so the security "
            "policy, retries, timeouts and circuit breakers all applied."
        )
    finally:
        await fetcher.close()


if __name__ == "__main__":
    asyncio.run(main())
