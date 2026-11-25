"""Strands Agents + dotfetch, through Strands' own mechanisms.

1. **Per-turn selection** -- ``adapter.agent(model=...)`` builds a Strands
   ``Agent`` whose ``BeforeModelCallEvent`` hook re-selects dotfetch tools for the
   latest user message before every model call.
2. **Readable errors** -- a failing tool returns an error ``ToolResult`` the
   model can react to.
3. **Approval** -- a ``requires_confirmation`` tool raises a Strands interrupt;
   resuming with "yes" runs it exactly once.
4. **Run context** -- a dotfetch tool reads the keyword arguments passed to
   ``agent(...)`` through ``current_tool_context()``.

Everything runs **offline**: a scripted ``Model`` streams the tool calls, so no
AWS credentials are needed. Drop ``model=`` (Bedrock is Strands' default) or
pass any Strands model to use it for real.

Install and run::

    pip install 'dotfetch[strands]'
    python examples/agent_frameworks/strands_example.py
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

from dotfetch import DotFetch
from dotfetch.adapters.embedders.simple import SimpleEmbedder


def build_fetcher() -> DotFetch:
    fetcher = DotFetch(embedder=SimpleEmbedder(dimension=256))

    @fetcher.register(tags=["crm"], examples=["find the customer record for Jo"])
    def lookup_customer(name: str) -> str:
        """Look up a customer record by name."""
        return f"Customer {name} found. Next step: send an email message to jo@example.com"

    @fetcher.register(tags=["email"], examples=["email the team"], requires_confirmation=True)
    def send_email(to: str) -> str:
        """Send an email message to a recipient."""
        return f"Email sent to {to}."

    @fetcher.register(tags=["math"], examples=["divide 10 by 2"])
    def divide(a: int, b: int) -> float:
        """Divide integer a by integer b."""
        return a / b

    @fetcher.register(tags=["identity"], examples=["who am I signed in as"])
    def whoami() -> str:
        """Return the name of the signed-in user."""
        from dotfetch.strands import current_tool_context

        ctx = current_tool_context()
        return f"You are signed in as {ctx.invocation_state['user']}." if ctx else "Unknown user."

    return fetcher


def scripted(*steps: list[tuple[str, dict[str, Any]]]) -> Any:
    """A Strands model that streams ``steps`` of tool calls, then answers."""
    from strands.models.model import Model

    class Scripted(Model):
        def __init__(self) -> None:
            self.steps = list(steps)
            self.offered: list[list[str]] = []

        def update_config(self, **kwargs: Any) -> None:
            pass

        def get_config(self) -> dict[str, Any]:
            return {}

        async def structured_output(self, *args: Any, **kwargs: Any) -> Any:
            raise NotImplementedError

        async def stream(self, messages: Any, tool_specs: Any = None, *args: Any, **kwargs: Any):
            self.offered.append(sorted(spec["name"] for spec in tool_specs or []))
            step = self.steps.pop(0) if self.steps else None
            yield {"messageStart": {"role": "assistant"}}
            for i, (name, args_) in enumerate(step or []):
                use = {"toolUseId": f"t{len(self.offered)}_{i}", "name": name}
                yield {"contentBlockStart": {"start": {"toolUse": use}}}
                yield {"contentBlockDelta": {"delta": {"toolUse": {"input": json.dumps(args_)}}}}
                yield {"contentBlockStop": {}}
            if not step:
                yield {"contentBlockStart": {"start": {}}}
                yield {"contentBlockDelta": {"delta": {"text": "done"}}}
                yield {"contentBlockStop": {}}
            yield {"messageStop": {"stopReason": "tool_use" if step else "end_turn"}}

    return Scripted()


def results(agent: Any) -> list[dict[str, Any]]:
    return [b["toolResult"] for m in agent.messages for b in m["content"] if "toolResult" in b]


async def main() -> None:
    fetcher = build_fetcher()
    try:
        await fetcher.sync()
        try:
            from strands import Agent
        except ImportError as exc:
            print(
                f"The Strands demo needs strands-agents ({exc}).\n"
                "Install it with `pip install 'dotfetch[strands]'`."
            )
            return

        from dotfetch.strands import StrandsAdapter

        adapter = StrandsAdapter(fetcher)

        # 1. Per-turn selection ------------------------------------------- #
        model = scripted([("lookup_customer", {"name": "Jo"})])
        agent = adapter.agent(model=model, limit=1, callback_handler=None)
        await agent.invoke_async("find the customer record for Jo")
        print(f"[per-turn] tools selected before each model call: {model.offered}\n")

        # 2. Readable errors ---------------------------------------------- #
        tools = await adapter.select("divide 10 by 2", limit=1)
        agent = Agent(
            model=scripted([("divide", {"a": 1, "b": 0})]), tools=tools, callback_handler=None
        )
        await agent.invoke_async("divide 1 by 0")
        error = results(agent)[0]
        print(f"[errors] status={error['status']} text={error['content'][0]['text']!r}\n")

        # 3. Approval ----------------------------------------------------- #
        tools = await adapter.select("send an email message", limit=1)
        agent = Agent(
            model=scripted([("send_email", {"to": "jo@example.com"})]),
            tools=tools,
            callback_handler=None,
        )
        paused = await agent.invoke_async("email Jo")
        [interrupt] = paused.interrupts
        print(f"[approval] stop_reason={paused.stop_reason} for {interrupt.reason['tool']}")
        await agent.invoke_async(
            [{"interruptResponse": {"interruptId": interrupt.id, "response": "yes"}}]
        )
        print(f"[approval] approved -> {results(agent)[-1]['content'][0]['text']}\n")

        # 4. Run context -------------------------------------------------- #
        tools = await adapter.select("who am I signed in as", limit=1)
        agent = Agent(model=scripted([("whoami", {})]), tools=tools, callback_handler=None)
        await agent.invoke_async("who am I?", user="Dan")
        print(f"[context] {results(agent)[0]['content'][0]['text']}")

        print(
            "\nEvery tool call above went through fetcher.execute, so the security "
            "policy, retries, timeouts and circuit breakers all applied."
        )
    finally:
        await fetcher.close()


if __name__ == "__main__":
    asyncio.run(main())
