"""Haystack + dotfetch integration example (haystack-ai >= 3.0).

Runs offline: a small scripted chat generator stands in for an LLM and issues
the tool calls, so Haystack's real ``Agent`` loop does the rest. It shows:

1. ``HaystackAdapter.agent()`` — dotfetch re-selects tools after every tool
   step: looking up a customer surfaces the email tool for the next step.
2. A failing tool — the Agent records an error tool result
   (``"<type>: <message>"``) and the model gets another step.
3. A confirmation-gated tool — Haystack's ``ConfirmationHook`` asks a
   ``ConfirmationUI`` (here a scripted one; ``SimpleConsoleUI()`` asks in the
   terminal) about the gated call only, and it runs once when confirmed.
4. Run context — a dotfetch tool reads the Agent's ``State`` through
   ``current_agent_state()``.

Set ``OPENAI_API_KEY`` to also run one turn on a real model.

Install:

    pip install 'dotfetch[haystack]'

Run:

    python examples/agent_frameworks/haystack_example.py
"""

from __future__ import annotations

import asyncio
import os
from typing import Any

from dotfetch import DotFetch
from dotfetch.adapters.embedders.simple import SimpleEmbedder


def build_fetcher() -> DotFetch:
    from dotfetch.haystack import current_agent_state

    fetcher = DotFetch(embedder=SimpleEmbedder(dimension=256))

    @fetcher.register(examples=["look up the customer record", "find customer details"])
    def lookup_customer(name: str) -> str:
        """Look up a customer's record by name."""
        return f"{name} prefers to be contacted by email message"

    @fetcher.register(examples=["how much is 50 euros in pounds", "convert 100 dollars to yen"])
    def convert_currency(amount: float, frm: str, to: str) -> str:
        """Convert an amount of money from one currency to another."""
        if frm == to:
            raise ValueError("pick two different currencies")
        return f"{amount} {frm} = {amount * 1.1:.2f} {to}"

    @fetcher.register(
        requires_confirmation=True,
        examples=["email the team about the delay", "send an email message to Sam"],
    )
    def send_email(to: str, body: str = "") -> str:
        """Compose and send an email message to a recipient."""
        return f"Email sent to {to}."

    @fetcher.register(examples=["who am I", "which account is this"])
    def whoami() -> str:
        """Say which user this run belongs to."""
        return f"user {current_agent_state().get('user')}"

    return fetcher


def scripted_generator(*steps: list[tuple[str, dict[str, Any]]]) -> Any:
    """A tiny offline chat generator: issues the scripted tool calls, then answers."""
    from haystack import component
    from haystack.dataclasses import ChatMessage, ToolCall

    @component
    class ScriptedGenerator:
        def __init__(self) -> None:
            self.steps = list(steps)
            self.offered: list[list[str]] = []

        @component.output_types(replies=list[ChatMessage])
        def run(self, messages: list[ChatMessage], tools: Any = None, **kwargs: Any) -> dict:
            self.offered.append(sorted(tool.name for tool in tools or []))
            step = self.steps.pop(0) if self.steps else None
            if not step:
                return {"replies": [ChatMessage.from_assistant("All done.")]}
            calls = [
                ToolCall(tool_name=name, arguments=args, id=f"call_{len(self.offered)}_{i}")
                for i, (name, args) in enumerate(step)
            ]
            return {"replies": [ChatMessage.from_assistant(tool_calls=calls)]}

    return ScriptedGenerator()


class ApproveEverything:
    """A ConfirmationUI that approves and prints what it was asked."""

    def get_user_confirmation(self, tool_name: str, tool_description: str, tool_params: dict):
        from haystack.hooks.human_in_the_loop import ConfirmationUIResult

        print(f"[approve] asked about {tool_name}({tool_params}) -> confirm")
        return ConfirmationUIResult(action="confirm")

    def to_dict(self) -> dict[str, Any]:
        return {}


def tool_results(result: dict) -> list[Any]:
    return [r for message in result["messages"] for r in message.tool_call_results or []]


async def main() -> None:
    try:
        from haystack.dataclasses import ChatMessage
        from haystack.hooks.protocol import VALID_HOOK_POINTS  # noqa: F401 - haystack >= 3.0
    except ImportError as exc:
        print(
            f"Haystack 3.x is not installed ({exc}).\n"
            "Install it with `pip install 'dotfetch[haystack]'`."
        )
        return

    from dotfetch.haystack import HaystackAdapter

    fetcher = build_fetcher()
    try:
        await fetcher.sync()
        adapter = HaystackAdapter(fetcher)

        # --- 1. Tools re-selected after every tool step --------------------- #
        generator = scripted_generator([("lookup_customer", {"name": "Jo"})])
        agent = adapter.agent(chat_generator=generator, limit=1, sticky=False)
        agent.run(messages=[ChatMessage.from_user("look up the customer record for Jo")])
        print(f"[per-step] tools offered per step: {generator.offered}")

        # --- 2. A failing tool: the model sees it and carries on ------------- #
        generator = scripted_generator(
            [("convert_currency", {"amount": 5, "frm": "EUR", "to": "EUR"})]
        )
        result = adapter.agent(chat_generator=generator, limit=1).run(
            messages=[ChatMessage.from_user("convert 5 euros to euros")]
        )
        [failed] = tool_results(result)
        print(f"[error]    error={failed.error}, model saw: {failed.result!r}")

        # --- 3. A gated tool is confirmed before it runs -------------------- #
        generator = scripted_generator([("send_email", {"to": "sam@example.com"})])
        agent = adapter.agent(
            chat_generator=generator, limit=1, confirmation_ui=ApproveEverything()
        )
        result = await agent.run_async(
            messages=[ChatMessage.from_user("send an email message to Sam")]
        )
        print(f"[approve]  after confirming: {tool_results(result)[0].result!r}")

        # --- 4. The Agent's State reaches the dotfetch tool ------------------- #
        generator = scripted_generator([("whoami", {})])
        agent = adapter.agent(
            chat_generator=generator, limit=1, state_schema={"user": {"type": str}}
        )
        result = agent.run(messages=[ChatMessage.from_user("who am I")], user="dan")
        print(f"[context]  {tool_results(result)[0].result!r}")

        # --- Optional: one turn on a real model ----------------------------- #
        if os.environ.get("OPENAI_API_KEY"):
            from haystack.components.generators.chat import OpenAIChatGenerator

            live = adapter.agent(chat_generator=OpenAIChatGenerator(model="gpt-5.5"), limit=2)
            result = live.run(messages=[ChatMessage.from_user("Convert 50 EUR to GBP.")])
            print(f"[live]     {result['last_message'].text}")
        else:
            print("(Set OPENAI_API_KEY to also run a turn on a real model.)")
    finally:
        await fetcher.close()


if __name__ == "__main__":
    asyncio.run(main())
