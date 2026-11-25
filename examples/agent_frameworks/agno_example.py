"""Agno (formerly Phidata) + dotfetch integration example.

Runs offline: a small scripted Agno ``Model`` stands in for an LLM and issues
the tool calls, so Agno's real run loop does the rest. It shows:

1. ``AgnoAdapter.agent()`` — one Agno ``Agent`` whose tools are a callable
   factory, so Agno asks dotfetch for each run's tools (two runs, two different
   tool slices).
2. A failing tool — Agno records the failed result (``"<type>: <message>"``)
   and the model gets another turn.
3. A confirmation-gated tool — the run pauses (``run.is_paused``); confirming
   the requirement and calling ``continue_run`` runs it exactly once.
4. Run context — a dotfetch tool reads Agno's ``RunContext`` (``user_id``,
   ``dependencies``) through ``current_run_context()``.

Set ``OPENAI_API_KEY`` to also run one turn on a real model.

Install:

    pip install 'dotfetch[agno]'

Run:

    python examples/agent_frameworks/agno_example.py
"""

from __future__ import annotations

import asyncio
import json
import os
from dataclasses import dataclass, field
from typing import Any

from dotfetch import DotFetch
from dotfetch.adapters.embedders.simple import SimpleEmbedder


def build_fetcher() -> DotFetch:
    from dotfetch.agno import current_run_context

    fetcher = DotFetch(embedder=SimpleEmbedder(dimension=256))

    @fetcher.register(examples=["what's the weather in London", "is it raining in Leeds"])
    def get_weather(city: str) -> str:
        """Get the current weather for a city."""
        return f"It is 21C and sunny in {city}."

    @fetcher.register(examples=["how much is 50 euros in pounds", "convert 100 dollars to yen"])
    def convert_currency(amount: float, frm: str, to: str) -> str:
        """Convert an amount of money from one currency to another."""
        if frm == to:
            raise ValueError("pick two different currencies")
        return f"{amount} {frm} = {amount * 1.1:.2f} {to}"

    @fetcher.register(
        requires_confirmation=True,
        examples=["email the team about the delay", "send a message to Sam"],
    )
    def send_email(to: str, body: str = "") -> str:
        """Compose and send an email message to a recipient."""
        return f"Email sent to {to}."

    @fetcher.register(examples=["who am I", "which account is this"])
    def whoami() -> str:
        """Say which user this run belongs to."""
        ctx = current_run_context()
        return f"user {ctx.user_id} on the {ctx.dependencies['plan']} plan"

    return fetcher


def scripted_model(*steps: list[tuple[str, dict[str, Any]]]) -> Any:
    """A tiny offline Agno model: issues the scripted tool calls, then answers."""
    from agno.models.base import Model
    from agno.models.response import ModelResponse

    @dataclass
    class ScriptedModel(Model):
        id: str = "scripted"
        name: str = "scripted"
        provider: str = "scripted"
        steps: list[Any] = field(default_factory=list)
        offered: list[list[str]] = field(default_factory=list)

        def _reply(self, tools: Any) -> ModelResponse:
            self.offered.append(sorted((t.get("function") or t)["name"] for t in tools or []))
            step = self.steps.pop(0) if self.steps else None
            if not step:
                return ModelResponse(role="assistant", content="All done.")
            calls = [
                {
                    "id": f"call_{len(self.offered)}_{i}",
                    "type": "function",
                    "function": {"name": name, "arguments": json.dumps(args)},
                }
                for i, (name, args) in enumerate(step)
            ]
            return ModelResponse(role="assistant", content="", tool_calls=calls)

        def invoke(self, *args: Any, tools: Any = None, **kwargs: Any) -> ModelResponse:
            return self._reply(tools)

        async def ainvoke(self, *args: Any, tools: Any = None, **kwargs: Any) -> ModelResponse:
            return self._reply(tools)

        def invoke_stream(self, *args: Any, **kwargs: Any) -> Any:
            raise NotImplementedError

        async def ainvoke_stream(self, *args: Any, **kwargs: Any) -> Any:
            raise NotImplementedError

        def _parse_provider_response(self, response: Any, **kwargs: Any) -> ModelResponse:
            return response

        def _parse_provider_response_delta(self, *args: Any, **kwargs: Any) -> Any:
            raise NotImplementedError

    return ScriptedModel(steps=list(steps))


async def main() -> None:
    try:
        import agno  # noqa: F401
    except ImportError as exc:
        print(
            f"Agno is not installed ({exc}).\nInstall it with `pip install 'dotfetch[agno]'`."
        )
        return

    from dotfetch.agno import AgnoAdapter

    fetcher = build_fetcher()
    try:
        await fetcher.sync()
        adapter = AgnoAdapter(fetcher)

        # --- 1. One agent, tools re-selected for every run ------------------ #
        model = scripted_model([("get_weather", {"city": "Paris"})])
        agent = adapter.agent(model=model, name="assistant", limit=1)
        await agent.arun("what's the weather in Paris?")
        await agent.arun("convert 50 euros to pounds")
        print(f"[per-run] tools offered per model call: {model.offered}")

        # --- 2. A failing tool: the model sees it and carries on ------------- #
        model = scripted_model([("convert_currency", {"amount": 5, "frm": "EUR", "to": "EUR"})])
        run = await adapter.agent(model=model, limit=1).arun("convert 5 euros to euros")
        [tool] = run.tools
        print(f"[error]   tool_call_error={tool.tool_call_error}, model saw: {tool.result!r}")

        # --- 3. A gated tool pauses the run until it is confirmed ------------ #
        model = scripted_model([("send_email", {"to": "sam@example.com"})])
        agent = adapter.agent(model=model, limit=1)
        run = await agent.arun("send a message to Sam about the delay")
        print(
            f"[approve] paused={run.is_paused}, waiting on "
            f"{[r.tool_execution.tool_name for r in run.active_requirements]}"
        )
        for requirement in run.active_requirements:
            requirement.confirm()
        run = await agent.acontinue_run(run_response=run)
        print(f"[approve] after confirming: {run.tools[0].result!r}")

        # --- 4. Run context reaches the dotfetch tool ------------------------- #
        model = scripted_model([("whoami", {})])
        run = await adapter.agent(model=model, limit=1).arun(
            "who am I", user_id="dan", dependencies={"plan": "team"}
        )
        print(f"[context] {run.tools[0].result!r}")

        # --- Optional: one turn on a real model ----------------------------- #
        if os.environ.get("OPENAI_API_KEY"):
            from agno.models.openai import OpenAIChat

            live = adapter.agent(model=OpenAIChat(id="gpt-5.5"), limit=2)
            response = await live.arun("what's the weather in Leeds?")
            print(f"[live]    {response.content}")
        else:
            print("(Set OPENAI_API_KEY to also run a turn on a real model.)")
    finally:
        await fetcher.close()


if __name__ == "__main__":
    asyncio.run(main())
