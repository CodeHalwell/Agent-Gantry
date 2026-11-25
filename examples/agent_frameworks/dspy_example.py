"""DSPy + dotfetch integration example (dspy >= 3.5).

Runs offline: DSPy's own scripted ``DummyLM`` stands in for an LLM and issues
the tool calls, so DSPy's real ``dspy.ReAct`` loop does the rest. It shows:

1. ``DSPyAdapter.select()`` — dotfetch picks the relevant ``dspy.Tool`` objects
   for a ``dspy.ReAct``.
2. A failing tool — ReAct records an error tool result whose text ends with
   dotfetch's ``"<type>: <message>"`` and gives the model another turn.
3. A confirmation-gated tool — ``approve=`` decides it before it runs, and an
   approved call runs once.
4. Run context — a dotfetch tool reads the ``dspy.ReAct`` running it through
   ``current_dspy_module()``.
5. ``agent_builder()`` — a fresh ``dspy.ReAct`` per call, with tools selected
   for that call's query.

Set ``OPENAI_API_KEY`` to also run one question on a real model.

Install:

    pip install 'dotfetch[dspy]'

Run:

    python examples/agent_frameworks/dspy_example.py
"""

from __future__ import annotations

import asyncio
import os
from typing import Any

from dotfetch import DotFetch
from dotfetch.adapters.embedders.simple import SimpleEmbedder


def build_fetcher() -> DotFetch:
    from dotfetch.dspy import current_dspy_module

    fetcher = DotFetch(embedder=SimpleEmbedder(dimension=256))

    @fetcher.register(examples=["what's the weather in London", "is it raining in Leeds"])
    def get_weather(location: str) -> str:
        """Get the current weather in a given location."""
        return f"The weather in {location} is sunny and 25C."

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

    @fetcher.register(examples=["which module is running", "who is calling this tool"])
    def whoami() -> str:
        """Say which DSPy module is running this tool."""
        return f"called by {type(current_dspy_module()).__name__}"

    @fetcher.register(examples=["what's Apple trading at", "current share price for MSFT"])
    def get_stock_price(symbol: str) -> str:
        """Get the current stock price for a symbol."""
        return f"The stock price for {symbol} is $150.00."

    return fetcher


def scripted_lm(*calls: tuple[str, dict[str, Any]]) -> Any:
    """A DummyLM that makes ``calls`` in one turn, then submits an answer."""
    from dspy.adapters.chat_adapter import ChatAdapter
    from dspy.utils.dummies import DummyLM

    return DummyLM(
        [
            {
                "next_thought": "Use the tools.",
                "tool_calls": {"tool_calls": [{"name": n, "args": a} for n, a in calls]},
            },
            {
                "next_thought": "That's everything I need.",
                "tool_calls": {"tool_calls": [{"name": "submit", "args": {"answer": "Done."}}]},
            },
        ],
        adapter=ChatAdapter(),
    )


def run(react: Any, lm: Any, question: str) -> Any:
    import dspy

    # dspy.context rather than dspy.configure: it is scoped to the block.
    with dspy.context(lm=lm, adapter=lm.adapter):
        return react(question=question)


def tool_results(pred: Any) -> list[Any]:
    return [
        result
        for event in pred.history.messages
        if event.get("tool_calls") is not None
        for result in event["tool_calls"].tool_call_results.tool_call_results
        if result.name != "submit"
    ]


def approve_everything(request: Any) -> bool:
    print(f"[approve]  asked about {request.tool_name}({request.arguments}) -> yes")
    return True


async def main() -> None:
    try:
        import dspy
        from dspy.adapters.types.tool import ToolCalls  # noqa: F401 - dspy >= 3.5
    except ImportError as exc:
        print(
            f"DSPy 3.5+ is not installed ({exc}).\n"
            "Install it with `pip install 'dotfetch[dspy]'`."
        )
        return

    from dotfetch.dspy import DSPyAdapter

    fetcher = build_fetcher()
    try:
        await fetcher.sync()
        adapter = DSPyAdapter(fetcher, approve=approve_everything)

        # --- 1. Select tools for a ReAct ------------------------------------ #
        tools = await adapter.select("what's the weather in London", limit=1)
        print(f"[select]   tools: {[tool.name for tool in tools]}")
        react = dspy.ReAct("question -> answer", tools=tools, max_iters=3)
        pred = run(react, scripted_lm(("get_weather", {"location": "London"})), "Weather?")
        print(f"[select]   tool result: {tool_results(pred)[0].value!r}")

        # --- 2. A failing tool: the model sees it and carries on ------------- #
        tools = await adapter.select("convert 5 euros to euros", limit=1)
        react = dspy.ReAct("question -> answer", tools=tools, max_iters=3)
        lm = scripted_lm(("convert_currency", {"amount": 5, "frm": "EUR", "to": "EUR"}))
        [failed] = tool_results(run(react, lm, "Convert 5 EUR to EUR."))
        print(f"[error]    is_error={failed.is_error}, ends: {failed.value.splitlines()[-1]!r}")

        # --- 3. A gated tool is approved before it runs --------------------- #
        tools = await adapter.select("send an email message to Sam", limit=1)
        react = dspy.ReAct("question -> answer", tools=tools, max_iters=3)
        pred = run(react, scripted_lm(("send_email", {"to": "sam@example.com"})), "Email Sam.")
        print(f"[approve]  result: {tool_results(pred)[0].value!r}")

        # --- 4. The running ReAct reaches the dotfetch tool ------------------- #
        tools = await adapter.select("which module is running", limit=1)
        react = dspy.ReAct("question -> answer", tools=tools, max_iters=3)
        pred = run(react, scripted_lm(("whoami", {})), "Who is calling?")
        print(f"[context]  {tool_results(pred)[0].value!r}")

        # --- 5. A fresh ReAct per call, tools selected for each query -------- #
        builder = adapter.agent_builder("question -> answer", max_iters=3, limit=1)
        finance = await builder.build("What's the stock price for AAPL?")
        print(f"[per-run] tools: {[name for name in finance.tools if name != 'submit']}")

        # --- Optional: one question on a real model ------------------------- #
        if os.environ.get("OPENAI_API_KEY"):
            weather = await builder.build("What's the weather in Leeds?")
            with dspy.context(lm=dspy.LM("openai/gpt-5.5")):
                print(f"[live]     {weather(question='What is the weather in Leeds?').answer}")
        else:
            print("(Set OPENAI_API_KEY to also run a question on a real model.)")
    finally:
        await fetcher.close()


if __name__ == "__main__":
    asyncio.run(main())
