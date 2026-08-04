"""
DSPy + Agent-Gantry integration example.

Demonstrates both tool-surfacing tiers Agent-Gantry offers for DSPy:

1. **Static** — ``DSPyAdapter(gantry).select(query, limit=...)`` selects and
   converts a fixed slice of tools to ``dspy.Tool`` objects in one call, ready
   for ``dspy.ReAct(signature, tools=[...])``.
2. **Per-call** — ``DSPyAdapter(gantry).agent_builder(signature, ...)``
   returns a builder whose ``await builder.build(query)`` re-selects tools and
   constructs a *fresh* ``dspy.ReAct`` for that call. ``dspy.ReAct`` fixes its
   tool list at construction with no runtime re-selection hook (unlike
   Strands/Google ADK), so per-call rebuild is the deepest tier DSPy permits
   — see the module docstring in
   ``agent_gantry/integrations/frameworks/dspy.py`` for the full rationale.

Actually *running* a ``dspy.ReAct`` agent needs a configured DSPy LM. When
``OPENAI_API_KEY`` is set, this example configures a real
``dspy.LM("openai/gpt-5.4-mini")`` and runs the agent for real. With no key
configured, it falls back to ``dspy.utils.dummies.DummyLM`` — a scripted,
offline stand-in DSPy ships for its own test suite — to drive one full
tool-calling round trip with **no network access and no API key at all**, so
this example completes end to end even in CI.
"""

import asyncio
import os

from agent_gantry import AgentGantry

# Importing the adapter namespace never requires dspy to be installed -- the
# real `dspy` import is deferred to the first call that actually needs it
# (`.select`/`.convert`/`.agent_builder`), caught gracefully below.
from agent_gantry.dspy import DSPyAdapter


def _run_with_dummy_lm(react, question: str):
    """Drive ``react`` end to end with a scripted, offline DummyLM.

    No network access or API key required. The script hard-codes the
    tool-call DSPy's ReAct loop should make (send_email, then finish) and the
    final answer, which is enough to prove the Gantry-sourced tool actually
    executes (through ``gantry.execute``) inside a real ``dspy.ReAct`` run.
    """
    import dspy
    from dspy.adapters.chat_adapter import ChatAdapter
    from dspy.utils.dummies import DummyLM

    adapter = ChatAdapter()
    lm = DummyLM(
        [
            {
                "next_thought": "I should look up the weather for London.",
                "next_tool_name": "get_weather",
                "next_tool_args": {"location": "London"},
            },
            {"next_thought": "That's everything I need.", "next_tool_name": "finish", "next_tool_args": {}},
            {
                "reasoning": "The weather tool already returned the answer.",
                "answer": "It's sunny and 25C in London.",
            },
        ],
        adapter=adapter,
    )
    dspy.configure(lm=lm, adapter=adapter)
    try:
        return react(question=question)
    finally:
        dspy.configure(lm=None, adapter=None)


def _run_with_real_lm(react, question: str):
    """Drive ``react`` with a real OpenAI-backed ``dspy.LM`` (requires OPENAI_API_KEY)."""
    import dspy

    dspy.configure(lm=dspy.LM("openai/gpt-5.4-mini"))
    try:
        return react(question=question)
    finally:
        dspy.configure(lm=None)


async def main() -> None:
    # 1. Initialize Agent-Gantry and register tools.
    gantry = AgentGantry()

    @gantry.register(tags=["weather"])
    def get_weather(location: str) -> str:
        """Get the current weather in a given location."""
        return f"The weather in {location} is sunny and 25C."

    @gantry.register(tags=["finance"])
    def get_stock_price(symbol: str) -> str:
        """Get the current stock price for a symbol."""
        return f"The stock price for {symbol} is $150.00."

    await gantry.sync()

    user_query = "What's the weather in London?"

    # 2. Static path: select + convert the top-k relevant tools in one call.
    #    Lowering the threshold for SimpleEmbedder compatibility in this example.
    #    The `dspy` package is only imported lazily, right here -- catch that
    #    gracefully so the example is useful to run even without it installed.
    try:
        dspy_tools = await DSPyAdapter(gantry).select(user_query, limit=1, score_threshold=0.1)
    except ImportError as exc:
        print(f"{exc}")
        return
    print(f"Gantry retrieved {len(dspy_tools)} tool(s) via the static .select(...) path:")
    for native_tool in dspy_tools:
        print(f"  - {native_tool.name}: {native_tool.desc}")

    import dspy

    react = dspy.ReAct("question -> answer", tools=dspy_tools, max_iters=3)
    print(f"\nBuilt a dspy.ReAct with {len(react.tools) - 1} Gantry tool(s) (excluding the built-in 'finish' tool).")

    # 3. Run the agent -- for real if OPENAI_API_KEY is configured, otherwise
    #    fully offline via DummyLM so this example always completes.
    if os.getenv("OPENAI_API_KEY"):
        print("\n--- Running dspy.ReAct with a real OpenAI-backed LM ---")
        pred = _run_with_real_lm(react, user_query)
    else:
        print("\n--- No OPENAI_API_KEY set -- running dspy.ReAct with DummyLM (fully offline) ---")
        pred = _run_with_dummy_lm(react, user_query)
    print(f"\nFinal Response: {pred.answer}")
    print(f"Tool observation: {pred.trajectory.get('observation_0')}")

    # 4. Per-call path: DSPy fixes a ReAct's tools at construction (no runtime
    #    re-selection hook), so `agent_builder` rebuilds a fresh ReAct per
    #    call, tools re-selected for that call's query -- the deepest tier
    #    DSPy allows (see the adapter module docstring for why).
    builder = DSPyAdapter(gantry).agent_builder("question -> answer", max_iters=3, limit=1, score_threshold=0.1)
    finance_react = await builder.build("What's the stock price for AAPL?")
    print(
        f"\nPer-call agent_builder rebuilt a ReAct with tools "
        f"{[name for name in finance_react.tools if name != 'finish']} for a finance query."
    )


if __name__ == "__main__":
    asyncio.run(main())
