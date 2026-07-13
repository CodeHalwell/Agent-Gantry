"""
AWS Strands Agents + Agent-Gantry integration example.

Demonstrates both re-selection tiers Agent-Gantry offers for Strands:

1. **Static** — ``StrandsAdapter(gantry).select(query, limit=...)`` selects and
   converts a fixed slice of tools in one call, ready for ``Agent(tools=[...])``.
2. **Deep, per-turn** — ``StrandsAdapter(gantry).agent(...)`` builds an
   ``Agent`` with *no* statically registered tools; a
   ``strands.hooks.BeforeModelCallEvent`` hook re-runs Gantry's semantic router
   before every model call and swaps the agent's tool registry in place. This
   is the deepest tool re-selection Strands supports (Strands reads the
   registry only *after* the hook fires), matching the depth of Google ADK's
   ``before_model_callback``.

Actually *running* the agent (i.e. calling the model) requires a configured
Strands model provider — Amazon Bedrock by default, needing AWS credentials.
Without credentials this example still builds both integration paths and
prints the selected tools, so it is useful to run with no keys configured at
all (e.g. in CI).
"""

import asyncio
import os

from agent_gantry import AgentGantry

# Importing the adapter namespace never requires strands-agents to be
# installed -- the real `strands` import is deferred to the first call that
# actually needs it (`.select`/`.convert`/`.agent`), caught gracefully below.
from agent_gantry.strands import StrandsAdapter


def _has_model_credentials() -> bool:
    """Best-effort check for a configured Strands model provider.

    Strands defaults to Amazon Bedrock; actually running the agent (not just
    building it) needs AWS credentials, or another provider's API key if the
    agent is built with a non-default ``model=``. Gating on this keeps the
    example runnable end-to-end with no keys set.
    """
    return bool(
        os.getenv("AWS_ACCESS_KEY_ID")
        or os.getenv("AWS_PROFILE")
        or os.getenv("AWS_BEARER_TOKEN_BEDROCK")
        or os.getenv("ANTHROPIC_API_KEY")
        or os.getenv("OPENAI_API_KEY")
    )


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

    user_query = "What's the weather in London and the stock price for AAPL?"

    # 2. Static path: select + convert the top-k relevant tools in one call.
    #    Lowering the threshold for SimpleEmbedder compatibility in this example.
    #    The `strands` package is only imported lazily, right here -- catch that
    #    gracefully so the example is useful to run even without it installed.
    try:
        strands_tools = await StrandsAdapter(gantry).select(
            user_query, limit=2, score_threshold=0.1
        )
    except ImportError as exc:
        print(f"{exc}")
        return
    print(f"Gantry retrieved {len(strands_tools)} tools via the static .select(...) path:")
    for native_tool in strands_tools:
        print(f"  - {native_tool.tool_name}: {native_tool.tool_spec['description']}")

    # 3. Deep path: build an Agent with NO static tools. StrandsAdapter wires a
    #    BeforeModelCallEvent hook that re-selects Gantry tools before every
    #    model call, so the tool surface tracks the conversation turn by turn.
    live_agent = StrandsAdapter(gantry).agent(
        limit=2,
        score_threshold=0.1,
        system_prompt="You are a helpful assistant with access to weather and finance tools.",
        callback_handler=None,
    )
    static_tool_count = len(list(live_agent.tool_registry.registry))
    print(
        f"\nBuilt a live Strands Agent with {static_tool_count} statically "
        "registered tools (0 expected -- the hook injects the relevant slice "
        "before every model call)."
    )

    # 4. Only actually run the agent (which calls the model) when credentials
    #    look configured, so this example completes cleanly with no keys set.
    if not _has_model_credentials():
        print(
            "\nNo model credentials detected (AWS/Anthropic/OpenAI) -- skipping "
            "the live agent run. The tool selection above already demonstrates "
            "both integration paths end to end."
        )
        return

    print("\n--- Running Strands Agent with Gantry-sourced tools ---")
    result = await live_agent.invoke_async(user_query)
    print(f"\nFinal Response: {result}")


if __name__ == "__main__":
    asyncio.run(main())
