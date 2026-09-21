"""
Fast Track Demo: Upgrade vanilla OpenAI to semantic tools in ~10 lines

This example shows how to take a basic OpenAI chat completion call and
upgrade it to use Agent-Gantry's semantic tool routing with minimal changes.

Runs without an API key: it registers three tools, syncs, and prints what
Gantry selects for each query. Set OPENAI_API_KEY (and install
``agent-gantry[openai]``) to also send the selected tools to a real model.
"""

import asyncio
import os

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass  # dotenv is optional


async def main():
    print("=== Fast Track Demo: Vanilla OpenAI → Semantic Tools ===\n")

    # ============================================================================
    # BEFORE: Basic OpenAI call (no tools)
    # ============================================================================
    print("📝 BEFORE: Basic OpenAI call with no tools\n")
    print("""
from openai import AsyncOpenAI

client = AsyncOpenAI()

async def chat(prompt: str):
    return await client.chat.completions.create(
        model="gpt-5.5",
        messages=[{"role": "user", "content": prompt}]
    )

response = await chat("What's the weather in Tokyo?")
# LLM has no tools, can only provide general information
    """)

    # ============================================================================
    # AFTER: Add Agent-Gantry in ~10 lines
    # ============================================================================
    print("\n✨ AFTER: Add Agent-Gantry's semantic tool routing\n")
    print("""
from openai import AsyncOpenAI
from agent_gantry import AgentGantry, with_semantic_tools, set_default_gantry

client = AsyncOpenAI()

# 1. Initialize Agent-Gantry (1 line)
gantry = AgentGantry()
set_default_gantry(gantry)

# 2. Register tools with simple decorators (3 lines)
@gantry.register(examples=["what's the weather in Tokyo", "is it raining in Leeds"])
def get_weather(city: str) -> str:
    '''Get current weather for a city.'''
    return f"Weather in {city}: Sunny, 72°F"

@gantry.register(examples=["what's AAPL trading at", "current share price for MSFT"])
def get_stock_price(symbol: str) -> str:
    '''Get current stock price for a symbol.'''
    return f"{symbol}: $150.00"

@gantry.register(examples=["email john about the meeting", "send a message to Sam"])
def send_email(to: str, subject: str) -> str:
    '''Send an email.'''
    return f"Email sent to {to}"

# 3. Add decorator to your chat function (1 line)
@with_semantic_tools(limit=1)
async def chat(prompt: str, *, tools=None):
    return await client.chat.completions.create(
        model="gpt-5.5",
        messages=[{"role": "user", "content": prompt}],
        tools=tools  # Tools automatically injected here
    )

# 4. Just call it - semantic routing happens automatically
response = await chat("What's the weather in Tokyo?")
# LLM receives only relevant tools (get_weather), not all 3 tools
# Token usage reduced by ~79%, accuracy improved
    """)

    # ============================================================================
    # Run it for real: selection needs no key, the model call does
    # ============================================================================
    # None of this needs a key, so it always runs: the selection *is* the
    # thing this demo is about, and printing a code listing instead would be
    # showing the reader a picture of the feature rather than the feature.
    from agent_gantry import AgentGantry, set_default_gantry, with_semantic_tools

    gantry = AgentGantry()
    set_default_gantry(gantry)

    try:
        # `examples=[...]` is the text the router embeds — the single
        # highest-value field on a tool definition.
        @gantry.register(examples=["what's the weather in Tokyo", "is it raining in Leeds"])
        def get_weather(city: str) -> str:
            """Get current weather for a city."""
            return f"Weather in {city}: Sunny, 72°F"

        @gantry.register(examples=["what's AAPL trading at", "current share price for MSFT"])
        def get_stock_price(symbol: str) -> str:
            """Get current stock price for a symbol."""
            return f"{symbol}: $150.00"

        @gantry.register(examples=["email john about the meeting", "send a message to Sam"])
        def send_email(to: str, subject: str) -> str:
            """Send an email."""
            return f"Email sent to {to}"

        await gantry.sync()

        queries = [
            "What's the weather in Tokyo?",
            "What's the price of AAPL stock?",
            "Send an email to john@example.com with subject 'Meeting'",
        ]

        print("\n🔍 What Gantry selects for each query (no API key needed):\n")
        for query in queries:
            # No score_threshold: it defaults to 0.0, and raising it is a
            # silent-drop trap on longer queries.
            selected = await gantry.retrieve_tools(query, limit=1)
            names = [t["function"]["name"] for t in selected]
            print(f"   📨 {query}")
            print(f"      → {names} (1 of 3 tools sent, not all 3)")

        if not os.environ.get("OPENAI_API_KEY"):
            print("\n⚠️  Set OPENAI_API_KEY to run the same thing against a real model.")
            print("    Everything above works without one.")
            return

        print("\n🚀 Running the same queries against the model...\n")

        from openai import AsyncOpenAI

        client = AsyncOpenAI()

        @with_semantic_tools(limit=1)
        async def chat(prompt: str, *, tools=None):
            print(f"   [Agent-Gantry] Injected {len(tools) if tools else 0} relevant tools")
            if tools:
                print(f"   [Agent-Gantry] Tools: {[t['function']['name'] for t in tools]}")
            return await client.chat.completions.create(
                model="gpt-5.5", messages=[{"role": "user", "content": prompt}], tools=tools
            )

        for query in queries:
            print(f"\n📨 Query: '{query}'")
            response = await chat(query)

            if response.choices[0].message.tool_calls:
                tool_call = response.choices[0].message.tool_calls[0]
                print(f"   [LLM] Called tool: {tool_call.function.name}")
            else:
                print(f"   [LLM] Response: {response.choices[0].message.content[:100]}...")
    finally:
        await gantry.close()

    # ============================================================================
    # Summary
    # ============================================================================
    print("\n" + "=" * 70)
    print("📊 Summary: What You Get")
    print("=" * 70)
    print("""
✅ Semantic Tool Selection: Only relevant tools sent to LLM
✅ Token Cost Reduction: ~79% fewer tokens (benchmark proven)
✅ Better Accuracy: LLM gets focused context, not tool overload
✅ Schema Transcoding: Works with OpenAI, Anthropic, Google, Mistral, Groq
✅ Minimal Code Changes: Just decorators, no refactoring needed
✅ Circuit Breakers: Built-in retry logic and error handling
✅ Observability: Telemetry and metrics out of the box

Total Lines Added: ~10 lines (3 imports, 1 init, 3 tool registrations, 1 decorator)
    """)

    print("\n🔗 Next Steps:")
    print("   - See examples/llm_integration/ for provider-specific examples")
    print(
        "   - Run examples/basics/plug_and_play_semantic_filter.py to import tools from a module with one decorator"
    )
    print("   - See examples/llm_integration/decorator_demo.py for the decorator's two call styles")
    print("   - Try examples/agent_frameworks/ for LangChain, CrewAI, LlamaIndex")


if __name__ == "__main__":
    asyncio.run(main())
