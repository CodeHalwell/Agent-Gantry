"""
Fast Track Demo: Upgrade vanilla OpenAI to semantic tools in ~10 lines

This example shows how to take a basic OpenAI chat completion call and
upgrade it to use Agent-Gantry's semantic tool routing with minimal changes.
"""
import asyncio
import os

from dotenv import load_dotenv

load_dotenv()


async def main():
    # Check for API key
    if not os.environ.get("OPENAI_API_KEY"):
        print("⚠️  No OPENAI_API_KEY found. This demo will show the code patterns.")
        print("    Set OPENAI_API_KEY in your environment to run the actual LLM calls.\n")

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
        model="gpt-4o",
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
@gantry.register
def get_weather(city: str) -> str:
    '''Get current weather for a city.'''
    return f"Weather in {city}: Sunny, 72°F"

@gantry.register
def get_stock_price(symbol: str) -> str:
    '''Get current stock price for a symbol.'''
    return f"{symbol}: $150.00"

@gantry.register
def send_email(to: str, subject: str) -> str:
    '''Send an email.'''
    return f"Email sent to {to}"

# 3. Add decorator to your chat function (1 line)
@with_semantic_tools(limit=3)
async def chat(prompt: str, *, tools=None):
    return await client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        tools=tools  # Tools automatically injected here
    )

# 4. Just call it - semantic routing happens automatically
response = await chat("What's the weather in Tokyo?")
# LLM receives only relevant tools (get_weather), not all 3 tools
# Token usage reduced by ~79%, accuracy improved
    """)

    # ============================================================================
    # Run actual demo if API key is available
    # ============================================================================
    if os.environ.get("OPENAI_API_KEY"):
        print("\n🚀 Running actual demo...\n")

        from openai import AsyncOpenAI
        from agent_gantry import AgentGantry, with_semantic_tools, set_default_gantry

        client = AsyncOpenAI()
        gantry = AgentGantry()
        set_default_gantry(gantry)

        # Register tools
        @gantry.register
        def get_weather(city: str) -> str:
            """Get current weather for a city."""
            return f"Weather in {city}: Sunny, 72°F"

        @gantry.register
        def get_stock_price(symbol: str) -> str:
            """Get current stock price for a symbol."""
            return f"{symbol}: $150.00"

        @gantry.register
        def send_email(to: str, subject: str) -> str:
            """Send an email."""
            return f"Email sent to {to}"

        # Add decorator
        @with_semantic_tools(limit=3, score_threshold=0.1)
        async def chat(prompt: str, *, tools=None):
            print(f"   [Agent-Gantry] Injected {len(tools) if tools else 0} relevant tools")
            if tools:
                print(f"   [Agent-Gantry] Tools: {[t['function']['name'] for t in tools]}")
            return await client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                tools=tools
            )

        # Test queries
        queries = [
            "What's the weather in Tokyo?",
            "What's the price of AAPL stock?",
            "Send an email to john@example.com with subject 'Meeting'",
        ]

        for query in queries:
            print(f"\n📨 Query: '{query}'")
            response = await chat(query)
            
            # Check if LLM called tools
            if response.choices[0].message.tool_calls:
                tool_call = response.choices[0].message.tool_calls[0]
                print(f"   [LLM] Called tool: {tool_call.function.name}")
            else:
                print(f"   [LLM] Response: {response.choices[0].message.content[:100]}...")

    # ============================================================================
    # Summary
    # ============================================================================
    print("\n" + "="*70)
    print("📊 Summary: What You Get")
    print("="*70)
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
    print("   - Read docs/semantic_tool_decorator.md for advanced usage")
    print("   - Try examples/agent_frameworks/ for LangChain, AutoGen, CrewAI")


if __name__ == "__main__":
    asyncio.run(main())
