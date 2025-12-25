"""
Anthropic Interleaved & Extended Thinking Demo.

This example demonstrates how to use Anthropic's advanced thinking features:
1. Interleaved Thinking - Shows the model's reasoning process
2. Extended Thinking - Allows deeper reasoning with budget tokens
3. Tool use with Agent-Gantry integration

Requirements:
    pip install anthropic agent-gantry

Environment:
    Set ANTHROPIC_API_KEY in your environment
"""

import asyncio
import os

from dotenv import load_dotenv

from agent_gantry import AgentGantry
from agent_gantry.integrations.anthropic_features import (
    AnthropicClient,
    AnthropicFeatures,
    create_anthropic_client,
)

load_dotenv()


async def demo_interleaved_thinking():
    """Demonstrate interleaved thinking - see the model's reasoning."""
    print("=" * 80)
    print("Demo 1: Interleaved Thinking")
    print("=" * 80)
    print()

    # Initialize client with interleaved thinking
    client = await create_anthropic_client(
        enable_thinking="interleaved",
    )

    # Ask a question that requires reasoning
    response, thinking = await client.chat_with_thinking(
        model="claude-3-5-sonnet-20241022",
        messages=[
            {
                "role": "user",
                "content": "If I have 3 apples and buy 2 more, then give away half, how many do I have?"
            }
        ],
        max_tokens=1024,
    )

    # Display thinking process
    if thinking:
        print("🧠 Model's Thinking Process:")
        print("-" * 80)
        for i, thought in enumerate(thinking, 1):
            print(f"\nThinking block {i}:")
            print(thought)
        print("-" * 80)
        print()

    # Display final answer
    print("💬 Final Answer:")
    for block in response.content:
        if hasattr(block, "text"):
            print(block.text)

    print()


async def demo_extended_thinking():
    """Demonstrate extended thinking with budget tokens."""
    print("=" * 80)
    print("Demo 2: Extended Thinking (Skills API)")
    print("=" * 80)
    print()

    # Initialize with extended thinking and budget
    features = AnthropicFeatures(
        enable_extended_thinking=True,
        thinking_budget_tokens=10000,  # Allow deeper reasoning
    )

    client = AnthropicClient(features=features)

    # Ask a complex problem
    response = await client.create_message(
        model="claude-3-5-sonnet-20241022",
        messages=[
            {
                "role": "user",
                "content": "Design a distributed caching system that can handle 1M requests per second."
            }
        ],
        max_tokens=2048,
    )

    print("💬 Response (with extended thinking):")
    for block in response.content:
        if hasattr(block, "text"):
            print(block.text)

    print("\nNote: Extended thinking uses more tokens but provides deeper reasoning.")
    print()


async def demo_thinking_with_tools():
    """Demonstrate thinking with tool use."""
    print("=" * 80)
    print("Demo 3: Interleaved Thinking + Tool Use")
    print("=" * 80)
    print()

    # Initialize Agent-Gantry
    gantry = AgentGantry()

    @gantry.register
    def calculate(expression: str) -> float:
        """Evaluate a mathematical expression safely."""
        # For demo purposes - in production use a safe evaluator
        try:
            return eval(expression, {"__builtins__": {}}, {})
        except Exception as e:
            return f"Error: {str(e)}"

    @gantry.register
    def get_weather(city: str) -> str:
        """Get the current weather for a city."""
        # Mock weather data
        return f"Weather in {city}: 72°F, Sunny"

    await gantry.sync()

    # Initialize client with thinking and tools
    client = await create_anthropic_client(
        gantry=gantry,
        enable_thinking="interleaved",
    )

    # Query that benefits from both thinking and tools
    response, thinking = await client.chat_with_thinking(
        model="claude-3-5-sonnet-20241022",
        messages=[
            {
                "role": "user",
                "content": "What's the weather in San Francisco, and calculate 15% tip on a $47.50 bill?"
            }
        ],
        max_tokens=2048,
        query="weather calculation math",
    )

    # Show thinking
    if thinking:
        print("🧠 Model's Reasoning:")
        for thought in thinking:
            print(f"  {thought[:200]}..." if len(thought) > 200 else f"  {thought}")
        print()

    # Execute tool calls
    tool_uses = [block for block in response.content if block.type == "tool_use"]
    if tool_uses:
        print("🔧 Tool Calls:")
        for tool_use in tool_uses:
            print(f"  • {tool_use.name}({tool_use.input})")

        # Execute tools
        tool_results = await client.execute_tool_calls(response)
        print()
        print("📊 Tool Results:")
        for result in tool_results:
            print(f"  • {result['content']}")

    print()


async def demo_comparison():
    """Compare responses with and without thinking."""
    print("=" * 80)
    print("Demo 4: Comparison - With vs Without Thinking")
    print("=" * 80)
    print()

    query = "Explain why the sky is blue in one sentence."

    # Without thinking
    print("Without thinking:")
    print("-" * 80)
    client_normal = AnthropicClient()
    response_normal = await client_normal.create_message(
        model="claude-3-5-sonnet-20241022",
        messages=[{"role": "user", "content": query}],
        max_tokens=512,
        auto_retrieve_tools=False,
    )
    for block in response_normal.content:
        if hasattr(block, "text"):
            print(block.text)
    print()

    # With thinking
    print("With interleaved thinking:")
    print("-" * 80)
    client_thinking = await create_anthropic_client(enable_thinking="interleaved")
    response_thinking, thinking = await client_thinking.chat_with_thinking(
        model="claude-3-5-sonnet-20241022",
        messages=[{"role": "user", "content": query}],
        max_tokens=512,
        auto_retrieve_tools=False,
    )

    if thinking:
        print("🧠 Reasoning:")
        for thought in thinking:
            print(f"  {thought}")
        print()

    for block in response_thinking.content:
        if hasattr(block, "text"):
            print(block.text)
    print()


async def main():
    """Run all demos."""
    # Check for API key
    if not os.getenv("ANTHROPIC_API_KEY"):
        print("❌ Error: ANTHROPIC_API_KEY not set in environment")
        print("Set it to run the demos:")
        print("  export ANTHROPIC_API_KEY='your-key-here'")
        return

    print("\n")
    print("╔" + "═" * 78 + "╗")
    print("║" + " " * 15 + "Anthropic Thinking Features Demo" + " " * 31 + "║")
    print("╚" + "═" * 78 + "╝")
    print()

    try:
        # Run demos
        await demo_interleaved_thinking()
        await demo_extended_thinking()
        await demo_thinking_with_tools()
        await demo_comparison()

        print("=" * 80)
        print("✅ All demos completed successfully!")
        print("=" * 80)
        print()
        print("Key Takeaways:")
        print("  • Interleaved thinking shows the model's reasoning process")
        print("  • Extended thinking allows deeper reasoning with budget control")
        print("  • Both features work seamlessly with Agent-Gantry tool use")
        print("  • Use thinking when you need transparency and better decision-making")
        print()

    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nNote: Some features may require specific model versions or beta access")


if __name__ == "__main__":
    asyncio.run(main())
