"""
LLM-based Intent Classification Example.

This example demonstrates how to enable LLM-based intent classification
for improved routing accuracy when keyword matching fails.

Requirements:
    pip install openai  # or anthropic, google-genai, mistralai, groq

Environment:
    Set OPENAI_API_KEY (or other provider's API key) in your environment
"""

import asyncio
import os

from agent_gantry import AgentGantry
from agent_gantry.schema.config import AgentGantryConfig, LLMConfig, RoutingConfig


async def main():
    """Demonstrate LLM-based intent classification."""

    # Method 1: Enable via config
    config = AgentGantryConfig(
        routing=RoutingConfig(
            use_llm_for_intent=True,
            llm=LLMConfig(
                provider="openai",
                model="gpt-4o-mini",
                api_key=os.getenv("OPENAI_API_KEY"),
                temperature=0.0,
                max_tokens=50,
            ),
        ),
    )

    gantry = AgentGantry(config=config)

    # Register tools with different intents
    @gantry.register
    def search_users(name: str) -> str:
        """Search for users by name."""
        return f"Found user: {name}"

    @gantry.register
    def create_user(name: str, email: str) -> str:
        """Create a new user account."""
        return f"Created user {name} with email {email}"

    @gantry.register
    def analyze_metrics() -> str:
        """Analyze system performance metrics."""
        return "CPU: 45%, Memory: 67%, Disk: 23%"

    @gantry.register
    def send_notification(user: str, message: str) -> str:
        """Send a notification to a user."""
        return f"Sent to {user}: {message}"

    @gantry.register
    def export_data(format: str = "csv") -> str:
        """Export data to a file."""
        return f"Exported data to {format}"

    await gantry.sync()

    print("=" * 70)
    print("LLM-Based Intent Classification Demo")
    print("=" * 70)
    print()

    # Test queries that work well with keyword matching
    print("📋 Test 1: Clear keyword-based queries (no LLM needed)")
    print("-" * 70)

    test_cases_keywords = [
        "search for John Doe",
        "create a new account",
        "send email notification",
    ]

    for query in test_cases_keywords:
        tools = await gantry.retrieve_tools(query, limit=1)
        if tools:
            print(f"Query: '{query}'")
            print(f"  → Tool: {tools[0]['function']['name']}")
            print()

    # Test queries that are ambiguous or need LLM understanding
    print()
    print("🤖 Test 2: Ambiguous queries (LLM-based classification)")
    print("-" * 70)

    test_cases_llm = [
        "What's the system status?",  # Could be metrics or query
        "Show me the trends",         # Analysis intent
        "Reach out to the team",      # Communication intent
        "Get me that information",    # Generic query
    ]

    for query in test_cases_llm:
        tools = await gantry.retrieve_tools(query, limit=1)
        if tools:
            print(f"Query: '{query}'")
            print(f"  → Tool: {tools[0]['function']['name']}")
            print()

    print()
    print("=" * 70)
    print("✅ Demo complete!")
    print()
    print("How it works:")
    print("1. First tries keyword-based intent classification (fast)")
    print("2. Falls back to LLM if no keywords match (accurate)")
    print("3. Uses intent to boost relevant tool scores")
    print()
    print("Benefits:")
    print("  • More accurate tool selection for ambiguous queries")
    print("  • No performance impact when keywords match")
    print("  • Configurable per-provider (OpenAI, Anthropic, etc.)")
    print("=" * 70)


async def demo_yaml_config():
    """Show how to configure via YAML."""
    print("\n" + "=" * 70)
    print("YAML Configuration Example")
    print("=" * 70)
    print()

    yaml_example = """
# config.yaml
routing:
  use_llm_for_intent: true
  llm:
    provider: openai  # or anthropic, google, mistral, groq
    model: gpt-4o-mini
    api_key: ${OPENAI_API_KEY}  # or set directly
    temperature: 0.0
    max_tokens: 50

  weights:
    semantic: 0.6
    intent: 0.15      # Intent matching weight
    conversation: 0.1
    health: 0.1
    cost: 0.05
"""

    print(yaml_example)
    print("\nUsage:")
    print("  gantry = AgentGantry.from_config('config.yaml')")
    print()


async def demo_providers():
    """Show different LLM provider configurations."""
    print("\n" + "=" * 70)
    print("Supported LLM Providers")
    print("=" * 70)
    print()

    providers = {
        "OpenAI": {
            "provider": "openai",
            "model": "gpt-4o-mini",
            "api_key_env": "OPENAI_API_KEY",
        },
        "Anthropic": {
            "provider": "anthropic",
            "model": "claude-3-5-haiku-20241022",
            "api_key_env": "ANTHROPIC_API_KEY",
        },
        "Google GenAI": {
            "provider": "google",
            "model": "gemini-2.0-flash-exp",
            "api_key_env": "GOOGLE_API_KEY",
        },
        "Mistral": {
            "provider": "mistral",
            "model": "mistral-small-latest",
            "api_key_env": "MISTRAL_API_KEY",
        },
        "Groq": {
            "provider": "groq",
            "model": "llama-3.3-70b-versatile",
            "api_key_env": "GROQ_API_KEY",
        },
    }

    for name, config in providers.items():
        print(f"• {name}")
        print(f"  Provider: {config['provider']}")
        print(f"  Model: {config['model']}")
        print(f"  API Key: ${config['api_key_env']}")
        print()


if __name__ == "__main__":
    # Check if API key is available
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️  Warning: OPENAI_API_KEY not set in environment")
        print("Set it to run the full demo, or use a different provider")
        print()
        # Still show configuration examples
        asyncio.run(demo_yaml_config())
        asyncio.run(demo_providers())
    else:
        asyncio.run(main())
        asyncio.run(demo_yaml_config())
        asyncio.run(demo_providers())
