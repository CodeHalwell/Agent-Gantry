import asyncio
import os
from typing import Any

from dotenv import load_dotenv

from agent_gantry import AgentGantry
from agent_gantry.schema.execution import ToolCall
from agent_gantry.schema.query import ConversationContext, ToolQuery

# Load environment variables
load_dotenv()

async def main():
    print("=== Agent-Gantry + Anthropic (Claude) Integration Demo ===\n")

    # 1. Check for API Key
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("❌ Error: ANTHROPIC_API_KEY not found in environment.")
        print("   Please set it in your .env file.")
        return

    # 2. Initialize Gantry
    try:
        from agent_gantry.adapters.embedders.nomic import NomicEmbedder
        gantry = AgentGantry(embedder=NomicEmbedder())
        print("✅ Initialized with Nomic Embeddings")
    except ImportError:
        gantry = AgentGantry()
        print("⚠️  Initialized with Simple Embeddings")

    # 3. Register Tools
    @gantry.register(tags=["system"])
    def get_server_status(env: str) -> str:
        """Check the status of a server environment (prod/staging)."""
        return f"Environment '{env}' is HEALTHY. CPU: 45%"

    await gantry.sync()
    print(f"✅ Registered {gantry.tool_count} tools\n")

    # 4. Initialize Anthropic Client
    from anthropic import AsyncAnthropic
    client = AsyncAnthropic(api_key=api_key)

    # --- Scenario: Dynamic Retrieval with Schema Conversion ---
    print("--- Scenario: Dynamic Retrieval with Anthropic Schema ---")
    query = "Check the production server status"
    print(f"User Query: '{query}'")

    # A. Retrieve Tools (Returns RetrievalResult)
    # We use the lower-level `retrieve` method to get the internal ToolDefinition objects
    # so we can convert them to Anthropic's format.
    retrieval_result = await gantry.retrieve(ToolQuery(
        context=ConversationContext(query=query),
        limit=1,
        score_threshold=0.1
    ))

    # B. Convert to Anthropic Schema
    # Agent-Gantry provides a helper method `to_anthropic_schema()` on ToolDefinition
    anthropic_tools = [t.tool.to_anthropic_schema() for t in retrieval_result.tools]

    print(f"Gantry retrieved {len(anthropic_tools)} tool(s)")
    # print(json.dumps(anthropic_tools, indent=2)) # Uncomment to see schema

    # C. Call Claude
    response = await client.messages.create(
        model="claude-3-opus-20240229",
        max_tokens=1024,
        messages=[{"role": "user", "content": query}],
        tools=anthropic_tools
    )

    # D. Handle Tool Use
    # Anthropic returns a list of content blocks. We look for 'tool_use'.
    for block in response.content:
        if block.type == "tool_use":
            print(f"Claude decided to call: {block.name}({block.input})")

            # Execute securely via Gantry
            result = await gantry.execute(ToolCall(
                tool_name=block.name,
                arguments=block.input
            ))
            print(f"Execution Result: {result.result}")

    # --- Scenario: Using the Decorator ---
    print("\n--- Scenario: Using @with_semantic_tools Decorator ---")
    from agent_gantry.integrations.decorator import with_semantic_tools

    # The decorator handles retrieval AND schema conversion (dialect="anthropic")
    @with_semantic_tools(gantry, limit=1, dialect="anthropic", score_threshold=0.1)
    async def chat_with_claude(messages: list[dict[str, str]], tools: list[dict[str, Any]] = None):
        print(f"   [Decorator] Injected {len(tools) if tools else 0} tools (Anthropic format)")
        return await client.messages.create(
            model="claude-3-opus-20240229",
            max_tokens=1024,
            messages=messages,
            tools=tools
        )

    query_dec = "Check staging status"
    print(f"User Query: '{query_dec}'")

    response_dec = await chat_with_claude(messages=[{"role": "user", "content": query_dec}])

    for block in response_dec.content:
        if block.type == "tool_use":
            print(f"Claude decided to call: {block.name}")

if __name__ == "__main__":
    asyncio.run(main())
