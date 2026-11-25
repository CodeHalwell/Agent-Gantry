"""
Anthropic (Claude) + dotfetch integration demo.

Demonstrates how to use dotfetch with Anthropic's Claude API, including:
- Dynamic tool retrieval with Anthropic schema conversion
- Using the @with_semantic_tools decorator with dialect="anthropic"
- The typed AnthropicAdapter convenience wrapper
- Tool execution handling specific to Anthropic's response format
"""

import asyncio
import os
from typing import Any

from dotenv import load_dotenv

from dotfetch import DotFetch, set_default_fetcher, with_semantic_tools
from dotfetch.schema.execution import ToolCall
from dotfetch.schema.query import ConversationContext, ToolQuery

# Load environment variables
load_dotenv()


async def main() -> None:
    print("=== dotfetch + Anthropic (Claude) Integration Demo ===\n")

    # 1. Check for API Key
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("❌ Error: ANTHROPIC_API_KEY not found in environment.")
        print("   Please set it in your .env file.")
        return

    # 2. Initialize dotfetch
    try:
        from dotfetch.adapters.embedders.nomic import NomicEmbedder

        fetcher = DotFetch(embedder=NomicEmbedder())
        print("✅ Initialized with Nomic Embeddings")
    except ImportError:
        fetcher = DotFetch()
        print(f"⚠️  Initialized with the default embedder ({type(fetcher.embedder).__name__})")

    try:
        # 3. Register Tools
        @fetcher.register(
            tags=["system"],
            examples=["is production healthy right now", "check the staging server status"],
        )
        def get_server_status(env: str) -> str:
            """Check the status of a server environment (prod/staging)."""
            return f"Environment '{env}' is HEALTHY. CPU: 45%"

        await fetcher.sync()
        print(f"✅ Registered {fetcher.tool_count} tools\n")

        # 4. Initialize Anthropic Client
        from anthropic import AsyncAnthropic

        client = AsyncAnthropic(api_key=api_key)

        # --- Scenario A: Dynamic Retrieval with Manual Schema Conversion ---
        print("--- Scenario A: Dynamic Retrieval with Anthropic Schema ---")
        query = "Check the production server status"
        print(f"User Query: '{query}'")

        # A. Retrieve Tools (Returns RetrievalResult)
        # We use the lower-level `retrieve` method to get the internal ToolDefinition objects
        # so we can convert them to Anthropic's format.
        retrieval_result = await fetcher.retrieve(
            ToolQuery(context=ConversationContext(query=query), limit=1, score_threshold=0.0)
        )

        # B. Convert to Anthropic Schema
        anthropic_tools = [t.tool.to_dialect("anthropic") for t in retrieval_result.tools]

        print(f"dotfetch retrieved {len(anthropic_tools)} tool(s)")
        # print(json.dumps(anthropic_tools, indent=2)) # Uncomment to see schema

        # C. Call Claude
        response = await client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=1024,
            messages=[{"role": "user", "content": query}],
            tools=anthropic_tools,
        )

        # D. Handle Tool Use
        # Anthropic returns a list of content blocks. We look for 'tool_use'.
        for block in response.content:
            if block.type == "tool_use":
                print(f"Claude decided to call: {block.name}({block.input})")

                # Execute securely via dotfetch
                result = await fetcher.execute(
                    ToolCall(tool_name=block.name, arguments=block.input)
                )
                print(f"Execution Result: {result.result}")

        # --- Scenario B: Using the Decorator (RECOMMENDED) ---
        print("\n--- Scenario B: Using @with_semantic_tools Decorator (RECOMMENDED) ---")

        # Set default fetcher for decorator usage
        set_default_fetcher(fetcher)

        # The decorator handles retrieval AND schema conversion (dialect="anthropic")
        @with_semantic_tools(limit=1, dialect="anthropic")
        async def chat_with_claude(
            messages: list[dict[str, str]],
            tools: list[dict[str, Any]] | None = None,
        ):
            print(f"   [Decorator] Injected {len(tools) if tools else 0} tools (Anthropic format)")
            return await client.messages.create(
                model="claude-sonnet-4-6", max_tokens=1024, messages=messages, tools=tools
            )

        query_dec = "Check staging status"
        print(f"User Query: '{query_dec}'")

        response_dec = await chat_with_claude(messages=[{"role": "user", "content": query_dec}])

        for block in response_dec.content:
            if block.type == "tool_use":
                print(f"Claude decided to call: {block.name}")

        # --- Scenario C: Typed AnthropicAdapter (provider-specific convenience) ---
        # AnthropicAdapter(fetcher).tools(query, limit=n) bakes in dialect="anthropic",
        # so it returns ready-to-send Claude tool schemas in one call — equivalent to
        # the retrieve + to_dialect("anthropic") dance shown in Scenario A.
        print("\n--- Scenario C: Typed AnthropicAdapter ---")
        from dotfetch.anthropic import AnthropicAdapter

        query_c = "Check the database server status"
        print(f"User Query: '{query_c}'")

        anthropic_tools_c = await AnthropicAdapter(fetcher).tools(query_c, limit=1)
        print(f"dotfetch retrieved {len(anthropic_tools_c)} tool(s): "
              f"{[t['name'] for t in anthropic_tools_c]}")

        response_c = await client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=1024,
            messages=[{"role": "user", "content": query_c}],
            tools=anthropic_tools_c,
        )
        for block in response_c.content:
            if block.type == "tool_use":
                print(f"Claude decided to call: {block.name}({block.input})")
                result = await fetcher.execute(
                    ToolCall(tool_name=block.name, arguments=block.input)
                )
                print(f"Execution Result: {result.result}")
    finally:
        await fetcher.close()


if __name__ == "__main__":
    asyncio.run(main())
