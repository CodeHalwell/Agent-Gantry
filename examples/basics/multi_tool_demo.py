"""
Semantic routing across a ten-tool catalogue.

Registers ten tools from unrelated domains, syncs, then shows which single
tool Gantry picks for each of ten queries before executing one of them.
Runs offline with no API key.

Run: python examples/basics/multi_tool_demo.py
"""

import asyncio

from agent_gantry import AgentGantry
from agent_gantry.schema.execution import ToolCall


async def main():
    print("Initializing AgentGantry...")
    gantry = AgentGantry()

    try:
        # --- Register 10 Different Tools ---

        @gantry.register(
            tags=["math", "calculation"],
            examples=["add 50 and 20", "what's 12 plus 7"],
        )
        def add_numbers(a: float, b: float) -> float:
            """Add two numbers together."""
            return a + b

        @gantry.register(
            tags=["string", "text"],
            examples=["join 'Hello' and 'World'", "stick these two words together"],
        )
        def concat_strings(s1: str, s2: str) -> str:
            """Concatenate two strings."""
            return s1 + s2

        @gantry.register(
            tags=["time", "date"],
            examples=["what time is it in Tokyo", "current time in New York"],
        )
        def get_current_time(timezone: str = "UTC") -> str:
            """Get the current time in a specific timezone."""
            return f"The time in {timezone} is 12:00 PM"

        @gantry.register(
            tags=["weather", "forecast"],
            examples=["what's the weather like in London", "will it rain in Paris tomorrow"],
        )
        def weather_forecast(city: str) -> str:
            """Get the weather forecast for a specific city."""
            return f"The weather in {city} is sunny."

        @gantry.register(
            tags=["communication", "email"],
            examples=["send an email to my boss", "email Sam the meeting notes"],
        )
        def send_email(recipient: str, subject: str, body: str) -> str:
            """Send an email to a recipient."""
            return f"Email sent to {recipient} with subject '{subject}'"

        @gantry.register(
            tags=["data", "search"],
            examples=["search for customer data", "look up orders in the database"],
        )
        def search_database(query: str) -> list[str]:
            """Search the internal database for a query string."""
            return [f"Result for {query}"]

        @gantry.register(
            tags=["file", "io"],
            examples=["create a file named notes.txt", "save this text to a new file"],
        )
        def create_file(filename: str, content: str) -> str:
            """Create a new file with the specified content."""
            return f"File '{filename}' created."

        @gantry.register(
            tags=["finance", "money"],
            examples=["convert 100 USD to EUR", "how much is 50 pounds in dollars"],
        )
        def convert_currency(amount: float, from_curr: str, to_curr: str) -> str:
            """Convert an amount from one currency to another."""
            return f"{amount} {from_curr} is equivalent to {amount * 1.2} {to_curr}"

        @gantry.register(
            tags=["translation", "language"],
            examples=["translate 'Hello' to Spanish", "say this in French"],
        )
        def translate_text(text: str, target_language: str) -> str:
            """Translate text to a target language."""
            return f"Translated '{text}' to {target_language}"

        @gantry.register(
            tags=["productivity", "calendar"],
            examples=["schedule a meeting with Alice and Bob", "set up a call for 3pm"],
        )
        def schedule_meeting(participants: list[str], time: str) -> str:
            """Schedule a meeting with participants at a specific time."""
            return f"Meeting scheduled with {', '.join(participants)} at {time}"

        # --- Sync Tools ---
        print("Syncing tools to vector store...")
        await gantry.sync()
        print(f"Registered {gantry.tool_count} tools.\n")

        # --- Test Queries ---
        test_queries = [
            "I need to add 50 and 20",
            "What's the weather like in London?",
            "Send an email to boss@example.com",
            "Translate 'Hello' to Spanish",
            "Create a file named notes.txt",
            "Convert 100 USD to EUR",
            "Schedule a meeting with Alice and Bob",
            "Search for customer data",
            "What time is it in Tokyo?",
            "Join 'Hello' and 'World'",
        ]

        print("--- Semantic Retrieval Demo ---")
        # AgentGantry() picks its embedder from what is installed: the local
        # sentence-transformers model when `agent-gantry[embeddings]` is present,
        # otherwise the hash-based SimpleEmbedder, which needs nothing but has
        # poor semantic understanding. Neither needs an API key.
        print(f"Embedder in use: {type(gantry.embedder).__name__}")
        print("(SimpleEmbedder = hash-based fallback; install agent-gantry[embeddings] for real semantics)")
        print("-" * 40)

        for query in test_queries:
            relevant_tools = await gantry.retrieve_tools(query, limit=1)

            print(f"Query: '{query}'")
            if relevant_tools:
                tool_name = relevant_tools[0]["function"]["name"]
                description = relevant_tools[0]["function"]["description"]
                print(f"  -> Top Match: {tool_name} ({description})")
            else:
                print("  -> No relevant tool found.")
            print("-" * 40)

        # --- Execution Demo ---
        print("\n--- Execution Demo ---")
        print("Executing 'convert_currency'...")
        result = await gantry.execute(
            ToolCall(
                tool_name="convert_currency",
                arguments={"amount": 100, "from_curr": "USD", "to_curr": "EUR"},
            )
        )
        print(f"Result: {result.result}")
    finally:
        await gantry.close()


if __name__ == "__main__":
    asyncio.run(main())
