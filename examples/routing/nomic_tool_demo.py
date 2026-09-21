"""
Routing ten tools with the Nomic embedder.

Uses ``NomicEmbedder`` (nomic-embed-text-v1.5, run locally through
sentence-transformers) instead of the default MiniLM model, then checks that
each of ten queries lands on the right tool. No API key, but it needs the
``nomic`` extra and downloads the model (~550 MB) on first run.

Run with::

    pip install agent-gantry[nomic]
    python examples/routing/nomic_tool_demo.py
"""

import asyncio
import sys

from agent_gantry import AgentGantry


async def main():
    print("Initializing AgentGantry with Nomic Embeddings...")

    # NomicEmbedder imports sentence-transformers lazily, on first use, so
    # probe for it here to fail before any tools are registered.
    try:
        import sentence_transformers  # noqa: F401

        from agent_gantry.adapters.embedders.nomic import NomicEmbedder
    except ImportError:
        print("\nError: 'nomic' extra dependencies not found.")
        print("Please install them using:")
        print("  pip install agent-gantry[nomic]")
        print("  # or")
        print("  pip install sentence-transformers numpy")
        sys.exit(1)

    # Matryoshka truncation: 256 dims for speed; Nomic supports up to 768.
    embedder = NomicEmbedder(dimension=256)
    gantry = AgentGantry(embedder=embedder)

    try:
        # --- Register 10 Different Tools ---

        @gantry.register(
            tags=["math", "calculation"], examples=["add 50 and 20", "what is 7 plus 12"]
        )
        def add_numbers(a: float, b: float) -> float:
            """Add two numbers together."""
            return a + b

        @gantry.register(
            tags=["string", "text"],
            examples=["join 'Hello' and 'World'", "stick these two words together"],
        )
        def concat_strings(s1: str, s2: str) -> str:
            """Concatenate or join two strings together."""
            return s1 + s2

        @gantry.register(
            tags=["time", "date"], examples=["what time is it in Tokyo", "current time in UTC"]
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
            examples=["send an email to my boss", "email Priya the quarterly report"],
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
            examples=["translate 'Hello' to Spanish", "how do you say thank you in French"],
        )
        def translate_text(text: str, target_language: str) -> str:
            """Translate text to a target language."""
            return f"Translated '{text}' to {target_language}"

        @gantry.register(
            tags=["productivity", "calendar"],
            examples=["schedule a meeting with Alice and Bob", "book a call for 3pm tomorrow"],
        )
        def schedule_meeting(participants: list[str], time: str) -> str:
            """Schedule a meeting with participants at a specific time."""
            return f"Meeting scheduled with {', '.join(participants)} at {time}"

        # --- Sync Tools ---
        print("Syncing tools to vector store (this may take a moment to download the model)...")
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

        print("--- Semantic Retrieval Demo (Nomic) ---")
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
    finally:
        await gantry.close()


if __name__ == "__main__":
    asyncio.run(main())
