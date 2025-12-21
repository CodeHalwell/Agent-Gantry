import asyncio
import sys

from agent_gantry import AgentGantry


async def main():
    print("Initializing AgentGantry with Nomic Embeddings...")

    try:
        from agent_gantry.adapters.embedders.nomic import NomicEmbedder
        # Initialize with Nomic embedder
        # We use a smaller dimension (256) for speed, but Nomic supports up to 768
        embedder = NomicEmbedder(dimension=256)
        gantry = AgentGantry(embedder=embedder)
    except ImportError:
        print("\nError: 'nomic' extra dependencies not found.")
        print("Please install them using:")
        print("  pip install agent-gantry[nomic]")
        print("  # or")
        print("  pip install sentence-transformers numpy")
        sys.exit(1)
    except Exception as e:
        print(f"\nError initializing Nomic embedder: {e}")
        sys.exit(1)

    # --- Register 10 Different Tools ---

    @gantry.register(tags=["math", "calculation"])
    def add_numbers(a: float, b: float) -> float:
        """Add two numbers together."""
        return a + b

    @gantry.register(tags=["string", "text"])
    def concat_strings(s1: str, s2: str) -> str:
        """Concatenate or join two strings together."""
        return s1 + s2

    @gantry.register(tags=["time", "date"])
    def get_current_time(timezone: str = "UTC") -> str:
        """Get the current time in a specific timezone."""
        return f"The time in {timezone} is 12:00 PM"

    @gantry.register(tags=["weather", "forecast"])
    def weather_forecast(city: str) -> str:
        """Get the weather forecast for a specific city."""
        return f"The weather in {city} is sunny."

    @gantry.register(tags=["communication", "email"])
    def send_email(recipient: str, subject: str, body: str) -> str:
        """Send an email to a recipient."""
        return f"Email sent to {recipient} with subject '{subject}'"

    @gantry.register(tags=["data", "search"])
    def search_database(query: str) -> list[str]:
        """Search the internal database for a query string."""
        return [f"Result for {query}"]

    @gantry.register(tags=["file", "io"])
    def create_file(filename: str, content: str) -> str:
        """Create a new file with the specified content."""
        return f"File '{filename}' created."

    @gantry.register(tags=["finance", "money"])
    def convert_currency(amount: float, from_curr: str, to_curr: str) -> str:
        """Convert an amount from one currency to another."""
        return f"{amount} {from_curr} is equivalent to {amount * 1.2} {to_curr}"

    @gantry.register(tags=["translation", "language"])
    def translate_text(text: str, target_language: str) -> str:
        """Translate text to a target language."""
        return f"Translated '{text}' to {target_language}"

    @gantry.register(tags=["productivity", "calendar"])
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
        "Join 'Hello' and 'World'"
    ]

    print("--- Semantic Retrieval Demo (Nomic) ---")
    for query in test_queries:
        # With Nomic, we can use a higher threshold (default 0.5)
        # because the embeddings are semantically meaningful.
        relevant_tools = await gantry.retrieve_tools(query, limit=1, score_threshold=0.4)

        print(f"Query: '{query}'")
        if relevant_tools:
            tool_name = relevant_tools[0]['function']['name']
            description = relevant_tools[0]['function']['description']
            print(f"  -> Top Match: {tool_name} ({description})")
        else:
            print("  -> No relevant tool found.")
        print("-" * 40)

if __name__ == "__main__":
    asyncio.run(main())
