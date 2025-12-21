import asyncio

from agent_gantry import AgentGantry
from agent_gantry.schema.execution import ToolCall


async def main():
    print("Initializing AgentGantry...")
    gantry = AgentGantry()

    # --- Register 10 Different Tools ---

    @gantry.register(tags=["math", "calculation"])
    def add_numbers(a: float, b: float) -> float:
        """Add two numbers together."""
        return a + b

    @gantry.register(tags=["string", "text"])
    def concat_strings(s1: str, s2: str) -> str:
        """Concatenate two strings."""
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
        "Join 'Hello' and 'World'"
    ]

    print("--- Semantic Retrieval Demo ---")
    print("Note: This demo uses the default 'SimpleEmbedder' which uses deterministic hashing.")
    print("It is fast and requires no API keys, but has poor semantic understanding.")
    print("For production accuracy, configure OpenAI or Nomic embeddings.")
    print("-" * 40)

    for query in test_queries:
        # Using score_threshold=0.1 for SimpleEmbedder compatibility
        relevant_tools = await gantry.retrieve_tools(query, limit=1, score_threshold=0.1)

        print(f"Query: '{query}'")
        if relevant_tools:
            tool_name = relevant_tools[0]['function']['name']
            description = relevant_tools[0]['function']['description']
            print(f"  -> Top Match: {tool_name} ({description})")
        else:
            print("  -> No relevant tool found.")
        print("-" * 40)

    # --- Execution Demo ---
    print("\n--- Execution Demo ---")
    print("Executing 'convert_currency'...")
    result = await gantry.execute(ToolCall(
        tool_name="convert_currency",
        arguments={"amount": 100, "from_curr": "USD", "to_curr": "EUR"}
    ))
    print(f"Result: {result.result}")

if __name__ == "__main__":
    asyncio.run(main())
