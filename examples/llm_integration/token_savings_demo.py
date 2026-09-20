import asyncio
import json
import os
from typing import Any

from dotenv import load_dotenv

from agent_gantry import AgentGantry

# Load environment variables from .env file
load_dotenv()

# Try to import OpenAI
try:
    from openai import AsyncOpenAI

    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

# Try to import tiktoken for exact token counting
try:
    import tiktoken

    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False


def count_tokens(tools: list[dict[str, Any]], model: str = "gpt-5.4-mini") -> int:
    """
    Count exact tokens for a list of tool schemas using tiktoken.
    """
    if not TIKTOKEN_AVAILABLE:
        # Fallback to estimation
        json_str = json.dumps(tools)
        return len(json_str) // 4

    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        # GPT-5.x models use o200k_base; fall back to cl100k_base if unknown
        try:
            encoding = tiktoken.get_encoding("o200k_base")
        except Exception:
            encoding = tiktoken.get_encoding("cl100k_base")

    # Use compact separators to match API behavior closer
    json_str = json.dumps(tools, separators=(",", ":"))
    return len(encoding.encode(json_str))


def estimate_tokens(tools: list[dict[str, Any]]) -> int:
    """
    Estimate token count for a list of tool schemas.
    Rule of thumb: ~4 characters per token for JSON.
    """
    return count_tokens(tools)


async def main():
    print("=== Agent-Gantry Token Savings Demo ===\n")

    # 1. Initialize Gantry
    print("1. Initializing AgentGantry...")
    try:
        from agent_gantry.adapters.embedders.nomic import NomicEmbedder

        embedder = NomicEmbedder(dimension=256)
        gantry = AgentGantry(embedder=embedder)
        print("   Using Nomic Embeddings (High Accuracy)")
    except ImportError:
        gantry = AgentGantry()
        print("   Using Simple Embeddings (Low Accuracy - Fallback)")

    try:
        # 2. Register 15 Diverse Tools
        print("2. Registering 15 diverse tools...")

        # Math
        @gantry.register(tags=["math"], examples=["add 7 and 12", "what is 50 plus 20"])
        def add_numbers(a: float, b: float) -> float:
            """Add two numbers."""
            return a + b

        @gantry.register(tags=["math"], examples=["what is 100 minus 37", "take 4 away from 10"])
        def subtract_numbers(a: float, b: float) -> float:
            """Subtract b from a."""
            return a - b

        @gantry.register(
            tags=["math"], examples=["what is 100 multiplied by 5", "multiply 6 by 7"]
        )
        def multiply_numbers(a: float, b: float) -> float:
            """Multiply two numbers."""
            return a * b

        @gantry.register(tags=["math"], examples=["divide 100 by 4", "what is 81 over 9"])
        def divide_numbers(a: float, b: float) -> float:
            """Divide a by b."""
            return a / b

        # Weather
        @gantry.register(
            tags=["weather"], examples=["what's the weather in London", "is it raining in Leeds"]
        )
        def get_weather(city: str) -> str:
            """Get current weather."""
            return f"Weather in {city}"

        @gantry.register(
            tags=["weather"],
            examples=["what's the forecast for Paris this week", "will it rain tomorrow in Oslo"],
        )
        def get_forecast(city: str, days: int) -> str:
            """Get weather forecast."""
            return f"Forecast for {city}"

        # Finance
        @gantry.register(
            tags=["finance"], examples=["what is AAPL trading at", "current share price for Tesco"]
        )
        def get_stock_price(ticker: str) -> float:
            """Get stock price."""
            return 100.0

        @gantry.register(
            tags=["finance"], examples=["convert 100 USD to EUR", "how much is 50 pounds in dollars"]
        )
        def convert_currency(amount: float, from_curr: str, to_curr: str) -> float:
            """Convert currency."""
            return amount * 1.2

        # Communication
        @gantry.register(
            tags=["email"], examples=["email the team about the outage", "send Sam a note"]
        )
        def send_email(to: str, subject: str, body: str) -> bool:
            """Send an email."""
            return True

        @gantry.register(
            tags=["email"], examples=["open the latest email from Priya", "read my newest message"]
        )
        def read_email(message_id: str) -> str:
            """Read an email."""
            return "Email content"

        # File System
        @gantry.register(
            tags=["file"], examples=["create a file called notes.txt", "save this text to a new file"]
        )
        def create_file(path: str, content: str) -> bool:
            """Create a file."""
            return True

        @gantry.register(
            tags=["file"], examples=["show me the contents of config.yaml", "open the README file"]
        )
        def read_file(path: str) -> str:
            """Read a file."""
            return "File content"

        # Language
        @gantry.register(
            tags=["translation"], examples=["translate this into Spanish", "say 'hello' in French"]
        )
        def translate_text(text: str, target_lang: str) -> str:
            """Translate text."""
            return "Translated text"

        @gantry.register(
            tags=["translation"],
            examples=["what language is this written in", "detect the language of this text"],
        )
        def detect_language(text: str) -> str:
            """Detect language."""
            return "en"

        # Search
        @gantry.register(
            tags=["search"], examples=["search the web for Python tutorials", "google the latest news"]
        )
        def search_web(query: str) -> list[str]:
            """Search the web."""
            return ["Result 1", "Result 2"]

        await gantry.sync()
        print(f"   Registered {gantry.tool_count} tools.\n")

        # 3. Define Query
        user_query = "What is 100 multiplied by 5?"
        print(f"--- Query: '{user_query}' ---\n")

        api_key = os.environ.get("OPENAI_API_KEY")
        client = None
        if OPENAI_AVAILABLE and api_key:
            client = AsyncOpenAI(api_key=api_key)

        # 4. Baseline: No Tools
        baseline_tokens = 0
        if client:
            print("--- Baseline: Query ONLY (No Tools) ---")
            try:
                response_base = await client.chat.completions.create(
                    model="gpt-5.4-mini",
                    messages=[{"role": "user", "content": user_query}],
                )
                baseline_tokens = response_base.usage.prompt_tokens
                print(f"   [Actual API Usage] Total Prompt Tokens: {baseline_tokens}")
                print("   (This is the cost of the query + system overhead without tools)\n")
            except Exception as e:
                print(f"   [API Error] {e}\n")

        # 5. Scenario A: All Tools (The "Context Window Tax")
        print("--- Scenario A: Passing ALL 15 Tools ---")

        # Get all tools as OpenAI schemas
        all_tool_defs = await gantry.list_tools()
        all_tools_schema = [t.to_dialect("openai") for t in all_tool_defs]

        est_tokens_all = count_tokens(all_tools_schema)
        print(f"   Tools passed: {len(all_tools_schema)}")
        print(f"   Exact Tool Definition Tokens (tiktoken, compact): {est_tokens_all}")

        if client:
            try:
                response_a = await client.chat.completions.create(
                    model="gpt-5.4-mini",
                    messages=[{"role": "user", "content": user_query}],
                    tools=all_tools_schema,
                    tool_choice="auto",
                )
                usage_a = response_a.usage.prompt_tokens
                print(f"   [Actual API Usage] Total Prompt Tokens: {usage_a}")

                if baseline_tokens > 0:
                    tool_cost_a = usage_a - baseline_tokens
                    print(f"   [Actual Tool Cost] ~{tool_cost_a} tokens (Total - Baseline)")

            except Exception as e:
                print(f"   [API Error] {e}")
                usage_a = est_tokens_all  # Fallback for comparison
        else:
            print("   [Skipping API Call] OPENAI_API_KEY not found or openai not installed.")
            usage_a = est_tokens_all

        print("\n")

        # 6. Scenario B: Agent Gantry (Context Reduction)
        print("--- Scenario B: Using Agent Gantry (Top 2 Tools) ---")

        # Retrieve only relevant tools
        relevant_tools_schema = await gantry.retrieve_tools(user_query, limit=2)

        est_tokens_filtered = count_tokens(relevant_tools_schema)
        print(f"   Tools passed: {len(relevant_tools_schema)}")
        for t in relevant_tools_schema:
            print(f"     - {t['function']['name']}")
        print(f"   Exact Tool Definition Tokens (tiktoken, compact): {est_tokens_filtered}")

        if client:
            try:
                response_b = await client.chat.completions.create(
                    model="gpt-5.4-mini",
                    messages=[{"role": "user", "content": user_query}],
                    tools=relevant_tools_schema,
                    tool_choice="auto",
                )
                usage_b = response_b.usage.prompt_tokens
                print(f"   [Actual API Usage] Total Prompt Tokens: {usage_b}")

                if baseline_tokens > 0:
                    tool_cost_b = usage_b - baseline_tokens
                    print(f"   [Actual Tool Cost] ~{tool_cost_b} tokens (Total - Baseline)")

                # Calculate Savings
                savings = usage_a - usage_b
                percent = (savings / usage_a) * 100
                print(f"\n   >>> SAVINGS: {savings} tokens ({percent:.1f}%) <<<")

            except Exception as e:
                print(f"   [API Error] {e}")
        else:
            # Calculate Estimated Savings
            savings = usage_a - est_tokens_filtered
            percent = (savings / usage_a) * 100 if usage_a > 0 else 0
            print(f"\n   >>> ESTIMATED SAVINGS: ~{savings} tokens ({percent:.1f}%) <<<")
    finally:
        await gantry.close()


if __name__ == "__main__":
    asyncio.run(main())
