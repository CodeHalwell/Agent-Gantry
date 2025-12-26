import asyncio
import os

from dotenv import load_dotenv
from openai import AsyncOpenAI

from agent_gantry import AgentGantry
from agent_gantry.metrics import calculate_token_savings

# Load environment variables
load_dotenv()

async def main():
    print("=== Agent-Gantry Token Savings Demo (OpenAI) ===\n")

    # 1. Setup
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("❌ Error: OPENAI_API_KEY not found in environment.")
        print("   Please set it in your .env file.")
        return

    # Initialize OpenAI Client
    client = AsyncOpenAI(api_key=api_key)

    # Initialize Gantry
    gantry = AgentGantry()

    # 2. Register 30 tools to create a "heavy" prompt
    # This simulates a real-world scenario with many available capabilities
    print("Registering 30 tools to simulate a large toolset...")
    for i in range(30):
        def create_handler(idx):
            async def handler(data: str):
                return f"Processed {data} with tool {idx}"
            handler.__name__ = f"tool_{idx}"
            handler.__doc__ = f"This tool is specifically designed to perform specialized operations for task category {idx}."
            return handler

        gantry.register(create_handler(i))

    await gantry.sync()
    print(f"✅ Registered {gantry.tool_count} tools.\n")

    query = "I need to perform a specialized operation for task category 7"
    print(f"User Query: '{query}'\n")

    # --- Baseline: Send ALL tools ---
    print("Step 1: Running Baseline (Sending ALL 30 tools to OpenAI)...")
    all_tools = [t.to_openai_schema() for t in await gantry.list_tools()]

    baseline_response = await client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": query}],
        tools=all_tools,
        tool_choice="auto"
    )

    baseline_usage = baseline_response.usage
    print(f"📊 Baseline Prompt Tokens: {baseline_usage.prompt_tokens}")

    # --- Optimized: Send only Top-K tools ---
    print("\nStep 2: Running Optimized (Using Gantry to send only Top-2 tools)...")
    # We lower the score_threshold for the SimpleEmbedder to ensure we get matches
    optimized_tools = await gantry.retrieve_tools(query, limit=2, score_threshold=0.1)
    print(f"Gantry selected: {[t['function']['name'] for t in optimized_tools]}")

    optimized_response = await client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": query}],
        tools=optimized_tools,
        tool_choice="auto"
    )

    optimized_usage = optimized_response.usage
    print(f"📊 Optimized Prompt Tokens: {optimized_usage.prompt_tokens}")

    # --- Calculate Savings ---
    print("\n" + "="*40)
    print("       TOKEN SAVINGS REPORT")
    print("="*40)

    # calculate_token_savings accepts dicts or ProviderUsage objects
    savings = calculate_token_savings(
        baseline=baseline_usage.model_dump(),
        optimized=optimized_usage.model_dump()
    )

    print(f"📉 Saved Prompt Tokens: {savings.saved_prompt_tokens}")
    print(f"💰 Prompt Savings:     {savings.prompt_savings_pct:.1f}%")
    print(f"📉 Saved Total Tokens:  {savings.saved_total_tokens}")
    print(f"💰 Total Savings:      {savings.total_savings_pct:.1f}%")
    print("="*40)

    if optimized_response.choices[0].message.tool_calls:
        tc = optimized_response.choices[0].message.tool_calls[0]
        print(f"\n✅ Verification: LLM correctly identified and called: {tc.function.name}")
    else:
        print("\n⚠️ Warning: LLM did not call any tools in the optimized run.")

if __name__ == "__main__":
    asyncio.run(main())
