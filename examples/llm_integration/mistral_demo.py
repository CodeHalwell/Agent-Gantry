import asyncio
import os
import json
from typing import Any

from dotenv import load_dotenv
from agent_gantry import AgentGantry
from agent_gantry.schema.execution import ToolCall

# Load environment variables
load_dotenv()

async def main():
    print("=== Agent-Gantry + Mistral AI Integration Demo ===\n")

    # 1. Check for API Key
    api_key = os.environ.get("MISTRAL_API_KEY")
    if not api_key:
        print("❌ Error: MISTRAL_API_KEY not found in environment.")
        print("   Please set it in your .env file.")
        return

    # 2. Initialize Gantry
    gantry = AgentGantry()

    # 3. Register Tools
    @gantry.register(tags=["translation"])
    def translate_text(text: str, target_lang: str) -> str:
        """Translate text to a target language."""
        return f"Translated '{text}' to {target_lang}: [Translated Text]"

    await gantry.sync()
    print(f"✅ Registered {gantry.tool_count} tools\n")

    # 4. Initialize Mistral Client
    from mistralai.client import MistralClient
    from mistralai.models.chat_completion import ChatMessage

    # Note: Mistral SDK might be sync or async depending on version. 
    # Assuming sync client for this demo, or async if available.
    # We'll use the sync client wrapped in a thread for simplicity if needed, 
    # but let's assume standard usage.
    client = MistralClient(api_key=api_key)

    # --- Scenario: Dynamic Retrieval ---
    print("--- Scenario: Dynamic Retrieval ---")
    query = "Translate 'Hello World' to French"
    print(f"User Query: '{query}'")

    # Retrieve tools (OpenAI format is compatible with Mistral)
    tools = await gantry.retrieve_tools(query, limit=1)
    print(f"Gantry retrieved {len(tools)} tool(s)")

    # Call Mistral
    # Mistral's `tools` parameter accepts the same JSON schema structure
    response = client.chat(
        model="mistral-large-latest",
        messages=[ChatMessage(role="user", content=query)],
        tools=tools,
        tool_choice="auto"
    )

    tool_calls = response.choices[0].message.tool_calls
    if tool_calls:
        for tc in tool_calls:
            print(f"Mistral decided to call: {tc.function.name}({tc.function.arguments})")
            
            # Execute securely via Gantry
            result = await gantry.execute(ToolCall(
                tool_name=tc.function.name,
                arguments=json.loads(tc.function.arguments)
            ))
            print(f"Execution Result: {result.result}")

    # --- Scenario: Using @with_semantic_tools Decorator ---
    print("\n--- Scenario: Using @with_semantic_tools Decorator ---")
    
    from agent_gantry.integrations.decorator import with_semantic_tools

    # Note: The decorator must be async if the wrapped function is async, 
    # or it can wrap a sync function if running in an async context.
    # Since MistralClient is sync here, we wrap a sync function but call it 
    # from our async main loop.
    
    @with_semantic_tools(gantry, limit=1)
    def chat_with_mistral(user_query: str, tools: list[dict[str, Any]] = None):
        """
        This function automatically gets relevant tools injected into the 'tools' argument
        based on the user_query.
        """
        print(f"Decorator injected {len(tools) if tools else 0} tools")
        
        response = client.chat(
            model="mistral-large-latest",
            messages=[ChatMessage(role="user", content=user_query)],
            tools=tools,
            tool_choice="auto"
        )
        
        if response.choices[0].message.tool_calls:
            tc = response.choices[0].message.tool_calls[0]
            print(f"Mistral (via decorator) called: {tc.function.name}")
            return tc.function.name
        return "No tool called"

    # The decorator handles the retrieval logic internally
    # Note: Since the decorated function is sync, we call it directly.
    # If the decorated function was async, we would await it.
    chat_with_mistral("Translate 'Good Morning' to Spanish")

if __name__ == "__main__":
    asyncio.run(main())
