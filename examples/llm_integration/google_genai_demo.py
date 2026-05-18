import asyncio
import os

from dotenv import load_dotenv

from agent_gantry import AgentGantry, set_default_gantry, with_semantic_tools
from agent_gantry.schema.execution import ToolCall
from agent_gantry.schema.query import ConversationContext, ToolQuery

# Load environment variables
load_dotenv()


async def main():
    print("=== Agent-Gantry + Google GenAI (Gemini) Integration Demo ===\n")

    # 1. Check for API Key
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        print("❌ Error: GOOGLE_API_KEY not found in environment.")
        print("   Please set it in your .env file.")
        return

    # 2. Initialize Gantry
    gantry = AgentGantry()
    set_default_gantry(gantry)  # Set default for decorator usage

    # 3. Register Tools
    @gantry.register(tags=["search"])
    def search_knowledge_base(query: str) -> str:
        """Search the internal knowledge base for documents."""
        return f"Found 2 documents for '{query}': [Doc A, Doc B]"

    await gantry.sync()
    print(f"✅ Registered {gantry.tool_count} tools\n")

    # 4. Initialize Google GenAI
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=api_key)

    # --- Scenario: Dynamic Retrieval with Gemini Schema ---
    print("--- Scenario: Dynamic Retrieval with Gemini Schema ---")
    user_query = "Find documents about project alpha"
    print(f"User Query: '{user_query}'")

    # A. Retrieve Tools
    retrieval_result = await gantry.retrieve(
        ToolQuery(context=ConversationContext(query=user_query), limit=1, score_threshold=0.1)
    )

    # B. Convert to Gemini Schema using to_dialect("gemini")
    gemini_tools = []
    for t in retrieval_result.tools:
        schema = t.tool.to_dialect("gemini")

        # Create Gemini FunctionDeclaration object
        func_decl = types.FunctionDeclaration(
            name=schema["name"], description=schema["description"], parameters=schema["parameters"]
        )
        gemini_tools.append(func_decl)

    # Wrap in Tool object
    if gemini_tools:
        tool = types.Tool(function_declarations=gemini_tools)
        config = types.GenerateContentConfig(tools=[tool])
    else:
        config = None

    print(f"Gantry retrieved {len(gemini_tools)} tool(s)")

    # C. Call Gemini asynchronously via client.aio
    response = await client.aio.models.generate_content(
        model="gemini-2.5-flash", contents=user_query, config=config
    )

    # Inspect response for function calls and complete the round-trip
    if response.function_calls:
        # Build function response parts for each call, then send them back.
        # For Gemini 2.x models each FunctionResponse must echo the matching
        # `id` from the FunctionCall so the model can correlate parallel calls.
        # Source: https://ai.google.dev/gemini-api/docs/function-calling
        function_response_parts = []
        for fn in response.function_calls:
            print(f"Gemini decided to call: {fn.name}({fn.args})")
            result = await gantry.execute(ToolCall(tool_name=fn.name, arguments=dict(fn.args)))
            print(f"Execution Result: {result.result}")
            fn_response = types.Part.from_function_response(
                name=fn.name,
                response={"result": result.result if result.result is not None else ""},
            )
            function_response_parts.append(fn_response)

        # Continue the conversation with the tool results
        followup = await client.aio.models.generate_content(
            model="gemini-2.5-flash",
            contents=[
                user_query,
                response.candidates[0].content,  # model's function_call turn
                types.Content(role="user", parts=function_response_parts),
            ],
            config=config,
        )
        print(f"Final answer: {followup.text}")

    # --- Scenario: Using the Decorator ---
    print("\n--- Scenario: Using @with_semantic_tools Decorator (RECOMMENDED) ---")

    # The decorator uses the default gantry set above and automatically
    # converts tools to Gemini format via dialect="gemini".
    # With dialect="gemini", each injected tool dict has the shape:
    #   {"name": "...", "description": "...", "parameters": {...}}
    @with_semantic_tools(limit=1, score_threshold=0.1, dialect="gemini")
    async def chat_with_gemini(prompt: str, tools: list = None):
        if tools:
            print(f"   [Decorator] Injected {len(tools)} tools")
            gemini_funcs = []
            for t in tools:
                # t is a Gemini function declaration dict: {"name": "...", "description": "...", "parameters": {...}}
                gemini_funcs.append(
                    types.FunctionDeclaration(
                        name=t["name"],
                        description=t["description"],
                        parameters=t["parameters"],
                    )
                )
            toolbox = types.Tool(function_declarations=gemini_funcs)
            cfg = types.GenerateContentConfig(tools=[toolbox])
        else:
            print("   [Decorator] No tools injected")
            cfg = None

        return await client.aio.models.generate_content(
            model="gemini-2.5-flash", contents=prompt, config=cfg
        )

    query_dec = "Find documents about project beta"
    print(f"User Query: '{query_dec}'")

    response_dec = await chat_with_gemini(prompt=query_dec)

    if response_dec.function_calls:
        for fn in response_dec.function_calls:
            print(f"Gemini decided to call: {fn.name}")


if __name__ == "__main__":
    asyncio.run(main())
