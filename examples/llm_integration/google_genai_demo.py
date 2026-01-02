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
    retrieval_result = await gantry.retrieve(ToolQuery(
        context=ConversationContext(query=user_query),
        limit=1,
        score_threshold=0.1
    ))

    # B. Convert to Gemini Schema
    # Agent-Gantry provides `to_gemini_schema()` which returns a dict compatible with FunctionDeclaration
    gemini_tools = []
    for t in retrieval_result.tools:
        schema = t.tool.to_gemini_schema()

        # Create Gemini FunctionDeclaration object
        func_decl = types.FunctionDeclaration(
            name=schema["name"],
            description=schema["description"],
            parameters=schema["parameters"]
        )
        gemini_tools.append(func_decl)

    # Wrap in Tool object
    if gemini_tools:
        tool = types.Tool(function_declarations=gemini_tools)
        config = types.GenerateContentConfig(tools=[tool])
    else:
        config = None

    print(f"Gantry retrieved {len(gemini_tools)} tool(s)")

    # C. Call Gemini
    # Note: Gemini 2.0 Flash is a good default
    response = client.models.generate_content(
        model='models/gemini-3-flash-preview',
        contents=user_query,
        config=config
    )

    # Inspect response for function calls
    for part in response.candidates[0].content.parts:
        if fn := part.function_call:
            print(f"Gemini decided to call: {fn.name}({fn.args})")

            # Execute securely via Gantry
            # Note: fn.args is a dict in the new SDK
            result = await gantry.execute(ToolCall(
                tool_name=fn.name,
                arguments=fn.args
            ))
            print(f"Execution Result: {result.result}")

    # --- Scenario: Using the Decorator ---
    print("\n--- Scenario: Using @with_semantic_tools Decorator (RECOMMENDED) ---")

    # The decorator uses the default gantry set above and automatically
    # converts tools to Gemini format via dialect="gemini"
    @with_semantic_tools(limit=1, score_threshold=0.1, dialect="gemini")
    async def chat_with_gemini(prompt: str, tools: list = None):
        # Convert the injected 'tools' (list of dicts) to Gemini Tool objects
        if tools:
            print(f"   [Decorator] Injected {len(tools)} tools")
            gemini_funcs = []
            for t in tools:
                # t is {'type': 'function', 'function': {...}}
                f_spec = t['function']
                gemini_funcs.append(types.FunctionDeclaration(
                    name=f_spec['name'],
                    description=f_spec['description'],
                    parameters=f_spec['parameters']
                ))
            toolbox = types.Tool(function_declarations=gemini_funcs)
            config = types.GenerateContentConfig(tools=[toolbox])
        else:
            print("   [Decorator] No tools injected")
            config = None

        return client.models.generate_content(
            model='models/gemini-3-flash-preview',
            contents=prompt,
            config=config
        )

    query_dec = "Find documents about project beta"
    print(f"User Query: '{query_dec}'")

    response_dec = await chat_with_gemini(prompt=query_dec)

    for part in response_dec.candidates[0].content.parts:
        if fn := part.function_call:
            print(f"Gemini decided to call: {fn.name}")

if __name__ == "__main__":
    asyncio.run(main())
