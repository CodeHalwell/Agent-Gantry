"""
Google GenAI (Gemini) + dotfetch integration demo.

Shows three ways to hand dotfetch-selected tools to Gemini: manual ``retrieve``
+ ``to_dialect("gemini")`` with the full function-call / function-response
round trip, the ``@with_semantic_tools(dialect="gemini")`` decorator, and the
typed ``GeminiAdapter``.

Requires GOOGLE_API_KEY; exits with a message if it is missing.

Install:
    pip install "dotfetch[google-genai]"
"""

import asyncio
import os

from dotenv import load_dotenv

from dotfetch import DotFetch, set_default_fetcher, with_semantic_tools
from dotfetch.schema.execution import ToolCall
from dotfetch.schema.query import ConversationContext, ToolQuery

# Load environment variables
load_dotenv()


async def main():
    print("=== dotfetch + Google GenAI (Gemini) Integration Demo ===\n")

    # 1. Check for API Key
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        print("❌ Error: GOOGLE_API_KEY not found in environment.")
        print("   Please set it in your .env file.")
        return

    # 2. Initialize dotfetch
    fetcher = DotFetch()
    set_default_fetcher(fetcher)  # Set default for decorator usage

    try:
        # 3. Register Tools
        @fetcher.register(
            tags=["search"],
            examples=["find documents about project alpha", "search the knowledge base for onboarding"],
        )
        def search_knowledge_base(query: str) -> str:
            """Search the internal knowledge base for documents."""
            return f"Found 2 documents for '{query}': [Doc A, Doc B]"

        await fetcher.sync()
        print(f"✅ Registered {fetcher.tool_count} tools\n")

        # 4. Initialize Google GenAI
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=api_key)

        # --- Scenario: Dynamic Retrieval with Gemini Schema ---
        print("--- Scenario: Dynamic Retrieval with Gemini Schema ---")
        user_query = "Find documents about project alpha"
        print(f"User Query: '{user_query}'")

        # A. Retrieve Tools
        retrieval_result = await fetcher.retrieve(
            ToolQuery(context=ConversationContext(query=user_query), limit=1, score_threshold=0.0)
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

        print(f"dotfetch retrieved {len(gemini_tools)} tool(s)")

        # C. Call Gemini asynchronously via client.aio
        response = await client.aio.models.generate_content(
            model="gemini-2.5-flash", contents=user_query, config=config
        )

        # Inspect response for function calls and send results back.
        # Gemini requires a multi-turn exchange: the model's function_call Part
        # and our function_response Part must be included in the next request so
        # the model can compose a final text answer.
        # Use types.Part.from_function_response() (recommended over raw dicts) and
        # echo back the call id so the API can correlate parallel calls.
        # Source: https://ai.google.dev/gemini-api/docs/function-calling
        if response.function_calls:
            tool_result_parts = []
            for fn in response.function_calls:
                print(f"Gemini decided to call: {fn.name}({fn.args})")

                # Execute securely via dotfetch
                result = await fetcher.execute(ToolCall(tool_name=fn.name, arguments=dict(fn.args)))
                print(f"Execution Result: {result.result}")

                # Build the function response Part using the SDK helper; include
                # the call id to support parallel function calls correctly.
                tool_result_parts.append(
                    types.Part.from_function_response(
                        name=fn.name,
                        response={"result": result.result},
                        # id is present on google-genai >= 1.x for parallel calls
                        **({"id": fn.id} if getattr(fn, "id", None) else {}),
                    )
                )

            # Send tool results back — include the original model response so Gemini
            # has context, then append our function_response parts.
            if tool_result_parts:
                followup = await client.aio.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=[
                        user_query,                      # original user message
                        response.candidates[0].content,  # model's function_call turn
                        types.Content(role="tool", parts=tool_result_parts),
                    ],
                    config=config,
                )
                print(f"Gemini final answer: {followup.text}")

        # --- Scenario: Using the Decorator ---
        print("\n--- Scenario: Using @with_semantic_tools Decorator (RECOMMENDED) ---")

        # The decorator uses the default fetcher set above and automatically
        # converts tools to Gemini format via dialect="gemini".
        # With dialect="gemini", each injected tool dict has the shape:
        #   {"name": "...", "description": "...", "parameters": {...}}
        @with_semantic_tools(limit=1, dialect="gemini")
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

        # --- Scenario: Typed GeminiAdapter (provider-specific convenience) ---
        # GeminiAdapter(fetcher).tools(query, limit=n) bakes in dialect="gemini" and
        # returns {"name", "description", "parameters"} dicts ready to feed straight
        # into types.FunctionDeclaration — no retrieve + to_dialect("gemini") needed.
        print("\n--- Scenario: Typed GeminiAdapter ---")
        from dotfetch.gemini import GeminiAdapter

        query_adapter = "Find documents about project gamma"
        print(f"User Query: '{query_adapter}'")

        tool_dicts = await GeminiAdapter(fetcher).tools(query_adapter, limit=1)
        print(f"dotfetch retrieved {len(tool_dicts)} tool(s)")

        if tool_dicts:
            decls = [
                types.FunctionDeclaration(
                    name=t["name"], description=t["description"], parameters=t["parameters"]
                )
                for t in tool_dicts
            ]
            cfg_adapter = types.GenerateContentConfig(
                tools=[types.Tool(function_declarations=decls)]
            )
            response_adapter = await client.aio.models.generate_content(
                model="gemini-2.5-flash", contents=query_adapter, config=cfg_adapter
            )
            if response_adapter.function_calls:
                for fn in response_adapter.function_calls:
                    print(f"Gemini (via adapter) called: {fn.name}")
    finally:
        await fetcher.close()


if __name__ == "__main__":
    asyncio.run(main())
