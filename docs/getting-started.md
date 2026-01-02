---
layout: default
title: Getting Started
nav_order: 2
description: "Get up and running with Agent-Gantry in 5 minutes"
---

# Getting Started with Agent-Gantry

This guide will get you up and running with Agent-Gantry in about 5 minutes.

---

## Prerequisites

- Python 3.10 or higher
- Basic knowledge of async/await in Python
- An LLM provider API key (OpenAI, Anthropic, Google, etc.)

## Installation

### Basic Installation

```bash
pip install agent-gantry
```

### With LLM Provider Support

```bash
# All LLM providers (OpenAI, Anthropic, Google, Mistral, Groq)
pip install agent-gantry[llm-providers]

# Individual providers
pip install agent-gantry[openai]        # OpenAI, Azure OpenAI
pip install agent-gantry[anthropic]     # Anthropic Claude
pip install agent-gantry[google-genai]  # Google Gemini
pip install agent-gantry[mistral]       # Mistral AI
pip install agent-gantry[groq]          # Groq
```

### With Local Persistence

```bash
# LanceDB for local vector storage + Nomic embeddings
pip install agent-gantry[lancedb,nomic]
```

### Everything

```bash
# All features, providers, and integrations
pip install agent-gantry[all]
```

## Your First Agent-Gantry Application

Let's build a simple agent that can help with weather and calculations.

### Step 1: Set Up Environment

Create a new file `my_agent.py` and import the necessary modules:

```python
import asyncio
import ast
import json
from openai import AsyncOpenAI
from agent_gantry import AgentGantry, with_semantic_tools, set_default_gantry

# Initialize OpenAI client
client = AsyncOpenAI()  # Requires OPENAI_API_KEY in environment

# Initialize Agent-Gantry
gantry = AgentGantry()
set_default_gantry(gantry)
```

<div class="callout note">
<div class="callout-title">💡 Why set_default_gantry?</div>

`set_default_gantry()` uses context variables for thread-safe and async-safe state management. This allows the `@with_semantic_tools` decorator to automatically access the gantry instance without explicit passing.

</div>

### Step 2: Register Tools

Define and register your tools using the `@gantry.register` decorator:

```python
@gantry.register(
    tags=["weather", "forecast"],
    description="Get current weather conditions for any city"
)
def get_weather(city: str, units: str = "fahrenheit") -> str:
    """
    Get the current weather for a city.

    Args:
        city: Name of the city
        units: Temperature units (fahrenheit or celsius)

    Returns:
        Weather description
    """
    # In production, this would call a real weather API
    return f"The weather in {city} is 72°{units[0].upper()} and sunny."


@gantry.register(
    tags=["math", "calculation"],
    description="Perform mathematical calculations"
)
def calculate(expression: str) -> float:
    """
    Evaluate a mathematical expression.

    Args:
        expression: Math expression to evaluate (e.g., "15 * 8")

    Returns:
        Result of the calculation
    """
    try:
        node = ast.parse(expression, mode="eval").body

        def _evaluate(node: ast.AST) -> float:
            if isinstance(node, ast.BinOp) and isinstance(
                node.op,
                (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Mod, ast.Pow),
            ):
                left = _evaluate(node.left)
                right = _evaluate(node.right)
                if isinstance(node.op, ast.Add):
                    return left + right
                if isinstance(node.op, ast.Sub):
                    return left - right
                if isinstance(node.op, ast.Mult):
                    return left * right
                if isinstance(node.op, ast.Div):
                    return left / right
                if isinstance(node.op, ast.Mod):
                    return left % right
                return left ** right
            if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
                value = _evaluate(node.operand)
                return value if isinstance(node.op, ast.UAdd) else -value
            if isinstance(node, ast.Constant):
                value = ast.literal_eval(node)
                if isinstance(value, (int, float)):
                    return float(value)
                raise ValueError("Only numeric literals are allowed")
            raise ValueError("Unsupported expression. Allowed operators: +, -, *, /, %, **")

        return float(_evaluate(node))
    except Exception as e:
        return f"Error: {e}"


@gantry.register(
    tags=["unit conversion"],
    description="Convert between different units of measurement"
)
def convert_temperature(value: float, from_unit: str, to_unit: str) -> float:
    """
    Convert temperature between Fahrenheit and Celsius.

    Args:
        value: Temperature value
        from_unit: Source unit (F or C)
        to_unit: Target unit (F or C)

    Returns:
        Converted temperature
    """
    if from_unit.upper() == "F" and to_unit.upper() == "C":
        return (value - 32) * 5/9
    elif from_unit.upper() == "C" and to_unit.upper() == "F":
        return (value * 9/5) + 32
    else:
        return value
```

### Step 3: Sync Tools to Vector Store

Before using semantic search, sync tools to the vector store:

```python
async def main():
    # Sync tools to enable semantic search
    await gantry.sync()
    print(f"✓ Synced {len(gantry._registry._tools)} tools to vector store")
```

### Step 4: Create Your LLM Function

Add the `@with_semantic_tools` decorator to automatically inject relevant tools:

```python
    @with_semantic_tools(limit=3, dialect="openai")
    async def chat(prompt: str, *, tools=None):
        """Chat with the LLM, automatically providing relevant tools."""
        response = await client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            tools=tools,  # Automatically injected by decorator
            tool_choice="auto"
        )
        return response
```

<div class="callout tip">
<div class="callout-title">✨ The Magic of @with_semantic_tools</div>

The decorator automatically:
1. Extracts the user prompt from your function arguments
2. Performs semantic search to find relevant tools
3. Converts tools to the target LLM provider format
4. Injects tools into your function call

No manual tool retrieval needed!

</div>

### Step 5: Test Your Agent

```python
    # Test queries
    queries = [
        "What's the weather like in Paris?",
        "Calculate 15% of 250",
        "Convert 72 degrees Fahrenheit to Celsius"
    ]

    for query in queries:
        print(f"\n{'='*60}")
        print(f"User: {query}")
        print(f"{'='*60}")

        response = await chat(query)

        # Handle tool calls if present
        if response.choices[0].message.tool_calls:
            print("🔧 Tool calls requested:")
            for tool_call in response.choices[0].message.tool_calls:
                print(f"  → {tool_call.function.name}({tool_call.function.arguments})")

                # Execute the tool
                from agent_gantry.schema.execution import ToolCall
                try:
                    parsed_args = json.loads(tool_call.function.arguments)
                except json.JSONDecodeError as err:
                    print(f"  ⚠️ Unable to parse tool arguments: {err}")
                    continue

                result = await gantry.execute(
                    ToolCall(
                        tool_name=tool_call.function.name,
                        arguments=parsed_args,
                    )
                )
                print(f"  ✓ Result: {result.output}")
        else:
            print(f"Assistant: {response.choices[0].message.content}")


# Run the agent
if __name__ == "__main__":
    asyncio.run(main())
```

### Complete Example

Here's the full code in one place:

```python
import asyncio
import ast
import json
from openai import AsyncOpenAI
from agent_gantry import AgentGantry, with_semantic_tools, set_default_gantry
from agent_gantry.schema.execution import ToolCall


def _evaluate_math_expression(expression: str) -> float:
    """Safely evaluate a basic math expression using the AST module."""
    node = ast.parse(expression, mode="eval").body

    def _evaluate(node: ast.AST) -> float:
        if isinstance(node, ast.BinOp) and isinstance(
            node.op,
            (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Mod, ast.Pow),
        ):
            left = _evaluate(node.left)
            right = _evaluate(node.right)
            if isinstance(node.op, ast.Add):
                return left + right
            if isinstance(node.op, ast.Sub):
                return left - right
            if isinstance(node.op, ast.Mult):
                return left * right
            if isinstance(node.op, ast.Div):
                return left / right
            if isinstance(node.op, ast.Mod):
                return left % right
            return left ** right
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            value = _evaluate(node.operand)
            return value if isinstance(node.op, ast.UAdd) else -value
        if isinstance(node, ast.Constant):
            value = ast.literal_eval(node)
            if isinstance(value, (int, float)):
                return float(value)
            raise ValueError("Only numeric literals are allowed")
        raise ValueError("Unsupported expression. Allowed operators: +, -, *, /, %, **")

    return float(_evaluate(node))

# Initialize
client = AsyncOpenAI()
gantry = AgentGantry()
set_default_gantry(gantry)


# Register tools
@gantry.register(tags=["weather"])
def get_weather(city: str, units: str = "fahrenheit") -> str:
    """Get the current weather for a city."""
    return f"The weather in {city} is 72°{units[0].upper()} and sunny."


@gantry.register(tags=["math"])
def calculate(expression: str) -> float:
    """Evaluate a mathematical expression."""
    try:
        return _evaluate_math_expression(expression)
    except Exception as e:
        return f"Error: {e}"


@gantry.register(tags=["conversion"])
def convert_temperature(value: float, from_unit: str, to_unit: str) -> float:
    """Convert temperature between Fahrenheit and Celsius."""
    if from_unit.upper() == "F" and to_unit.upper() == "C":
        return (value - 32) * 5/9
    elif from_unit.upper() == "C" and to_unit.upper() == "F":
        return (value * 9/5) + 32
    return value


async def main():
    # Sync tools
    await gantry.sync()
    print(f"✓ Synced {len(gantry._registry._tools)} tools")

    # Define chat function with semantic tools
    @with_semantic_tools(limit=3, dialect="openai")
    async def chat(prompt: str, *, tools=None):
        response = await client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            tools=tools,
            tool_choice="auto"
        )
        return response

    # Test
    query = "What's the weather like in Paris?"
    response = await chat(query)
    print(f"Query: {query}")
    print(f"Response: {response.choices[0].message}")


if __name__ == "__main__":
    asyncio.run(main())
```

### Run It

```bash
export OPENAI_API_KEY="your-api-key-here"
python my_agent.py
```

## What Just Happened?

1. **Tool Registration**: You registered 3 tools with descriptions and tags
2. **Automatic Embedding**: Agent-Gantry embedded your tools into a vector store
3. **Semantic Routing**: When you called `chat()`, Agent-Gantry:
   - Extracted the prompt ("What's the weather like in Paris?")
   - Performed vector search to find relevant tools
   - Found `get_weather` as the most relevant tool
   - Converted it to OpenAI format
   - Injected it into your LLM call
4. **Context Window Savings**: Instead of sending all 3 tools, only the top 1-3 relevant tools were sent, saving ~70-90% on tokens

## Next Steps

Now that you have a working agent, explore more advanced features:

### Learn More About Semantic Routing

- [Semantic Tool Decorator]({{ '/guides/semantic_tool_decorator' | relative_url }}) - Deep dive into the decorator
- [Vector Store Integration]({{ '/guides/vector_store_llm_integration' | relative_url }}) - Advanced vector store usage

### Add More Features

- **[Dynamic MCP Selection]({{ '/guides/dynamic_mcp_selection' | relative_url }})** - Connect to MCP servers on-demand
- **[Local Persistence]({{ '/guides/local_persistence_and_skills' | relative_url }})** - Use LanceDB for persistent tool storage
- **[Configuration]({{ '/reference/configuration' | relative_url }})** - Customize embedders, rerankers, and more

### Integrate with Different LLM Providers

- **[LLM SDK Compatibility]({{ '/reference/llm_sdk_compatibility' | relative_url }})** - Use with Anthropic, Google, Mistral, etc.

### Production Best Practices

- **[Architecture Overview]({{ '/architecture/overview' | relative_url }})** - Understand the system design
- **[Best Practices]({{ '/architecture/best-practices' | relative_url }})** - Security, performance, and error handling

## Common Issues

### ImportError: No module named 'agent_gantry'

Make sure you've installed the package:

```bash
pip install agent-gantry
```

### Tools not being selected correctly

1. Make sure you called `await gantry.sync()` after registering tools
2. Add more descriptive `description` and `tags` to your tools
3. Ensure your tool docstrings are clear and detailed

### "No default gantry set" error

Call `set_default_gantry(gantry)` after creating your AgentGantry instance:

```python
gantry = AgentGantry()
set_default_gantry(gantry)  # This line is required
```

---

## Questions or Issues?

- Check the [Troubleshooting Guide]({{ '/troubleshooting' | relative_url }})
- Review the [API Reference]({{ '/reference/api-reference' | relative_url }})
- [Open an issue on GitHub](https://github.com/CodeHalwell/Agent-Gantry/issues)

---

<div style="display: flex; justify-content: space-between; margin-top: 3rem; padding-top: 1.5rem; border-top: 1px solid var(--border-color);">

<a href="{{ '/' | relative_url }}" style="display: flex; flex-direction: column; padding: 1rem; max-width: 45%;">
<span style="font-size: 0.75rem; color: var(--text-muted); text-transform: uppercase;">Previous</span>
<span style="font-weight: 600; color: var(--text-primary);">← Home</span>
</a>

<a href="{{ '/guides/semantic_tool_decorator' | relative_url }}" style="display: flex; flex-direction: column; padding: 1rem; max-width: 45%; text-align: right;">
<span style="font-size: 0.75rem; color: var(--text-muted); text-transform: uppercase;">Next</span>
<span style="font-weight: 600; color: var(--text-primary);">Semantic Tool Decorator →</span>
</a>

</div>
