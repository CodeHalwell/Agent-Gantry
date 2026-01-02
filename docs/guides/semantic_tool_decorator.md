# Semantic Tool Selection Decorator

This document describes the `with_semantic_tools` decorator for automatic semantic tool selection in LLM generate functions.

## Overview

The `with_semantic_tools` decorator wraps LLM client generate functions to automatically perform semantic tool selection using Agent Gantry before forwarding to the underlying LLM API.

## Installation

The decorator is part of the core Agent Gantry library:

```python
from agent_gantry import AgentGantry, with_semantic_tools, set_default_gantry
```

## Recommended Pattern: set_default_gantry()

The recommended way to use the decorator is with `set_default_gantry()` for cleaner, more maintainable code:

```python
from agent_gantry import AgentGantry, set_default_gantry, with_semantic_tools
from openai import OpenAI

# Initialize and set default once at startup
gantry = AgentGantry()
set_default_gantry(gantry)

# Register tools...
@gantry.register
def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    return f"Weather in {city}: Sunny, 72°F"

# Create OpenAI client
client = OpenAI()

# Cleaner decorator syntax - no gantry parameter needed
@with_semantic_tools(limit=3)
async def generate(prompt: str, *, tools: list | None = None):
    """Generate a response with automatically selected tools."""
    return client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        tools=tools,
    )

# Usage
response = await generate("What's the weather in Paris?")
```

**Benefits:**
- Cleaner decorator syntax (no gantry parameter on every decorator)
- Set once, use everywhere pattern
- Thread-safe and async-safe (uses contextvars)
- Still allows explicit gantry parameter for advanced use cases

## Basic Usage

### With Explicit Gantry Parameter

The explicit `gantry` parameter still works for backward compatibility and multi-instance scenarios:

```python
from agent_gantry import AgentGantry, with_semantic_tools
from openai import OpenAI

# Initialize Agent Gantry
gantry = AgentGantry()

# Register tools
@gantry.register
def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    return f"Weather in {city}: Sunny, 72°F"

@gantry.register
def search_web(query: str) -> str:
    """Search the web for information."""
    return f"Results for: {query}"

@gantry.register
def send_email(to: str, subject: str, body: str) -> str:
    """Send an email to a recipient."""
    return f"Email sent to {to}"

# Create OpenAI client
client = OpenAI()

# Wrap your generate function with explicit gantry
@with_semantic_tools(gantry, limit=3)
async def generate(prompt: str, *, tools: list | None = None):
    """Generate a response with automatically selected tools."""
    return client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        tools=tools,
    )

# Usage - tools are automatically selected based on the prompt
response = await generate("What's the weather in Paris?")
# The decorator automatically selects get_weather as a relevant tool
```

### With Messages Format (OpenAI/Anthropic Style)

```python
@with_semantic_tools(limit=3)  # Using default gantry
async def chat(messages: list[dict], *, tools: list | None = None):
    """Chat with automatic tool selection."""
    return client.chat.completions.create(
        model="gpt-4",
        messages=messages,
        tools=tools,
    )

# The decorator extracts the user message for tool selection
response = await chat([
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Send an email to john@example.com"},
])
# Automatically selects send_email tool
```

### With Anthropic

```python
from anthropic import Anthropic

client = Anthropic()

@with_semantic_tools(dialect="anthropic", limit=5)
async def claude_chat(messages: list[dict], *, tools: list | None = None):
    """Chat with Claude using automatic tool selection."""
    return client.messages.create(
        model="claude-3-opus-20240229",
        messages=messages,
        tools=tools,  # Tools are in Anthropic format
    )
```

## Configuration Options

The decorator accepts several configuration options:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `prompt_param` | str | "prompt" | Parameter name containing the user prompt |
| `tools_param` | str | "tools" | Parameter name for injecting tools |
| `limit` | int | 5 | Maximum number of tools to retrieve |
| `dialect` | str | "openai" | Schema dialect ("openai", "anthropic", "gemini") |
| `auto_sync` | bool | True | Sync tools before retrieval |
| `score_threshold` | float | 0.5 | Minimum relevance score for tools |

### Example with All Options

```python
@with_semantic_tools(
    prompt_param="query",       # Custom prompt parameter name
    tools_param="functions",    # Custom tools parameter name
    limit=3,                    # Return top 3 tools
    dialect="anthropic",        # Use Anthropic tool format
    auto_sync=True,             # Sync tools before each request
    score_threshold=0.3,        # Lower threshold for more results
)
async def custom_generate(query: str, *, functions: list | None = None):
    ...
```

## Alternative: SemanticToolsDecorator Factory

For reusable configuration across multiple functions:

```python
from agent_gantry import SemanticToolsDecorator

# Create a reusable decorator factory
decorator = SemanticToolsDecorator(
    gantry,
    dialect="openai",
    limit=5,
    score_threshold=0.4,
)

# Apply to multiple functions
@decorator.wrap
async def generate_openai(prompt: str, *, tools=None):
    ...

@decorator.wrap(limit=2)  # Override specific options
async def generate_azure(prompt: str, *, tools=None):
    ...
```

## How It Works

1. **Function Interception**: The decorator wraps your generate function
2. **Prompt Extraction**: Extracts the user prompt from arguments
   - Supports direct `prompt` parameter
   - Supports OpenAI/Anthropic `messages` format
   - Handles multimodal content (extracts text from content arrays)
3. **Semantic Retrieval**: Uses Agent Gantry to find relevant tools
4. **Tool Injection**: Injects selected tools into the function call
5. **Execution**: Calls the original function with tools

```
User Request → Decorator → Agent Gantry → LLM API
    ↑              ↓              ↓           ↓
"weather?"    Extract prompt   Semantic   tools=[...]
                              Retrieval
```

## Architectural Tradeoffs

### Advantages

1. **Clean Separation**: Tool selection logic is decoupled from LLM client code
2. **Dynamic Selection**: Tools are selected per-request based on context
3. **Framework Agnostic**: Works with any LLM provider (OpenAI, Anthropic, etc.)
4. **Reduced Token Usage**: Only relevant tools are sent to the LLM
5. **Easy Integration**: Minimal code changes to existing applications

### Disadvantages

1. **Added Latency**: Semantic retrieval adds ~10-50ms per request
2. **Sync Limitations**: Sync function wrapper may not work in existing event loops
3. **Single-Turn Focus**: Tool selection is based only on the current prompt, not full conversation history
4. **Embedding Quality**: Results depend on the quality of the embedder

### When to Use

**Good Use Cases:**
- Applications with many tools (10+) where context window is a concern
- Dynamic tool sets that change frequently
- Multi-tenant applications with user-specific tools

**Consider Alternatives When:**
- You have only a few tools (< 5)
- Tools are always needed together
- You need conversation-aware tool selection

## Advanced: Direct SemanticToolSelector Usage

For more control, use `SemanticToolSelector` directly:

```python
from agent_gantry import SemanticToolSelector

selector = SemanticToolSelector(
    gantry,
    prompt_param="prompt",
    tools_param="tools",
    limit=5,
)

# Wrap functions for use
wrapped = selector.wrap_async(my_async_function)

# Use wrapped in place of my_async_function
result = await wrapped(...)
```

## Performance Considerations

1. **Caching**: Consider caching retrieved tools for repeated queries
2. **Batch Requests**: For high-volume scenarios, consider batching tool retrieval
3. **Threshold Tuning**: Adjust `score_threshold` based on your embedder quality
4. **Limit Optimization**: Set `limit` based on your LLM's context window

## Error Handling

The decorator handles errors gracefully:

```python
@with_semantic_tools()
async def generate(prompt: str, *, tools: list | None = None):
    try:
        return await client.chat.completions.create(...)
    except Exception as e:
        # Original exceptions are not suppressed
        raise

# If tool retrieval fails, the function is called without tools
# If prompt extraction fails, the function is called without tools
```

## Testing

```python
import pytest
from agent_gantry import AgentGantry, with_semantic_tools, set_default_gantry

@pytest.mark.asyncio
async def test_tool_injection():
    gantry = AgentGantry()
    set_default_gantry(gantry)

    @gantry.register
    def my_tool(x: int) -> str:
        """A test tool for demonstration."""
        return str(x)

    received_tools = None

    @with_semantic_tools(score_threshold=0.0)
    async def generate(prompt: str, *, tools: list | None = None):
        nonlocal received_tools
        received_tools = tools
        return "response"

    await generate("test prompt")
    assert received_tools is not None
```

## Migration Guide

### From Static Tool Lists

Before:
```python
TOOLS = [tool1, tool2, tool3]  # Static list

async def generate(prompt: str):
    return client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        tools=TOOLS,  # Always sends all tools
    )
```

After:
```python
from agent_gantry import AgentGantry, set_default_gantry, with_semantic_tools

gantry = AgentGantry()
set_default_gantry(gantry)
# Register tools dynamically...

@with_semantic_tools()
async def generate(prompt: str, *, tools: list | None = None):
    return client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        tools=tools,  # Only relevant tools
    )
```

### From Manual Tool Selection

Before:
```python
async def generate(prompt: str):
    # Manual tool selection logic
    if "weather" in prompt:
        tools = [weather_tool]
    elif "email" in prompt:
        tools = [email_tool]
    else:
        tools = []

    return client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        tools=tools,
    )
```

After:
```python
@with_semantic_tools()
async def generate(prompt: str, *, tools: list | None = None):
    # Semantic selection replaces manual logic
    return client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        tools=tools,
    )
```

## See Also

- [Agent Gantry Documentation](./index.md)
- [LLM SDK Compatibility](./llm_sdk_compatibility.md)
- [Vector Store Integration](./vector_store_llm_integration.md)
