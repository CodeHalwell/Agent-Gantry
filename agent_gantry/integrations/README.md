# agent_gantry/integrations - The "Plug & Play" Hub

This module is your gateway to seamless LLM integration. It provides decorators and adapters that let you add Agent-Gantry's semantic tool routing to any LLM framework with minimal code changes.

## Quick Start: The `@with_semantic_tools` Decorator

The `@with_semantic_tools` decorator is the easiest way to add semantic tool selection to your LLM calls:

```python
from agent_gantry import AgentGantry, with_semantic_tools, set_default_gantry
from openai import AsyncOpenAI

# Initialize once
gantry = AgentGantry()
set_default_gantry(gantry)
client = AsyncOpenAI()

# Register your tools
@gantry.register
def get_weather(city: str) -> str:
    """Get current weather for a city."""
    return f"Weather in {city}: Sunny, 72°F"

# Apply decorator - tools are automatically selected and injected
@with_semantic_tools(limit=3)
async def chat(prompt: str, *, tools=None):
    return await client.chat.completions.create(
        model="gpt-5.5",
        messages=[{"role": "user", "content": prompt}],
        tools=tools  # Agent-Gantry injects relevant tools here
    )

# Just call it - semantic routing happens automatically
response = await chat("What's the weather in Paris?")
```

## Using `set_default_gantry()` for Global Configuration

Instead of passing the `gantry` instance to every decorator, you can set a global default:

```python
from agent_gantry import AgentGantry, set_default_gantry, with_semantic_tools

# Set once at application startup
gantry = AgentGantry()
set_default_gantry(gantry)

# Now you can use decorators without passing gantry explicitly
@with_semantic_tools(limit=3)
async def chat(prompt: str, *, tools=None):
    # ... your LLM call
    pass

# Works across multiple functions
@with_semantic_tools(limit=5)
async def analyze(text: str, *, tools=None):
    # ... your LLM call
    pass
```

**Note:** In multi-threaded applications or parallel tests, be aware that `set_default_gantry()` sets module-level global state. For thread-safe usage, pass the gantry instance explicitly to decorators.

## Dialect Support: Works with Any LLM Provider

Agent-Gantry automatically converts tool schemas to match your LLM provider's format:

```python
# OpenAI / Azure OpenAI / Groq / Mistral (default)
@with_semantic_tools(limit=3)
async def chat_openai(...):
    pass

# Anthropic (Claude)
@with_semantic_tools(dialect="anthropic", limit=3)
async def chat_anthropic(...):
    pass

# Google Gemini
@with_semantic_tools(dialect="gemini", limit=3)
async def chat_gemini(...):
    pass
```

Supported dialects: `openai` (default), `anthropic`, `gemini`

## Configuration Options

The `@with_semantic_tools` decorator accepts several configuration options:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `gantry` | AgentGantry | (from default) | Gantry instance to use (optional if `set_default_gantry()` was called) |
| `limit` | int | 5 | Maximum number of tools to retrieve |
| `dialect` | str | "openai" | Tool schema format ("openai", "anthropic", "gemini") |
| `score_threshold` | float | 0.5 | Minimum relevance score for tools (lower for SimpleEmbedder) |
| `auto_sync` | bool | True | Automatically sync tools before retrieval |
| `prompt_param` | str | "prompt" | Parameter name containing the user prompt |
| `tools_param` | str | "tools" | Parameter name for injecting tools |

Example with custom configuration:

```python
@with_semantic_tools(
    gantry=my_gantry,         # Explicit gantry instance
    limit=2,                  # Return only top 2 tools
    dialect="anthropic",      # Use Anthropic tool format
    score_threshold=0.3,      # Lower threshold for more results
    auto_sync=False,          # Skip automatic syncing (if already synced)
)
async def chat(messages, *, tools=None):
    # ... your LLM call
    pass
```

## Framework-Specific Adapters

### Generic Framework Helper

For frameworks that accept OpenAI-style tool schemas:

```python
from agent_gantry import AgentGantry
from agent_gantry.integrations import fetch_framework_tools

gantry = AgentGantry()

@gantry.register
def send_email(to: str, body: str) -> str:
    """Send an email."""
    return f"Email sent to {to}"

# Retrieve tools with framework-specific formatting
tools = await fetch_framework_tools(
    gantry,
    "send a follow-up email",
    framework="langgraph",  # or "autogen", "crewai", etc.
    limit=2,
)
# Pass tools directly to your framework
```

### LangChain Integration

```python
from langchain_openai import ChatOpenAI
from agent_gantry import AgentGantry, with_semantic_tools

gantry = AgentGantry()
# Register tools...

# Use decorator with LangChain
@with_semantic_tools(gantry, limit=3)
async def chat_with_langchain(prompt: str, *, tools=None):
    llm = ChatOpenAI(model="gpt-5.5")
    # Convert to LangChain tools and use with your agent
    # See examples/agent_frameworks/langchain_example.py for details
```

### AutoGen Integration

```python
from autogen_agentchat import ConversableAgent
from agent_gantry import AgentGantry

gantry = AgentGantry()
# Register tools...

# Retrieve relevant tools
tools = await gantry.retrieve_tools("task description", limit=3)

# Pass to AutoGen agent
agent = ConversableAgent(
    name="assistant",
    llm_config={"model": "gpt-5.5", "tools": tools}
)
```

### CrewAI Integration

```python
from crewai import Agent  # Task and Crew also available for full workflows
from agent_gantry import AgentGantry

gantry = AgentGantry()
# Register tools...

# Convert Gantry tools to CrewAI format
from agent_gantry.integrations.framework_adapters import to_crewai_tools

tools = await gantry.retrieve_tools("task description", limit=3)
crewai_tools = to_crewai_tools(tools, gantry)

# Use with CrewAI agent
agent = Agent(
    role="assistant",
    goal="Complete tasks",
    tools=crewai_tools
)
```

## Modules

- `semantic_tools.py`: Core `with_semantic_tools` decorator and `SemanticToolSelector` class for automatic tool injection
- `framework_adapters.py`: Helpers for converting tools to framework-specific formats (LangGraph, Semantic Kernel, CrewAI, Google ADK, Strands)
- `agent_framework_bridge.py`: Microsoft Agent Framework 1.0 GA bridge — `GantryToolBridge` wraps Gantry tools as AF `FunctionTool`s with `approval_mode` auto-derived from Gantry `ToolCapability`. Exposes three agent construction helpers:
  - `build_agent(client, query, ...)` — one-liner using `client.as_agent()`, fine for single-agent flows.
  - `as_agent(client, query, ...)` — direct `Agent(client, ...)` construction; preferred when the result feeds `WorkflowBuilder`.
  - `build_workflow([specs], edges=[...])` — assembles a `WorkflowAgent` from multiple Gantry-equipped agents; each agent receives its own semantically-selected tool subset via a per-spec `query`.
- `agent_framework_middleware.py`: Function middleware for AF 1.0 GA — `GantryApprovalMiddleware` (routes tool execution through Gantry's `SecurityPolicy`, raising `MiddlewareTermination` for `require_confirmation` patterns and `PermissionDeniedError` for policy-denied calls) and `GantryObservabilityMiddleware` (records per-invocation timing onto Gantry's telemetry).

### Microsoft Agent Framework 1.0 GA quickstart

```python
from agent_framework.openai import OpenAIChatClient
from agent_gantry import AgentGantry
from agent_gantry.core.security import SecurityPolicy
from agent_gantry.integrations.agent_framework_bridge import GantryToolBridge
from agent_gantry.integrations.agent_framework_middleware import (
    GantryApprovalMiddleware,
    GantryObservabilityMiddleware,
)

gantry = AgentGantry()
# ... register tools ...
await gantry.sync()

bridge = GantryToolBridge(gantry)
policy = SecurityPolicy(require_confirmation=["delete_*", "refund_*"])

agent = await bridge.build_agent(
    client=OpenAIChatClient(),
    query="What's the weather in London?",
    name="WeatherAgent",
    instructions="Use tools to answer weather questions.",
    middleware=[
        GantryApprovalMiddleware(policy),
        GantryObservabilityMiddleware(gantry),
    ],
)
response = await agent.run("What's the weather in London?")
```

For multi-agent workflows use `bridge.as_agent()` to build first-class `Agent` objects and wire them with `WorkflowBuilder`:

```python
from agent_framework import WorkflowAgent, WorkflowBuilder

triage  = await bridge.as_agent(client, "triage classify",    name="Triage",  instructions="Route.")
billing = await bridge.as_agent(client, "billing invoices",   name="Billing", instructions="Handle billing.")
support = await bridge.as_agent(client, "support tickets",    name="Support", instructions="Handle support.")

workflow = (
    WorkflowBuilder(start_executor=triage)
    .add_edge(triage, billing, condition=lambda ctx: "invoice" in str(ctx).lower())
    .add_edge(triage, support)
    .build()
)
agent = WorkflowAgent(workflow, name="CustomerService")
```

Or use the `bridge.build_workflow()` convenience helper to do the same in one call.

For Sequential, Concurrent, Handoff, and Group Chat patterns see `examples/agent_frameworks/`.

## Advanced Usage

### Reusable Decorator Factory

Create a decorator factory for consistent configuration across multiple functions:

```python
from agent_gantry.integrations.decorator import SemanticToolsDecorator

# Create factory with default configuration
decorator = SemanticToolsDecorator(
    gantry,
    dialect="openai",
    limit=5,
    score_threshold=0.4,
)

# Apply to multiple functions
@decorator.wrap
async def chat_function_1(prompt: str, *, tools=None):
    # ... LLM call
    pass

@decorator.wrap(limit=2)  # Override specific options
async def chat_function_2(prompt: str, *, tools=None):
    # ... LLM call
    pass
```

### Direct SemanticToolSelector Usage

For maximum control:

```python
from agent_gantry.integrations.decorator import SemanticToolSelector

selector = SemanticToolSelector(
    gantry,
    prompt_param="query",
    tools_param="functions",
    limit=5,
)

# Wrap individual functions
wrapped_fn = selector.wrap_async(my_async_function)
result = await wrapped_fn(query="...", ...)
```

## Examples

See `examples/llm_integration/` for complete end-to-end examples:
- `decorator_demo.py`: Basic decorator usage
- `openai_demo.py`: OpenAI integration with multiple scenarios
- `anthropic_demo.py`: Anthropic/Claude integration
- `google_genai_demo.py`: Google Gemini integration
- `mistral_demo.py`: Mistral AI integration
- `groq_demo.py`: Groq integration

See `examples/agent_frameworks/` for framework-specific examples:
- `langchain_example.py`: LangChain agent integration
- `autogen_example.py`: AutoGen agent integration
- `crewai_example.py`: CrewAI agent integration

## See Also

- [Semantic Tool Decorator Documentation](../../docs/semantic_tool_decorator.md)
- [LLM SDK Compatibility Guide](../../docs/llm_sdk_compatibility.md)
- [Core README](../core/README.md) - Understanding the router and executor
