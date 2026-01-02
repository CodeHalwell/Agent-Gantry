---
layout: default
title: Home
nav_order: 1
description: "Universal Tool Orchestration Platform for LLM-Based Agent Systems"
permalink: /
---

# Agent-Gantry Documentation

**Universal Tool Orchestration Platform for LLM-Based Agent Systems**

_Context is precious. Execution is sacred. Trust is earned._

---

## Welcome

Agent-Gantry is a Python library that solves three critical problems in LLM-based agent systems:

<div class="callout note">
<div class="callout-title">✨ What Agent-Gantry Does</div>

1. **Context Window Tax**: Reduces token costs by ~90% through semantic routing instead of sending all tools in every prompt
2. **Tool/Protocol Fragmentation**: Write Once, Run Anywhere - supports OpenAI, Claude, Gemini, A2A agents, and MCP clients
3. **Operational Blindness**: Zero-trust security with policies, capabilities, and circuit breakers

</div>

## Quick Links

<div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 1rem; margin: 2rem 0;">

<div style="padding: 1.5rem; background: var(--bg-secondary); border-radius: 8px; border-left: 4px solid var(--primary-color);">
<h3 style="margin-top: 0;">🚀 Get Started</h3>
<p>Install and run your first example in 5 minutes</p>
<a href="{{ '/getting-started' | relative_url }}">Quick Start Guide →</a>
</div>

<div style="padding: 1.5rem; background: var(--bg-secondary); border-radius: 8px; border-left: 4px solid var(--accent-color);">
<h3 style="margin-top: 0;">📚 Guides</h3>
<p>Learn key concepts and advanced patterns</p>
<a href="{{ '/guides/semantic_tool_decorator' | relative_url }}">Browse Guides →</a>
</div>

<div style="padding: 1.5rem; background: var(--bg-secondary); border-radius: 8px; border-left: 4px solid var(--secondary-color);">
<h3 style="margin-top: 0;">📖 API Reference</h3>
<p>Complete API documentation and examples</p>
<a href="{{ '/reference/api-reference' | relative_url }}">API Docs →</a>
</div>

<div style="padding: 1.5rem; background: var(--bg-secondary); border-radius: 8px; border-left: 4px solid var(--warning-color);">
<h3 style="margin-top: 0;">🏗️ Architecture</h3>
<p>Understand the system design and best practices</p>
<a href="{{ '/architecture/overview' | relative_url }}">Architecture →</a>
</div>

</div>

## Installation

```bash
# Basic installation
pip install agent-gantry

# With all LLM providers
pip install agent-gantry[llm-providers]

# With local persistence (LanceDB + Nomic embeddings)
pip install agent-gantry[lancedb,nomic]

# Everything
pip install agent-gantry[all]
```

## 5-Minute Quick Start

Transform your existing LLM code into a semantically-aware agent system:

```python
from openai import AsyncOpenAI
from agent_gantry import AgentGantry, with_semantic_tools, set_default_gantry

# Initialize
client = AsyncOpenAI()
gantry = AgentGantry()
set_default_gantry(gantry)

# Register tools
@gantry.register(tags=["weather"])
def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    return f"The weather in {city} is 72°F and sunny."

# Apply decorator - tools are automatically injected!
@with_semantic_tools(limit=3)
async def ask_llm(prompt: str, *, tools=None):
    return await client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        tools=tools  # Agent-Gantry injects relevant tools here
    )

# Just call it - semantic routing happens automatically
await ask_llm("What's the weather in San Francisco?")
```

**That's it!** Agent-Gantry automatically:
- 🎯 Selects only relevant tools based on the query (reducing token costs by ~79%)
- 🔄 Converts tool schemas to any LLM provider format
- 🛡️ Executes tools with circuit breakers and security policies

## Key Features

### Semantic Routing
Intelligent tool selection using vector similarity, reducing context window usage by ~90%

### Multi-Protocol Support
Native support for:
- **MCP** (Model Context Protocol) - Client and Server
- **A2A** (Agent-to-Agent Protocol)
- **OpenAI**, **Anthropic**, **Google Gemini**, **Mistral**, **Groq**

### Production-Ready
- Circuit breakers and health tracking
- Retries with exponential backoff
- Structured logging and telemetry
- Zero-trust security with capability-based permissions

### Framework Agnostic
Works seamlessly with:
- LangChain
- AutoGen
- CrewAI
- LlamaIndex
- Semantic Kernel
- Custom agents

## What's New in v0.1.2

<div class="callout tip">
<div class="callout-title">✨ Dynamic MCP Server Selection</div>

Register MCP servers with rich metadata and let Agent-Gantry intelligently select which servers to connect to based on your query:

```python
# Register servers with metadata (no immediate connection)
gantry.register_mcp_server(
    name="filesystem",
    command=["npx", "-y", "@modelcontextprotocol/server-filesystem"],
    description="Provides tools for reading and writing files",
    tags=["filesystem", "files", "io"],
    examples=["read a file", "write to a file"],
)

# Semantic search finds relevant servers
servers = await gantry.retrieve_mcp_servers(
    query="I need to read a configuration file",
    limit=2
)

# Connect only to selected servers
for server in servers:
    await gantry.discover_tools_from_server(server.name)
```

<a href="{{ '/guides/dynamic_mcp_selection' | relative_url }}">Learn more about Dynamic MCP Selection →</a>

</div>

## Context Window Savings

Agent-Gantry significantly reduces token usage by dynamically surfacing only the most relevant tools.

**Benchmark Results:**

| Scenario | Tools Passed | Prompt Tokens | Cost Reduction |
|----------|--------------|---------------|----------------|
| **Standard** (All Tools) | 15 | 366 | - |
| **Agent-Gantry** (Top 2) | 2 | 78 | **~79%** |

*Measured using `gpt-3.5-turbo` with provider-reported token usage.*

### Stress Test: 100 Tools

| Metric | Value |
|--------|-------|
| **Total Tools** | 100 |
| **Retrieval Limit** | Top 2 |
| **Accuracy** | **100%** (10/10 queries) |
| **Embedder** | Nomic (`nomic-embed-text-v1.5`) |

## Documentation Structure

- **[Getting Started]({{ '/getting-started' | relative_url }})** - Installation, quick start, and first steps
- **[Guides]({{ '/guides/semantic_tool_decorator' | relative_url }})** - Topic-specific tutorials and patterns
  - [Semantic Tool Decorator]({{ '/guides/semantic_tool_decorator' | relative_url }})
  - [Dynamic MCP Selection]({{ '/guides/dynamic_mcp_selection' | relative_url }})
  - [Vector Store & LLM Integration]({{ '/guides/vector_store_llm_integration' | relative_url }})
  - [Local Persistence & Skills]({{ '/guides/local_persistence_and_skills' | relative_url }})
- **[Reference]({{ '/reference/api-reference' | relative_url }})** - API documentation and configuration
  - [API Reference]({{ '/reference/api-reference' | relative_url }})
  - [Configuration]({{ '/reference/configuration' | relative_url }})
  - [LLM SDK Compatibility]({{ '/reference/llm_sdk_compatibility' | relative_url }})
  - [CLI]({{ '/reference/cli' | relative_url }})
- **[Architecture]({{ '/architecture/overview' | relative_url }})** - System design and best practices
  - [System Overview]({{ '/architecture/overview' | relative_url }})
  - [Best Practices]({{ '/architecture/best-practices' | relative_url }})
- **[Troubleshooting]({{ '/troubleshooting' | relative_url }})** - Common issues and solutions

## Community & Support

- **[GitHub Repository](https://github.com/CodeHalwell/Agent-Gantry)** - Source code, issues, and contributions
- **[Report a Bug](https://github.com/CodeHalwell/Agent-Gantry/issues/new)** - Found an issue? Let us know
- **[Feature Requests](https://github.com/CodeHalwell/Agent-Gantry/discussions)** - Suggest improvements

## License

Agent-Gantry is open-source software licensed under the [MIT License](https://github.com/CodeHalwell/Agent-Gantry/blob/main/LICENSE).

---

<div style="text-align: center; margin: 3rem 0; padding: 2rem; background: var(--bg-secondary); border-radius: 8px;">

### Ready to Get Started?

<p style="margin-bottom: 1.5rem;">Transform your LLM agent system with semantic tool orchestration</p>

<a href="{{ '/getting-started' | relative_url }}" style="display: inline-block; padding: 0.75rem 2rem; background: var(--primary-color); color: white; border-radius: 6px; font-weight: 600; text-decoration: none; transition: background-color 150ms;">
Get Started Now →
</a>

</div>
