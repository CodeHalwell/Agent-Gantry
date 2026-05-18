# Agent-Gantry

**Universal Tool Orchestration Platform for LLM-Based Agent Systems**

*Context is precious. Execution is sacred. Trust is earned.*

---

## Overview

Agent-Gantry is a Python library and service for intelligent, secure tool orchestration in LLM-based agent systems. It solves three tightly coupled problems:

1. **Context Window Tax**: Reduces token costs by ~90% through semantic routing and dynamic tool surfacing instead of dumping 100+ tools into every prompt.

2. **Tool/Protocol Fragmentation**: Write Once, Run Anywhere - register tools once, use with OpenAI, Claude, Gemini, A2A agents, and MCP clients.

3. **Operational Blindness**: Zero-Trust security with tools guarded by policies, capabilities, and circuit breakers.

## Installation

```bash
pip install agent-gantry
```

For development:

```bash
pip install agent-gantry[dev]
```

### Optional: install the bundled Claude Skill

Agent-Gantry ships a Claude Skill — a self-contained reference an agent can read to learn the library — bundled inside the wheel. Install it next to your other skills with one command:

```bash
agent-gantry install-skill --target ./skills
```

This drops a `skills/agent-gantry/` directory in your project. Wire it into an AF agent via `SkillsProvider(skill_paths=["./skills"])` or point Claude Code at it directly. To use the bundled copy without copying:

```python
from agent_gantry.skills import skill_path
print(skill_path())  # /…/site-packages/agent_gantry/skills/agent-gantry
```

The skill covers each integration (Microsoft Agent Framework, LangChain, AutoGen, CrewAI, LlamaIndex, Semantic Kernel, Google ADK, plain SDK use), the new introspection APIs, and a debugging playbook.

### LLM Provider SDKs

Install with specific LLM provider support:

```bash
# All LLM providers
pip install agent-gantry[llm-providers]

# Individual providers
pip install agent-gantry[openai]        # OpenAI, Azure OpenAI, OpenRouter
pip install agent-gantry[anthropic]     # Anthropic (Claude)
pip install agent-gantry[google-genai]  # Google GenAI
pip install agent-gantry[google-vertexai]  # Google Vertex AI
pip install agent-gantry[mistral]       # Mistral AI
pip install agent-gantry[groq]          # Groq

# Everything
pip install agent-gantry[all]
```

See [docs/llm_sdk_compatibility.md](docs/llm_sdk_compatibility.md) for detailed provider documentation.

### Optional components

- **Vector stores**: `pip install agent-gantry[vector-stores]` (Qdrant/Chroma stubs)
- **Local persistence (LanceDB)**: `pip install agent-gantry[lancedb]`
- **Local embeddings (Nomic Matryoshka)**: `pip install agent-gantry[nomic]`
- **Agent framework integrations**: `pip install agent-gantry[agent-frameworks]` (LangChain, AutoGen, CrewAI, LlamaIndex, Semantic Kernel, etc.)
- **Example extras**: `pip install agent-gantry[example-tools]` for optional libraries used by the example scripts
- **Protocols**: `pip install agent-gantry[mcp]` and `pip install agent-gantry[a2a]`

Combine as needed, e.g.:

```bash
pip install agent-gantry[lancedb,nomic,mcp,a2a]
```

## Quick Start: The "Plug and Play" Experience

Transform your existing LLM code into a semantically-aware agent system with just a decorator:

```python
from openai import AsyncOpenAI
from agent_gantry import AgentGantry, with_semantic_tools, set_default_gantry

# Initialize
client = AsyncOpenAI()
gantry = AgentGantry()
set_default_gantry(gantry)

# Register tools with a simple decorator
@gantry.register(tags=["weather"])
def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    return f"The weather in {city} is 72°F and sunny."

# Apply decorator to your existing LLM function - tools are automatically injected!
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
- 🔄 Converts tool schemas to any LLM provider format (OpenAI, Anthropic, Google, etc.)
- 🛡️ Executes tools with circuit breakers, retries, and security policies

### Architecture Flow

```mermaid
graph LR
    A[LLM Call] --> B[with_semantic_tools]
    B --> C[Agent-Gantry<br/>Semantic Router]
    C --> D[Vector Search]
    D --> E[Top-K Tools]
    E --> F[Schema Transcoding]
    F --> G[LLM Call<br/>with Injected Tools]
    style C fill:#4CAF50
    style B fill:#2196F3
```

### Manual Control (Power Users)

For fine-grained control over tool retrieval and execution:

```python
from agent_gantry import AgentGantry
from agent_gantry.schema.execution import ToolCall

# Initialize
gantry = AgentGantry()

# Register a tool
@gantry.register(tags=["finance"])
def calculate_tax(amount: float) -> float:
    """Calculates US sales tax for a given amount."""
    return amount * 0.08

# Sync tools to enable semantic search
await gantry.sync()

# Retrieve relevant tools (returns OpenAI-compatible schemas)
tools = await gantry.retrieve_tools("What is the tax on $100?", limit=5)

# Execute a tool directly
result = await gantry.execute(ToolCall(
    tool_name="calculate_tax",
    arguments={"amount": 100.0},
))
```

See [docs/configuration.md](docs/configuration.md) for full config options and
[docs/local_persistence_and_skills.md](docs/local_persistence_and_skills.md) for LanceDB/Nomic setup
plus skill storage.

### Factory Function for Easy Setup

For quick setup with sensible defaults, use the `create_default_gantry()` factory function:

```python
from agent_gantry import create_default_gantry

# Creates an AgentGantry instance with automatic embedder selection
# (NomicEmbedder if available, falls back to SimpleEmbedder)
tools = create_default_gantry()

@tools.register(tags=["math"])
def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b
```

This is ideal for:
- Rapid prototyping and testing
- Projects that need multiple independent AgentGantry instances
- Avoiding module-level instantiation issues

### Framework Ready: Works with Any LLM Provider

Agent-Gantry seamlessly integrates with all major LLM providers. Just use the `dialect` parameter:

**OpenAI / Azure OpenAI / OpenRouter / Groq:**
```python
from openai import AsyncOpenAI
from agent_gantry import with_semantic_tools

client = AsyncOpenAI()

@with_semantic_tools(limit=3)  # Default dialect is "openai"
async def chat(messages, *, tools=None):
    return await client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        tools=tools
    )
```

**Anthropic (Claude):**
```python
from anthropic import AsyncAnthropic
from agent_gantry import with_semantic_tools

client = AsyncAnthropic()

@with_semantic_tools(dialect="anthropic", limit=3)
async def chat(messages, *, tools=None):
    return await client.messages.create(
        model="claude-sonnet-4-6",
        messages=messages,
        tools=tools,  # Automatically converted to Anthropic format
        max_tokens=1024
    )
```

**Google Gemini:**
```python
from google import genai
from agent_gantry import with_semantic_tools

client = genai.Client()

@with_semantic_tools(dialect="gemini", limit=3)
async def chat(prompt, *, tools=None):
    return client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
        tools=tools  # Automatically converted to Gemini format
    )
```

**Mistral** (OpenAI-compatible endpoint; `mistralai` SDK quarantined on PyPI — use the `openai` SDK):
```python
from openai import AsyncOpenAI
from agent_gantry import with_semantic_tools

client = AsyncOpenAI(api_key="your-mistral-key", base_url="https://api.mistral.ai/v1")

@with_semantic_tools(limit=3)  # Mistral uses OpenAI-compatible format
async def chat(messages, *, tools=None):
    return await client.chat.completions.create(
        model="mistral-large-latest",
        messages=messages,
        tools=tools,
    )
```

See [docs/llm_sdk_compatibility.md](docs/llm_sdk_compatibility.md) for detailed integration guides.

### Microsoft Agent Framework (native integration)

For Microsoft Agent Framework 1.x, `GantryContextProvider` plugs into AF's context-engineering pipeline as a first-class `ContextProvider` — no manual schema wiring, no static tool list, no overrides of `Agent(tools=...)`. Two modes:

```python
from agent_framework import Agent
from agent_framework.openai import OpenAIChatClient
from agent_gantry import AgentGantry, GantryContextProvider

gantry = AgentGantry()
# ... register tools, await gantry.sync() ...

# Per-run mode: tool set is fixed for one agent.run() call.
provider = GantryContextProvider(gantry, top_k=5)
agent = Agent(OpenAIChatClient(), "...", context_providers=[provider])

# Per-call mode: re-runs retrieval every chat-completion round (multi-step agents).
provider = GantryContextProvider(gantry, top_k=3, query_strategy="per_call")
agent = Agent(OpenAIChatClient(), "...")
provider.attach_to(agent)  # one-call helper — attaches provider + middleware
```

`GantryContextProvider` co-exists with `SkillsProvider`, flows through `WorkflowBuilder` / `SequentialBuilder` / `HandoffBuilder`, and supports `required=[...]`, `always_include=[...]`, `static_tools=[...]`, and `verbose=True` for one-line per-round logging. See `examples/agent_frameworks/agent_framework_provider_example.py` for the full walkthrough.

### Debugging tool routing

When the LLM "doesn't see" a tool you expect, Agent-Gantry surfaces the decision so you don't have to write middleware to find out:

```python
# After any agent.run() call:
decision = provider.last_selection
print(decision.summary())         # query="..." → top5: [tool_a:0.61, ...]
for c in decision.candidates:
    print(c.name, c.score, c.kept)  # full ranked list, kept/dropped

# Or, offline — same code path as the live middleware:
decision = await provider.dry_run_retrieve("the user's actual query")
```

For registry-level mistakes — descriptions that name other tools, near-duplicate tools, overly generic tags — run the linter:

```bash
agent-gantry lint
agent-gantry sim factorial fibonacci   # cosine similarity between two tools
```

### Robust score thresholds for long queries

Absolute cosine thresholds collapse on long instructional queries because the embedding gets diluted. Use the relative mode for length-robust filtering:

```python
provider = GantryContextProvider(gantry, score_threshold="relative:0.8")
# Keep anything within 80% of the top score, regardless of query length.
```

The default is `score_threshold=0.0` (no filtering) — opt-in to filtering, don't get filtered by surprise.

### Persistent embedding cache

`InMemoryVectorStore` (the default) re-embeds every tool on each cold start — real money on paid embedders. Wrap with `CachedEmbedder` for a disk-backed SQLite cache keyed by `(embedder_id, text_hash)`:

```python
from agent_gantry import AgentGantry
from agent_gantry.adapters.embedders.openai import OpenAIEmbedder
from agent_gantry.adapters.embedders.cached import CachedEmbedder
from agent_gantry.schema.config import EmbedderConfig

base = OpenAIEmbedder(EmbedderConfig(type="openai", model="text-embedding-3-large"))
embedder = CachedEmbedder(base)  # default: ~/.cache/agent_gantry/embeddings.sqlite
gantry = AgentGantry(embedder=embedder)
```

### Load tools from multiple modules

Organize tools in a `tools/` directory with separate files for each category, then import them into your main file:

```python
# tools/web_tools.py
from agent_gantry import AgentGantry

tools = AgentGantry()

@tools.register
def search_web(query: str) -> str:
    """Search the web for information."""
    return f"Results for: {query}"

@tools.register
def fetch_url(url: str) -> str:
    """Fetch content from a URL."""
    return f"Content from {url}"

# tools/math_tools.py
from agent_gantry import AgentGantry

tools = AgentGantry()

@tools.register
def calculate(expression: str) -> float:
    """Evaluate a mathematical expression safely."""
    # Use ast.literal_eval for safe evaluation of simple expressions
    import ast
    import operator
    
    # Define safe operators
    ops = {ast.Add: operator.add, ast.Sub: operator.sub,
           ast.Mult: operator.mul, ast.Div: operator.truediv}
    
    def safe_eval(node):
        if isinstance(node, ast.Num):
            return node.n
        elif isinstance(node, ast.BinOp):
            return ops[type(node.op)](safe_eval(node.left), safe_eval(node.right))
        raise ValueError(f"Unsupported expression")
    
    return safe_eval(ast.parse(expression, mode='eval').body)

@tools.register
def convert_units(value: float, from_unit: str, to_unit: str) -> float:
    """Convert between units."""
    return value  # simplified

# main.py
import asyncio
from openai import AsyncOpenAI
from agent_gantry import AgentGantry, set_default_gantry, with_semantic_tools

client = AsyncOpenAI()

async def main():
    # Import tools from multiple module files
    gantry = await AgentGantry.from_modules([
        "tools.web_tools",
        "tools.math_tools",
    ], attr="tools")
    
    await gantry.sync()
    set_default_gantry(gantry)
    
    # Now all tools are available for semantic selection
    @with_semantic_tools(limit=3)
    async def generate(prompt: str, *, tools=None):
        return await client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            tools=tools,
        )
    
    # Tools are automatically selected based on query
    response = await generate("Search for Python tutorials")  # selects web_tools
    response = await generate("Calculate 15% of 200")  # selects math_tools

asyncio.run(main())
```

You can also pass `modules=[...]` to `AgentGantry(...)` for deferred loading or call
`collect_tools_from_modules([...])` to import into an existing instance. Duplicate tools are skipped
with a warning so shared modules can be safely combined.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        AGENT LAYER                              │
│  (LangChain / AutoGen / LlamaIndex / CrewAI / Custom Agents)    │
└─────────────────────────┬───────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                       AGENT-GANTRY                              │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌────────┐ │
│  │  Semantic   │  │  Execution  │  │ Observability│ │ Policy │ │
│  │   Router    │  │   Engine    │  │  / Telemetry │ │ Engine │ │
│  └─────────────┘  └─────────────┘  └─────────────┘  └────────┘ │
└─────────────────────────┬───────────────────────────────────────┘
                          │
          ┌───────────────┼───────────────┬───────────────┐
          ▼               ▼               ▼               ▼
    ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐
    │  Python  │   │   MCP    │   │   REST   │   │   A2A    │
    │Functions │   │ Servers  │   │   APIs   │   │  Agents  │
    └──────────┘   └──────────┘   └──────────┘   └──────────┘
```

## Features

- **Semantic Routing**: Intelligent tool selection using vector similarity, intent classification, and conversation context
- **Native Microsoft Agent Framework integration**: `GantryContextProvider` plugs into AF as a first-class `ContextProvider` — per-run or per-call retrieval
- **Multi-Protocol Support**: Native support for MCP (Model Context Protocol) and A2A (Agent-to-Agent)
- **Schema Transcoding**: Automatic conversion between OpenAI, Anthropic, and Gemini tool formats
- **LLM Provider Compatibility**: Works with OpenAI, Azure OpenAI, Anthropic, Google GenAI, Vertex AI, Mistral, Groq, OpenRouter — plus any OpenAI-compatible endpoint via `api_base`
- **First-class introspection**: `RetrievalDecision`, `provider.last_selection`, `provider.dry_run_retrieve(query)`, verbose mode — debug routing without writing middleware
- **Length-robust thresholds**: `score_threshold="relative:0.8"` keeps anything within 80% of the top score, surviving long pipeline-style queries that collapse absolute cosine cutoffs
- **Registry linter**: `agent-gantry lint` flags tool descriptions that pull each other into the wrong queries (cross-references, near-duplicates, generic tags)
- **Persistent embedding cache**: `CachedEmbedder` wraps any embedder with a SQLite cache so cold starts don't re-embed everything
- **Bundled Claude Skill**: ship usage docs to your agents via `agent-gantry install-skill` or `from agent_gantry.skills import skill_path`
- **Circuit Breakers**: Automatic failure detection and recovery
- **Observability**: Built-in structured logging and telemetry for tracing and metrics, including `gantry.bridge_retrieval` spans carrying the full ranked candidate list
- **Zero-Trust Security**: Capability-based permissions and policy enforcement
- **Modular tool loading**: Import and deduplicate tool registries from other modules or packages
- **Local persistence & skills**: LanceDB-backed tool/skill storage, Matryoshka embeddings, and skill schemas for prompt guidance
- **Argument Validation**: Defensive validation against tool schemas
- **Async-Native**: Full async support for tools and execution
- **Retries & Timeouts**: Automatic retries with exponential backoff and configurable timeouts
- **Health Tracking**: Per-tool health metrics including success rate, latency, and circuit breaker state

## Context Window Savings

Agent-Gantry significantly reduces token usage by dynamically surfacing only the most relevant tools.

**Benchmark Results (15 Tools Registered, provider-reported usage):**

| Scenario | Tools Passed | Prompt Tokens | Cost Reduction |
|----------|--------------|---------------|----------------|
| **Standard** (All Tools) | 15 | 366 | - |
| **Agent-Gantry** (Top 2) | 2 | 78 | **~79%** |

*Measured using provider `usage` fields from `gpt-3.5-turbo` responses (no token estimators). See `examples/llm_integration/token_savings_demo.py` for the full benchmark.*

### Semantic Routing Accuracy

Agent-Gantry maintains high accuracy even with large toolsets by leveraging semantic embeddings.

**Stress Test Results (100 Tools Registered):**

| Metric | Value |
|--------|-------|
| **Total Tools** | 100 |
| **Retrieval Limit** | Top 2 |
| **Accuracy** | **100%** (10/10 queries) |
| **Embedder** | Nomic (`nomic-embed-text-v1.5`) |

*See `examples/testing_limits/stress_test_100_tools.py` for the full stress test.*

### Real-World End-to-End Verification

Agent-Gantry has been verified in end-to-end scenarios with real LLMs and tangible tools.

**Real-World Test (30 Tangible Tools + GPT-4o):**

| Metric | Result |
|--------|--------|
| **Scenario** | 30 functional tools (Math, File System, Network, etc.) |
| **Retrieval Accuracy** | **100%** (Correct tool always in Top 3) |
| **LLM Selection** | **100%** (GPT-4o correctly selected the tool) |
| **Execution** | **100%** (Tools executed and returned correct results) |

*See `examples/testing_limits/real_world_30_tools_test.py` for the complete end-to-end verification script.*

## Project Structure

```
agent_gantry/
├── core/                 # Main facade, registry, router, executor, security
├── schema/               # Data models (tools, queries, events, config)
├── adapters/             # Protocol adapters
│   ├── vector_stores/    # Qdrant, Chroma, LanceDB, In-Memory, PGVector
│   ├── embedders/        # OpenAI, Azure, Nomic, SentenceTransformers, Simple, Cached
│   ├── rerankers/        # Cohere, CrossEncoder
│   └── executors/        # Direct, Sandbox, MCP, HTTP, A2A
├── providers/            # Tool import from various sources (A2A, …)
├── servers/              # MCP and A2A server implementations
├── integrations/         # Microsoft Agent Framework bridge/provider/middleware,
│                         # LangChain, AutoGen, LlamaIndex, CrewAI, Semantic Kernel,
│                         # Google ADK, Anthropic Skills
├── query/                # Query generators (last_user_text, last_tool_result, …)
├── metrics/              # Token-savings metrics
├── observability/        # Telemetry adapters (console, OpenTelemetry, Prometheus)
├── utils/                # Fingerprinting, registry linter, async helpers
├── skills/               # Bundled Claude Skill (Agent-Gantry usage docs)
└── cli/                  # Command-line interface
```

## Development

### Setup

```bash
# Clone the repository
git clone https://github.com/CodeHalwell/Agent-Gantry.git
cd Agent-Gantry

# Preferred: uv for reproducible environments
# We use `pip install uv` here for convenience in Python-first environments.
# You can also install uv via curl, pipx, or system packages; see:
# https://docs.astral.sh/uv/getting-started/installation/
pip install uv
uv sync --extra dev
uv run pytest

# Or use pip directly
pip install -e ".[dev]"
pytest
```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=agent_gantry

# Run specific test file
pytest tests/test_tool.py
```

## MCP Integration

Agent-Gantry provides first-class support for the Model Context Protocol (MCP), enabling seamless integration with Claude Desktop and other MCP clients.

### Dynamic MCP Server Selection (NEW ✨)

Register MCP servers with rich metadata and let Agent-Gantry intelligently select which servers to connect to based on your query:

```python
from agent_gantry import AgentGantry

gantry = AgentGantry()

# Register servers with metadata (no immediate connection)
gantry.register_mcp_server(
    name="filesystem",
    command=["npx", "-y", "@modelcontextprotocol/server-filesystem"],
    description="Provides tools for reading and writing files on the local filesystem",
    args=["--path", "/home/user/documents"],
    tags=["filesystem", "files", "io"],
    examples=["read a file", "write to a file", "list directory"],
    capabilities=["read_files", "write_files"],
)

gantry.register_mcp_server(
    name="database",
    command=["python", "-m", "mcp_postgresql"],
    description="Access PostgreSQL databases for querying and data manipulation",
    tags=["database", "sql", "data"],
    examples=["query database", "insert record"],
    capabilities=["read_data", "write_data"],
)

# Sync server metadata for semantic search
await gantry.sync_mcp_servers()

# Semantic search finds relevant servers
servers = await gantry.retrieve_mcp_servers(
    query="I need to read a configuration file",
    limit=2
)

# Connect and load tools only from selected servers
for server in servers:
    await gantry.discover_tools_from_server(server.name)

# Now use the tools
tools = await gantry.retrieve_tools("read my config.yaml")
```

> **Note:** Dynamic MCP server selection and the associated semantic search APIs are currently a
> preview/placeholder implementation. The underlying semantic search logic is not yet fully
> implemented and behavior may change in future releases. See the dynamic MCP selection docs
> for the current status and roadmap.

**Benefits:**
- 🎯 **Semantic Routing**: Automatically finds relevant servers based on query context
- ⚡ **Lazy Loading**: Connects to servers only when needed
- 🔒 **Security**: Minimizes attack surface by connecting only to necessary servers
- 📊 **Health Tracking**: Monitors server availability and connection success

See [docs/dynamic_mcp_selection.md](docs/dynamic_mcp_selection.md) for complete documentation.

### Serve as MCP Server

```python
from agent_gantry import AgentGantry

gantry = AgentGantry()

@gantry.register
def calculate_sum(a: int, b: int) -> int:
    """Calculate the sum of two numbers."""
    return a + b

await gantry.sync()

# Serve as MCP server (dynamic mode for context window minimization)
await gantry.serve_mcp(transport="stdio", mode="dynamic")
```

**Dynamic Mode Benefits:**
- Exposes only 2 meta-tools: `find_relevant_tools` and `execute_tool`
- Reduces context window usage by ~90%
- Tools discovered on-demand through semantic search
- Perfect for Claude Desktop integration

### Connect to MCP Servers

```python
from agent_gantry.schema.config import MCPServerConfig

config = MCPServerConfig(
    name="filesystem",
    command=["npx", "-y", "@modelcontextprotocol/server-filesystem"],
    args=["--path", "/tmp"],
    namespace="fs",
)

# Discover and register tools from external MCP server
count = await gantry.add_mcp_server(config)
print(f"Added {count} tools from MCP server")
```

See `examples/mcp_integration_demo.py` for a complete demonstration.

## A2A Integration

Agent-Gantry implements the Agent-to-Agent (A2A) protocol, enabling interoperability between agents.

### Serve as A2A Agent

```python
from agent_gantry import AgentGantry

gantry = AgentGantry()

@gantry.register
def analyze_data(data: str) -> str:
    """Analyze data and provide insights."""
    return f"Analysis: {data}"

await gantry.sync()

# Serve as A2A agent (requires FastAPI and uvicorn)
# Agent Card will be available at: http://localhost:8080/.well-known/agent.json
gantry.serve_a2a(host="0.0.0.0", port=8080)
```

**Skills Exposed:**
- **tool_discovery**: Find relevant tools using semantic search
- **tool_execution**: Execute tools with retries and circuit breakers

### Consume External A2A Agents

```python
from agent_gantry.schema.config import A2AAgentConfig

config = A2AAgentConfig(
    name="external-agent",
    url="https://external-agent.example.com",
    namespace="external",
)

# Discover and register external agent's skills as tools
count = await gantry.add_a2a_agent(config)
print(f"Added {count} skills from external agent")

# External agent skills are now available as tools
tools = await gantry.retrieve_tools("translate text")
```

### Agent Card

Agent-Gantry automatically generates an Agent Card following the A2A protocol:

```json
{
  "name": "AgentGantry",
  "description": "Intelligent tool routing and execution service",
  "url": "http://localhost:8080",
  "version": "1.0.0",
  "skills": [
    {
      "id": "tool_discovery",
      "name": "Tool Discovery",
      "description": "Find relevant tools for a given task using semantic search"
    },
    {
      "id": "tool_execution",
      "name": "Tool Execution",
      "description": "Execute registered tools with retries and timeouts"
    }
  ]
}
```

See `examples/a2a_integration_demo.py` for a complete demonstration.

## CLI

A lightweight CLI ships with the package for quick inspection and diagnostics:

```bash
agent-gantry list                              # list registered tools
agent-gantry search "refund an order" --limit 3
agent-gantry lint                              # detect tool-description authoring mistakes
agent-gantry sim factorial fibonacci           # cosine similarity between two tools
agent-gantry sync --dry-run                    # which tools would (re-)embed and why
agent-gantry install-skill --target ./skills   # install the bundled Claude Skill
```

`lint` flags three patterns that silently degrade routing quality: tool descriptions that name *other* registered tools (the embedding pulls them toward the wrong queries), pairs of tools with >0.85 cosine similarity (probably should be merged or differentiated), and tags that appear on more than half the registry (low discriminative value). Exit code is `1` when any issues are flagged, `0` when the registry is clean — so it drops into any CI runner that respects exit codes.

The bundled `agent-gantry` entrypoint boots with demo tools, so for your own registry wire a tiny wrapper that imports your `AgentGantry` instance and calls `analyze_registry()` directly:

```python
# scripts/gantry_lint.py
import asyncio, sys
from my_app.tools import gantry  # your configured AgentGantry instance

async def _run() -> int:
    analysis = await gantry.analyze_registry()
    print(analysis.format_text())
    return 1 if not analysis.empty else 0

if __name__ == "__main__":
    sys.exit(asyncio.run(_run()))
```

### `lint` as a pre-commit hook

Add a `local` hook to `.pre-commit-config.yaml` so the registry gets checked on every commit that touches tool definitions:

```yaml
repos:
  - repo: local
    hooks:
      - id: agent-gantry-lint
        name: agent-gantry lint
        entry: python -m scripts.gantry_lint
        language: system
        pass_filenames: false
        files: ^my_app/tools/.*\.py$
```

`pass_filenames: false` keeps the hook running once per commit instead of once per file, and the `files:` filter skips it on commits that don't touch the registry.

### `lint` in GitHub Actions

```yaml
# .github/workflows/gantry-lint.yml
name: gantry-lint
on:
  pull_request:
    paths:
      - 'my_app/tools/**'
      - 'pyproject.toml'
  push:
    branches: [main]

jobs:
  lint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.12'
          cache: pip
      - run: pip install -e .
      - run: python -m scripts.gantry_lint
```

The bundled CLI boots with demo tools and an in-memory embedder. For details and customization options, see [docs/cli.md](docs/cli.md).

## Documentation

📚 **[View Full Documentation](https://codehalwell.github.io/Agent-Gantry/)** - Complete guides, API reference, and tutorials

### Quick Links

- **[Getting Started](https://codehalwell.github.io/Agent-Gantry/getting-started.html)** - 5-minute tutorial
- **[API Reference](https://codehalwell.github.io/Agent-Gantry/reference/api-reference.html)** - Complete API docs
- **[Architecture](https://codehalwell.github.io/Agent-Gantry/architecture/overview.html)** - System design and patterns
- **[Best Practices](https://codehalwell.github.io/Agent-Gantry/architecture/best-practices.html)** - Production deployment guide
- **[Troubleshooting](https://codehalwell.github.io/Agent-Gantry/troubleshooting.html)** - Common issues and solutions

## Roadmap

- **Phase 1**: ✅ Core Foundation - Data models, in-memory vector store, basic routing
- **Phase 2**: ✅ Robustness - Execution engine, retries, circuit breakers, security
- **Phase 3**: ✅ Context-Aware Routing - Intent classification, MMR diversity
- **Phase 4**: ✅ Production Adapters - Qdrant, Chroma, OpenAI embeddings
- **Phase 5**: ✅ MCP Integration - MCP client and server, dynamic tool discovery
- **Phase 6**: ✅ A2A Integration - Agent-to-Agent protocol, Agent Card, skill mapping
- **Phase 7**: ✅ Framework Integrations - LangChain, AutoGen, CrewAI, LlamaIndex, Semantic Kernel
- **Phase 8**: ✅ Documentation & Quality - Professional docs site, type safety, comprehensive examples

## License

MIT

## Contributing

Contributions are welcome! Please read the contributing guidelines before submitting a pull request.
