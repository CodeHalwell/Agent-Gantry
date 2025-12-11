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

## Quick Start

```python
from agent_gantry import AgentGantry

# Initialize
gantry = AgentGantry()

# Register a tool
@gantry.register(tags=["finance"])
def calculate_tax(amount: float) -> float:
    """Calculates US sales tax for a given amount."""
    return amount * 0.08

# Retrieve relevant tools (returns OpenAI-compatible schemas)
tools = await gantry.retrieve_tools("What is the tax on $100?", limit=5)

# Execute a tool
from agent_gantry.schema.execution import ToolCall
result = await gantry.execute(ToolCall(
    tool_name="calculate_tax",
    arguments={"amount": 100.0},
))
```

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
- **Multi-Protocol Support**: Native support for MCP (Model Context Protocol) and A2A (Agent-to-Agent)
- **Schema Transcoding**: Automatic conversion between OpenAI, Anthropic, and Gemini tool formats
- **Circuit Breakers**: Automatic failure detection and recovery
- **Observability**: Built-in OpenTelemetry integration for tracing and metrics
- **Zero-Trust Security**: Capability-based permissions and policy enforcement

## Project Structure

```
agent_gantry/
├── core/                 # Main facade, registry, router, executor
├── schema/               # Data models (tools, queries, events, config)
├── adapters/             # Protocol adapters
│   ├── vector_stores/    # Qdrant, Chroma, In-Memory, etc.
│   ├── embedders/        # OpenAI, SentenceTransformers, etc.
│   ├── rerankers/        # Cohere, CrossEncoder, etc.
│   └── executors/        # Direct, Sandbox, MCP, HTTP, A2A
├── providers/            # Tool import from various sources
├── servers/              # MCP and A2A server implementations
├── integrations/         # LangChain, AutoGen, LlamaIndex, CrewAI
├── observability/        # Telemetry, metrics, logging
└── cli/                  # Command-line interface
```

## Development

### Setup

```bash
# Clone the repository
git clone https://github.com/CodeHalwell/Agent-Gantry.git
cd Agent-Gantry

# Install dependencies
pip install -e ".[dev]"

# Run tests
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

## Roadmap

See [plan.md](plan.md) for the detailed development roadmap.

- **Phase 1**: Core Foundation - Data models, in-memory vector store, basic routing
- **Phase 2**: Robustness - Execution engine, retries, circuit breakers
- **Phase 3**: Context-Aware Routing - Intent classification, MMR diversity
- **Phase 4**: Production Adapters - Qdrant, Chroma, OpenAI embeddings
- **Phase 5**: MCP Integration - MCP client and server
- **Phase 6**: A2A Integration - Agent-to-Agent protocol
- **Phase 7**: Framework Integrations - LangChain, AutoGen, etc.

## License

MIT

## Contributing

Contributions are welcome! Please read the contributing guidelines before submitting a pull request.
