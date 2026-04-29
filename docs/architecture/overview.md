---
layout: default
title: System Overview
parent: Architecture
nav_order: 1
description: "Understanding Agent-Gantry's architecture and design"
---

# Architecture Overview

Learn how Agent-Gantry is designed and how its components work together.

---

## System Architecture

Agent-Gantry follows a layered architecture with clear separation of concerns:

```
┌─────────────────────────────────────────────────────────────────┐
│                        APPLICATION LAYER                        │
│       (Your Agent / LangChain / Agent Framework / CrewAI)       │
└─────────────────────────┬───────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                       AGENT-GANTRY FACADE                       │
│                      (AgentGantry Class)                        │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌────────┐│
│  │   @with_    │  │  register() │  │   sync()    │  │execute││
│  │semantic_tools│  │retrieve_tools│  │add_mcp_server│  │       ││
│  └─────────────┘  └─────────────┘  └─────────────┘  └────────┘│
└─────────────────────────┬───────────────────────────────────────┘
                          │
          ┌───────────────┼───────────────┬───────────────┐
          ▼               ▼               ▼               ▼
    ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐
    │ Registry │   │  Router  │   │ Executor │   │  MCP     │
    │          │   │          │   │          │   │ Router   │
    └────┬─────┘   └────┬─────┘   └────┬─────┘   └────┬─────┘
         │              │              │              │
         └──────────────┴──────────────┴──────────────┘
                          │
          ┌───────────────┼───────────────┬───────────────┐
          ▼               ▼               ▼               ▼
    ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐
    │  Vector  │   │ Embedder │   │ Reranker │   │  LLM     │
    │  Store   │   │          │   │          │   │ Client   │
    └──────────┘   └──────────┘   └──────────┘   └──────────┘
                          │
          ┌───────────────┼───────────────┬───────────────┐
          ▼               ▼               ▼               ▼
    ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐
    │  Python  │   │   MCP    │   │   REST   │   │   A2A    │
    │ Functions│   │  Servers │   │   APIs   │   │  Agents  │
    └──────────┘   └──────────┘   └──────────┘   └──────────┘
```

## Core Components

### 1. AgentGantry (Facade)

**Purpose:** Main entry point providing a unified interface for all operations.

**Responsibilities:**
- Tool registration and lifecycle management
- Orchestrating semantic search
- Coordinating tool execution
- Managing MCP and A2A integrations

**Key Methods:**
- `register()` - Register tools
- `sync()` - Sync tools to vector store
- `retrieve_tools()` - Semantic tool search
- `execute()` - Execute tools with policies
- `add_mcp_server()` - Integrate MCP servers
- `serve_mcp()` - Serve as MCP server

**Why a Facade?**

The facade pattern simplifies the API by hiding complex subsystem interactions. Users interact with a single `AgentGantry` object rather than managing registry, router, executor, etc. separately.

### 2. Tool Registry

**Purpose:** Centralized storage and management of tool definitions.

**Responsibilities:**
- Store tool metadata and implementations
- Deduplication of tools
- Namespace management
- Tool lookup and retrieval

**Key Features:**
- Fast in-memory lookup by name/namespace
- Support for tool versioning
- Metadata indexing
- Namespace isolation

**Implementation:** `agent_gantry/core/registry.py`

### 3. Semantic Router

**Purpose:** Intelligent tool selection using semantic similarity.

**Responsibilities:**
- Embed tool descriptions into vectors
- Perform similarity search against user queries
- Apply filters (namespace, tags, capabilities)
- Rerank results for relevance

**Key Workflow:**

```python
1. User query → Embed query into vector
2. Vector search in tool database
3. Apply filters (tags, namespace, score threshold)
4. Optional: Rerank using LLM or cross-encoder
5. Return top-K tools
```

**Implementation:** `agent_gantry/core/router.py`

**Why Semantic Search?**

Traditional keyword matching fails for natural language queries. Semantic search uses embeddings to understand meaning, enabling queries like:
- "I need to get the current time" → matches `get_timestamp()` tool
- "Calculate percentage" → matches `calculate()` tool

### 4. MCP Router (NEW in v0.1.2)

**Purpose:** Semantic selection of MCP servers before connecting.

**Responsibilities:**
- Store MCP server metadata
- Embed server descriptions into vectors
- Semantic search for relevant servers
- Health tracking for connected servers

**Key Workflow:**

```python
1. Register MCP servers with metadata
2. Sync server metadata to vector store
3. User query → Find relevant servers
4. Connect only to selected servers
5. Discover and register tools from servers
```

**Implementation:** `agent_gantry/core/mcp_router.py`

**Benefits:**
- 🎯 Lazy loading - connect only when needed
- 🔒 Security - minimize attack surface
- ⚡ Performance - avoid unnecessary connections
- 📊 Health tracking - monitor server availability

### 5. Executor

**Purpose:** Secure, reliable tool execution with observability.

**Responsibilities:**
- Execute tool implementations
- Apply security policies and permissions
- Circuit breaker pattern for fault tolerance
- Retry logic with exponential backoff
- Telemetry and structured logging

**Key Features:**

**Security:**
- Capability-based permissions
- Input validation against tool schemas
- Sandboxing support (optional)

**Reliability:**
- Circuit breakers (open/half-open/closed states)
- Automatic retries with backoff
- Timeout protection
- Health metrics tracking

**Observability:**
- Structured logging of all executions
- Success/failure metrics
- Execution duration tracking
- Error categorization

**Implementation:** `agent_gantry/core/executor.py`

### 6. Adapters (Extensibility Layer)

Agent-Gantry uses the **Adapter Pattern** for all external integrations. This allows swapping implementations without changing core logic.

#### Vector Store Adapters

**Purpose:** Abstract vector storage and search.

**Implementations:**
- `InMemoryVectorStore` - Fast, ephemeral storage (default)
- `LanceDBVectorStore` - Local persistent storage with LanceDB
- `QdrantVectorStore` - Qdrant cloud/self-hosted
- `ChromaVectorStore` - Chroma DB integration
- `PGVectorStore` - PostgreSQL with pgvector extension

**Protocol:** `agent_gantry/adapters/vector_stores/base.py`

**Key Methods:**
- `add()` - Add tool embeddings
- `search()` - Vector similarity search
- `delete()` - Remove tools
- `get_stored_fingerprints()` - Change detection

#### Embedder Adapters

**Purpose:** Convert text to vector embeddings.

**Implementations:**
- `SimpleEmbedder` - Deterministic TF-IDF (no API required)
- `NomicEmbedder` - Nomic AI Matryoshka embeddings (768D, local)
- `OpenAIEmbedder` - OpenAI text-embedding-3-small/large
- `SentenceTransformersEmbedder` - HuggingFace models (local)

**Protocol:** `agent_gantry/adapters/embedders/base.py`

**Key Methods:**
- `embed()` - Single text to vector
- `embed_batch()` - Batch processing
- `dimension` - Embedding dimensionality

**Choosing an Embedder:**

| Embedder | Dimension | Quality | Speed | Cost |
|----------|-----------|---------|-------|------|
| SimpleEmbedder | ~1000 | Low | Fast | Free |
| NomicEmbedder | 768 | High | Fast | Free (local) |
| OpenAIEmbedder | 1536/3072 | Highest | Medium | Paid API |
| SentenceTransformers | Varies | High | Slow (CPU) | Free (local) |

**Recommendation:** Use NomicEmbedder for production, SimpleEmbedder for testing.

#### Reranker Adapters (Optional)

**Purpose:** Improve search quality by reranking results.

**Implementations:**
- `CohereReranker` - Cohere Rerank API
- `CrossEncoderReranker` - Local sentence-transformers cross-encoder

**When to use:**
- Large tool libraries (100+ tools)
- When top-3 accuracy is critical
- When initial search returns many similar results

**Trade-off:** Reranking adds latency but improves accuracy.

#### Executor Adapters

**Purpose:** Different execution strategies for tools.

**Implementations:**
- `DirectExecutor` - Direct Python function calls (default)
- `SandboxExecutor` - Execute in isolated environment (future)
- `MCPExecutor` - Execute tools via MCP protocol
- `HTTPExecutor` - Call remote REST APIs
- `A2AExecutor` - Execute A2A agent skills

## Data Flow

### Tool Registration Flow

```
┌───────────────┐
│  @register    │
│  decorator    │
└───────┬───────┘
        │
        ▼
┌───────────────────┐
│ Extract metadata  │
│ (name, params,    │
│  description)     │
└───────┬───────────┘
        │
        ▼
┌───────────────────┐
│ Store in Registry │
│ (in-memory map)   │
└───────┬───────────┘
        │
        ▼
┌───────────────────┐
│ Mark as pending   │
│ for sync          │
└───────────────────┘
```

### Semantic Search Flow

```
┌─────────────────┐
│  User Query     │
│ "get weather"   │
└────────┬────────┘
         │
         ▼
┌─────────────────────┐
│ Embed query using   │
│ embedder adapter    │
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│ Vector search in    │
│ vector store        │
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│ Apply filters       │
│ (namespace, tags)   │
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│ Optional: Rerank    │
│ using LLM/reranker  │
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│ Return top-K tools  │
│ (ToolDefinition[])  │
└─────────────────────┘
```

### Tool Execution Flow

```
┌──────────────────┐
│   ToolCall       │
│ (name, args)     │
└────────┬─────────┘
         │
         ▼
┌─────────────────────┐
│ Validate arguments  │
│ against schema      │
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│ Check permissions   │
│ & capabilities      │
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│ Check circuit       │
│ breaker state       │
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│ Execute via         │
│ executor adapter    │
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│ Update health       │
│ metrics             │
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│ Log telemetry       │
│ event               │
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│ Return ToolResult   │
│ (output, metadata)  │
└─────────────────────┘
```

## Design Principles

### 1. Async-First

All core operations are async. This enables:
- Non-blocking I/O for network calls
- Concurrent tool execution
- Better scalability

**Pattern:**

```python
# All core methods are async
await gantry.sync()
tools = await gantry.retrieve_tools("query")
result = await gantry.execute(call)
```

### 2. Schema-First with Pydantic v2

Data models are defined using Pydantic before implementation:
- Type safety
- Automatic validation
- JSON serialization
- Clear contracts

**Example:**

```python
from pydantic import BaseModel, Field

class ToolDefinition(BaseModel):
    name: str
    description: str
    parameters: dict[str, Any]
    # ... validation happens automatically
```

### 3. Protocol-Based Adapters

External integrations use Python protocols (structural typing):
- No inheritance required
- Duck typing with type safety
- Easy to extend

**Example:**

```python
from typing import Protocol

class VectorStoreAdapter(Protocol):
    async def add(self, tools: list[ToolDefinition]) -> None: ...
    async def search(self, query_vector: list[float], limit: int) -> list[tuple]: ...
```

Any class implementing these methods can be a vector store.

### 4. Separation of Concerns

Each component has a single responsibility:
- **Registry**: Storage only
- **Router**: Search and ranking only
- **Executor**: Execution and observability only
- **AgentGantry**: Orchestration only

### 5. Context-Local State

Uses `contextvars` for thread-safe and async-safe state:

```python
from agent_gantry import set_default_gantry

set_default_gantry(gantry)  # Context-local, safe for async
```

## Extensibility Points

### 1. Custom Vector Store

```python
class MyVectorStore:
    async def add(self, tools, embeddings): ...
    async def search(self, query_vector, limit): ...
    async def delete(self, tool_ids): ...

gantry = AgentGantry(vector_store=MyVectorStore())
```

### 2. Custom Embedder

```python
class MyEmbedder:
    @property
    def dimension(self) -> int: return 768

    async def embed(self, text: str) -> list[float]: ...
    async def embed_batch(self, texts: list[str]) -> list[list[float]]: ...

gantry = AgentGantry(embedder=MyEmbedder())
```

### 3. Custom Executor

```python
class MyExecutor:
    async def execute(self, call: ToolCall, func: Callable) -> ToolResult: ...

gantry = AgentGantry(executor=MyExecutor())
```

### 4. Custom Tool Provider

Implement a provider to import tools from external sources:

```python
from agent_gantry.providers.base import ToolProvider

class MyProvider(ToolProvider):
    async def discover_tools(self) -> list[ToolDefinition]: ...
```

## Performance Considerations

### Vector Search Latency

- **InMemoryVectorStore**: <10ms for 1000 tools
- **LanceDB**: <50ms for 100k tools
- **Qdrant/Chroma**: Varies by network latency

**Optimization:** Use local storage (LanceDB) for low latency.

### Embedding Latency

- **SimpleEmbedder**: <1ms (deterministic)
- **NomicEmbedder**: <20ms (local inference)
- **OpenAI**: 50-200ms (API call)

**Optimization:** Batch embeddings during sync, cache query embeddings.

### Tool Execution

Execution time depends on tool implementation. Use:
- Timeouts to prevent long-running tools
- Circuit breakers to isolate failing tools
- Async execution to run tools concurrently

## Security Architecture

### 1. Capability-Based Permissions

Tools declare required capabilities:

```python
@gantry.register(capabilities=["file_write"])
def write_file(path: str, content: str): ...
```

Executor checks permissions before execution.

### 2. Input Validation

All tool arguments validated against JSON Schema:

```python
# Automatic validation
tool.parameters = {
    "type": "object",
    "properties": {"city": {"type": "string"}},
    "required": ["city"]
}
```

### 3. Circuit Breakers

Prevent cascading failures:
- **Closed**: Normal operation
- **Open**: Tool disabled after N failures
- **Half-Open**: Test recovery

### 4. Sandboxing (Future)

Execute untrusted tools in isolated environments.

## Next Steps

- **[Best Practices]({{ '/architecture/best-practices' | relative_url }})** - Production deployment patterns
- **[API Reference]({{ '/reference/api-reference' | relative_url }})** - Detailed API documentation
- **[Configuration]({{ '/reference/configuration' | relative_url }})** - Configuration options

---

<div style="display: flex; justify-content: space-between; margin-top: 3rem; padding-top: 1.5rem; border-top: 1px solid var(--border-color);">

<a href="{{ '/reference/cli' | relative_url }}" style="display: flex; flex-direction: column; padding: 1rem; max-width: 45%;">
<span style="font-size: 0.75rem; color: var(--text-muted); text-transform: uppercase;">Previous</span>
<span style="font-weight: 600; color: var(--text-primary);">← CLI</span>
</a>

<a href="{{ '/architecture/best-practices' | relative_url }}" style="display: flex; flex-direction: column; padding: 1rem; max-width: 45%; text-align: right;">
<span style="font-size: 0.75rem; color: var(--text-muted); text-transform: uppercase;">Next</span>
<span style="font-weight: 600; color: var(--text-primary);">Best Practices →</span>
</a>

</div>
