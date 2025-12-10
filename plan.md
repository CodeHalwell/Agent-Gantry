ToolArsenal v3.0 Enterprise
Universal Tool Orchestration Platform (Gold Standard PRD – 2025)
Status: Approved for Development / v3.0.0
Last Updated: 2025-01-10
Target: Production-Grade / Industry Standard
Core Philosophy: Context is precious. Execution is sacred. Trust is earned.

⸻

0. Preamble: Why This Spec Exists

This document is the merged, ratified, “Gold Standard” PRD for ToolArsenal in 2025.

It combines:
	•	The v3.0 draft (core models, routing, adapters, MCP/A2A integration, frameworks, CLI, tests, roadmap)
	•	The additional Aegis layer for policy / security / observability
	•	The Schema Dialect / Transcoding layer for cross-vendor tool interoperability

It aims to:
	•	Bridge the current “wild west” of agent/tool integration and the disciplined engineering needed for enterprise deployment.
	•	Serve as the reference spec for an MCP-centric, cross-framework tool orchestration layer.

High-level value prop:
	•	Reduce token costs by ~90% by semantic routing and dynamic tool surfacing instead of dumping 100+ tools into every prompt.
	•	Write Once, Run Anywhere: register once, use with OpenAI / Claude / Gemini / A2A agents / MCP clients.
	•	Zero-Trust security: tools guarded by policies, capabilities, and circuit breakers, not just vibes and prompts.

⸻

1. Executive Summary

ToolArsenal is a Python library and service for intelligent, secure tool orchestration in LLM-based agent systems.

It solves three tightly coupled problems:
	1.	Context Window Tax / Pollution
Agents today often receive dozens or hundreds of tool schemas in their prompt. This:
	•	Bloats token usage and latency
	•	Makes models more likely to pick the wrong tool
	•	Exposes internal tool surfaces unnecessarily
ToolArsenal instead:
	•	Uses semantic retrieval + context-aware routing
	•	Returns only 5–10 truly relevant tools per turn
	•	Supports dynamic MCP mode where the “tool” is actually “find relevant tools”.
	2.	Tool / Protocol Fragmentation
Tools currently show up from:
	•	Python functions
	•	MCP servers
	•	REST/OpenAPI specs
	•	Other agents (A2A)
Each uses different discovery, auth, and calling conventions. ToolArsenal provides:
	•	A unified internal ToolDefinition format
	•	A schema transcoder (OpenAI / Anthropic / Gemini / custom)
	•	Adapters for MCP, A2A, OpenAPI, and local Python functions
	3.	Operational Blindness
Without observability, you don’t know:
	•	Which tools were considered versus chosen
	•	Why a given tool was ranked highly
	•	How often it fails, how long it takes, or whether its circuit breaker is open
ToolArsenal:
	•	Emits structured telemetry events for retrieval and execution
	•	Integrates with OpenTelemetry / Datadog / Prometheus
	•	Tracks per-tool health, success rate, latency, and circuit breaker state

Core Positioning

At a stack level:

┌─────────────────────────────────────────────────────────────────┐
│                        AGENT LAYER                              │
│  (LangChain / AutoGen / LlamaIndex / CrewAI / Custom Agents)    │
└─────────────────────────┬───────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                         TOOL ARSENAL                            │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌────────┐ │
│  │  Semantic   │  │  Execution  │  │ Observability│ │ Aegis  │ │
│  │   Router    │  │   Engine    │  │   / Telemetry│ │Policy  │ │
│  └─────────────┘  └─────────────┘  └─────────────┘  └────────┘ │
└─────────────────────────┬───────────────────────────────────────┘
                          │
          ┌───────────────┼───────────────┬───────────────┐
          ▼               ▼               ▼               ▼
    ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐
    │  Python  │   │   MCP    │   │   REST   │   │   A2A    │
    │Functions │   │ Servers  │   │   APIs   │   │  Agents  │
    └──────────┘   └──────────┘   └──────────┘   └──────────┘


⸻

2. Problem Statement (Why This Exists)

2.1 Context Window Tax

Enterprise setups routinely have:
	•	50+ internal tools
	•	10+ MCP servers
	•	Multiple external SaaS integrations

Naively serialising all tool definitions into the system prompt costs:
	•	10k+ tokens per request
	•	Higher latency & cost
	•	Confusing tool selection
	•	Oversharing of internal capabilities

ToolArsenal’s approach:
	•	Store tools in a vector store with rich metadata
	•	Retrieve on demand using semantic search, MMR, and context-aware routing
	•	Expose dynamic MCP meta-tools (e.g. find_relevant_tools) instead of raw tool lists

2.2 Integration Fragmentation

Different sources:

Source	Interface	Discovery	Auth
Python functions	Direct callable	Static decorator	None
MCP Servers	JSON-RPC stdio/SSE	tools/list + capabilities	OAuth/API Key/Env
REST APIs	HTTP	OpenAPI spec / docs	API keys / OAuth
A2A Agents	Agent Protocol	Agent Card / registry	mTLS / JWT / token

Without ToolArsenal, developers write bespoke glue for each combination. ToolArsenal defines:
	•	A single ToolDefinition
	•	A VectorStoreAdapter / EmbeddingAdapter / ExecutorAdapter / TelemetryAdapter set of protocols
	•	High-level APIs (register, retrieve, execute) plus MCP/A2A servers

2.3 Observability Gap

Without an orchestration layer, it’s hard to answer:
	•	What tools did the agent consider?
	•	Why did it choose tool X instead of tool Y?
	•	What were the scores, penalties, and heuristics applied?
	•	What is the P95 latency of retrieval, per namespace, per tenant?
	•	Which tools are flaky or causing circuit breakers to open?

ToolArsenal bakes in:
	•	Per-call and per-retrieval spans
	•	Structured events (retrieval, execution, health change)
	•	Metrics for latency, error rate, success rate, cost

⸻

3. Architecture

3.1 Design Principles
	1.	Protocol-First: All core interfaces are Python Protocols; implementations are adapters.
	2.	Hexagonal Architecture: Clear separation between App Core and Ports/Adapters.
	3.	Async-Native: Core APIs are async; sync wrappers provided where necessary.
	4.	Zero Runtime Dependencies in Core: Adapters own their dependencies.
	5.	Fail-Safe Defaults: Conservative behaviour by default (rate-limited, confirmed for destructive operations).
	6.	Observability Built-In: OTel spans, metrics, and structured logs from day one.
	7.	Zero-Trust Security: Capabilities, confirmation thresholds, and policies enforced before execution.

3.2 Hexagonal Architecture + Aegis Layer

Conceptual block diagram (extended with the Aegis layer):

graph TD
    subgraph "Driving Ports (Inputs)"
        API[Public API]
        MCP_In[MCP Server Interface]
        A2A_In[A2A Agent Interface]
    end

    subgraph "The Aegis Layer (Middleware)"
        Auth[Policy Engine (OPA-lite)]
        Transcoder[Schema Transcoder]
        Telemetry[OTel Tracing / Metrics]
    end

    subgraph "Application Core"
        Router[Semantic Router (MMR, signals)]
        Executor[Execution Engine]
        Registry[Tool Registry]
    end

    subgraph "Driven Ports (Outputs)"
        VDB[(Vector Store Adapter)]
        Embedder[(Embedding Adapter)]
        MCP_Out[MCP Client Adapter]
        Py[Python Runtime]
        HTTP[HTTP / OpenAPI]
        A2A_Out[A2A Client]
    end

    API --> Auth
    MCP_In --> Auth
    A2A_In --> Auth

    Auth --> Telemetry
    Telemetry --> Transcoder
    Transcoder --> Router
    
    Router --> Registry
    Router --> VDB
    Router --> Embedder
    
    Executor --> MCP_Out
    Executor --> Py
    Executor --> HTTP
    Executor --> A2A_Out

Concrete code layout:

tool_arsenal/
├── core/
│   ├── arsenal.py          # Main ToolArsenal façade
│   ├── registry.py         # @register decorator, lifecycle
│   ├── router.py           # Semantic routing
│   ├── executor.py         # Execution engine (retries, CB, timeouts)
│   └── context.py          # Conversation context
├── schema/
│   ├── tool.py             # ToolDefinition, ToolCost, ToolHealth, SchemaDialect
│   ├── query.py            # ConversationContext, ToolQuery, RetrievalResult
│   ├── events.py           # Observability event types
│   └── config.py           # ToolArsenalConfig, *Config models
├── adapters/
│   ├── vector_stores/      # Qdrant, Chroma, PGVector, InMemory, etc.
│   ├── embedders/          # OpenAI, Azure, HF, local
│   ├── rerankers/          # Cohere, CrossEncoder, LLM
│   └── executors/          # Sandbox, Docker, Direct, MCP, HTTP, A2A
├── providers/
│   ├── python_function.py  # @register implementation
│   ├── mcp_client.py       # MCPClient
│   ├── openapi.py          # OpenAPI import
│   └── a2a_client.py       # A2AClient
├── servers/
│   ├── mcp_server.py       # MCPServer
│   └── a2a_server.py       # A2A FastAPI server
├── integrations/
│   ├── langchain.py        # LangChain tools
│   ├── autogen.py          # AutoGen functions
│   ├── llamaindex.py       # LlamaIndex tools
│   └── crewai.py           # CrewAI tools
├── observability/
│   ├── tracer.py           # TelemetryAdapter impls
│   ├── metrics.py          # Metrics exporters
│   └── logger.py           # Structured logging
└── cli/
    ├── serve.py            # Run MCP/A2A servers
    ├── inspect.py          # Registry inspection
    └── benchmark.py        # Latency benchmarks


⸻

4. Data Models (schema/)

4.1 Schema Dialects & Universal Tool Definition

We support multiple schema dialects for differing LLM ecosystems:

from enum import Enum
from pydantic import BaseModel, Field, ConfigDict, field_validator
from typing import List, Dict, Any, Optional
from datetime import datetime
import hashlib


class SchemaDialect(str, Enum):
    OPENAI = "openai"       # Strict JSON Schema, function calling
    ANTHROPIC = "anthropic" # input_schema nested under 'input_schema'
    GEMINI = "gemini"       # Google A2A / function schema format
    AUTO = "auto"           # Infer based on target client

Canonical ToolDefinition (merged version, with dialect transcoding and enterprise metadata):

from enum import Enum
from typing import List, Dict, Any, Optional, Literal
from pydantic import BaseModel, Field, ConfigDict
from datetime import datetime
import hashlib


class ToolSource(str, Enum):
    PYTHON_FUNCTION = "python_function"
    MCP_SERVER = "mcp_server"
    OPENAPI = "openapi"
    A2A_AGENT = "a2a_agent"
    MANUAL = "manual"


class ToolCapability(str, Enum):
    READ_DATA = "read_data"
    WRITE_DATA = "write_data"
    DELETE_DATA = "delete_data"
    EXECUTE_CODE = "execute_code"
    NETWORK_ACCESS = "network_access"
    FILE_SYSTEM = "file_system"
    FINANCIAL = "financial"
    PII_ACCESS = "pii_access"
    EXTERNAL_API = "external_api"


class ToolCost(BaseModel):
    estimated_latency_ms: int = Field(default=100)
    monetary_cost: Optional[float] = Field(default=None)
    rate_limit: Optional[int] = Field(default=None, description="Max calls per minute")
    context_tokens: int = Field(default=0, description="Tokens added when selected")


class ToolHealth(BaseModel):
    success_rate: float = Field(default=1.0, ge=0.0, le=1.0)
    avg_latency_ms: float = Field(default=0.0)
    total_calls: int = Field(default=0)
    consecutive_failures: int = Field(default=0)
    last_success: Optional[datetime] = None
    last_failure: Optional[datetime] = None
    circuit_breaker_open: bool = Field(default=False)


class ToolDefinition(BaseModel):
    """
    Universal representation of a tool.

    Canonical internal format, regardless of original source:
    - Python function
    - MCP server tool
    - OpenAPI operation
    - A2A agent skill
    """

    # Identity
    name: str = Field(..., min_length=1, max_length=128, pattern=r'^[a-z][a-z0-9_]*$')
    version: str = Field(default="1.0.0", pattern=r'^\d+\.\d+\.\d+$')
    namespace: str = Field(default="default")

    # Discovery
    description: str = Field(..., min_length=10, max_length=2000)
    extended_description: Optional[str] = Field(default=None, max_length=10000)
    examples: List[str] = Field(default_factory=list, max_length=10)
    tags: List[str] = Field(default_factory=list)

    # Schema (canonical JSON Schema for input/output)
    parameters_schema: Dict[str, Any] = Field(
        ...,
        description="JSON Schema for input parameters (OpenAI-style function calling)"
    )
    returns_schema: Optional[Dict[str, Any]] = Field(default=None)

    # Provenance
    source: ToolSource = Field(default=ToolSource.PYTHON_FUNCTION)
    source_uri: Optional[str] = Field(default=None)

    # Capabilities & permissions
    capabilities: List[ToolCapability] = Field(default_factory=list)
    requires_confirmation: bool = Field(default=False)

    # Cost model
    cost: ToolCost = Field(default_factory=ToolCost)

    # Runtime (non-persisted)
    health: ToolHealth = Field(default_factory=ToolHealth, exclude=True)

    # Metadata / lifecycle
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    deprecated: bool = Field(default=False)
    deprecation_message: Optional[str] = Field(default=None)
    superseded_by: Optional[str] = Field(default=None)

    model_config = ConfigDict(extra='ignore', validate_assignment=True)

    @field_validator("name")
    @classmethod
    def validate_name(cls, v: str) -> str:
        reserved = {"register", "retrieve", "execute", "list", "delete"}
        if v in reserved:
            raise ValueError(f"Tool name '{v}' is reserved")
        return v

    @property
    def qualified_name(self) -> str:
        """namespace.name:version"""
        return f"{self.namespace}.{self.name}:{self.version}"

    @property
    def content_hash(self) -> str:
        """
        Deterministic hash for change detection and efficient syncing.

        NOTE: Used to avoid re-embedding / re-indexing when nothing changed.
        """
        content = f"{self.name}:{self.version}:{self.description}:{self.parameters_schema}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    # --- Dialect / protocol export helpers ---

    def to_openai_schema(self) -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters_schema,
            },
        }

    def to_anthropic_schema(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.parameters_schema,
        }

    def to_gemini_schema(self) -> Dict[str, Any]:
        # Minimal placeholder; can be extended per A2A / Gemini spec
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters_schema,
        }

    def to_dialect(self, dialect: SchemaDialect) -> Dict[str, Any]:
        """Just-in-Time transcoding for specific LLMs or protocols."""
        if dialect == SchemaDialect.ANTHROPIC:
            return self.to_anthropic_schema()
        if dialect == SchemaDialect.GEMINI:
            return self.to_gemini_schema()
        # Default is OpenAI style
        return self.to_openai_schema()

Tool dependencies:

class ToolDependency(BaseModel):
    tool_name: str
    dependency_type: Literal["requires", "suggests", "conflicts"]
    reason: Optional[str] = None

4.2 Query & Context Models

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

class ConversationContext(BaseModel):
    """
    Conversation state for context-aware routing.

    Key idea: routing isn’t just about the current query; we use:
    - conversation summary
    - recent messages
    - which tools have been tried / failed
    - user capabilities
    """

    # Immediate request
    query: str = Field(...)

    # Coarse history
    conversation_summary: Optional[str] = Field(default=None)
    recent_messages: List[Dict[str, str]] = Field(default_factory=list, max_length=10)

    # Tool usage in this conversation
    tools_already_used: List[str] = Field(default_factory=list)
    tools_failed: List[str] = Field(default_factory=list)

    # Optional high-level intent classification
    inferred_intent: Optional[str] = Field(default=None)

    # Permission context
    user_capabilities: List[ToolCapability] = Field(
        default_factory=lambda: list(ToolCapability)
    )
    require_confirmation_for: List[ToolCapability] = Field(default_factory=list)


class ToolQuery(BaseModel):
    """Request to find relevant tools."""

    context: ConversationContext

    limit: int = Field(default=5, ge=1, le=50)
    score_threshold: float = Field(default=0.5, ge=0.0, le=1.0)

    # Filters
    namespaces: Optional[List[str]] = None
    required_capabilities: Optional[List[ToolCapability]] = None
    excluded_capabilities: Optional[List[ToolCapability]] = None
    sources: Optional[List[ToolSource]] = None
    exclude_deprecated: bool = True
    exclude_unhealthy: bool = True

    # Advanced
    enable_reranking: bool = False
    include_dependencies: bool = True
    diversity_factor: float = Field(default=0.0, ge=0.0, le=1.0)

Scored tool and retrieval result:

from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any

class ScoredTool(BaseModel):
    tool: ToolDefinition
    semantic_score: float = Field(ge=0.0, le=1.0)
    rerank_score: Optional[float] = None
    context_score: float = 0.0
    health_penalty: float = 0.0

    @property
    def final_score(self) -> float:
        base = self.rerank_score if self.rerank_score is not None else self.semantic_score
        return max(0.0, base + self.context_score - self.health_penalty)


class RetrievalResult(BaseModel):
    tools: List[ScoredTool]

    query_embedding_time_ms: float
    vector_search_time_ms: float
    rerank_time_ms: Optional[float] = None
    total_time_ms: float

    candidate_count: int
    filtered_count: int

    trace_id: str

    def to_openai_tools(self) -> List[Dict[str, Any]]:
        return [t.tool.to_openai_schema() for t in self.tools]

    def to_anthropic_tools(self) -> List[Dict[str, Any]]:
        return [t.tool.to_anthropic_schema() for t in self.tools]

4.3 Execution Models

from typing import Any, Optional, Dict, List, Literal
from pydantic import BaseModel, Field
from datetime import datetime
from enum import Enum


class ExecutionStatus(str, Enum):
    SUCCESS = "success"
    FAILURE = "failure"
    TIMEOUT = "timeout"
    PERMISSION_DENIED = "permission_denied"
    CIRCUIT_OPEN = "circuit_open"
    PENDING_CONFIRMATION = "pending_confirmation"
    CANCELLED = "cancelled"


class ToolCall(BaseModel):
    tool_name: str
    arguments: Dict[str, Any]

    timeout_ms: int = Field(default=30000, ge=100, le=300000)
    retry_count: int = Field(default=0, ge=0, le=5)
    require_confirmation: Optional[bool] = None

    trace_id: Optional[str] = None
    parent_span_id: Optional[str] = None


class ToolResult(BaseModel):
    tool_name: str
    status: ExecutionStatus

    result: Optional[Any] = None
    error: Optional[str] = None
    error_type: Optional[str] = None

    queued_at: datetime
    started_at: Optional[datetime] = None
    completed_at: datetime

    attempt_number: int = Field(default=1)

    trace_id: str
    span_id: str

    @property
    def latency_ms(self) -> float:
        if self.started_at:
            return (self.completed_at - self.started_at).total_seconds() * 1000
        return 0.0


class BatchToolCall(BaseModel):
    calls: List[ToolCall]
    execution_strategy: Literal["parallel", "sequential", "adaptive"] = "adaptive"
    fail_fast: bool = False


class BatchToolResult(BaseModel):
    results: List[ToolResult]
    total_time_ms: float
    successful_count: int
    failed_count: int


⸻

5. Protocol Interfaces (Ports)

5.1 Vector Store Protocol

from typing import Protocol, List, Tuple, Optional, Dict, Any
from abc import abstractmethod
import numpy as np


class VectorStoreAdapter(Protocol):
    """
    Vector DB abstraction for tools.

    Implementations: QdrantAdapter, ChromaAdapter, PGVectorAdapter,
                     PineconeAdapter, WeaviateAdapter, InMemoryAdapter.
    """

    @abstractmethod
    async def initialize(self) -> None:
        """Idempotent setup of collections / indexes."""
        ...

    @abstractmethod
    async def add_tools(
        self,
        tools: List[ToolDefinition],
        embeddings: List[np.ndarray],
        upsert: bool = True,
    ) -> int:
        ...

    @abstractmethod
    async def search(
        self,
        query_vector: np.ndarray,
        limit: int,
        filters: Optional[Dict[str, Any]] = None,
        score_threshold: Optional[float] = None,
    ) -> List[Tuple[ToolDefinition, float]]:
        ...

    @abstractmethod
    async def get_by_name(self, name: str, namespace: str = "default") -> Optional[ToolDefinition]:
        ...

    @abstractmethod
    async def delete(self, name: str, namespace: str = "default") -> bool:
        ...

    @abstractmethod
    async def list_all(
        self,
        namespace: Optional[str] = None,
        limit: int = 1000,
        offset: int = 0,
    ) -> List[ToolDefinition]:
        ...

    @abstractmethod
    async def count(self, namespace: Optional[str] = None) -> int:
        ...

    @abstractmethod
    async def health_check(self) -> bool:
        ...

5.2 Embedding Protocol

from typing import Protocol, List, Optional
from abc import abstractmethod


class EmbeddingAdapter(Protocol):
    """
    Text embedding abstraction.

    Implementations: OpenAIEmbedder, AzureOpenAIEmbedder, CohereEmbedder,
                     HuggingFaceEmbedder, SentenceTransformerEmbedder, OllamaEmbedder.
    """

    @property
    @abstractmethod
    def dimension(self) -> int:
        ...

    @property
    @abstractmethod
    def model_name(self) -> str:
        ...

    @abstractmethod
    async def embed_text(self, text: str) -> List[float]:
        ...

    @abstractmethod
    async def embed_batch(
        self,
        texts: List[str],
        batch_size: Optional[int] = None,
    ) -> List[List[float]]:
        ...

    @abstractmethod
    async def health_check(self) -> bool:
        ...

5.3 Reranker Protocol

from typing import Protocol, List, Tuple
from abc import abstractmethod


class RerankerAdapter(Protocol):
    """
    Rerank tools after vector search for higher precision.
    """

    @abstractmethod
    async def rerank(
        self,
        query: str,
        tools: List[Tuple[ToolDefinition, float]],
        top_k: int,
    ) -> List[Tuple[ToolDefinition, float]]:
        ...

5.4 Executor Protocol

from typing import Protocol, Any, Dict, Optional, Callable, Awaitable
from abc import abstractmethod


class ExecutorAdapter(Protocol):
    """
    Execution backend for tools.

    Implementations: DirectExecutor, SandboxExecutor, DockerExecutor,
                     MCPExecutor, A2AExecutor, HTTPExecutor.
    """

    @abstractmethod
    async def execute(
        self,
        tool: ToolDefinition,
        call: ToolCall,
        handler: Optional[Callable[..., Awaitable[Any]]] = None,
    ) -> ToolResult:
        ...

    @abstractmethod
    async def validate_arguments(
        self,
        tool: ToolDefinition,
        arguments: Dict[str, Any],
    ) -> Tuple[bool, Optional[str]]:
        ...

    @abstractmethod
    def supports_source(self, source: ToolSource) -> bool:
        ...

5.5 Telemetry Protocol

from typing import Protocol, Dict, Any, Optional
from abc import abstractmethod
from contextlib import asynccontextmanager


class TelemetryAdapter(Protocol):
    """
    Observability backend for ToolArsenal.

    Implementations: OpenTelemetryAdapter, DatadogAdapter, PrometheusAdapter,
                     ConsoleAdapter, NoOpAdapter.
    """

    @abstractmethod
    @asynccontextmanager
    async def span(self, name: str, attributes: Optional[Dict[str, Any]] = None):
        yield

    @abstractmethod
    async def record_retrieval(self, query: ToolQuery, result: RetrievalResult) -> None:
        ...

    @abstractmethod
    async def record_execution(self, call: ToolCall, result: ToolResult) -> None:
        ...

    @abstractmethod
    async def record_health_change(
        self,
        tool_name: str,
        old_health: ToolHealth,
        new_health: ToolHealth,
    ) -> None:
        ...


⸻

6. Core Logic: Routing, Security, Execution

6.1 Context-Aware Routing

6.1.1 Routing Signals & Scoring

from dataclasses import dataclass


@dataclass
class RoutingSignals:
    semantic_similarity: float
    intent_match: float
    conversation_relevance: float
    health_score: float
    cost_score: float

    already_used_penalty: float
    already_failed_penalty: float
    deprecated_penalty: float


@dataclass
class RoutingWeights:
    semantic: float = 0.6
    intent: float = 0.15
    conversation: float = 0.1
    health: float = 0.1
    cost: float = 0.05


def compute_final_score(signals: RoutingSignals, weights: RoutingWeights) -> float:
    base_score = (
        signals.semantic_similarity * weights.semantic
        + signals.intent_match * weights.intent
        + signals.conversation_relevance * weights.conversation
        + signals.health_score * weights.health
        + signals.cost_score * weights.cost
    )
    penalties = (
        signals.already_used_penalty
        + signals.already_failed_penalty
        + signals.deprecated_penalty
    )
    return max(0.0, base_score - penalties)

The router:
	1.	Embeds query → vector
	2.	Retrieves top-K (e.g. 20) tools from vector store
	3.	For each candidate computes RoutingSignals based on:
	•	Semantic similarity
	•	Intent alignment (see below)
	•	Whether tool was recently mentioned/used in history
	•	Tool health and cost
	4.	Optionally passes candidates through a RerankerAdapter
	5.	Applies MMR to ensure diversity when diversity_factor > 0
	6.	Applies capability and policy filters (see Aegis section)

6.1.2 Intent Classification

from typing import Optional, Dict, List
from enum import Enum


class TaskIntent(str, Enum):
    DATA_QUERY = "data_query"
    DATA_MUTATION = "data_mutation"
    ANALYSIS = "analysis"
    COMMUNICATION = "communication"
    FILE_OPERATIONS = "file_operations"
    CUSTOMER_SUPPORT = "customer_support"
    ADMIN = "admin"
    UNKNOWN = "unknown"


INTENT_TAG_MAPPING: Dict[TaskIntent, List[str]] = {
    TaskIntent.DATA_QUERY: ["query", "search", "get", "list", "fetch", "read"],
    TaskIntent.DATA_MUTATION: ["create", "update", "delete", "write", "modify"],
    TaskIntent.ANALYSIS: ["analyze", "compute", "aggregate", "calculate", "report"],
    TaskIntent.COMMUNICATION: ["email", "message", "notify", "slack", "send"],
    TaskIntent.FILE_OPERATIONS: ["file", "upload", "download", "convert", "export"],
    TaskIntent.CUSTOMER_SUPPORT: ["ticket", "refund", "support", "customer"],
    TaskIntent.ADMIN: ["user", "permission", "setting", "config", "admin"],
}


async def classify_intent(
    query: str,
    conversation_summary: Optional[str] = None,
    use_llm: bool = False,
) -> TaskIntent:
    query_lower = query.lower()
    scores = {}

    for intent, keywords in INTENT_TAG_MAPPING.items():
        scores[intent] = sum(1 for kw in keywords if kw in query_lower)

    if max(scores.values()) > 0:
        return max(scores, key=scores.get)

    if use_llm and conversation_summary:
        # Optional: call an LLM for classification
        pass

    return TaskIntent.UNKNOWN

6.1.3 MMR Diversity

import numpy as np
from typing import List, Tuple


def mmr_rerank(
    query_embedding: np.ndarray,
    candidates: List[Tuple[ToolDefinition, np.ndarray, float]],
    k: int,
    lambda_param: float = 0.7,
) -> List[Tuple[ToolDefinition, float]]:
    if len(candidates) <= k:
        return [(t, s) for t, _, s in candidates]

    selected = []
    selected_embeddings = []
    remaining = list(candidates)

    for _ in range(k):
        if not remaining:
            break

        mmr_scores = []
        for tool, emb, rel_score in remaining:
            if not selected_embeddings:
                diversity = 0.0
            else:
                similarities = [
                    np.dot(emb, sel_emb)
                    / (np.linalg.norm(emb) * np.linalg.norm(sel_emb))
                    for sel_emb in selected_embeddings
                ]
                diversity = max(similarities)

            mmr = lambda_param * rel_score - (1 - lambda_param) * diversity
            mmr_scores.append((tool, emb, rel_score, mmr))

        best = max(mmr_scores, key=lambda x: x[3])
        selected.append((best[0], best[2]))
        selected_embeddings.append(best[1])
        remaining = [(t, e, s) for t, e, s, _ in mmr_scores if t.name != best[0].name]

    return selected

6.2 Aegis: Security Policy & Zero-Trust Controls

At the heart of Aegis is a policy engine and capability-based access control, sitting between the agents and the execution engine.

6.2.1 Security Policy Model
Example SecurityPolicy (pattern-based, with require-confirmation):

from pydantic import BaseModel
from typing import List
import fnmatch


class ConfirmationRequired(Exception):
    pass


class PermissionDenied(Exception):
    pass


class SecurityPolicy(BaseModel):
    """
    Rules of Engagement for tools.
    """

    require_confirmation: List[str] = ["delete_*", "payment_*"]
    allowed_domains: List[str] = ["internal-api.com"]
    max_requests_per_minute: int = 60

    def check_permission(self, tool_name: str, arguments: dict) -> None:
        """
        Raises PermissionDenied or ConfirmationRequired.
        """
        for pattern in self.require_confirmation:
            if fnmatch.fnmatch(tool_name, pattern):
                raise ConfirmationRequired(
                    f"Tool {tool_name} requires human approval."
                )
        # Domain / rate limits enforced via other middleware

6.2.2 Capability-Based Security

class PermissionChecker:
    """Enforce capability-based access control."""

    def __init__(self, user_capabilities: List[ToolCapability]):
        self.allowed = set(user_capabilities)

    def can_use(self, tool: ToolDefinition) -> Tuple[bool, Optional[str]]:
        required = set(tool.capabilities)
        missing = required - self.allowed
        if missing:
            return False, f"Missing capabilities: {', '.join(c.value for c in missing)}"
        return True, None

    def filter_tools(self, tools: List[ToolDefinition]) -> List[ToolDefinition]:
        return [t for t in tools if self.can_use(t)[0]]

6.2.3 Input Validation (Defensive)

import re
from typing import Tuple, Optional


def validate_tool_name(name: str) -> Tuple[bool, Optional[str]]:
    if not re.match(r'^[a-z][a-z0-9_]{0,127}$', name):
        return False, "Name must be lowercase alphanumeric with underscores, 1-128 chars"
    return True, None


def validate_description(desc: str) -> Tuple[bool, Optional[str]]:
    suspicious_patterns = [
        r'\{\{.*\}\}',
        r'<script',
        r'javascript:',
    ]
    for pattern in suspicious_patterns:
        if re.search(pattern, desc, re.IGNORECASE):
            return False, "Description contains suspicious pattern"
    return True, None

6.3 Execution Engine

The execution engine is responsible for:
	•	Permission checks (policy + capabilities)
	•	Argument validation
	•	Circuit breaker logic
	•	Retries, back-off, and timeouts
	•	Health metric updates
	•	Telemetry emission

High-level code (abridged from draft):

import asyncio
from datetime import datetime
from typing import Any

class ExecutionEngine:
    def __init__(
        self,
        arsenal: "ToolArsenal",
        default_timeout_ms: int = 30000,
        max_retries: int = 3,
        circuit_breaker_threshold: int = 5,
        circuit_breaker_timeout_s: int = 60,
    ):
        self._arsenal = arsenal
        self._default_timeout = default_timeout_ms
        self._max_retries = max_retries
        self._cb_threshold = circuit_breaker_threshold
        self._cb_timeout = circuit_breaker_timeout_s

    async def execute(self, call: ToolCall) -> ToolResult:
        trace_id = call.trace_id or self._generate_trace_id()
        span_id = self._generate_span_id()
        queued_at = datetime.utcnow()

        tool = await self._arsenal._vector_store.get_by_name(call.tool_name)
        if not tool:
            return ToolResult(
                tool_name=call.tool_name,
                status=ExecutionStatus.FAILURE,
                error=f"Tool '{call.tool_name}' not found",
                error_type="ToolNotFound",
                queued_at=queued_at,
                completed_at=datetime.utcnow(),
                trace_id=trace_id,
                span_id=span_id,
            )

        # Circuit breaker short-circuit
        if tool.health.circuit_breaker_open and not self._should_attempt_recovery(tool):
            return ToolResult(
                tool_name=call.tool_name,
                status=ExecutionStatus.CIRCUIT_OPEN,
                error="Circuit breaker is open due to repeated failures",
                queued_at=queued_at,
                completed_at=datetime.utcnow(),
                trace_id=trace_id,
                span_id=span_id,
            )

        # Argument validation via ExecutorAdapter
        is_valid, error = await self._validate_arguments(tool, call.arguments)
        if not is_valid:
            return ToolResult(
                tool_name=call.tool_name,
                status=ExecutionStatus.FAILURE,
                error=error,
                error_type="ValidationError",
                queued_at=queued_at,
                completed_at=datetime.utcnow(),
                trace_id=trace_id,
                span_id=span_id,
            )

        # Confirmation logic (Aegis)
        needs_confirm = call.require_confirmation
        if needs_confirm is None:
            needs_confirm = tool.requires_confirmation
        if needs_confirm:
            return ToolResult(
                tool_name=call.tool_name,
                status=ExecutionStatus.PENDING_CONFIRMATION,
                queued_at=queued_at,
                completed_at=datetime.utcnow(),
                trace_id=trace_id,
                span_id=span_id,
            )

        # Retry loop with timeout
        last_error = None
        last_error_type = None
        for attempt in range(1, (call.retry_count or self._max_retries) + 1):
            started_at = datetime.utcnow()
            try:
                result = await self._execute_with_timeout(
                    tool,
                    call,
                    call.timeout_ms or self._default_timeout,
                )
                completed_at = datetime.utcnow()
                await self._record_success(tool, (completed_at - started_at).total_seconds() * 1000)
                return ToolResult(
                    tool_name=call.tool_name,
                    status=ExecutionStatus.SUCCESS,
                    result=result,
                    queued_at=queued_at,
                    started_at=started_at,
                    completed_at=completed_at,
                    attempt_number=attempt,
                    trace_id=trace_id,
                    span_id=span_id,
                )
            except asyncio.TimeoutError:
                last_error = "Execution timed out"
                last_error_type = "TimeoutError"
            except Exception as e:
                last_error = str(e)
                last_error_type = type(e).__name__

            if attempt < (call.retry_count or self._max_retries):
                await asyncio.sleep(2 ** attempt * 0.1)

        completed_at = datetime.utcnow()
        await self._record_failure(tool)

        return ToolResult(
            tool_name=call.tool_name,
            status=ExecutionStatus.TIMEOUT if last_error and "timeout" in last_error.lower()
            else ExecutionStatus.FAILURE,
            error=last_error,
            error_type=last_error_type,
            queued_at=queued_at,
            completed_at=completed_at,
            attempt_number=call.retry_count or self._max_retries,
            trace_id=trace_id,
            span_id=span_id,
        )

Batch execution supports parallel, sequential, and adaptive modes and is specified in the draft (kept as-is).

⸻

7. MCP Integration (Model Context Protocol)

MCP is a core target. ToolArsenal:
	1.	Consumes external MCP servers (via MCPClient) and converts their tools into ToolDefinition entries.
	2.	Exposes itself as an MCP server, either:
	•	Static mode: all tools listed directly
	•	Dynamic mode: only a meta-tool find_relevant_tools plus execute_tool
	•	Hybrid: “common” tools directly + meta-tools for the rest

7.1 MCP Client

Key behaviour:
	•	Connects via stdio (subprocess), SSE, or WebSocket.
	•	Performs MCP initialize handshake.
	•	Calls tools/list and converts output to ToolDefinition with source=ToolSource.MCP_SERVER.
	•	Executes tools via tools/call.

The full MCPClient implementation from the draft is retained (not repeating here for brevity).

7.2 MCP Server – Dynamic Meta-Tool Mode

The “killer feature”: dynamic mode does not expose all tools at once. Instead:
	•	Exposes:
	•	find_relevant_tools(query, limit)
	•	execute_tool(tool_name, arguments)
	•	This lets clients (e.g. Claude Desktop) search for tools based on the immediate task and only then call them.

Snippet from MCPServer._handle_tools_list:

async def _handle_tools_list(self, params: Dict[str, Any]) -> Dict[str, Any]:
    if self.mode == "dynamic":
        return {
            "tools": [
                {
                    "name": "find_relevant_tools",
                    "description": (
                        "Search for tools relevant to your current task. "
                        "Use this before calling other tools."
                    ),
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "What you're trying to accomplish"
                            },
                            "limit": {
                                "type": "integer",
                                "description": "Max tools to return",
                                "default": 5
                            }
                        },
                        "required": ["query"]
                    }
                },
                {
                    "name": "execute_tool",
                    "description": (
                        "Execute a tool by name. Use find_relevant_tools first "
                        "to discover available tools and their schemas."
                    ),
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "tool_name": {"type": "string"},
                            "arguments": {"type": "object"}
                        },
                        "required": ["tool_name", "arguments"]
                    }
                }
            ]
        }
    else:
        tools = await self.arsenal._vector_store.list_all()
        return {
            "tools": [
                {
                    "name": t.name,
                    "description": t.description,
                    "inputSchema": t.parameters_schema
                }
                for t in tools
            ]
        }

tools/call routes either to these meta-tools or directly to ToolArsenal.execute.

⸻

8. A2A Integration (Agent-to-Agent Protocol)

ToolArsenal participates in A2A in two directions:
	1.	Consuming A2A agents as tools (via A2AClient)
	2.	Exposing itself as an A2A agent with an Agent Card and skills

8.1 Agent Card

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any


class AgentSkill(BaseModel):
    id: str
    name: str
    description: str
    input_modes: List[str] = ["text"]
    output_modes: List[str] = ["text"]


class AgentCard(BaseModel):
    name: str = Field(default="ToolArsenal")
    description: str = Field(default="Intelligent tool routing and execution service")
    url: str
    version: str = Field(default="3.0.0")

    skills: List[AgentSkill] = Field(default_factory=lambda: [
        AgentSkill(
            id="tool_discovery",
            name="Tool Discovery",
            description="Find relevant tools for a given task using semantic search",
        ),
        AgentSkill(
            id="tool_execution",
            name="Tool Execution",
            description="Execute registered tools with retries, timeouts, and policies",
        ),
    ])

    authentication: Optional[Dict[str, Any]] = None
    provider: Dict[str, str] = Field(default_factory=lambda: {
        "organization": "Your Org",
        "url": "https://your-org.com",
    })


def generate_agent_card(arsenal: "ToolArsenal", base_url: str) -> AgentCard:
    return AgentCard(
        url=base_url,
        description=f"ToolArsenal with {arsenal.tool_count} tools available",
    )

8.2 A2A Client

The A2AClient:
	•	Fetches .well-known/agent.json
	•	Maps skills → ToolDefinition with names like a2a_{agent_name}_{skill_id}
	•	Exposes a send_task method to send tasks to the remote agent

Implementation from the draft stands as spec.

8.3 A2A Server

create_a2a_server(arsenal) is a FastAPI app that:
	•	Serves the Agent Card at /.well-known/agent.json
	•	Handles JSON-RPC tasks/send by:
	•	Extracting text from message parts
	•	Using arsenal.retrieve to find relevant tools
	•	Optionally executing them; in the draft it returns discovered tool descriptions

⸻

9. Configuration & Developer Experience

9.1 Configuration Schema

ToolArsenalConfig is the single source of truth for configuration.

Key structures (abridged):

from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List, Literal


class VectorStoreConfig(BaseModel):
    type: Literal["memory", "qdrant", "chroma", "pgvector", "pinecone", "weaviate"]
    url: Optional[str] = None
    api_key: Optional[str] = None
    collection_name: str = "tool_arsenal"
    dimension: Optional[int] = None
    options: Dict[str, Any] = Field(default_factory=dict)


class EmbedderConfig(BaseModel):
    type: Literal["openai", "azure", "cohere", "huggingface", "sentence_transformers", "ollama"]
    model: str = "text-embedding-3-small"
    api_key: Optional[str] = None
    api_base: Optional[str] = None
    batch_size: int = 100
    max_retries: int = 3


class RerankerConfig(BaseModel):
    enabled: bool = False
    type: Literal["cohere", "cross_encoder", "llm"] = "cross_encoder"
    model: Optional[str] = None
    top_k: int = 10


class RoutingConfig(BaseModel):
    weights: Dict[str, float] = Field(default_factory=lambda: {
        "semantic": 0.6,
        "intent": 0.15,
        "conversation": 0.1,
        "health": 0.1,
        "cost": 0.05,
    })
    enable_intent_classification: bool = True
    use_llm_for_intent: bool = False
    enable_mmr: bool = True
    mmr_lambda: float = 0.7


class ExecutionConfig(BaseModel):
    default_timeout_ms: int = 30000
    max_retries: int = 3
    circuit_breaker_threshold: int = 5
    circuit_breaker_timeout_s: int = 60
    enable_sandbox: bool = False
    sandbox_type: Literal["none", "subprocess", "docker"] = "none"


class TelemetryConfig(BaseModel):
    enabled: bool = True
    type: Literal["console", "opentelemetry", "datadog", "prometheus"] = "console"
    otlp_endpoint: Optional[str] = None
    service_name: str = "tool_arsenal"
    expose_prometheus: bool = False
    prometheus_port: int = 9090


class MCPConfig(BaseModel):
    servers: List[MCPServerConfig] = Field(default_factory=list)
    serve_mcp: bool = False
    mcp_mode: Literal["dynamic", "static", "hybrid"] = "dynamic"


class A2AConfig(BaseModel):
    agents: List[A2AAgentConfig] = Field(default_factory=list)
    serve_a2a: bool = False
    a2a_port: int = 8080


class ToolArsenalConfig(BaseModel):
    vector_store: VectorStoreConfig = Field(default_factory=lambda: VectorStoreConfig(type="memory"))
    embedder: EmbedderConfig = Field(
        default_factory=lambda: EmbedderConfig(type="sentence_transformers", model="all-MiniLM-L6-v2")
    )
    reranker: RerankerConfig = Field(default_factory=RerankerConfig)
    routing: RoutingConfig = Field(default_factory=RoutingConfig)
    execution: ExecutionConfig = Field(default_factory=ExecutionConfig)
    telemetry: TelemetryConfig = Field(default_factory=TelemetryConfig)
    mcp: MCPConfig = Field(default_factory=MCPConfig)
    a2a: A2AConfig = Field(default_factory=A2AConfig)

    auto_sync: bool = True
    sync_on_register: bool = False

    @classmethod
    def from_yaml(cls, path: str) -> "ToolArsenalConfig":
        import yaml
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(**data)

9.2 Example Configuration Files

Development (config.dev.yaml):

vector_store:
  type: memory

embedder:
  type: sentence_transformers
  model: all-MiniLM-L6-v2

reranker:
  enabled: false

routing:
  enable_mmr: false

execution:
  default_timeout_ms: 60000

telemetry:
  type: console

mcp:
  serve_mcp: false

Production (config.prod.yaml – merged from both specs):

vector_store:
  type: qdrant
  url: ${QDRANT_URL}
  api_key: ${QDRANT_API_KEY}
  collection_name: production_tools

embedder:
  type: openai
  model: text-embedding-3-small
  api_key: ${OPENAI_API_KEY}

reranker:
  enabled: true
  type: cohere
  model: rerank-english-v3.0

routing:
  weights:
    semantic: 0.5
    intent: 0.2
    conversation: 0.1
    health: 0.15
    cost: 0.05
  enable_mmr: true
  mmr_lambda: 0.7

execution:
  default_timeout_ms: 30000
  circuit_breaker_threshold: 3

telemetry:
  type: opentelemetry
  otlp_endpoint: ${OTLP_ENDPOINT}
  service_name: tool_arsenal_prod

mcp:
  serve_mcp: true
  mcp_mode: dynamic
  servers:
    - name: database
      command: ["npx", "-y", "@anthropic/mcp-database"]
    - name: filesystem
      command: ["npx", "-y", "@anthropic/mcp-filesystem", "/data"]

a2a:
  serve_a2a: false

security:
  policy: strict
  human_in_the_loop:
    - refund_user
    - drop_table

9.3 Developer Usage (Python)

from tool_arsenal import ToolArsenal

# 1. Initialise from config
arsenal = ToolArsenal.from_config("config.prod.yaml")

# 2. Register tools
@arsenal.register(tags=["finance"])
def calculate_tax(amount: float) -> float:
    """Calculates US sales tax."""
    return amount * 0.08

# 3. Use inside an agent loop
user_query = "What is the tax on $100?"
tools = await arsenal.retrieve_tools(user_query, limit=1)
# Tools is a list of OpenAI-compatible function calling schemas

# 4. Execute a tool directly
from tool_arsenal.schema import ToolCall
result = await arsenal.execute(ToolCall(
    tool_name="calculate_tax",
    arguments={"amount": 100.0},
))


⸻

10. Framework Integrations

10.1 LangChain

ToolArsenalToolkit allows dynamic tool selection from ToolArsenal into LangChain agents. Implementation from draft is retained.

Usage:

from tool_arsenal.integrations.langchain import ToolArsenalToolkit
from langchain.agents import create_react_agent

toolkit = ToolArsenalToolkit(arsenal)
tools = toolkit.get_tools("customer refund", limit=5)
agent = create_react_agent(llm, tools)

10.2 AutoGen

to_autogen_functions and create_autogen_executor allow AutoGen to:
	•	Discover tools via semantic retrieval
	•	Execute them via the ToolArsenal execution engine

The existing code from the draft is treated as spec.

Other integrations (LlamaIndex, CrewAI) follow the same pattern: convert ToolDefinition → framework-specific tool type, and use arsenal.execute in their handler.

⸻

11. CLI Interface

High-level CLI commands:
	•	serve – run as MCP server (stdio or sse)
	•	list_tools – list tools from the vector store
	•	search – semantic search over tools from the shell
	•	benchmark – run latency benchmarks (enforce P95 < 50ms)

The CLI implementation (using click + rich) from the draft is considered canonical.

⸻

12. Testing Strategy

12.1 Unit Tests & Fixtures
	•	arsenal fixture – a fresh ToolArsenal instance
	•	sample_tools fixture – standard tool set: query_database, send_email, create_user, process_refund, generate_report

12.2 Retrieval Quality Tests

Key requirements:
	•	Correct top-1 tool for common queries
	•	Multi-tool retrieval for compound tasks
	•	Low scores for irrelevant queries (e.g. weather queries vs internal business tools)
	•	Penalisation when a tool has recently failed in the conversation

The tests in tests/test_retrieval_quality.py from the draft stand as normative examples.

12.3 Performance Tests

Performance contract:
	•	P95 retrieval latency < 50ms for in-memory vector store and local embedder with 1k tools.

tests/test_performance.py:
	•	test_retrieval_latency_memory – 100 iterations, compute P95
	•	test_registration_throughput – register 1,000 tools with throughput > 100 tools/sec

⸻

13. Development Roadmap (Merged)

This merges the earlier 3-phase roadmap with the 7-phase revised roadmap. The 7 phases are canonical; the first three map to:
	•	Phase 1–3: Core + Security + Observability + Context-Aware Routing
	•	Phase 4–5: Production adapters + MCP integration
	•	Phase 6: A2A
	•	Phase 7: Framework integrations & polish

Phase 1: Core Foundation (Weeks 1–3)

Goal: Working prototype with semantic retrieval and basic security.

Deliverables:
	•	Data models (schema/)
	•	In-memory vector store
	•	SentenceTransformer embedder (local)
	•	Basic @register decorator and ToolRegistry
	•	retrieve() with vector similarity
	•	Unit tests for retrieval quality
	•	Minimal CLI (list, search)

Success:
	•	Register Python functions
	•	Retrieve correct tools
	•	P95 retrieval latency < 50ms (in-memory)

Phase 2: Robustness (Weeks 4–5)

Goal: Production-grade execution core.

Deliverables:
	•	Execution engine (retries, timeouts, circuit breaker, health)
	•	Argument validation
	•	Async-native kit
	•	Error handling & structured logging
	•	Basic SecurityPolicy / PermissionChecker integration

Success:
	•	Graceful failure modes
	•	Circuit breaker opens coherently
	•	Full async API

Phase 3: Context-Aware Routing (Weeks 6–7)

Goal: Intelligent multi-signal routing.

Deliverables:
	•	Intent classification (keyword + optional LLM)
	•	Conversation context tracking
	•	Multi-signal scoring (semantic + intent + context + health + cost)
	•	MMR diversity
	•	Optional reranker support
	•	Tests for context-aware & failure-penalising routing

Success:
	•	Failed tools deprioritised within same conversation
	•	Intent matching improves retrieval quality
	•	Reranker improves top-k precision

Phase 4: Production Adapters (Weeks 8–10)

Goal: Cloud-ready infrastructure.

Deliverables:
	•	Qdrant, Chroma, PGVector adapters
	•	OpenAI & Azure embedder adapters
	•	Cohere reranker adapter
	•	OpenTelemetry integration
	•	Prometheus metrics export

Success:
	•	Swappable backends via config only
	•	Distributed tracing functioning
	•	Metrics scraped / exported

Phase 5: MCP Integration (Weeks 11–13)

Goal: First-class MCP citizen.

Deliverables:
	•	MCPClient (consume arbitrary MCP servers)
	•	MCPServer (stdio and SSE)
	•	Dynamic/hybrid tool exposure mode
	•	Auto-discovery of MCP tools
	•	Claude Desktop integration tested

Success:
	•	Consume tools from external MCP servers
	•	Claude Desktop can use ToolArsenal as a dynamic router
	•	Context window utilisation lower vs static listing

Phase 6: A2A Integration (Weeks 14–16)

Goal: Agent-to-Agent interoperability.

Deliverables:
	•	A2AClient
	•	A2A server (FastAPI) implementation
	•	Agent Card generation
	•	Skill-to-ToolDefinition mapping
	•	Basic authentication support

Success:
	•	ToolArsenal can use external agents as tools
	•	ToolArsenal can be used as an A2A agent

Phase 7: Framework Integrations & Polish (Weeks 17–18)

Goal: Ecosystem adoption.

Deliverables:
	•	LangChain, AutoGen, LlamaIndex, CrewAI integrations
	•	Comprehensive docs & examples
	•	Example notebooks (Jupyter)
	•	PyPI package published
	•	“Gold Standard” README and reference docs

Success:
	•	“Drop-in” usage for major frameworks
	•	Clear documentation path for new integrators
	•	PyPI release tool-arsenal==3.0.0

⸻

14. Public API Reference – ToolArsenal

class ToolArsenal:
    """
    Main façade.

    Example:
        arsenal = ToolArsenal()

        @arsenal.register
        def my_tool(x: int) -> str:
            '''Does something useful.'''
            return str(x * 2)

        tools = await arsenal.retrieve_tools("double a number")
    """

    def __init__(
        self,
        config: Optional[ToolArsenalConfig] = None,
        vector_store: Optional[VectorStoreAdapter] = None,
        embedder: Optional[EmbeddingAdapter] = None,
        reranker: Optional[RerankerAdapter] = None,
        telemetry: Optional[TelemetryAdapter] = None,
    ):
        ...

    @classmethod
    def from_config(cls, path: str) -> "ToolArsenal":
        ...

    def register(
        self,
        func: Optional[Callable] = None,
        *,
        name: Optional[str] = None,
        namespace: str = "default",
        capabilities: Optional[List[ToolCapability]] = None,
        requires_confirmation: bool = False,
        tags: Optional[List[str]] = None,
        examples: Optional[List[str]] = None,
    ) -> Callable:
        """
        Decorator to register Python functions as tools.
        """
        ...

    async def add_tool(self, tool: ToolDefinition) -> None:
        ...

    async def add_mcp_server(self, config: MCPServerConfig) -> int:
        ...

    async def add_a2a_agent(self, config: A2AAgentConfig) -> int:
        ...

    async def sync(self) -> int:
        """
        Sync pending registrations to vector store.
        """
        ...

    async def retrieve(self, query: ToolQuery) -> RetrievalResult:
        """
        Core semantic routing function.
        """
        ...

    async def retrieve_tools(
        self,
        query: str,
        limit: int = 5,
        **kwargs,
    ) -> List[Dict[str, Any]]:
        """
        Convenience wrapper: returns OpenAI-compatible schemas.
        """
        ...

    async def execute(self, call: ToolCall) -> ToolResult:
        """
        Execute a tool call with full Aegis protections.
        """
        ...

    async def execute_batch(self, batch: BatchToolCall) -> BatchToolResult:
        ...

    def serve_mcp(self, transport: str = "stdio") -> None:
        ...

    def serve_a2a(self, host: str = "0.0.0.0", port: int = 8080) -> None:
        ...

    @property
    def tool_count(self) -> int:
        ...

    async def get_tool(self, name: str, namespace: str = "default") -> Optional[ToolDefinition]:
        ...

    async def list_tools(
        self,
        namespace: Optional[str] = None,
        source: Optional[ToolSource] = None,
    ) -> List[ToolDefinition]:
        ...

    async def delete_tool(self, name: str, namespace: str = "default") -> bool:
        ...

    async def health_check(self) -> Dict[str, bool]:
        ...


⸻

15. Security Considerations

15.1 Threat Model & Mitigations

Threat	Mitigation
Prompt injection via tool metadata	Validation and sanitisation of descriptions, names
Malicious tool execution	Sandboxing, capability filtering, manual confirmation
Credential leakage	No secrets in ToolDefinition metadata; use env/secret stores
Denial of Service	Rate limits, circuit breakers, timeouts
Unauthorised tool access	Capability-based permissions, SecurityPolicy
MCP server impersonation	TLS verification, server authentication

15.2 Zero-Trust Defaults
	•	Destructive tools (delete_*, payment_*, drop_table, refund_*) require confirmation by default.
	•	Network / file system / financial tools must declare appropriate ToolCapability flags.
	•	External MCP servers and A2A agents must be explicitly configured and can be scoped to namespaces.

⸻

16. Glossary
	•	Agent – LLM-based system that can call tools.
	•	Tool – A callable function or operation (Python, MCP, HTTP, A2A).
	•	Semantic Routing – Selecting tools based on meaning, not just names.
	•	MMR (Maximal Marginal Relevance) – Balances relevance and diversity in retrieval.
	•	MCP – Model Context Protocol, standard for tool integration.
	•	A2A – Agent-to-Agent protocol for agent interoperability.
	•	Circuit Breaker – Pattern that stops calling a failing service for a cooldown.
	•	Embedding – Dense numeric vector representation of text.
	•	Reranking – Second-stage scoring with more precise models.

⸻

17. References
	1.	Model Context Protocol Specification – https://modelcontextprotocol.io/specification/2025-11-25
	2.	Google A2A Protocol – https://google.github.io/A2A/
	3.	OpenAI Function Calling – https://platform.openai.com/docs/guides/function-calling
	4.	Anthropic Tool Use – https://docs.anthropic.com/en/docs/build-with-claude/tool-use
	5.	Hexagonal Architecture – https://alistair.cockburn.us/hexagonal-architecture/
	6.	Circuit Breaker Pattern – https://martinfowler.com/bliki/CircuitBreaker.html

⸻

This document is the ratified, merged ToolArsenal v3.0 Enterprise PRD.
Version: 3.0.0 – “Context is precious. Execution is sacred. Trust is earned.”
