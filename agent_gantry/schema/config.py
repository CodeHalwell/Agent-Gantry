"""
Configuration models for Agent-Gantry.

Single source of truth for all configuration options.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class VectorStoreConfig(BaseModel):
    """Configuration for vector store backend."""

    type: Literal["memory", "qdrant", "chroma", "pgvector", "pinecone", "weaviate"] = "memory"
    url: str | None = None
    api_key: str | None = None
    collection_name: str = "agent_gantry"
    dimension: int | None = None
    options: dict[str, Any] = Field(default_factory=dict)


class EmbedderConfig(BaseModel):
    """Configuration for embedding backend."""

    type: Literal[
        "openai", "azure", "cohere", "huggingface", "sentence_transformers", "ollama"
    ] = "sentence_transformers"
    model: str = "all-MiniLM-L6-v2"
    api_key: str | None = None
    api_base: str | None = None
    batch_size: int = 100
    max_retries: int = 3


class RerankerConfig(BaseModel):
    """Configuration for reranker backend."""

    enabled: bool = False
    type: Literal["cohere", "cross_encoder", "llm"] = "cross_encoder"
    model: str | None = None
    top_k: int = 10


class RoutingConfig(BaseModel):
    """Configuration for semantic routing."""

    weights: dict[str, float] = Field(
        default_factory=lambda: {
            "semantic": 0.6,
            "intent": 0.15,
            "conversation": 0.1,
            "health": 0.1,
            "cost": 0.05,
        }
    )
    enable_intent_classification: bool = True
    use_llm_for_intent: bool = False
    enable_mmr: bool = True
    mmr_lambda: float = 0.7


class ExecutionConfig(BaseModel):
    """Configuration for tool execution."""

    default_timeout_ms: int = 30000
    max_retries: int = 3
    circuit_breaker_threshold: int = 5
    circuit_breaker_timeout_s: int = 60
    enable_sandbox: bool = False
    sandbox_type: Literal["none", "subprocess", "docker"] = "none"


class TelemetryConfig(BaseModel):
    """Configuration for observability."""

    enabled: bool = True
    type: Literal["console", "opentelemetry", "datadog", "prometheus"] = "console"
    otlp_endpoint: str | None = None
    service_name: str = "agent_gantry"
    expose_prometheus: bool = False
    prometheus_port: int = 9090


class MCPServerConfig(BaseModel):
    """Configuration for an MCP server to connect to."""

    name: str
    command: list[str]
    args: list[str] = Field(default_factory=list)
    env: dict[str, str] = Field(default_factory=dict)
    namespace: str = "default"


class MCPConfig(BaseModel):
    """Configuration for MCP integration."""

    servers: list[MCPServerConfig] = Field(default_factory=list)
    serve_mcp: bool = False
    mcp_mode: Literal["dynamic", "static", "hybrid"] = "dynamic"


class A2AAgentConfig(BaseModel):
    """Configuration for an A2A agent to connect to."""

    name: str
    url: str
    namespace: str = "default"


class A2AConfig(BaseModel):
    """Configuration for A2A integration."""

    agents: list[A2AAgentConfig] = Field(default_factory=list)
    serve_a2a: bool = False
    a2a_port: int = 8080


class AgentGantryConfig(BaseModel):
    """
    Main configuration for Agent-Gantry.

    Can be loaded from YAML files or constructed programmatically.
    """

    vector_store: VectorStoreConfig = Field(default_factory=VectorStoreConfig)
    embedder: EmbedderConfig = Field(default_factory=EmbedderConfig)
    reranker: RerankerConfig = Field(default_factory=RerankerConfig)
    routing: RoutingConfig = Field(default_factory=RoutingConfig)
    execution: ExecutionConfig = Field(default_factory=ExecutionConfig)
    telemetry: TelemetryConfig = Field(default_factory=TelemetryConfig)
    mcp: MCPConfig = Field(default_factory=MCPConfig)
    a2a: A2AConfig = Field(default_factory=A2AConfig)

    auto_sync: bool = True
    sync_on_register: bool = False

    @classmethod
    def from_yaml(cls, path: str) -> AgentGantryConfig:
        """Load configuration from a YAML file."""
        import yaml  # type: ignore[import-untyped]

        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(**data)
