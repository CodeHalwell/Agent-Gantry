"""Configuration-driven construction of Agent-Gantry's pluggable adapters.

These factories translate the declarative ``*Config`` models into concrete
adapter instances (vector store, embedder, reranker, telemetry). They are kept
out of :class:`~agent_gantry.core.gantry.AgentGantry` because they hold no
instance state — each is a pure ``config -> adapter`` mapping — which keeps the
facade focused on orchestration. Imports for the optional backends (Qdrant,
Chroma, PGVector, LanceDB, Nomic, sentence-transformers, Cohere, …) stay lazy —
inside the branch that needs them — so the base install never requires them; the
always-available adapters (in-memory store, simple/OpenAI embedders, telemetry)
are imported at module load.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, cast

from agent_gantry.adapters.embedders.openai import AzureOpenAIEmbedder, OpenAIEmbedder
from agent_gantry.adapters.embedders.simple import SimpleEmbedder
from agent_gantry.adapters.vector_stores.memory import InMemoryVectorStore
from agent_gantry.observability.console import ConsoleTelemetryAdapter, NoopTelemetryAdapter
from agent_gantry.observability.opentelemetry_adapter import (
    OpenTelemetryAdapter,
    PrometheusTelemetryAdapter,
)

if TYPE_CHECKING:
    from agent_gantry.adapters.embedders.base import EmbeddingAdapter
    from agent_gantry.adapters.rerankers.base import RerankerAdapter
    from agent_gantry.adapters.vector_stores.base import VectorStoreAdapter
    from agent_gantry.observability.telemetry import TelemetryAdapter
    from agent_gantry.schema.config import (
        EmbedderConfig,
        RerankerConfig,
        TelemetryConfig,
        VectorStoreConfig,
    )

logger = logging.getLogger(__name__)


def build_vector_store(config: VectorStoreConfig) -> VectorStoreAdapter:
    """Construct a vector store adapter from configuration."""
    if config.type == "qdrant":
        from agent_gantry.adapters.vector_stores.remote import QdrantVectorStore

        if not config.url:
            raise ValueError("Qdrant requires 'url' in configuration")
        return cast(
            "VectorStoreAdapter",
            QdrantVectorStore(
                url=config.url,
                api_key=config.api_key,
                collection_name=config.collection_name,
                dimension=config.dimension or 1536,
            ),
        )
    if config.type == "chroma":
        from agent_gantry.adapters.vector_stores.remote import ChromaVectorStore

        return cast(
            "VectorStoreAdapter",
            ChromaVectorStore(
                url=config.url,
                collection_name=config.collection_name,
                persist_directory=config.db_path,
            ),
        )
    if config.type == "pgvector":
        from agent_gantry.adapters.vector_stores.remote import PGVectorStore

        if not config.url:
            raise ValueError("PGVector requires 'url' (connection string) in configuration")
        return cast(
            "VectorStoreAdapter",
            PGVectorStore(
                url=config.url,
                table_name=config.collection_name,
                dimension=config.dimension or 1536,
            ),
        )
    if config.type == "lancedb":
        from agent_gantry.adapters.vector_stores.lancedb import LanceDBVectorStore

        return cast(
            "VectorStoreAdapter",
            LanceDBVectorStore(
                db_path=config.db_path,
                tools_table=config.collection_name,
                dimension=config.dimension or 768,
            ),
        )
    return InMemoryVectorStore()


def build_embedder(config: EmbedderConfig) -> EmbeddingAdapter:
    """Construct an embedder from configuration."""
    if config.type == "openai" and config.api_key:
        return OpenAIEmbedder(config)
    if config.type == "azure" and config.api_key:
        return AzureOpenAIEmbedder(config)
    if config.type == "nomic":
        from agent_gantry.adapters.embedders.nomic import NomicEmbedder

        return NomicEmbedder(
            model=config.model or "nomic-ai/nomic-embed-text-v1.5",
            dimension=config.dimension,
            task_type=config.task_type or "search_document",
        )
    if config.type == "sentence_transformers":
        try:
            import sentence_transformers as _st  # noqa: F401

            from agent_gantry.adapters.embedders.sentence_transformers import (
                SentenceTransformersEmbedder,
            )

            return SentenceTransformersEmbedder(
                model=config.model or "all-MiniLM-L6-v2",
                dimension=config.dimension,
            )
        except ImportError:
            logger.debug("sentence-transformers not available, falling back to SimpleEmbedder")
    return SimpleEmbedder()


def build_reranker(config: RerankerConfig) -> RerankerAdapter | None:
    """Construct a reranker from configuration."""
    if not config.enabled:
        return None
    if config.type == "cohere":
        from agent_gantry.adapters.rerankers.cohere import CohereReranker

        return CohereReranker(model=config.model)
    if config.type == "cross_encoder":
        from agent_gantry.adapters.rerankers.cross_encoder import CrossEncoderReranker

        return CrossEncoderReranker(
            model=config.model or "cross-encoder/ms-marco-MiniLM-L-6-v2",
        )
    return None


def build_telemetry(config: TelemetryConfig) -> TelemetryAdapter:
    """Construct telemetry adapter from configuration."""
    if not config.enabled:
        return NoopTelemetryAdapter()
    if config.type == "opentelemetry":
        return OpenTelemetryAdapter(
            service_name=config.service_name,
            otlp_endpoint=config.otlp_endpoint,
        )
    if config.type == "prometheus":
        return PrometheusTelemetryAdapter(
            service_name=config.service_name,
            prometheus_port=config.prometheus_port,
        )
    return ConsoleTelemetryAdapter()
