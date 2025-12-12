"""
Tests for Phase 4: production adapters and observability.
"""

from __future__ import annotations

import pytest
import yaml

from agent_gantry import AgentGantry
from agent_gantry.adapters.embedders.openai import OpenAIEmbedder
from agent_gantry.adapters.rerankers.cohere import CohereReranker
from agent_gantry.adapters.vector_stores.remote import QdrantVectorStore
from agent_gantry.observability.opentelemetry_adapter import (
    OpenTelemetryAdapter,
    PrometheusTelemetryAdapter,
)


@pytest.mark.asyncio
async def test_config_switches_adapters(tmp_path) -> None:
    """Ensure adapters swap purely via YAML config."""
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "vector_store": {"type": "qdrant", "url": "http://localhost:6333"},
                "embedder": {"type": "openai", "api_key": "test-key", "model": "text-embedding-3-small"},
                "reranker": {"enabled": True, "type": "cohere", "model": "rerank-english-v3.0"},
                "telemetry": {"type": "opentelemetry", "service_name": "agent_gantry_phase4"},
            }
        )
    )

    gantry = AgentGantry.from_config(str(config_path))

    assert isinstance(gantry._vector_store, QdrantVectorStore)
    assert isinstance(gantry._embedder, OpenAIEmbedder)
    assert isinstance(gantry._reranker, CohereReranker)
    assert isinstance(gantry._telemetry, OpenTelemetryAdapter)

    health = await gantry.health_check()
    assert health["vector_store"]
    assert health["embedder"]
    assert health["telemetry"]


@pytest.mark.asyncio
async def test_prometheus_metrics_and_health(tmp_path) -> None:
    """Prometheus adapter should emit metrics and report healthy."""
    config_path = tmp_path / "config.prom.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "telemetry": {
                    "type": "prometheus",
                    "service_name": "agent_gantry_metrics",
                    "prometheus_port": 9100,
                }
            }
        )
    )
    gantry = AgentGantry.from_config(str(config_path))

    @gantry.register
    def echo(text: str) -> str:
        """Echo text for telemetry tests."""
        return text

    await gantry.sync()
    await gantry.retrieve_tools("echo some text", limit=1)

    assert isinstance(gantry._telemetry, PrometheusTelemetryAdapter)
    metrics = gantry._telemetry.export_metrics()
    assert "agent_gantry_retrievals_total" in metrics

    health = await gantry.health_check()
    assert health["telemetry"]


@pytest.mark.asyncio
async def test_remote_vector_store_requires_endpoint() -> None:
    """Remote stores should surface unhealthy when misconfigured."""
    store = QdrantVectorStore()
    await store.initialize()
    assert not await store.health_check()
