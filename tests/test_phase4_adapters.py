"""
Tests for Phase 4: production adapters and observability.
"""

from __future__ import annotations

import pytest
import yaml

from agent_gantry import AgentGantry
from agent_gantry.observability.opentelemetry_adapter import PrometheusTelemetryAdapter




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


