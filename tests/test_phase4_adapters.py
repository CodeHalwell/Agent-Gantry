"""
Tests for Phase 4: production adapters and observability.
"""

from __future__ import annotations

import pytest
import yaml

from dotfetch import DotFetch
from dotfetch.observability.opentelemetry_adapter import PrometheusTelemetryAdapter


@pytest.mark.asyncio
async def test_prometheus_metrics_and_health(tmp_path) -> None:
    """Prometheus adapter should emit metrics and report healthy."""
    config_path = tmp_path / "config.prom.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "telemetry": {
                    "type": "prometheus",
                    "service_name": "dotfetch_metrics",
                    "prometheus_port": 9100,
                }
            }
        )
    )
    fetcher = DotFetch.from_config(str(config_path))

    @fetcher.register
    def echo(text: str) -> str:
        """Echo text for telemetry tests."""
        return text

    await fetcher.sync()
    await fetcher.retrieve_tools("echo some text", limit=1)

    assert isinstance(fetcher._telemetry, PrometheusTelemetryAdapter)
    metrics = fetcher._telemetry.export_metrics()
    assert "dotfetch_retrievals_total" in metrics

    health = await fetcher.health_check()
    assert health["telemetry"]


