"""
Observability modules for dotfetch.

Telemetry, metrics, and logging.
"""

from dotfetch.observability.opentelemetry_adapter import (
    OpenTelemetryAdapter,
    PrometheusTelemetryAdapter,
)
from dotfetch.observability.telemetry import TelemetryAdapter

__all__ = [
    "TelemetryAdapter",
    "OpenTelemetryAdapter",
    "PrometheusTelemetryAdapter",
]
