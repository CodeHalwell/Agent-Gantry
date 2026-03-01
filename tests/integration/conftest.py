"""
Integration test configuration and fixtures.

These tests require external services and are skipped by default.
Run with: pytest -m integration
"""

from __future__ import annotations

import os

import pytest


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Skip integration tests unless explicitly requested."""
    if config.getoption("-m", default="") and "integration" in config.getoption("-m", default=""):
        return  # Don't skip if integration marker is explicitly requested

    skip_integration = pytest.mark.skip(reason="Integration tests require -m integration")
    for item in items:
        if "integration" in item.keywords:
            item.add_marker(skip_integration)


@pytest.fixture
def qdrant_url() -> str:
    """Qdrant connection URL from environment or default."""
    return os.getenv("QDRANT_URL", "http://localhost:6333")


@pytest.fixture
def pgvector_url() -> str:
    """PGVector connection URL from environment or default."""
    return os.getenv("PGVECTOR_URL", "postgresql://postgres:postgres@localhost:5432/agent_gantry")
