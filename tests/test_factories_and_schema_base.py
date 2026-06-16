"""Unit tests for the extracted ``core/factories.py`` and ``schema/base.py``.

These modules were introduced by the simplicity refactor; the tests pin the
behaviour that callers rely on: config-driven adapter selection (including the
URL-validation error paths), the shared identifier newline-rejection security
invariant, and the shared ``HealthMetrics`` field constraints.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from agent_gantry.adapters.embedders.simple import SimpleEmbedder
from agent_gantry.adapters.vector_stores.memory import InMemoryVectorStore
from agent_gantry.core.factories import (
    build_embedder,
    build_reranker,
    build_telemetry,
    build_vector_store,
)
from agent_gantry.observability.console import ConsoleTelemetryAdapter, NoopTelemetryAdapter
from agent_gantry.schema.base import HealthMetrics, reject_newlines
from agent_gantry.schema.config import (
    EmbedderConfig,
    RerankerConfig,
    TelemetryConfig,
    VectorStoreConfig,
)


class TestBuildVectorStore:
    def test_memory_is_the_default(self) -> None:
        assert isinstance(build_vector_store(VectorStoreConfig()), InMemoryVectorStore)

    def test_qdrant_requires_url(self) -> None:
        with pytest.raises(ValueError, match="Qdrant requires 'url'"):
            build_vector_store(VectorStoreConfig(type="qdrant"))

    def test_pgvector_requires_url(self) -> None:
        with pytest.raises(ValueError, match="PGVector requires 'url'"):
            build_vector_store(VectorStoreConfig(type="pgvector"))


class TestBuildEmbedder:
    def test_falls_back_to_simple_without_credentials(self) -> None:
        # "openai" without an api_key falls through every branch to SimpleEmbedder.
        assert isinstance(build_embedder(EmbedderConfig(type="openai")), SimpleEmbedder)


class TestBuildReranker:
    def test_disabled_returns_none(self) -> None:
        assert build_reranker(RerankerConfig(enabled=False)) is None


class TestBuildTelemetry:
    def test_disabled_returns_noop(self) -> None:
        assert isinstance(build_telemetry(TelemetryConfig(enabled=False)), NoopTelemetryAdapter)

    def test_console_is_the_default(self) -> None:
        assert isinstance(build_telemetry(TelemetryConfig()), ConsoleTelemetryAdapter)


class TestRejectNewlines:
    def test_clean_value_passes_through_unchanged(self) -> None:
        assert reject_newlines("default") == "default"

    def test_non_string_passes_through(self) -> None:
        assert reject_newlines(None) is None

    @pytest.mark.parametrize("bad", ["trailing\n", "carriage\r", "mid\ndle"])
    def test_newlines_are_rejected(self, bad: str) -> None:
        with pytest.raises(ValueError, match="newline"):
            reject_newlines(bad)


class TestBuildImplClassCache:
    def test_same_base_returns_cached_class(self) -> None:
        from agent_gantry.integrations.agent_framework_provider import _build_impl_class

        class _FakeContextProvider:
            def __init__(self, *, source_id: str) -> None:
                self.source_id = source_id

        cls1 = _build_impl_class(_FakeContextProvider)
        cls2 = _build_impl_class(_FakeContextProvider)
        assert cls1 is cls2
        assert issubclass(cls1, _FakeContextProvider)


class TestHealthMetrics:
    def test_defaults(self) -> None:
        h = HealthMetrics()
        assert h.success_rate == 1.0
        assert h.consecutive_failures == 0
        assert h.last_success is None and h.last_failure is None

    @pytest.mark.parametrize("bad_rate", [-0.1, 1.5])
    def test_success_rate_is_bounded(self, bad_rate: float) -> None:
        with pytest.raises(ValidationError):
            HealthMetrics(success_rate=bad_rate)
