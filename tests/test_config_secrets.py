"""API keys in configuration stay out of repr() and str()."""

from __future__ import annotations

import pytest

from dotfetch.schema.config import (
    DotFetchConfig,
    EmbedderConfig,
    LLMConfig,
    VectorStoreConfig,
)

SECRET = "sk-test-not-a-real-key-4f9c"


@pytest.mark.parametrize(
    "config",
    [
        EmbedderConfig(type="openai", api_key=SECRET),
        VectorStoreConfig(type="qdrant", url="http://localhost:6333", api_key=SECRET),
        LLMConfig(api_key=SECRET),
    ],
    ids=["embedder", "vector_store", "llm"],
)
def test_api_key_is_not_in_repr_or_str(config: object) -> None:
    assert SECRET not in repr(config)
    assert SECRET not in str(config)
    assert SECRET not in f"{config}"
    # Still readable where it is needed.
    assert config.api_key == SECRET  # type: ignore[attr-defined]


def test_nested_config_does_not_reveal_keys() -> None:
    config = DotFetchConfig(
        embedder=EmbedderConfig(type="openai", api_key=SECRET),
        vector_store=VectorStoreConfig(type="qdrant", url="http://x", api_key=SECRET),
    )
    assert SECRET not in repr(config)
    assert SECRET not in str(config)


def test_dump_still_carries_the_key() -> None:
    # repr=False hides the key from printing and logging only; a dump is how
    # a config is saved, so it keeps the key. Callers redact dumps themselves.
    assert EmbedderConfig(type="openai", api_key=SECRET).model_dump()["api_key"] == SECRET
