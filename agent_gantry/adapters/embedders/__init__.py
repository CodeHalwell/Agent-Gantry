"""
Embedding adapters for Agent-Gantry.
"""

from agent_gantry.adapters.embedders.base import EmbeddingAdapter
from agent_gantry.adapters.embedders.simple import SimpleEmbedder

__all__ = [
    "EmbeddingAdapter",
    "SimpleEmbedder",
]
