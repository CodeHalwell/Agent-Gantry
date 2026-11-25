"""Evaluation harnesses. See :mod:`dotfetch.evaluation.retrieval`."""

from __future__ import annotations

from dotfetch.evaluation.retrieval import (
    CaseOutcome,
    RetrievalCase,
    RetrievalReport,
    Retriever,
    ThresholdCalibration,
    calibrate_threshold,
    compare_to_baseline,
    evaluate_retrieval,
    format_reports,
    mcp_server_retriever,
    skill_retriever,
    tool_retriever,
)

__all__ = [
    "CaseOutcome",
    "RetrievalCase",
    "RetrievalReport",
    "Retriever",
    "ThresholdCalibration",
    "calibrate_threshold",
    "compare_to_baseline",
    "evaluate_retrieval",
    "format_reports",
    "mcp_server_retriever",
    "skill_retriever",
    "tool_retriever",
]
