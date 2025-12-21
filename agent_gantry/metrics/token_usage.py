"""
Token usage accounting helpers that rely on provider-reported usage fields.

These helpers intentionally avoid approximate token estimators (e.g., tiktoken)
and instead consume the `usage` blocks that major providers (OpenAI, Anthropic,
Google) return alongside model responses. This makes the reported savings
auditable and reproducible in tests.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping


@dataclass(frozen=True)
class ProviderUsage:
    """
    Normalized provider usage block.

    Providers typically return a `usage` dictionary with prompt, completion, and
    total token counts. Only ``prompt_tokens`` is required; total tokens will be
    derived when missing.
    """

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

    @classmethod
    def from_usage(cls, usage: Mapping[str, int]) -> "ProviderUsage":
        """
        Build from a provider usage mapping (e.g., OpenAI/Anthropic response).
        """
        prompt = int(usage.get("prompt_tokens", 0))
        completion = int(usage.get("completion_tokens", 0))
        total = int(usage.get("total_tokens", prompt + completion))
        if total == 0 and (prompt or completion):
            total = prompt + completion
        return cls(
            prompt_tokens=prompt,
            completion_tokens=completion,
            total_tokens=total,
        )


@dataclass(frozen=True)
class TokenSavings:
    """
    Savings summary between a baseline prompt and an optimized (top-k) prompt.
    """

    baseline: ProviderUsage
    optimized: ProviderUsage
    saved_prompt_tokens: int
    saved_total_tokens: int
    prompt_savings_pct: float
    total_savings_pct: float


def calculate_token_savings(
    baseline: ProviderUsage | Mapping[str, int],
    optimized: ProviderUsage | Mapping[str, int],
) -> TokenSavings:
    """
    Compute token savings using provider-reported usage blocks.

    Args:
        baseline: Usage for the "all tools" (or unfiltered) invocation.
        optimized: Usage for the top-k / filtered invocation.

    Returns:
        TokenSavings with raw and percentage savings.
    """
    base_usage = baseline if isinstance(baseline, ProviderUsage) else ProviderUsage.from_usage(baseline)
    opt_usage = optimized if isinstance(optimized, ProviderUsage) else ProviderUsage.from_usage(optimized)

    saved_prompt = max(0, base_usage.prompt_tokens - opt_usage.prompt_tokens)
    saved_total = max(0, base_usage.total_tokens - opt_usage.total_tokens)

    prompt_pct = 0.0
    if base_usage.prompt_tokens:
        prompt_pct = (saved_prompt / base_usage.prompt_tokens) * 100

    total_pct = 0.0
    if base_usage.total_tokens:
        total_pct = (saved_total / base_usage.total_tokens) * 100

    return TokenSavings(
        baseline=base_usage,
        optimized=opt_usage,
        saved_prompt_tokens=saved_prompt,
        saved_total_tokens=saved_total,
        prompt_savings_pct=prompt_pct,
        total_savings_pct=total_pct,
    )
