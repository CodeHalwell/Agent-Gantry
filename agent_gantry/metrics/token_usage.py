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
    derived when missing. If ``total_tokens`` is explicitly provided (even as 0),
    that value is preserved.
    """

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

    @staticmethod
    def _coerce_token_value(value: int | float, field_name: str) -> int:
        """
        Coerce a provider-reported token value to an int with validation.

        Providers sometimes return floats; we only accept integer-equivalent values
        to avoid silently truncating non-integer token counts.
        """
        if isinstance(value, float) and not value.is_integer():
            raise ValueError(
                f"{field_name} must be an integer token count, got non-integer float: {value!r}"
            )
        return int(value)

    @classmethod
    def from_usage(cls, usage: Mapping[str, int | float]) -> "ProviderUsage":
        """
        Build from a provider usage mapping (e.g., OpenAI/Anthropic response).
        """
        prompt_raw: int | float | None = usage.get("prompt_tokens")
        completion_raw: int | float | None = usage.get("completion_tokens")

        prompt = cls._coerce_token_value(prompt_raw, "prompt_tokens") if prompt_raw is not None else 0
        completion = (
            cls._coerce_token_value(completion_raw, "completion_tokens")
            if completion_raw is not None
            else 0
        )

        if "total_tokens" in usage:
            total_raw = usage["total_tokens"]
            total = (
                cls._coerce_token_value(total_raw, "total_tokens")
                if total_raw is not None
                else prompt + completion
            )
        else:
            total = prompt + completion

        # If total is missing we derive it; if explicitly provided as 0 we preserve that
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
    baseline: ProviderUsage | Mapping[str, int | float],
    optimized: ProviderUsage | Mapping[str, int | float],
) -> TokenSavings:
    """
    Compute token savings using provider-reported usage blocks.

    Note: savings are clamped at zero to avoid negative values when an optimized
    request unexpectedly uses more tokens than the baseline. Callers should
    inspect their inputs if negative deltas were expected.

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
