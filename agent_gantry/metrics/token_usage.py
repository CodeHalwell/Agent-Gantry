"""
Token usage accounting helpers that rely on provider-reported usage fields.

These helpers intentionally avoid approximate token estimators (e.g., tiktoken)
and instead consume the `usage` blocks that major providers (OpenAI, Anthropic,
Google) return alongside model responses. This makes the reported savings
auditable and reproducible in tests.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

# Provider spellings of each usage figure: OpenAI, Anthropic, Google.
_PROMPT_KEYS = ("prompt_tokens", "input_tokens", "prompt_token_count")
_COMPLETION_KEYS = ("completion_tokens", "output_tokens", "candidates_token_count")
_CACHE_KEYS = ("cache_creation_input_tokens", "cache_read_input_tokens")
_TOTAL_KEYS = ("total_tokens", "total_token_count")
_USAGE_KEYS = _PROMPT_KEYS + _COMPLETION_KEYS + _CACHE_KEYS + _TOTAL_KEYS


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
    #: Portion of ``prompt_tokens`` that Anthropic reported as cache writes or
    #: reads. Broken out so a caller can still see the uncached figure, but
    #: counted inside ``prompt_tokens`` because those tokens were processed --
    #: excluding them makes a cached run look dramatically cheaper than an
    #: uncached baseline and inflates any savings computed against it.
    cached_prompt_tokens: int = 0

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
    def from_usage(cls, usage: Mapping[str, int | float]) -> ProviderUsage:
        """
        Build from a provider usage mapping (e.g., OpenAI/Anthropic/Google response).

        Supports multiple provider token field conventions:
        - OpenAI: prompt_tokens, completion_tokens, total_tokens
        - Anthropic: input_tokens, output_tokens, and the cache counters
          (cache_creation_input_tokens / cache_read_input_tokens), which are
          added to the prompt total and also surfaced as
          ``cached_prompt_tokens``
        - Google: prompt_token_count, candidates_token_count, total_token_count

        Checks each provider convention in order and uses the first present field,
        even if its value is 0. This ensures we don't skip valid zero token counts.
        """
        prompt_raw = next((usage[key] for key in _PROMPT_KEYS if key in usage), None)
        completion_raw = next((usage[key] for key in _COMPLETION_KEYS if key in usage), None)
        total_raw = next((usage[key] for key in _TOTAL_KEYS if key in usage), None)
        # Anthropic reports cached prompt tokens separately from
        # ``input_tokens``; both were processed, so both count as prompt tokens.
        cached = sum(cls._coerce_token_value(usage[key], key) for key in _CACHE_KEYS if key in usage)

        prompt = (
            cls._coerce_token_value(prompt_raw, "prompt_tokens") if prompt_raw is not None else 0
        ) + cached
        completion = (
            cls._coerce_token_value(completion_raw, "completion_tokens")
            if completion_raw is not None
            else 0
        )
        # A missing total is derived; an explicit one (even 0) is preserved.
        total = (
            cls._coerce_token_value(total_raw, "total_tokens")
            if total_raw is not None
            else prompt + completion
        )
        return cls(
            prompt_tokens=prompt,
            completion_tokens=completion,
            total_tokens=total,
            cached_prompt_tokens=cached,
        )

    @classmethod
    def from_response_usage(cls, response: Any) -> ProviderUsage | None:
        """Build from a provider *response*, or ``None`` when it carries no usage.

        The block is ``usage`` on OpenAI and Anthropic responses and
        ``usage_metadata`` on google-genai ones, as an attribute or a key. SDK
        usage objects are attribute holders, so the known fields are read off
        them into a mapping first.
        """
        is_mapping = isinstance(response, Mapping)
        usage = None
        for key in ("usage", "usage_metadata"):
            usage = response.get(key) if is_mapping else getattr(response, key, None)
            if usage is not None:
                break
        if usage is None:
            return None
        if not isinstance(usage, Mapping):
            usage = {
                key: value
                for key in _USAGE_KEYS
                if isinstance(value := getattr(usage, key, None), (int, float))
            }
        return cls.from_usage(usage) if usage else None


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
    base_usage = (
        baseline if isinstance(baseline, ProviderUsage) else ProviderUsage.from_usage(baseline)
    )
    opt_usage = (
        optimized if isinstance(optimized, ProviderUsage) else ProviderUsage.from_usage(optimized)
    )

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
