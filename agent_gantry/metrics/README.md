# agent_gantry/metrics

Utilities for measuring and reporting token usage and savings. This package is intentionally small
and dependency-free so it can be reused from both library code and examples.

- `token_usage.py`: Normalizes provider-reported `usage` blocks into a common `ProviderUsage` data
  class and computes `TokenSavings` between a baseline request (e.g., all tools) and an optimized
  request (e.g., top-k tools). Calculations are derived from provider-reported token counts rather
  than estimators to keep results auditable.

## Quick example

```python
from agent_gantry.metrics.token_usage import ProviderUsage, calculate_token_savings

baseline = ProviderUsage.from_usage({"prompt_tokens": 366, "completion_tokens": 0})
optimized = ProviderUsage.from_usage({"prompt_tokens": 78, "completion_tokens": 0})

savings = calculate_token_savings(baseline, optimized)
print(savings.prompt_savings_pct)  # ~79%
```

The helper is used throughout the examples (see `examples/llm_integration/token_savings_demo.py`) to
report end-to-end context window reductions when using semantic routing.***
