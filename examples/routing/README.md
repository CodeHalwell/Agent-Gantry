# examples/routing

Routing-focused demos that highlight semantic retrieval, filtering, health weighting, and custom
adapters.

## Files
- `custom_adapter_demo.py`: Illustrates building a minimal custom embedder adapter and plugging it
  into the router.
- `filtering_demo.py`: Shows namespace filtering to restrict retrieval to specific tool groups.
- `health_aware_routing_demo.py`: Demonstrates how tool health (success rate, circuit breaker state)
  affects ranking.
- `nomic_tool_demo.py`: Uses the high-accuracy Nomic embedder for better semantic matches on nuanced
  queries.

## Run commands

```bash
python examples/routing/custom_adapter_demo.py
python examples/routing/filtering_demo.py
python examples/routing/health_aware_routing_demo.py
python examples/routing/nomic_tool_demo.py
```

These scripts log retrieval scores and include inline comments explaining how routing weights and
adapters change the top-k results.
