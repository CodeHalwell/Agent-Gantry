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
- `jev_selection_demo.py`: Replaces semantic matching with a decision model — `JevSelector` for
  tools and skills, `JevReranker` over the vector-search shortlist. Runs without an API key, where
  it demonstrates the fail-open path instead.
- `jev_threshold_tuning_demo.py`: Picks a selector threshold from its own scores rather than
  guessing, by sweeping thresholds over a labelled query set and printing what each keeps and cuts.
- `asymmetric_embedder_demo.py`: Writing an embedder whose query side differs from its document
  side (`embed_query` vs `embed_text`), as Nomic, E5 and BGE all want. No API key or model download.

## Run commands

```bash
python examples/routing/custom_adapter_demo.py
python examples/routing/filtering_demo.py
python examples/routing/health_aware_routing_demo.py
python examples/routing/nomic_tool_demo.py
python examples/routing/jev_selection_demo.py          # needs agent-gantry[jev]
python examples/routing/jev_threshold_tuning_demo.py   # needs TYPESAFE_API_KEY
python examples/routing/asymmetric_embedder_demo.py    # no key, no download
```

These scripts log retrieval scores and include inline comments explaining how routing weights and
adapters change the top-k results.
