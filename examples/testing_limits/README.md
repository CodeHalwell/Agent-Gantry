# examples/testing_limits

Stress tests that measure routing accuracy and token savings with larger tool sets.

## Files
- `stress_test_100_tools.py`: Registers 100 synthetic tools (10 services x 10 actions) with the Nomic
  embedder and reports top-2 retrieval accuracy over ten queries.
- `real_world_30_tools_test.py`: End-to-end test with 30 working tools: retrieval is scored on its
  own, and with `OPENAI_API_KEY` set the retrieved slice is handed to `gpt-5.5` and the tool it
  picks is executed through `gantry.execute`.

## Run commands

```bash
python examples/testing_limits/stress_test_100_tools.py
python examples/testing_limits/real_world_30_tools_test.py
```

Both use the Nomic embedder, so they need `agent-gantry[nomic]` and download the model on first run
(they fall back to the hashing `SimpleEmbedder`, with lower accuracy, if sentence-transformers is
missing). Neither needs an API key; set `OPENAI_API_KEY` to run the real-world test's LLM step.
Useful for benchmarking embedding/reranking choices before deploying a large catalogue.
