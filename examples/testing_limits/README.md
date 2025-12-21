# examples/testing_limits

Stress tests that measure routing accuracy and token savings with larger tool sets.

## Files
- `stress_test_100_tools.py`: Registers 100 synthetic tools to validate retrieval accuracy under
  load and prints per-query scores.
- `real_world_30_tools_test.py`: End-to-end test with 30 tangible tools plus GPT-4o to validate tool
  selection and execution in realistic scenarios.

## Run commands

```bash
python examples/testing_limits/stress_test_100_tools.py
python examples/testing_limits/real_world_30_tools_test.py
```

These scripts are useful for benchmarking embedding/reranking choices and verifying performance
before deploying a large catalog. Set provider API keys if you want to run the real-world test with
actual LLM calls.
