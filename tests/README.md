# tests

Guide to the pytest suite. Tests mirror the roadmap phases and the primary user flows:

- `conftest.py`: Shared fixtures for a pre-wired gantry instance, sample tools, telemetry fakes, and
  temporary directories.
- `test_gantry.py`: Core facade behavior (registration, syncing, execution, health checks).
- `test_tool.py`: Validation rules and schema conversions for `ToolDefinition`.
- `test_retrieval.py`: Semantic routing and retrieval result behavior.
- `test_phase2.py`: Phase 2 robustness milestones (retries, timeouts, circuit breakers).
- `test_phase3_routing.py`: Routing weight configuration and diversity controls.
- `test_phase4_adapters.py`: Adapter compatibility checks for embedders, rerankers, and vector stores.
- `test_llm_sdk_compatibility.py`: Import/init coverage for OpenAI, Anthropic, Google GenAI, Vertex,
  and Mistral SDKs (ensures optional deps do not regress schema compatibility).
- `test_phase5_mcp.py`: MCP server/client discovery and dynamic mode behaviors.
- `test_phase6_a2a.py`: A2A agent serving and client discovery flows.
- `test_decorator.py`: Ensures the semantic tool injection decorator wires tool schemas correctly for
  OpenAI/Anthropic/Google SDKs.
- `test_lancedb_nomic.py`: LanceDB + Nomic integration and skill storage.
- `test_token_savings_and_accuracy.py`: Validates token savings math and retrieval accuracy on
  synthetic workloads.

To run the suite:

```bash
pytest
```

Use `pytest -k "<name>"` to target a single area while iterating.***
