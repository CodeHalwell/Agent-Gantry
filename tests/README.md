# tests

Guide to the test suite.

- `conftest.py`: Shared pytest fixtures for gantry instances, sample tools, and telemetry fakes.
- `test_gantry.py`: Core facade behavior (registration, syncing, execution, health checks).
- `test_tool.py`: Validation rules and schema conversions for `ToolDefinition`.
- `test_retrieval.py`: Semantic routing and retrieval result behavior.
- `test_phase2.py`: Phase 2 robustness milestones (retries, timeouts, circuit breakers).
- `test_phase3_routing.py`: Routing weight configuration and diversity controls.
- `test_phase4_adapters.py`: Adapter compatibility checks for embedders, rerankers, and vector stores.
- `test_llm_sdk_compatibility.py`: Import/init coverage for OpenAI, Anthropic, Google GenAI, Vertex,
  and Mistral SDKs.
- `test_phase5_mcp.py`: MCP server/client discovery and dynamic mode behaviors.
- `test_phase6_a2a.py`: A2A agent serving and client discovery flows.
