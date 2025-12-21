# Agent-Gantry Examples

Hands-on examples demonstrating Agent-Gantry features. Each subdirectory has its own README with more
detail and run commands.

## Directory map

- `basics/`: Tool registration, async execution, and multi-tool routing patterns.
- `routing/`: Advanced semantic routing, custom adapters, health-aware ranking.
- `execution/`: Circuit breakers, batch execution, and security policy enforcement.
- `llm_integration/`: End-to-end loops with OpenAI/Anthropic/Google/Groq/Mistral plus the semantic
  tool decorator.
- `observability/`: Console telemetry demonstration.
- `protocols/`: MCP and A2A integration demos (including Claude Desktop config).
- `testing_limits/`: Stress tests for token savings and accuracy.

## Running examples

All examples are plain Python scripts. From the repo root:

```bash
uv run python examples/basics/tool_demo.py           # or: python examples/basics/tool_demo.py
python examples/routing/health_aware_routing_demo.py
python examples/protocols/mcp_integration_demo.py
```

Most scripts print step-by-step output so you can see retrieval scores, telemetry spans, and results
as they execute. Many demos rely only on the in-memory embedder/vector store; provider-specific demos
(OpenAI, Anthropic, Google GenAI, etc.) will read credentials from the environment when needed.***
