# agent_gantry package

Central package for Agent-Gantry. It wires together the semantic router, execution
engine, telemetry, and protocol adapters.

- `__init__.py`: Exposes the public Agent-Gantry API surface.
- `adapters/`: Adapter implementations for vector stores, embedders, rerankers, and executors.
- `cli/`: Entry points and helpers for the command-line interface.
- `core/`: Core orchestration classes (facade, router, executor, registry, security).
- `integrations/`: Stubs for framework integrations (LangChain, AutoGen, LlamaIndex, CrewAI).
- `observability/`: Telemetry and logging adapters.
- `providers/`: Clients that import tools from external systems (e.g., A2A agents).
- `schema/`: Pydantic models for configs, tools, queries, events, and execution.
- `servers/`: MCP and A2A server entry points.
