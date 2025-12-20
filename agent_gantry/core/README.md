# agent_gantry/core

Core orchestration logic for Agent-Gantry.

- `__init__.py`: Package exports for core classes.
- `context.py`: Conversation context handling and tool query modeling helpers.
- `executor.py`: Execution engine that runs tool calls with retries, timeouts, and circuit breakers.
- `gantry.py`: Main AgentGantry facade, registration API, retrieval, execution, and server helpers.
- `registry.py`: Internal registry that maps tool names to Python callables.
- `router.py`: Semantic routing logic that scores tools using vector search and reranking weights.
- `security.py`: Capability-based security policy enforcement used during execution.
