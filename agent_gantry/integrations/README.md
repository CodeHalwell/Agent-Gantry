# agent_gantry/integrations

Integration shims for popular agent frameworks.

- `decorator.py`: Semantic tool injection decorator for LLM clients.
- `framework_adapters.py`: Thin, dependency-free helpers for LangGraph, Semantic Kernel, CrewAI, Google ADK, and Strands.

Example:

```python
from agent_gantry import AgentGantry
from agent_gantry.integrations import fetch_framework_tools

gantry = AgentGantry()
# ... register tools ...

tools = await fetch_framework_tools(
    gantry,
    "send a follow-up email",
    framework="langgraph",
    limit=2,
)
# Pass `tools` directly into the framework's OpenAI-style tool slot
```
