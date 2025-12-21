# agent_gantry/integrations

Integration shims for popular agent frameworks. These helpers keep framework dependencies optional
while making it easy to reuse Agent-Gantry's semantic routing and schema conversion.

## Modules

- `decorator.py`: Implements `with_semantic_tools`, a decorator that retrieves relevant tools at call
  time and injects them into LLM SDK calls (OpenAI, Anthropic, Google, Groq, etc.).
- `framework_adapters.py`: Dependency-free helpers for LangGraph, Semantic Kernel, CrewAI, Google
  ADK, and Strands. They primarily convert `ToolDefinition` objects into the framework's preferred
  tool schema.

## Example: fetch tools for a framework

```python
from agent_gantry import AgentGantry
from agent_gantry.integrations import fetch_framework_tools

gantry = AgentGantry()

@gantry.register
def send_email(to: str, body: str) -> str:
    ...

tools = await fetch_framework_tools(
    gantry,
    "send a follow-up email",
    framework="langgraph",
    limit=2,
)
# Pass `tools` directly into the framework's OpenAI-style tool slot
```

See `examples/llm_integration/decorator_demo.py` and `examples/llm_integration/llm_demo.py` for
end-to-end usage.***
