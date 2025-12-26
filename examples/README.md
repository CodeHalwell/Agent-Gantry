# Agent-Gantry Examples

Hands-on examples demonstrating Agent-Gantry features. Each subdirectory has its own README with more
detail and run commands.

## 🚀 Start Here: Fast Track Demo

**New to Agent-Gantry?** Start with the Fast Track Demo to see how to upgrade vanilla OpenAI code to semantic tools in ~10 lines:

```bash
python examples/fast_track_demo.py
# or: uv run python examples/fast_track_demo.py
```

This demo shows the "before and after" of adding Agent-Gantry to a basic LLM call, with clear side-by-side comparison.

## Directory map

- `fast_track_demo.py`: **START HERE** - Shows how to upgrade vanilla OpenAI to semantic tools in ~10 lines.
- `basics/`: Tool registration, async execution, multi-tool routing, and plug-and-play imports.
- `routing/`: Advanced semantic routing, custom adapters, health-aware ranking.
- `execution/`: Circuit breakers, batch execution, and security policy enforcement.
- `llm_integration/`: End-to-end loops with OpenAI/Anthropic/Google/Groq/Mistral plus the semantic
  tool decorator. **All examples use the `@with_semantic_tools` decorator** for consistent "Plug & Play" experience.
- `agent_frameworks/`: Integration with LangChain, AutoGen, CrewAI, LlamaIndex, Semantic Kernel, and more.
- `observability/`: Console telemetry demonstration and token savings analysis.
- `protocols/`: MCP and A2A integration demos (including Claude Desktop config).
- `testing_limits/`: Stress tests for token savings and accuracy with large toolsets (100 tools).

## Running examples

All examples are plain Python scripts. From the repo root:

```bash
# Start with the Fast Track Demo to see the "Plug & Play" experience
python examples/fast_track_demo.py

# Or use uv (recommended for reproducible environments)
uv run python examples/fast_track_demo.py

# Basic examples (plug-and-play ready)
uv run python examples/basics/tool_demo.py
python examples/basics/plug_and_play_semantic_filter.py
python examples/routing/health_aware_routing_demo.py

# Protocol examples
python examples/protocols/mcp_integration_demo.py
```

Provider-specific or framework demos may need extras:

```bash
# Install all example dependencies
pip install -e ".[example-tools,agent-frameworks,mcp,a2a]"

# Or install only what you need
pip install -e ".[openai,anthropic]"  # For LLM provider examples
pip install -e ".[agent-frameworks]"   # For framework integration examples
pip install -e ".[mcp]"                # For Claude Desktop integration
pip install -e ".[a2a]"                # For Agent-to-Agent protocol
```

### Plug-and-play tool catalogs

- Import prebuilt tools from `examples.basics.toolpack` with one line:
  ```bash
  python examples/basics/plug_and_play_semantic_filter.py
  ```
- Swap in your own tool modules using `AgentGantry.from_modules([...])` to keep code changes minimal.

## Key Patterns Demonstrated

### "Plug & Play" Decorator Pattern (Recommended)

Most examples in `llm_integration/` demonstrate the `@with_semantic_tools` decorator:

```python
from agent_gantry import AgentGantry, with_semantic_tools

gantry = AgentGantry()

@gantry.register
def my_tool(...):
    """Tool description."""
    pass

@with_semantic_tools(gantry, limit=3)
async def chat(prompt: str, *, tools=None):
    # Tools automatically injected by decorator
    return await client.chat.completions.create(
        model="...",
        messages=[{"role": "user", "content": prompt}],
        tools=tools
    )
```

### Manual Control Pattern (Power Users)

Examples in `basics/` and `routing/` show fine-grained control:

```python
from agent_gantry import AgentGantry

gantry = AgentGantry()

# Register tools
@gantry.register
def my_tool(...):
    pass

await gantry.sync()

# Manually retrieve tools
tools = await gantry.retrieve_tools("query", limit=5)

# Manually execute tools
result = await gantry.execute(ToolCall(...))
```

Install only the extras you need (e.g., `.[mcp]` for Claude Desktop, `.[a2a]` for the FastAPI agent).
Multi-module tool catalogs can be stitched together with `AgentGantry.from_modules` before running a
script when you want to reuse tool packages across demos.

Most scripts print step-by-step output so you can see retrieval scores, telemetry spans, and results
as they execute. Many demos rely only on the in-memory embedder/vector store; provider-specific demos
(OpenAI, Anthropic, Google GenAI, etc.) will read credentials from the environment when needed.
