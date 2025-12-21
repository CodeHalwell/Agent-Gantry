# Agent-Gantry Examples

This directory contains examples demonstrating various features of Agent-Gantry, organized by category.

## Directory Structure

### 1. Basics (`examples/basics/`)
Fundamental concepts of Agent-Gantry.
- `tool_demo.py`: Basic "Hello World" example demonstrating tool registration and retrieval.
- `multi_tool_demo.py`: Demonstrates semantic routing across a larger set of 10 diverse tools using the default embedder.
- `async_demo.py`: Shows native support for asynchronous tool execution.

### 2. Routing & Retrieval (`examples/routing/`)
Advanced semantic routing, filtering, and custom adapters.
- `nomic_tool_demo.py`: Shows how to use the high-accuracy Nomic embedder for better semantic matching.
- `filtering_demo.py`: Demonstrates filtering tools by namespace during retrieval.
- `health_aware_routing_demo.py`: Demonstrates how tool health status affects retrieval (excluding unhealthy tools).
- `custom_adapter_demo.py`: Shows how to implement and use a custom Embedding Adapter.

### 3. Execution & Reliability (`examples/execution/`)
Robust execution patterns, security, and error handling.
- `circuit_breaker_demo.py`: Demonstrates the circuit breaker pattern blocking a failing tool.
- `batch_execution_demo.py`: Shows parallel execution of multiple tools using `execute_batch`.
- `security_demo.py`: Demonstrates the Security Policy system, including confirmation requirements for sensitive tools.

### 4. LLM Integration (`examples/llm_integration/`)
Integrating Agent-Gantry with Large Language Models.
- `llm_demo.py`: A complete loop demonstrating Query -> Retrieve -> LLM -> Execute.
- `decorator_demo.py`: Shows how to use the `@with_semantic_tools` decorator to automatically inject relevant tools into LLM functions.
- `token_savings_demo.py`: Benchmarks token usage, proving a ~79% reduction in context window costs.

### 5. Observability (`examples/observability/`)
Monitoring and debugging.
- `telemetry_demo.py`: Demonstrates observability features using the Console Telemetry Adapter.

### 6. Protocols (`examples/protocols/`)
Integration with standard agent protocols (MCP, A2A).
- `mcp_integration_demo.py`: Runs Agent-Gantry as an MCP server and connects to other MCP servers.
- `a2a_integration_demo.py`: Shows how to serve Agent-Gantry over the A2A protocol and call remote agents.
- `claude_desktop_config.json`: Sample Claude Desktop configuration pointing at an Agent-Gantry MCP server.

### 7. Stress Testing (`examples/testing_limits/`)
Performance and scalability testing.
- `stress_test_100_tools.py`: Registers 100 distinct tools and verifies semantic routing accuracy.

## MCP Integration Demo

Run the comprehensive MCP integration demo:

```bash
python examples/protocols/mcp_integration_demo.py
```

This demonstrates:
- Serving Agent-Gantry as an MCP server in dynamic mode
- Connecting to external MCP servers as a client
- Meta-tool discovery and execution flow
- Context window savings (90%+ reduction)

## Claude Desktop Integration

To integrate Agent-Gantry with Claude Desktop:

1. Create your Agent-Gantry server script:

```python
# my_agent_gantry_server.py
import asyncio
from agent_gantry import AgentGantry

async def main():
    gantry = AgentGantry()
    
    # Register your tools
    @gantry.register
    def my_tool(x: int) -> int:
        """My custom tool."""
        return x * 2
    
    await gantry.sync()
    
    # Start MCP server in dynamic mode
    await gantry.serve_mcp(transport="stdio", mode="dynamic")

if __name__ == "__main__":
    asyncio.run(main())
```

2. Add to your Claude Desktop configuration (`~/Library/Application Support/Claude/claude_desktop_config.json` on macOS):

```json
{
  "mcpServers": {
    "agent-gantry": {
      "command": "python",
      "args": ["/path/to/my_agent_gantry_server.py"]
    }
  }
}
```

3. Restart Claude Desktop

4. Claude will now be able to discover and use your tools dynamically through the `find_relevant_tools` and `execute_tool` meta-tools!
