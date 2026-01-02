---
layout: default
title: Best Practices
parent: Architecture
nav_order: 2
description: "Production deployment patterns and best practices for Agent-Gantry"
---

# Best Practices

Learn how to build production-ready agent systems with Agent-Gantry.

---

## Tool Design

### Write Clear Descriptions

Tool descriptions are embedded and used for semantic search. Make them descriptive and include key terms:

<div class="callout danger">
<div class="callout-title">❌ Bad Example</div>

```python
@gantry.register
def get(city: str) -> str:
    """Get something."""
    return weather_api.get(city)
```

**Problems:** Vague description, no keywords, unclear purpose.

</div>

<div class="callout tip">
<div class="callout-title">✅ Good Example</div>

```python
@gantry.register(
    tags=["weather", "forecast", "meteorology"],
    description="Retrieve current weather conditions and forecast for any city worldwide"
)
def get_weather(city: str, units: str = "fahrenheit") -> str:
    """
    Get current weather and 5-day forecast for a city.

    Provides temperature, humidity, wind speed, and precipitation data
    from the OpenWeatherMap API.

    Args:
        city: City name (e.g., "San Francisco" or "Paris, France")
        units: Temperature units - "fahrenheit" or "celsius"

    Returns:
        Weather summary with current conditions and forecast
    """
    return weather_api.get(city, units=units)
```

**Why this is better:**
- Clear, specific description
- Includes search keywords (weather, forecast, conditions)
- Detailed docstring for LLM understanding
- Parameter validation and defaults

</div>

### Use Descriptive Tags

Tags improve search accuracy and enable filtering:

```python
@gantry.register(
    tags=["database", "sql", "query", "data", "postgres"],
    description="Execute SQL queries against PostgreSQL database"
)
def query_database(sql: str) -> list[dict]:
    """Execute SQL query and return results."""
    ...
```

**Tag Best Practices:**
- Use 3-6 tags per tool
- Include domain terms (e.g., "database", "sql")
- Add common synonyms (e.g., "postgres", "postgresql")
- Use lowercase for consistency

### Define JSON Schemas for Parameters

Explicit schemas enable better validation and LLM understanding:

```python
from pydantic import BaseModel, Field

class WeatherParams(BaseModel):
    city: str = Field(description="City name or coordinates")
    units: str = Field(default="fahrenheit", pattern="^(fahrenheit|celsius)$")
    include_forecast: bool = Field(default=False, description="Include 5-day forecast")

@gantry.register
def get_weather(params: WeatherParams) -> str:
    """Get weather with validated parameters."""
    ...
```

## Performance Optimization

### Sync Once, Query Many

Tool syncing is expensive. Do it once at startup:

```python
# ✅ Good: Sync once at startup
async def main():
    gantry = AgentGantry()

    # Register all tools
    from myapp.tools import weather, math, database

    # Sync once after all registrations
    await gantry.sync()

    # Now ready for queries
    set_default_gantry(gantry)
```

**Avoid syncing in request handlers** - it re-embeds all tools.

### Use Fingerprinting for Change Detection

Agent-Gantry automatically detects unchanged tools:

```python
# First sync: embeds all 100 tools
await gantry.sync()

# Later: only re-embeds changed tools
await gantry.sync()  # Skips unchanged tools automatically
```

Force re-sync with `force=True`:

```python
await gantry.sync(force=True)  # Re-embed everything
```

### Batch Tool Registration

Register tools in batches for better performance:

```python
# Instead of:
for tool_def in tool_definitions:
    gantry._registry.register(tool_def)
    await gantry.sync()  # ❌ Syncs 100 times

# Do this:
for tool_def in tool_definitions:
    gantry._registry.register(tool_def)

await gantry.sync()  # ✅ Syncs once
```

### Choose the Right Vector Store

| Use Case | Recommended Vector Store |
|----------|-------------------------|
| Development/Testing | InMemoryVectorStore |
| Small deployments (<10k tools) | LanceDBVectorStore |
| Large deployments (>10k tools) | QdrantVectorStore |
| Existing PostgreSQL | PGVectorStore |

### Optimize Embedder Selection

| Embedder | Best For |
|----------|----------|
| SimpleEmbedder | Quick testing, CI/CD |
| NomicEmbedder | Production (best quality, local) |
| OpenAIEmbedder | Maximum accuracy, budget allows |
| SentenceTransformers | Self-hosted, GPU available |

**Recommendation:** Start with NomicEmbedder (768D, local, fast).

## Security Best Practices

### Validate All Tool Inputs

Never trust external input:

```python
from pydantic import BaseModel, Field, validator

class FileParams(BaseModel):
    path: str = Field(max_length=255)

    @validator('path')
    def validate_path(cls, v):
        if '..' in v or v.startswith('/'):
            raise ValueError("Invalid path")
        return v

@gantry.register
def read_file(params: FileParams) -> str:
    """Read file with validated path."""
    ...
```

### Use Capabilities for Permission Control

Declare required capabilities:

```python
@gantry.register(
    capabilities=["file_write", "network_access"],
    description="Download file from URL"
)
def download_file(url: str, destination: str) -> bool:
    """Download file (requires file_write and network_access)."""
    ...
```

Check capabilities in production:

```python
config = AgentGantryConfig(
    security=SecurityConfig(
        enforce_capabilities=True,
        allowed_capabilities=["file_read", "file_write"]
    )
)
```

### Never Execute Arbitrary Code

<div class="callout danger">
<div class="callout-title">⚠️ Security Risk</div>

```python
# ❌ DANGEROUS - Never do this
@gantry.register
def run_code(code: str) -> Any:
    return eval(code)  # Arbitrary code execution!
```

</div>

Use safe alternatives:

```python
import ast
import operator

# ✅ Safe math evaluation
@gantry.register
def calculate(expression: str) -> float:
    """Safely evaluate math expressions."""
    allowed_ops = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
    }

    def eval_expr(node):
        if isinstance(node, ast.Num):
            return node.n
        elif isinstance(node, ast.BinOp):
            return allowed_ops[type(node.op)](
                eval_expr(node.left),
                eval_expr(node.right)
            )
        raise ValueError("Unsupported operation")

    return eval_expr(ast.parse(expression, mode='eval').body)
```

### Use Environment Variables for Secrets

Never hardcode API keys:

```python
import os

# ✅ Good
api_key = os.getenv("WEATHER_API_KEY")

# ❌ Bad
api_key = "abc123..."  # Hardcoded secret
```

## Error Handling

### Implement Retries with Exponential Backoff

```python
from tenacity import retry, stop_after_attempt, wait_exponential

@gantry.register
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10)
)
async def call_external_api(query: str) -> dict:
    """Call external API with retries."""
    async with httpx.AsyncClient() as client:
        response = await client.get(f"https://api.example.com?q={query}")
        response.raise_for_status()
        return response.json()
```

### Use Circuit Breakers

Agent-Gantry includes built-in circuit breakers. Configure them:

```python
from agent_gantry.schema.config import ExecutorConfig, CircuitBreakerConfig

config = AgentGantryConfig(
    executor=ExecutorConfig(
        circuit_breaker=CircuitBreakerConfig(
            failure_threshold=5,      # Open after 5 failures
            success_threshold=2,       # Close after 2 successes
            timeout_seconds=60.0       # Wait 60s before half-open
        )
    )
)

gantry = AgentGantry(config=config)
```

### Log Errors with Context

```python
import logging

logger = logging.getLogger(__name__)

@gantry.register
async def fetch_data(source: str) -> dict:
    """Fetch data with error logging."""
    try:
        result = await data_source.fetch(source)
        return result
    except ConnectionError as e:
        logger.error(f"Connection failed to {source}: {e}", exc_info=True)
        raise
    except Exception as e:
        logger.error(f"Unexpected error fetching from {source}: {e}", exc_info=True)
        raise
```

## Testing Strategies

### Unit Test Individual Tools

```python
import pytest

@pytest.mark.asyncio
async def test_get_weather():
    result = get_weather(city="Paris", units="celsius")
    assert "Paris" in result
    assert "°C" in result
```

### Integration Test Semantic Routing

```python
@pytest.mark.asyncio
async def test_semantic_routing():
    gantry = AgentGantry()

    @gantry.register(tags=["weather"])
    def get_weather(city: str) -> str:
        return f"Weather in {city}"

    await gantry.sync()

    # Test semantic search
    tools = await gantry.retrieve_tools("what's the weather", limit=1)
    assert len(tools) == 1
    assert tools[0].name == "get_weather"
```

### Use Fixtures for Gantry Instances

```python
@pytest.fixture
async def gantry():
    g = AgentGantry()
    # Register test tools
    yield g
    # Cleanup if needed

@pytest.mark.asyncio
async def test_with_fixture(gantry):
    @gantry.register
    def test_tool() -> str:
        return "test"

    await gantry.sync()
    tools = await gantry.retrieve_tools("test", limit=1)
    assert len(tools) == 1
```

## Production Deployment

### Use Configuration Files

```yaml
# config.yaml
embedder:
  type: nomic
  dimension: 768

vector_store:
  type: lancedb
  db_path: /data/gantry_tools.lancedb
  collection_name: tools
  dimension: 768

executor:
  timeout: 30.0
  max_retries: 3
  circuit_breaker:
    failure_threshold: 5
    success_threshold: 2
    timeout_seconds: 60.0

telemetry:
  enabled: true
  log_level: INFO
  export_metrics: true
```

Load config:

```python
from agent_gantry import AgentGantry
from agent_gantry.schema.config import AgentGantryConfig
import yaml

with open("config.yaml") as f:
    config_dict = yaml.safe_load(f)

config = AgentGantryConfig(**config_dict)
gantry = AgentGantry(config=config)
```

### Enable Structured Logging

```python
import structlog

structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.JSONRenderer()
    ]
)

# Agent-Gantry will use structured logging automatically
```

### Monitor Health Metrics

```python
# Expose metrics via Prometheus
from prometheus_client import Counter, Histogram, start_http_server

tool_executions = Counter('gantry_tool_executions_total', 'Total tool executions', ['tool_name', 'status'])
tool_duration = Histogram('gantry_tool_duration_seconds', 'Tool execution duration', ['tool_name'])

# Start metrics server
start_http_server(9090)
```

### Use Health Checks

```python
from fastapi import FastAPI

app = FastAPI()

@app.get("/health")
async def health_check():
    # Check if gantry is initialized
    if not gantry._initialized:
        return {"status": "initializing"}, 503

    # Check vector store
    try:
        await gantry._vector_store.search([0.0] * 768, limit=1)
    except Exception:
        return {"status": "unhealthy", "reason": "vector_store_error"}, 503

    return {"status": "healthy"}
```

## LLM Integration Best Practices

### Handle Tool Call Loops

Prevent infinite loops when LLMs call tools repeatedly:

```python
MAX_ITERATIONS = 5

@with_semantic_tools(limit=3)
async def chat_with_tools(prompt: str, *, tools=None):
    messages = [{"role": "user", "content": prompt}]

    for i in range(MAX_ITERATIONS):
        response = await client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            tools=tools
        )

        # No tool calls - we're done
        if not response.choices[0].message.tool_calls:
            return response.choices[0].message.content

        # Execute tools
        for tool_call in response.choices[0].message.tool_calls:
            result = await gantry.execute(...)
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": str(result.output)
            })

    return "Maximum iterations reached"
```

### Validate LLM Tool Calls

LLMs sometimes generate invalid tool calls:

```python
import json

for tool_call in response.choices[0].message.tool_calls:
    try:
        # Parse arguments
        args = json.loads(tool_call.function.arguments)

        # Validate tool exists
        tool = gantry.get_tool(tool_call.function.name)
        if not tool:
            logger.warning(f"LLM called non-existent tool: {tool_call.function.name}")
            continue

        # Execute
        result = await gantry.execute(ToolCall(
            tool_name=tool_call.function.name,
            arguments=args
        ))
    except json.JSONDecodeError:
        logger.error(f"Invalid JSON from LLM: {tool_call.function.arguments}")
    except Exception as e:
        logger.error(f"Tool execution failed: {e}")
```

## Next Steps

- **[Troubleshooting]({{ '/troubleshooting' | relative_url }})** - Common issues and solutions
- **[API Reference]({{ '/reference/api-reference' | relative_url }})** - Detailed API documentation
- **[Examples](https://github.com/CodeHalwell/Agent-Gantry/tree/main/examples)** - Production-ready code examples

