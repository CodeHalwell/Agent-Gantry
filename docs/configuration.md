# Configuration reference

Agent-Gantry is configured through the `AgentGantryConfig` model in `agent_gantry.schema.config`
and can be loaded directly from YAML via `AgentGantry.from_config("config.yaml")`. This guide
summarizes the available knobs and shows ready-to-use snippets for common deployments.

## Loading from YAML

```python
from agent_gantry import AgentGantry

gantry = AgentGantry.from_config("config.yaml")
await gantry.sync()
```

```yaml
# config.yaml
vector_store:
  type: memory
embedder:
  type: sentence_transformers  # Uses lightweight simple embedder stub by default
routing:
  weights:
    semantic: 0.6
    intent: 0.15
    conversation: 0.1
    health: 0.1
    cost: 0.05
execution:
  default_timeout_ms: 30000
telemetry:
  type: console
```

## Module-based tool loading

When your tools live in multiple files, you can import and deduplicate their registries without
sharing vector stores:

- Pass `modules=[...]` into `AgentGantry(...)` to defer loading until the first `sync()`
- Use `AgentGantry.from_modules([...], attr="tools")` to build and populate a new gantry in one call
- Call `collect_tools_from_modules([...], module_attr="tools")` to merge into an existing instance;
  pending unsynced tools are imported and duplicates are skipped with a warning

```python
# tools/payments.py
from agent_gantry import AgentGantry

tools = AgentGantry()

@tools.register
def refund(order_id: str, amount: float) -> str:
    """Issue a refund."""
    return f"Refunded {amount} for {order_id}"

# bootstrap.py
import asyncio
from agent_gantry import AgentGantry

async def main() -> None:
    gantry = await AgentGantry.from_modules(["tools.payments"], attr="tools")
    await gantry.sync()
    print(await gantry.retrieve_tools("refund an order"))

asyncio.run(main())
```

## Core components

### Vector stores

| Type | Notes |
| ---- | ----- |
| `memory` | Default in-memory store (no persistence) |
| `lancedb` | Local, on-device persistence (requires `pip install agent-gantry[lancedb]`) |
| `qdrant`, `chroma`, `pgvector`, `pinecone`, `weaviate` | Stub adapters that proxy to the in-memory store and surface health as unhealthy if no endpoint is provided |

LanceDB supports `db_path`, `dimension`, and separate `tools` / `skills` tables. Remote store
configs accept `url`, `api_key`, and `collection_name`.

### Embedders

| Type | Notes |
| ---- | ----- |
| `openai` / `azure` | Uses OpenAI or Azure OpenAI embeddings (needs API key) |
| `nomic` | Local Matryoshka embeddings via `nomic-embed-text-v1.5` (requires `pip install agent-gantry[nomic]`) |
| `sentence_transformers`, `huggingface` (Hugging Face), `cohere`, `ollama` | Currently fall back to the simple embedder stub; keep `type` as-is for forward compatibility |

`EmbedderConfig` supports `model`, `dimension` (for Matryoshka truncation), and `task_type`
(`search_document`, `search_query`, `clustering`, `classification` for Nomic).

### Reranker

`RerankerConfig` toggles second-stage reranking. The Cohere reranker is implemented; other types
fall back to `None`.

```yaml
reranker:
  enabled: true
  type: cohere
  model: rerank-english-v3.0
```

### Routing

`RoutingConfig` controls the semantic router:

- `weights`: blend semantic, intent, conversation, health, and cost signals
- `enable_intent_classification`, `use_llm_for_intent`
- `enable_mmr`, `mmr_lambda`

### Execution

`ExecutionConfig` configures retries, timeouts, and circuit breaker thresholds:

- `default_timeout_ms`
- `max_retries`
- `circuit_breaker_threshold`
- `circuit_breaker_timeout_s`
- `sandbox_type` (`none`, `subprocess`, `docker`) – sandboxing not yet wired in the default executor

### Telemetry

`TelemetryConfig` chooses the observability backend:

- `enabled`: set `false` for `NoopTelemetryAdapter`
- `type`: `console` (default), `opentelemetry`, or `prometheus` (`PrometheusTelemetryAdapter`)
- `service_name`, `otlp_endpoint`, `prometheus_port`

`datadog` is accepted by the schema but currently falls back to console logging.

### MCP / A2A

- `mcp.servers`: list of `MCPServerConfig` entries (name, command, args, env, namespace)
- `mcp.serve_mcp`: enable serving as an MCP server (`mode`: `dynamic`, `static`, `hybrid`)
- `a2a.agents`: list of `A2AAgentConfig` entries (name, url, namespace)
- `a2a.serve_a2a`: expose the FastAPI-based A2A server on `a2a_port`

## Ready-made configuration snippets

### Local-first: LanceDB + Nomic + Prometheus metrics

```yaml
vector_store:
  type: lancedb
  db_path: .agent_gantry/lancedb
  dimension: 256
embedder:
  type: nomic
  model: nomic-ai/nomic-embed-text-v1.5
  dimension: 256
  task_type: search_document
telemetry:
  type: prometheus
  service_name: agent_gantry_local
  prometheus_port: 9090
reranker:
  enabled: false
execution:
  default_timeout_ms: 30000
```

Install extras:

```bash
pip install agent-gantry[lancedb,nomic]
```

### Remote-ready: Qdrant + OpenAI embeddings + Cohere reranker

```yaml
vector_store:
  type: qdrant
  url: https://qdrant.example.com
  api_key: ${QDRANT_API_KEY}
  collection_name: agent_gantry
embedder:
  type: openai
  api_key: ${OPENAI_API_KEY}
  model: text-embedding-3-small
reranker:
  enabled: true
  type: cohere
  model: rerank-english-v3.0
telemetry:
  type: opentelemetry
  service_name: agent_gantry_prod
  otlp_endpoint: https://otel.example.com
mcp:
  serve_mcp: true
  mcp_mode: dynamic
```

Install extras:

```bash
pip install agent-gantry[vector-stores,openai]
pip install cohere  # reranker model
```

## Tuning sync behavior

- `auto_sync`: automatically embed and upsert pending tools on retrieval / execution
- `sync_on_register`: immediately embed and upsert each tool when registered

For large catalogs, disable `auto_sync` and call `await gantry.sync()` after batch registration.

## Health checks

`AgentGantry.health_check()` reports the status of the vector store, embedder, and telemetry
adapter. Remote adapters without endpoints will report unhealthy, which is expected for stubbed
vector stores until real endpoints are configured.
