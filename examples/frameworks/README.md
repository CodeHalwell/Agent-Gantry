# Universal framework adapters — examples & verification

These examples exercise the native tool adapters
(`agent_gantry.integrations.frameworks`) and the multi-turn `ToolRefresher`.
They all run **offline** — no API keys, no LLM, and no third-party agent
framework required (uninstalled frameworks are skipped cleanly).

| File | What it does |
|---|---|
| `verify_all.py` | **Verification harness.** Asserts the P0 fixes, the universal `GantryToolset`/`ToolSpec` core, every `<Framework>Adapter` (built or cleanly skipped), and the multi-turn `ToolRefresher` pivot. Exits non-zero on any core failure. |
| `universal_adapters_example.py` | Readable walkthrough: select once with `GantryToolset`, invoke a `ToolSpec`, then export the selection to every framework's native tool object. |
| `multi_turn_refresher_example.py` | Direction-changing tool selection within one run via `ToolRefresher`. |

## Run

```bash
python examples/frameworks/verify_all.py                 # asserts everything; exit 0 = OK
python examples/frameworks/universal_adapters_example.py
python examples/frameworks/multi_turn_refresher_example.py
```

`verify_all.py` uses the strongest embedder available (SentenceTransformers if
installed) and falls back to the offline `SimpleEmbedder`. Install a framework
(e.g. `pip install langchain-core`) and re-run to see its adapter switch from
`SKIP` to `OK`.

## How this maps to the integrations

- **Select** → `GantryToolset(gantry).select(query, limit=3)` returns ranked
  `ToolSpec`s. Each `ToolSpec` is invocable directly (`ainvoke`/`invoke`) and
  routes through `gantry.execute`.
- **Export** → `LangChainAdapter(gantry).select(query, limit=3)` (and
  `LlamaIndexAdapter` / `CrewAIAdapter` / `PydanticAIAdapter` / `OpenAIAgentsAdapter` /
  `SmolagentsAdapter` / `HaystackAdapter` / `AgnoAdapter` / `AutoGenAdapter` /
  `LangGraphAdapter`) build that framework's native tool objects; use
  `<Adapter>.convert(spec)` for a single `ToolSpec`.
- **Multi-turn** → `ToolRefresher(gantry).refresh(messages)` re-selects fresh
  each turn so the agent can pivot to a different tool as the task changes.

The framework-specific examples that use real LLMs and the older manual
wrapping pattern live in [`../agent_frameworks/`](../agent_frameworks/).
