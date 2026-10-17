# Universal framework adapters — examples & verification

These examples exercise the native tool adapters
(`agent_gantry.integrations.frameworks`) and the multi-turn `ToolRefresher`.
No API key or LLM is required. Uninstalled agent frameworks are skipped cleanly.
The universal adapter example uses a local hash embedder. The verification and
multi-turn demos use SentenceTransformers when installed, which may download its
model on first run; otherwise they use the hash embedder.

| File | What it does |
|---|---|
| `verify_all.py` | **Verification harness.** Asserts the P0 fixes, the universal `GantryToolset`/`ToolSpec` core, every `<Framework>Adapter` (built or cleanly skipped), and the multi-turn `ToolRefresher` pivot. Exits non-zero on any core failure. |
| `universal_adapters_example.py` | Readable walkthrough: select once with `GantryToolset`, invoke a `ToolSpec`, then export the selection to every framework's native tool object. |
| `multi_turn_refresher_example.py` | Direction-changing tool selection within one run via `ToolRefresher`. |
| `importers_example.py` | **Reverse direction.** Registers existing LangChain/CrewAI/LlamaIndex native tool objects *into* Gantry (`agent_gantry.integrations.importers`), then retrieves and executes them through the normal `gantry.execute()` path — the "already built tools with a framework, want them in Gantry" half of "register once, use anywhere." |

## Run

```bash
python examples/frameworks/verify_all.py                 # asserts everything; exit 0 = OK
python examples/frameworks/universal_adapters_example.py
python examples/frameworks/multi_turn_refresher_example.py
python examples/frameworks/importers_example.py
```

`verify_all.py` uses the strongest embedder available (SentenceTransformers if
installed) and falls back to the offline `SimpleEmbedder`. Install a framework
(e.g. `pip install langchain-core`) and re-run to see its adapter switch from
`SKIP` to `OK`.

## How this maps to the integrations

- **Select** → `GantryToolset(gantry).select(query, limit=3)` returns ranked
  `ToolSpec`s. Each `ToolSpec` is invocable directly (`ainvoke`/`invoke`) and
  routes through `gantry.execute`.
- **Export** → `gantry.framework("langchain").select(query, limit=3)` builds
  native tools. All twelve names from `supported_frameworks()` share this API,
  including Microsoft Agent Framework. Use `gantry.framework(name).convert(spec)`
  for one `ToolSpec`; `spec.native_name` is the safe framework-facing name.
- **Multi-turn** → `ToolRefresher(gantry).refresh(messages)` re-selects fresh
  each turn so the agent can pivot to a different tool as the task changes.
- **Import (reverse)** → `register_langchain_tools(gantry, tools)` /
  `register_crewai_tools` / `register_llamaindex_tools`
  (`agent_gantry.integrations.importers`) convert already-built native tool
  objects into `ToolDefinition`s with an execution handler wired through
  `gantry.add_tool(tool, handler=...)`, so they run through `gantry.execute`
  exactly like a `@gantry.register`-ed tool — and can then be **exported**
  again via any adapter above, to a *different* framework than the one they
  came from.

Framework-specific static and live integration examples are in [`../agent_frameworks/`](../agent_frameworks/).
