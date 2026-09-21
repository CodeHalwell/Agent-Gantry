# examples/basics

Introductory examples that show how to register tools, sync embeddings, retrieve them by meaning, and execute them.

## Files
- `tool_demo.py`: Small "hello world" walkthrough that registers a single tool, retrieves it, and executes it.
- `multi_tool_demo.py`: Registers ten tools across unrelated domains to illustrate semantic routing, then executes one.
- `async_demo.py`: Demonstrates native async tool execution (an `async def` tool awaited by the executor).
- `tool_creation_patterns.py`: Five registration patterns side by side — decorator, direct call, async function, bound method, and renaming on registration.
- `plug_and_play_semantic_filter.py`: Loads tools from `toolpack.py` and injects only the relevant ones into an existing (mocked) LLM call with a single decorator. Run from the repo root.
- `toolpack.py`: Reusable tool catalogue exported as `tools` for plug-and-play imports; not meant to be run directly.
- `skills_example.py`: Registers two skills (procedural guidance, never executed), retrieves them by meaning, and formats them for system-prompt injection.

## Run commands

```bash
python examples/basics/tool_demo.py
python examples/basics/multi_tool_demo.py
python examples/basics/async_demo.py
python examples/basics/tool_creation_patterns.py
python examples/basics/plug_and_play_semantic_filter.py
python examples/basics/skills_example.py
```

Every script runs offline with no credentials. `AgentGantry()` uses the in-memory vector store and picks its embedder from what is installed: the local sentence-transformers model with `agent-gantry[embeddings]`, otherwise the hash-based `SimpleEmbedder`. Use these as starting points when wiring Agent-Gantry into your own agents.
