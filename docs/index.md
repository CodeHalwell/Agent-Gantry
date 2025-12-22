# Agent-Gantry documentation home

Start here for an overview of the available guides and how to navigate the docs.

- **Quick start**: See `README.md` at the repository root for installation and basic usage.
- **Configuration reference**: `configuration.md` describes all config knobs and ready-made YAML.
- **Module-based loading**: `configuration.md` also shows how to merge tools from multiple modules.
- **Local persistence + skills**: `local_persistence_and_skills.md` shows LanceDB/Nomic setup and skill storage.
- **Vector store + LLM wiring**: `vector_store_llm_integration.md` shows how to register many tools,
  build a vector index, and present tools to OpenAI, Google GenAI, and Claude SDKs.
- **CLI**: `cli.md` documents the bundled `agent-gantry` CLI commands.
- **Roadmap phases**: `phase2.md`, `phase5_mcp.md`, and `phase6_a2a.md` track planned milestones.
- **SDK compatibility**: `llm_sdk_compatibility.md` summarizes what each provider expects.
- **Semantic tool decorator**: `semantic_tool_decorator.md` covers automatic tool selection for LLM calls.

All docs are Markdown files to make it easy to copy code into your projects. Contributions are
welcome—add new guides in this folder and link them from here.
