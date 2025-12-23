# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Initial public release preparation
- Comprehensive test suite with 69% coverage
- Production-ready linting and type checking setup

## [0.1.0] - 2025-12-23

### Added
- Core foundation with semantic routing and tool orchestration
- Multi-protocol support (OpenAI, Anthropic, Google GenAI, Vertex AI, Mistral, Groq)
- Vector store adapters (In-Memory, Qdrant, Chroma)
- Embedder adapters (Sentence Transformers, OpenAI)
- Reranker support (Cohere, Cross-Encoder)
- Execution engine with retries, timeouts, and circuit breakers
- Zero-trust security with capability-based permissions and policies
- MCP (Model Context Protocol) client and server support
- A2A (Agent-to-Agent) protocol implementation
- Health tracking and observability
- OpenTelemetry integration
- CLI interface for tool management
- Comprehensive documentation and examples

### Features
- **Semantic Routing**: Intelligent tool selection using vector similarity
- **Context Window Optimization**: Reduce token usage by ~90%
- **Circuit Breakers**: Automatic failure detection and recovery
- **Argument Validation**: Defensive validation against tool schemas
- **Async-Native**: Full async support for tools and execution
- **Schema Transcoding**: Automatic conversion between tool formats
- **Intent Classification**: Enhanced routing with intent matching
- **MMR Diversity**: Maximal Marginal Relevance for diverse tool selection

### Documentation
- Comprehensive README with quick start guide
- MCP integration examples
- A2A integration examples
- Phase documentation (Phase 2-6)
- LLM SDK compatibility guide
- Architecture diagrams

[Unreleased]: https://github.com/CodeHalwell/Agent-Gantry/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/CodeHalwell/Agent-Gantry/releases/tag/v0.1.0
