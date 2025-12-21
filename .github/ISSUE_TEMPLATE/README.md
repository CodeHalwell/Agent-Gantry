# GitHub Issue Templates

This directory contains issue templates for Agent-Gantry. These templates help contributors submit well-structured feature requests and bug reports aligned with the project's strategic roadmap.

## Available Templates

### Strategic Feature Request Templates

These templates correspond to Agent-Gantry's strategic expansion areas:

1. **🎯 Advanced Routing & Orchestration** (`01-advanced-routing-orchestration.md`)
   - Hierarchical / Domain Routing (Router of Routers)
   - Tool Chaining / Pipelines (Atomic Workflows)
   - Parameter-Aware Routing (Contextual Feasibility)

2. **🔒 Enterprise Security & Governance** (`02-enterprise-security-governance.md`)
   - Human-in-the-Loop (HITL) Protocol
   - PII Redaction / Data Loss Prevention (DLP)
   - Rate Limiting & Quotas

3. **🚀 Frictionless Onboarding (Importers)** (`03-frictionless-onboarding.md`)
   - OpenAPI / Swagger Ingestion
   - Database Introspection
   - Other Tool Source Integrations

4. **🎨 Developer Experience & UI** (`04-developer-experience-ui.md`)
   - Gantry Dashboard (Web UI)
   - Simulation & Replay (Snapshot Testing)
   - Visualization and Debugging Tools

5. **🧠 Smart Learning** (`05-smart-learning.md`)
   - Feedback Loop (RLHF for Tools)
   - Adaptive Embeddings
   - Tool Performance Learning

6. **💾 State Management** (`06-state-management.md`)
   - Session Memory / Context Store
   - Cross-Tool State Sharing
   - Persistent State Storage

### General Templates

7. **🐛 Bug Report** (`bug_report.md`)
   - Standard template for reporting bugs and unexpected behavior

8. **✨ General Feature Request** (`general_feature_request.md`)
   - For feature requests that don't fit into strategic categories above

## How to Use

When creating a new issue on GitHub, you'll be presented with a list of templates to choose from. Select the template that best matches your request:

1. **For strategic features**: Choose the appropriate strategic area template
2. **For bugs**: Use the Bug Report template
3. **For other features**: Use the General Feature Request template

## Template Structure

Each strategic template includes:

- **Feature Area**: Checkboxes for specific sub-areas
- **Problem Statement**: Current vs. desired behavior
- **Proposed Solution**: Technical details and examples
- **Use Case**: Real-world scenarios and expected benefits
- **Context**: How this relates to current capabilities
- **Additional Context**: For supporting information

## Configuration

The `config.yml` file configures the issue template chooser:

- Disables blank issues (encourages using templates)
- Provides links to documentation and discussions

## Contributing

When submitting an issue:

1. Choose the most appropriate template
2. Fill out all required sections
3. Provide code examples where applicable
4. Link to related issues if they exist
5. Be specific about use cases and expected benefits

## Strategic Roadmap Alignment

These templates align with Agent-Gantry's vision to become a comprehensive orchestration platform. Each template corresponds to a strategic expansion area designed to:

- Scale to enterprise-level tool inventories (10,000+ tools)
- Provide production-grade security and compliance
- Reduce friction in tool onboarding
- Enhance developer experience
- Enable continuous learning and improvement
- Support stateful agent workflows

For more information, see the project's [roadmap](../../plan.md) and [README](../../README.md).
