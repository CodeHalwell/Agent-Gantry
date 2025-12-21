---
name: "🚀 Frictionless Onboarding (Importers)"
about: Feature requests for automatic tool import from various sources (OpenAPI, databases, etc.)
title: "[Importer] "
labels: ["enhancement", "importer", "dx", "onboarding"]
assignees: []
---

## Feature Area
<!-- Check the area(s) this feature request relates to -->
- [ ] OpenAPI / Swagger Ingestion
- [ ] Database Introspection
- [ ] Other Tool Source Integration

## Problem Statement
<!-- Describe the current onboarding friction this feature would eliminate -->

**Current Behavior:**
<!-- How do users currently register tools from this source? -->

**Desired Behavior:**
<!-- What one-line import experience should exist? -->

## Proposed Solution

### Overview
<!-- High-level description of the importer -->

### Import Source
<!-- Describe the source system, format, or protocol -->

### Technical Details
<!-- Technical implementation details, if applicable -->

### Example Usage
<!-- Provide a code example showing the one-line import -->

```python
# Example usage
from agent_gantry import AgentGantry

gantry = AgentGantry()

# One-line import example
```

## Use Case
<!-- Describe a real-world scenario where this importer would be valuable -->

**Example Scenario:**


**Expected Benefit:**


## Context: Current Tool Registration

Currently, Agent-Gantry supports tool registration from:
- ✅ Python functions (via `@gantry.register` decorator)
- ✅ MCP servers (via `add_mcp_server`)
- ✅ A2A agents (via `add_a2a_agent`)

This feature request aims to expand "one-line import" capabilities for additional sources.

### OpenAPI / Swagger Ingestion
- **Goal**: One-line import for existing REST APIs
- **Example**: `gantry.ingest_openapi("https://api.stripe.com/v1/spec.json")`
- **Benefit**: Auto-generate tool definitions and schemas for every endpoint
- **Output**: Creates tools like `stripe_create_charge`, `stripe_list_customers`, etc.

### Database Introspection
- **Goal**: Auto-generate tools for SQL queries based on schema
- **Example**: `gantry.ingest_postgres(connection_string, tables=["users", "orders"])`
- **Benefit**: Creates CRUD tools like `search_users`, `get_user_by_id`, `update_order_status`
- **Output**: Type-safe tools based on database schema

### Other Integration Ideas
- GraphQL schema introspection
- gRPC service definitions
- Kubernetes API resources
- Cloud provider APIs (AWS, GCP, Azure)
- SaaS APIs (Slack, GitHub, Jira, etc.)

## Schema Mapping
<!-- Describe how the source schema would map to Agent-Gantry tool definitions -->

**Source Schema:**
```
<!-- Example of source format -->
```

**Generated Tool:**
```python
# Example of generated AgentGantry tool
```

## Authentication & Configuration
<!-- How would authentication and configuration be handled? -->

## Additional Context
<!-- Add any other context, API documentation links, or references -->

## Related Issues
<!-- Link to related issues or PRs -->

---
**Note**: This feature aligns with the strategic goal of reducing friction in tool onboarding and accelerating time-to-value for Agent-Gantry users.
