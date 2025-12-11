"""
Pytest configuration and fixtures for Agent-Gantry tests.
"""

from __future__ import annotations

import pytest

from agent_gantry import AgentGantry
from agent_gantry.schema.tool import ToolDefinition


@pytest.fixture
def gantry() -> AgentGantry:
    """Create a fresh AgentGantry instance for testing."""
    return AgentGantry()


@pytest.fixture
def sample_tools() -> list[ToolDefinition]:
    """Create sample tool definitions for testing."""
    return [
        ToolDefinition(
            name="query_database",
            description="Query the database to retrieve information based on SQL-like queries.",
            parameters_schema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The query to execute"},
                    "limit": {"type": "integer", "description": "Maximum results to return"},
                },
                "required": ["query"],
            },
            tags=["database", "query", "read"],
        ),
        ToolDefinition(
            name="send_email",
            description="Send an email to specified recipients with a subject and body.",
            parameters_schema={
                "type": "object",
                "properties": {
                    "to": {"type": "string", "description": "Email recipient"},
                    "subject": {"type": "string", "description": "Email subject"},
                    "body": {"type": "string", "description": "Email body"},
                },
                "required": ["to", "subject", "body"],
            },
            tags=["email", "communication", "send"],
        ),
        ToolDefinition(
            name="create_user",
            description="Create a new user account in the system with the specified details.",
            parameters_schema={
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "User's full name"},
                    "email": {"type": "string", "description": "User's email address"},
                    "role": {"type": "string", "description": "User's role"},
                },
                "required": ["name", "email"],
            },
            tags=["user", "create", "admin"],
        ),
        ToolDefinition(
            name="process_refund",
            description="Process a refund for a customer order or transaction.",
            parameters_schema={
                "type": "object",
                "properties": {
                    "order_id": {"type": "string", "description": "Order ID to refund"},
                    "amount": {"type": "number", "description": "Amount to refund"},
                    "reason": {"type": "string", "description": "Reason for refund"},
                },
                "required": ["order_id", "amount"],
            },
            tags=["refund", "financial", "customer"],
            requires_confirmation=True,
        ),
        ToolDefinition(
            name="generate_report",
            description="Generate a report based on specified parameters and date range.",
            parameters_schema={
                "type": "object",
                "properties": {
                    "report_type": {"type": "string", "description": "Type of report"},
                    "start_date": {"type": "string", "description": "Start date"},
                    "end_date": {"type": "string", "description": "End date"},
                },
                "required": ["report_type"],
            },
            tags=["report", "analysis", "generate"],
        ),
    ]
