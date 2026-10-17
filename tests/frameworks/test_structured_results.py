"""Text-based framework tools retain structured Python return values as JSON."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal
from enum import Enum
from uuid import UUID

import pytest
from pydantic import BaseModel

from agent_gantry.integrations.agent_framework_bridge import GantryToolBridge


class Payload(BaseModel):
    value: int


@dataclass
class Record:
    value: int


class Mode(Enum):
    READ = "read"


@pytest.mark.parametrize("framework", ["bridge", "openai_agents"])
@pytest.mark.parametrize(
    "value,expected",
    [
        (Payload(value=3), {"value": 3}),
        (Record(value=4), {"value": 4}),
        (
            {"when": datetime(2026, 10, 16, tzinfo=timezone.utc)},
            {"when": "2026-10-16T00:00:00+00:00"},
        ),
        ({"amount": Decimal("1.234567890123456789")}, {"amount": "1.234567890123456789"}),
        ({"id": UUID(int=0)}, {"id": "00000000-0000-0000-0000-000000000000"}),
        ({Mode.READ: "ok", (1, 2): "pair"}, {"Mode.READ": "ok", "(1, 2)": "pair"}),
        ({"values": {1}}, {"values": [1]}),
        ({"body": b"hello"}, {"body": "hello"}),
    ],
)
async def test_structured_result_stays_machine_readable(gantry, framework, value, expected):
    if framework == "openai_agents":
        pytest.importorskip("agents")

    @gantry.register
    def read_record():
        """Read a record with structured Python values."""
        return value

    if framework == "bridge":
        tools = await GantryToolBridge(gantry, as_function_tool=False).get_tools(
            "read a record", limit=1
        )
        result = await tools[0]()
    else:
        tools = await gantry.framework("openai_agents").select("read a record", limit=1)
        result = await tools[0].on_invoke_tool(None, "{}")
    assert json.loads(result) == expected
