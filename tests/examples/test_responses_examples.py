"""The larger demos must send valid structured tool outputs back to Responses."""

from __future__ import annotations

import importlib.util
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import AsyncMock

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize(
    ("script", "catalogue"),
    [
        ("project_demo/main.py", "examples.project_demo.tools.tools"),
        ("project_demo/main_persistent.py", "examples.project_demo.tools.tools_persistent"),
        ("tool_vector_db/main.py", "examples.tool_vector_db.tools"),
    ],
)
async def test_responses_round_trip(script, catalogue, gantry, monkeypatch, capsys):
    pytest.importorskip("openai")
    pytest.importorskip("dotenv")
    monkeypatch.setenv("PYTHON_DOTENV_DISABLED", "1")
    monkeypatch.setenv("OPENAI_API_KEY", "offline-example-test")

    @gantry.register(examples=["what time is it"])
    def lookup_time() -> dict:
        """Return a structured value that plain json.dumps cannot serialize."""
        return {"at": datetime(2026, 10, 16, tzinfo=timezone.utc)}

    # Supply a tiny catalogue without importing the demo's heavy tool dependencies
    # or touching its persistent database. Load the script under a private name.
    tools_module = ModuleType(catalogue)
    tools_module.tools = gantry
    monkeypatch.setitem(sys.modules, catalogue, tools_module)
    spec = importlib.util.spec_from_file_location(
        "_gantry_responses_example", ROOT / "examples" / script
    )
    assert spec and spec.loader
    example = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(example)

    first = SimpleNamespace(
        id="response_1",
        output=[
            SimpleNamespace(
                type="function_call",
                name="lookup_time",
                arguments="{}",
                call_id="call_clock",
            )
        ],
    )
    final = SimpleNamespace(
        output_text="The clock was checked.",
        output=[
            SimpleNamespace(
                type="message",
                content=[SimpleNamespace(text="The clock was checked.")],
            )
        ],
    )
    create = AsyncMock(side_effect=[first, final])
    client = SimpleNamespace(responses=SimpleNamespace(create=create))
    if script.startswith("project_demo"):
        monkeypatch.setattr(example, "client", lambda: client)
        text, calls, outputs = await example.generate_response("what time is it")
        assert text == "The clock was checked."
        assert len(calls) == len(outputs) == 1
    else:
        monkeypatch.setattr(example, "AsyncOpenAI", lambda: client)
        await example.main()
        assert "The clock was checked." in capsys.readouterr().out

    assert create.await_count == 2
    follow_up = create.await_args_list[1].kwargs
    assert follow_up["previous_response_id"] == "response_1"
    [output] = follow_up["input"]
    assert output["type"] == "function_call_output"
    assert output["call_id"] == "call_clock"
    assert json.loads(output["output"]) == {"at": "2026-10-16T00:00:00+00:00"}
