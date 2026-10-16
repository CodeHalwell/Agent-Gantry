"""Exercise the documented no-credential paths with real installed SDKs."""

from __future__ import annotations

import ast
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize(
    ("example", "sdk"),
    [
        ("langchain", "langchain_core"),
        ("langgraph", "langchain_core"),
        ("llamaindex", "llama_index.core"),
        ("crewai", "crewai"),
        ("google_adk", "google.adk"),
        ("pydantic_ai", "pydantic_ai"),
        ("openai_agents", "agents"),
        ("haystack", "haystack"),
        ("agno", "agno"),
        ("strands", "strands"),
        ("dspy", "dspy"),
        ("generic_adapters", None),
        ("agent_framework", "agent_framework"),
        ("agent_framework_provider", "agent_framework"),
        ("agent_framework_orchestration", "agent_framework"),
        ("agent_framework_harness", "agent_framework"),
        ("agent_framework_trace_events", "agent_framework"),
    ],
)
def test_framework_example_without_credentials(example: str, sdk: str | None) -> None:
    if sdk:
        try:
            installed = importlib.util.find_spec(sdk) is not None
        except ModuleNotFoundError:
            installed = False
        if not installed:
            pytest.skip(f"Optional SDK {sdk} is not installed")
    env = {
        key: value
        for key, value in os.environ.items()
        if not any(
            word in key.upper()
            for word in (
                "KEY",
                "TOKEN",
                "SECRET",
                "PASSWORD",
                "AZURE_OPENAI",
                "AWS_",
                "GOOGLE_APPLICATION_CREDENTIALS",
            )
        )
    }
    env.update(
        PYTHON_DOTENV_DISABLED="1",
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        AWS_EC2_METADATA_DISABLED="true",
        CREWAI_DISABLE_TELEMETRY="true",
        OTEL_SDK_DISABLED="true",
    )
    result = subprocess.run(
        [sys.executable, "-m", f"examples.agent_frameworks.{example}_example"],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "tool" in result.stdout.lower(), result.stdout


def test_all_example_sources_support_python_310() -> None:
    for path in (ROOT / "examples").rglob("*.py"):
        ast.parse(path.read_text(encoding="utf-8"), filename=str(path), feature_version=(3, 10))


def test_token_count_falls_back_without_download(monkeypatch: pytest.MonkeyPatch) -> None:
    from examples.llm_integration import token_savings_demo as demo

    attempts = []

    def unavailable(model: str) -> None:
        attempts.append(model)
        raise OSError("Vocabulary not cached and network is unavailable")

    monkeypatch.setattr(demo, "TIKTOKEN_AVAILABLE", True)
    monkeypatch.setattr(
        demo, "tiktoken", SimpleNamespace(encoding_for_model=unavailable), raising=False
    )
    demo._tokenizer.cache_clear()
    tools = [{"type": "function", "function": {"name": "example"}}]
    try:
        expected = len(json.dumps(tools, separators=(",", ":"))) // 4
        assert demo.count_tokens(tools) == expected
        assert demo.count_tokens(tools) == expected
        assert len(attempts) == 1  # A failed download is not repeated per count.
    finally:
        demo._tokenizer.cache_clear()


@pytest.mark.parametrize("name", ["llm_demo", "token_savings_demo"])
async def test_llm_demo_without_embedding_download(
    name: str, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import importlib

    demo = importlib.import_module(f"examples.llm_integration.{name}")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(demo, "OPENAI_AVAILABLE", False)
    if name == "token_savings_demo":
        monkeypatch.setattr(demo, "TIKTOKEN_AVAILABLE", False)
        demo._tokenizer.cache_clear()
    await demo.main()
    output = capsys.readouterr().out
    if name == "llm_demo":
        assert "add_numbers" in output
        assert "70" in output
    else:
        assert "Registered 15 tools" in output
        assert "estimate" in output
        demo._tokenizer.cache_clear()
