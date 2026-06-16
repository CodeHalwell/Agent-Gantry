#!/usr/bin/env python3
"""Agent-Gantry × Microsoft Agent Framework — Textual TUI demo.

Visualises semantic tool routing: full registry vs top-k selection per round,
and live tool execution through Gantry's execution engine.

**Screenshot-friendly demo (no API key):** press ``D`` or click *Run demo*.
Scripts a four-step flow with **unrelated** sub-tasks chained in one user prompt (UTC
time, BBC News, UUID, password) using the same ``GantryContextProvider.dry_run_retrieve``
path the live middleware uses.

**Live agent (requires chat credentials):** press ``L`` or click *Run live* to
drive a real AF ``Agent`` with ``query_strategy="per_call"``. Set ``OPENAI_API_KEY``
(for OpenAI) or ``AZURE_OPENAI_ENDPOINT`` / ``AZURE_OPENAI_BASE_URL`` plus
``AZURE_OPENAI_API_KEY`` (for Azure OpenAI). A ``.env`` file in the repo root is
loaded automatically.

Install (in this repo)::

    uv sync --extra agent-frameworks
    uv pip install textual

Install (standalone project)::

    uv add textual "agent-gantry[agent-frameworks]"
    # Optional stronger embeddings (sync runs before the TUI starts):
    uv add "agent-gantry[nomic]"

Run::

    uv run examples/agent_frameworks/agent_framework_tui_demo.py
    uv run examples/agent_frameworks/agent_framework_tui_demo.py --demo-on-start
"""

from __future__ import annotations

import os

# Avoid tqdm / HF progress bars spawning multiprocessing locks inside Textual.
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TQDM_DISABLE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import argparse
import asyncio
import hashlib
import secrets
import string
import sys
import uuid
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Annotated, ClassVar
from urllib.error import URLError
from urllib.request import Request, urlopen

from textual import on, work
from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical
from textual.message import Message
from textual.widgets import Button, DataTable, Footer, Header, RichLog, Static
from textual.widgets.data_table import ColumnKey, RowKey

from agent_gantry import AgentGantry, ToolCallEvent
from agent_gantry.adapters.embedders.simple import SimpleEmbedder
from agent_gantry.agent_framework import AgentFrameworkAdapter
from agent_gantry.integrations.agent_framework_bridge import RetrievalDecision
from agent_gantry.schema.execution import ExecutionStatus, ToolCall

# ---------------------------------------------------------------------------
# Tool registry (stdlib-only — same shape as agent_framework_trace_events_example)
# ---------------------------------------------------------------------------

DEMO_TASK = (
    "What is the current UTC date and time? "
    "Then get the latest news on bbc.co.uk. "
    "Also generate a random UUID v4. "
    "Finally, create a secure 12-character password."
)


@dataclass(frozen=True)
class DemoRound:
    query: str
    execute: ToolCall


DEMO_ROUNDS: tuple[DemoRound, ...] = (
    DemoRound(
        DEMO_TASK,
        ToolCall(tool_name="get_current_datetime", arguments={}),
    ),
    DemoRound(
        "Get the latest news on bbc.co.uk.",
        ToolCall(tool_name="get_bbc_news", arguments={"max_headlines": 5}),
    ),
    DemoRound(
        "Generate a random UUID v4.",
        ToolCall(tool_name="generate_uuid", arguments={}),
    ),
    DemoRound(
        "Create a secure 12-character password.",
        ToolCall(tool_name="generate_password", arguments={"length": 12}),
    ),
)


def _format_tool_call(call: ToolCall) -> str:
    if not call.arguments:
        return f"{call.tool_name}()"
    parts: list[str] = []
    for key, value in call.arguments.items():
        text = repr(value)
        if len(text) > 24:
            text = text[:21] + "…"
        parts.append(f"{key}={text}")
    return f"{call.tool_name}({', '.join(parts)})"


def _short_query(query: str, *, limit: int = 56) -> str:
    q = query.replace("\n", " ")
    if len(q) > limit:
        return q[: limit - 1] + "…"
    return q


def _resolve_embedder(kind: str):
    if kind == "simple":
        return SimpleEmbedder(dimension=256)
    if kind == "nomic":
        from agent_gantry.adapters.embedders.nomic import NomicEmbedder

        return NomicEmbedder()
    raise ValueError(f"Unknown embedder: {kind!r} (expected simple or nomic)")


def build_gantry(*, embedder_kind: str = "simple") -> AgentGantry:
    gantry = AgentGantry(embedder=_resolve_embedder(embedder_kind))

    @gantry.register(
        tags=["password", "security", "secret", "random"],
        examples=["generate a secure password", "create a strong random password"],
    )
    def generate_password(
        length: Annotated[int, "Password length (8-128)"] = 20,
    ) -> str:
        """Generate a cryptographically strong random password."""
        length = max(8, min(length, 128))
        alphabet = string.ascii_letters + string.digits + "!@#$%^&*()-_=+"
        return "".join(secrets.choice(alphabet) for _ in range(length))

    @gantry.register(
        tags=["hashing", "crypto", "checksum", "sha256"],
        examples=["hash this string with SHA-256", "compute the SHA-256 of this text"],
    )
    def hash_text(
        text: Annotated[str, "The text to hash"],
        algorithm: Annotated[str, "md5, sha1, or sha256"] = "sha256",
    ) -> str:
        """Compute a cryptographic hash digest of some text."""
        algo = algorithm.lower()
        if algo not in ("md5", "sha1", "sha256"):
            return "ERROR: algorithm must be one of md5, sha1, sha256"
        return hashlib.new(algo, text.encode()).hexdigest()

    @gantry.register(
        tags=["math", "arithmetic", "calculator"],
        examples=["what is 12 * (4 + 3)?", "compute 2 to the power of 10"],
    )
    def calculate(expression: Annotated[str, "An arithmetic expression"]) -> str:
        """Evaluate a simple arithmetic expression."""
        return expression  # demo placeholder — not invoked in the scripted flow

    @gantry.register(
        tags=["time", "date", "clock", "calendar"],
        examples=["what time is it now?", "what is today's date?"],
    )
    def get_current_datetime() -> str:
        """Get the current UTC date and time as an ISO-8601 timestamp."""
        return datetime.now(timezone.utc).isoformat()

    @gantry.register(
        tags=["news", "web", "bbc", "headlines", "fetch"],
        examples=[
            "get the latest news from bbc.co.uk",
            "what are today's BBC News headlines?",
        ],
    )
    def get_bbc_news(
        max_headlines: Annotated[int, "Number of headlines (1-10)"] = 5,
    ) -> str:
        """Fetch the latest BBC News headline stories from bbc.co.uk."""
        max_headlines = max(1, min(max_headlines, 10))
        request = Request(
            "https://feeds.bbci.co.uk/news/rss.xml",
            headers={"User-Agent": "agent-gantry-tui-demo/1.0"},
        )
        try:
            with urlopen(request, timeout=15) as response:
                root = ET.fromstring(response.read())
        except URLError as exc:
            return f"ERROR: could not fetch BBC News ({exc.reason})"

        headlines: list[str] = []
        for item in root.findall("./channel/item")[:max_headlines]:
            title = (item.findtext("title") or "").strip()
            if title:
                headlines.append(f"- {title}")

        if not headlines:
            return "No BBC News headlines found."
        return "BBC News headlines:\n" + "\n".join(headlines)

    @gantry.register(
        tags=["uuid", "identifier", "random"],
        examples=["generate a unique id", "give me a random UUID"],
    )
    def generate_uuid() -> str:
        """Generate a random UUID (version 4) identifier."""
        return str(uuid.uuid4())

    @gantry.register(
        tags=["text", "analysis", "word-count"],
        examples=["how many words are in this text?", "count the characters"],
    )
    def text_statistics(text: Annotated[str, "Text to analyse"]) -> str:
        """Report word and character counts for a block of text."""
        return f"words={len(text.split())} characters={len(text)}"

    return gantry


async def prepare_session(*, embedder_kind: str = "simple") -> tuple[
    AgentGantry, AgentFrameworkAdapter, list[str]
]:
    """Build Gantry, sync embeddings, and return ready-to-use session state."""
    gantry = build_gantry(embedder_kind=embedder_kind)
    provider = AgentFrameworkAdapter(gantry).context_provider(
        top_k=3,
        query_strategy="per_call",
    )
    await gantry.sync()
    registry_names = sorted(tool.name for tool in await gantry.list_tools())
    return gantry, provider, registry_names


def build_live_chat_client():
    """Create an AF chat client with explicit credential routing and clear errors."""
    from agent_framework.exceptions import SettingNotFoundError
    from dotenv import load_dotenv
    from agent_framework.openai import OpenAIChatClient

    load_dotenv()

    openai_api_key = os.getenv("OPENAI_API_KEY")
    azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    azure_base_url = os.getenv("AZURE_OPENAI_BASE_URL")
    azure_api_key = os.getenv("AZURE_OPENAI_API_KEY")
    chat_model = (
        os.getenv("OPENAI_CHAT_MODEL")
        or os.getenv("OPENAI_MODEL")
        or os.getenv("AZURE_OPENAI_CHAT_MODEL")
        or os.getenv("AZURE_OPENAI_MODEL")
    )

    if openai_api_key:
        kwargs: dict[str, str] = {"api_key": openai_api_key}
        if chat_model:
            kwargs["model"] = chat_model
        try:
            return OpenAIChatClient(**kwargs)
        except SettingNotFoundError as exc:
            raise ValueError(
                f"{exc} Set OPENAI_MODEL or OPENAI_CHAT_MODEL."
            ) from exc

    if azure_endpoint or azure_base_url:
        kwargs = {}
        if chat_model:
            kwargs["model"] = chat_model
        if azure_endpoint:
            kwargs["azure_endpoint"] = azure_endpoint
        if azure_base_url:
            kwargs["base_url"] = azure_base_url
        if azure_api_key:
            kwargs["api_key"] = azure_api_key
        try:
            return OpenAIChatClient(**kwargs)
        except SettingNotFoundError as exc:
            raise ValueError(
                f"{exc} Set AZURE_OPENAI_CHAT_MODEL or AZURE_OPENAI_MODEL."
            ) from exc

    raise ValueError(
        "Live mode needs chat credentials. Set OPENAI_API_KEY for OpenAI, or "
        "AZURE_OPENAI_ENDPOINT (or AZURE_OPENAI_BASE_URL) plus AZURE_OPENAI_API_KEY "
        "for Azure OpenAI. Optional: OPENAI_MODEL / AZURE_OPENAI_CHAT_MODEL."
    )


# ---------------------------------------------------------------------------
# Textual messages
# ---------------------------------------------------------------------------


@dataclass
class RoundState:
    number: int
    query: str
    decision: RetrievalDecision
    planned_call: ToolCall | None = None


class RoundUpdated(Message):
    def __init__(self, state: RoundState) -> None:
        self.state = state
        super().__init__()


class ToolExecutionLogged(Message):
    """UI notification that a Gantry tool call finished."""

    def __init__(self, tool_event: ToolCallEvent, *, round_num: int = 0) -> None:
        self.tool_event = tool_event
        self.round_num = round_num
        super().__init__()


class StatusUpdated(Message):
    def __init__(self, text: str) -> None:
        self.text = text
        super().__init__()


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------


class GantryAgentFrameworkTUI(App[None]):
    """Semantic tool routing dashboard for Agent Framework integrators."""

    TITLE = "Agent-Gantry × Microsoft Agent Framework"
    SUB_TITLE = "Context is precious · retrieve top-k, not the whole registry"

    CSS = """
    Screen {
        background: #070a12;
    }

    #hero {
        height: auto;
        padding: 0 1 1 1;
        border-bottom: solid #26344d;
    }

    #task-label {
        color: #a9b6cc;
        margin-bottom: 0;
    }

    #task-text {
        color: #edf4ff;
        text-style: bold;
    }

    #controls {
        height: auto;
        padding: 0 1 1 1;
    }

    #controls Button {
        margin-right: 1;
    }

    #panels {
        height: 1fr;
        padding: 0 1;
    }

    .panel {
        width: 1fr;
        border: solid #26344d;
        background: #101827;
        padding: 0 1 1 1;
        margin: 0 1 1 0;
    }

    .panel-title {
        color: #6ee7f9;
        text-style: bold;
        padding: 1 0 0 0;
    }

    .panel-subtitle {
        color: #7d8da8;
        padding: 0 0 1 0;
    }

    #registry-table, #selection-table {
        height: 1fr;
    }

    #execution-log {
        height: 1fr;
        border: none;
        background: transparent;
        overflow-x: hidden;
    }

    #status-bar {
        dock: bottom;
        height: 3;
        padding: 0 2;
        background: #141f33;
        color: #95f985;
        content-align: left middle;
        border-top: solid #26344d;
    }

    Button.primary {
        background: #6ee7f9;
        color: #06101a;
    }

    Button.warning {
        background: #c4a7ff;
        color: #06101a;
    }
    """

    BINDINGS: ClassVar[list[tuple[str, str, str]]] = [
        ("d", "run_demo", "Demo"),
        ("l", "run_live", "Live"),
        ("q", "quit", "Quit"),
    ]

    def __init__(
        self,
        *,
        gantry: AgentGantry,
        provider: AgentFrameworkAdapter,
        registry_names: list[str],
        demo_on_start: bool = False,
    ) -> None:
        super().__init__()
        self._demo_on_start = demo_on_start
        self._gantry = gantry
        self._provider = provider
        self._registry_names = registry_names
        self._registry_rows: dict[str, RowKey] = {}
        self._status_col: ColumnKey | None = None
        self._busy = False
        self._current_round = 0
        self._execution_round = 0

    def compose(self) -> ComposeResult:
        yield Header(show_clock=False)
        with Vertical(id="hero"):
            yield Static("Task", id="task-label")
            yield Static(DEMO_TASK, id="task-text")
        with Horizontal(id="controls"):
            yield Button("Run demo (D)", id="demo-btn", variant="primary")
            yield Button("Run live agent (L)", id="live-btn", classes="warning")
        with Horizontal(id="panels"):
            with Vertical(classes="panel"):
                yield Static("Tool registry", classes="panel-title")
                yield Static(
                    "All tools registered with Gantry",
                    classes="panel-subtitle",
                )
                yield DataTable(id="registry-table", zebra_stripes=True)
            with Vertical(classes="panel"):
                yield Static("Semantic selection", classes="panel-title")
                yield Static(
                    "Top-k surfaced this round (AF ContextProvider)",
                    classes="panel-subtitle",
                )
                yield Static("Press D to start", id="round-banner")
                yield DataTable(id="selection-table", zebra_stripes=True)
            with Vertical(classes="panel"):
                yield Static("Execution log", classes="panel-title")
                yield Static(
                    "Calls routed through gantry.execute",
                    classes="panel-subtitle",
                )
                yield RichLog(
                    id="execution-log",
                    highlight=True,
                    markup=True,
                    wrap=True,
                    min_width=1,
                )
        yield Static(
            "Ready · demo mode needs no API key",
            id="status-bar",
        )
        yield Footer()

    async def on_mount(self) -> None:
        self._gantry.on_tool_call(self._on_tool_executed)
        registry = self.query_one("#registry-table", DataTable)
        _tool_col, status_col = registry.add_columns("Tool", "Status")
        self._status_col = status_col
        for name in self._registry_names:
            self._registry_rows[name] = registry.add_row(name, "idle", key=name)

        selection = self.query_one("#selection-table", DataTable)
        selection.add_columns("Round", "Tool", "Score", "Surfaced")

        if self._demo_on_start:
            self.call_after_refresh(self.run_demo)

    def _on_tool_executed(self, event: ToolCallEvent) -> None:
        round_num = self._execution_round or self._current_round
        self.post_message(ToolExecutionLogged(event, round_num=round_num))

    def _set_busy(self, busy: bool) -> None:
        self._busy = busy
        self.query_one("#demo-btn", Button).disabled = busy
        self.query_one("#live-btn", Button).disabled = busy

    @on(Button.Pressed, "#demo-btn")
    def handle_demo_button(self) -> None:
        self.run_demo()

    @on(Button.Pressed, "#live-btn")
    def handle_live_button(self) -> None:
        self.run_live()

    def action_run_demo(self) -> None:
        self.run_demo()

    def action_run_live(self) -> None:
        self.run_live()

    @work(exclusive=True)
    async def run_demo(self) -> None:
        if self._busy:
            return
        self._set_busy(True)
        log = self.query_one("#execution-log", RichLog)
        log.clear()
        self.query_one("#selection-table", DataTable).clear(columns=False)
        self._reset_registry_status()
        self._current_round = 0
        self.post_message(StatusUpdated("Running scripted demo…"))

        try:
            for round_num, step in enumerate(DEMO_ROUNDS, start=1):
                await self._demo_round(
                    round_num=round_num,
                    query=step.query,
                    execute=step.execute,
                )
            self.post_message(
                StatusUpdated(
                    f"Demo complete · {len(DEMO_ROUNDS)} rounds · top-k retrieval per round"
                )
            )
        finally:
            self._set_busy(False)

    async def _demo_round(
        self,
        *,
        round_num: int,
        query: str,
        execute: ToolCall,
    ) -> str:
        self._current_round = round_num
        self._execution_round = round_num
        decision = await self._provider.dry_run_retrieve(query)
        self.post_message(
            RoundUpdated(RoundState(round_num, query, decision, planned_call=execute))
        )
        try:
            result = await self._gantry.execute(execute)
        finally:
            self._execution_round = 0
        if result.status != ExecutionStatus.SUCCESS:
            raise RuntimeError(result.error or "tool execution failed")
        return str(result.result)

    @work(exclusive=True)
    async def run_live(self) -> None:
        if self._busy:
            return
        try:
            from agent_framework import Agent
        except ImportError:
            self.post_message(
                StatusUpdated("Live mode requires agent-framework — uv add agent-gantry[agent-frameworks]")
            )
            return

        self._set_busy(True)
        log = self.query_one("#execution-log", RichLog)
        log.clear()
        self.query_one("#selection-table", DataTable).clear(columns=False)
        self._reset_registry_status()
        self._current_round = 0

        try:
            client = build_live_chat_client()
        except ValueError as exc:
            self.post_message(StatusUpdated(f"Live run failed: {exc}"))
            self._set_busy(False)
            return

        self.post_message(StatusUpdated("Running live AF agent…"))

        try:
            agent = Agent(
                client,
                name="GantryTUI",
                instructions=(
                    "You are a helpful assistant. Use the provided tools to complete "
                    "the user's request, one step at a time."
                ),
            )
            self._provider.attach_to(agent, trace=False)

            async def poll_selections() -> None:
                seen = 0
                while True:
                    selections = self._provider.selections
                    if len(selections) > seen:
                        for idx, decision in enumerate(
                            selections[seen:], start=seen + 1
                        ):
                            self.post_message(
                                RoundUpdated(
                                    RoundState(idx, decision.query, decision)
                                )
                            )
                        seen = len(selections)
                    await asyncio.sleep(0.15)

            poll_task = asyncio.create_task(poll_selections())
            try:
                await agent.run(DEMO_TASK)
            finally:
                poll_task.cancel()
                with asyncio.suppress(asyncio.CancelledError):
                    await poll_task

            self.post_message(
                StatusUpdated(
                    f"Live run complete · {len(self._provider.selections)} retrieval round(s)"
                )
            )
        except Exception as exc:  # noqa: BLE001 — surface to TUI status bar
            self.post_message(StatusUpdated(f"Live run failed: {exc}"))
        finally:
            self._set_busy(False)

    def _update_registry_status(self, tool_name: str, status: str) -> None:
        row_key = self._registry_rows.get(tool_name)
        if row_key is None or self._status_col is None:
            return
        self.query_one("#registry-table", DataTable).update_cell(
            row_key,
            self._status_col,
            status,
        )

    def _reset_registry_status(self) -> None:
        for name in self._registry_names:
            self._update_registry_status(name, "idle")

    @on(RoundUpdated)
    def handle_round_updated(self, message: RoundUpdated) -> None:
        state = message.state
        self._current_round = state.number
        injected = set(state.decision.injected)
        banner = self.query_one("#round-banner", Static)
        q = _short_query(state.query, limit=72)
        banner.update(
            f"Round {state.number} · {len(injected)} surfaced of "
            f"{len(self._registry_names)} registered · query: {q}"
        )

        selection = self.query_one("#selection-table", DataTable)
        for candidate in state.decision.candidates:
            surfaced = "yes" if candidate.name in injected else "no"
            selection.add_row(
                str(state.number),
                candidate.name,
                f"{candidate.score:.3f}",
                surfaced,
            )

        log = self.query_one("#execution-log", RichLog)
        surfaced_names = ", ".join(state.decision.injected) or "(none)"
        log.write(
            f"[#6ee7f9]Round {state.number}[/]  "
            f"[#7d8da8]query:[/] {_short_query(state.query)}"
        )
        log.write(f"  [#7d8da8]surfaced:[/] {surfaced_names}")
        if state.planned_call is not None:
            log.write(
                f"  [#c4a7ff]call[/]    {_format_tool_call(state.planned_call)}"
            )

        registry = self.query_one("#registry-table", DataTable)
        for name in self._registry_names:
            if name in injected:
                self._update_registry_status(name, "surfaced")
            elif (
                self._status_col is not None
                and (row_key := self._registry_rows.get(name)) is not None
                and registry.get_cell(row_key, self._status_col) == "surfaced"
            ):
                self._update_registry_status(name, "idle")

    @on(ToolExecutionLogged)
    def handle_tool_executed(self, message: ToolExecutionLogged) -> None:
        tool_event = message.tool_event
        log = self.query_one("#execution-log", RichLog)
        status = (
            f"[#95f985]✓ result[/]"
            if tool_event.ok
            else f"[red]✗ failed[/]"
        )
        preview = str(tool_event.result.result)
        if len(preview) > 48:
            preview = preview[:48] + "…"
        round_label = (
            f"Round {message.round_num} · "
            if message.round_num
            else ""
        )
        log.write(
            f"  {status}  {round_label}"
            f"{_format_tool_call(tool_event.call)}  "
            f"[#7d8da8]{tool_event.latency_ms:.0f} ms[/]  {preview}"
        )
        if tool_event.tool_name in self._registry_names:
            self._update_registry_status(tool_event.tool_name, "executed")

    @on(StatusUpdated)
    def handle_status_updated(self, message: StatusUpdated) -> None:
        self.query_one("#status-bar", Static).update(message.text)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--demo-on-start",
        action="store_true",
        help="Start the scripted demo immediately (good for screenshots).",
    )
    parser.add_argument(
        "--embedder",
        choices=("simple", "nomic"),
        default="simple",
        help=(
            "Embedding backend for semantic retrieval. "
            "'simple' needs no ML deps (default). "
            "'nomic' uses sentence-transformers for better ranking."
        ),
    )
    args = parser.parse_args()
    try:
        gantry, provider, registry_names = asyncio.run(
            prepare_session(embedder_kind=args.embedder)
        )
        GantryAgentFrameworkTUI(
            gantry=gantry,
            provider=provider,
            registry_names=registry_names,
            demo_on_start=args.demo_on_start,
        ).run()
    except KeyboardInterrupt:
        sys.exit(0)


if __name__ == "__main__":
    main()
