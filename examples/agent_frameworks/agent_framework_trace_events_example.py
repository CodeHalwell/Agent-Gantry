"""Per-call tool routing with the built-in trace + event hooks (AF 1.5+).

This is the idiomatic version of the "register a pile of tools, route to the
right one each round, and watch what happens" pattern. It mirrors a hand-rolled
demo where you'd normally wire up your own tracing middleware and result
stringifier — except every piece of that glue now ships in the library:

* ``provider.attach_to(agent, trace=True)`` installs the per-call retrieval
  middleware **and** a console trace that prints, for every tool the model
  calls, the round number, the call, the set the router surfaced, and a short
  preview of the result. No bespoke ``@function_middleware`` needed.
* ``gantry.on_tool_call(cb)`` is a framework-agnostic event hook fired at
  ``gantry.execute`` — the single choke point every tool call flows through —
  so logging/metrics work the same whether the call came from Agent Framework,
  another framework, or a direct ``gantry.execute``.
* ``provider.selections`` keeps the per-round retrieval history so you can see
  exactly which tools were offered at each step (not just the last one).
* ``agent_gantry.render_result`` turns any tool result — including AF ``Content``
  block lists — into readable text.

Logging note: importing ``agent_gantry`` no longer attaches a handler or raises
the log level, so there's nothing to silence. If you *want* Gantry's INFO lines
on the console, opt in with ``agent_gantry.enable_console_logging()``.

The task — "generate a password, then hash it" — is a genuine two-step where
step 2 depends on step 1's output, which is exactly what ``per_call`` retrieval
(driven by the previous tool's result) is for: round 1 surfaces the password
tool, round 2 surfaces the hashing tool.
"""

from __future__ import annotations

import ast
import asyncio
import hashlib
import operator
import secrets
import string
import uuid
from datetime import datetime, timezone
from typing import Annotated

from agent_framework import Agent
from agent_framework.openai import OpenAIChatClient
from dotenv import load_dotenv

from agent_gantry import AgentGantry, ToolCallEvent, render_result
from agent_gantry.agent_framework import AgentFrameworkAdapter

load_dotenv()


# Safe arithmetic via an AST walk — deliberately *not* eval(). `{"__builtins__":
# {}}` is not a sandbox in CPython, and examples get copy-pasted into real code,
# so the tool below evaluates a whitelisted node set instead.
_ARITH_BINOPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
}
_ARITH_UNARY = {ast.UAdd: operator.pos, ast.USub: operator.neg}


def _safe_arith(expression: str) -> str:
    def _ev(node: ast.AST) -> float:
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in _ARITH_BINOPS:
            return _ARITH_BINOPS[type(node.op)](_ev(node.left), _ev(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in _ARITH_UNARY:
            return _ARITH_UNARY[type(node.op)](_ev(node.operand))
        raise ValueError("unsupported expression")

    try:
        return str(_ev(ast.parse(expression, mode="eval").body))
    except Exception:  # noqa: BLE001 - any parse/eval error → friendly message
        return "ERROR: could not evaluate expression"


# ---------------------------------------------------------------------------
# A small, self-contained tool registry (stdlib only — no external API keys).
# Tags + example queries sharpen the semantic router's recall.
# ---------------------------------------------------------------------------
def build_gantry() -> AgentGantry:
    gantry = AgentGantry()

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
        """Evaluate a simple arithmetic expression (safe AST walk, no eval())."""
        return _safe_arith(expression)

    @gantry.register(
        tags=["time", "date", "clock", "calendar"],
        examples=["what time is it now?", "what is today's date?"],
    )
    def get_current_datetime() -> str:
        """Get the current UTC date and time as an ISO-8601 timestamp."""
        return datetime.now(timezone.utc).isoformat()

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


async def main() -> None:
    # Want to see Gantry's own INFO lines too? Uncomment:
    # from agent_gantry import enable_console_logging
    # enable_console_logging()

    gantry = build_gantry()
    await gantry.sync()

    # Framework-agnostic event hook: fires once per tool call at gantry.execute,
    # regardless of which framework drove it. Great for logging/metrics.
    def log_event(event: ToolCallEvent) -> None:
        status = "ok" if event.ok else f"FAILED ({event.result.error})"
        print(
            f"    · event: {event.tool_name} -> {status} "
            f"({event.latency_ms:.0f} ms)"
        )

    gantry.on_tool_call(log_event)

    # Per-call routing: retrieval re-runs every chat round, so the tool surface
    # adapts as the agent reasons. The default per_call query generator is driven
    # by the previous tool's result, which is what makes step 2 (hashing) find
    # the hash tool after step 1 (password generation) runs.
    provider = AgentFrameworkAdapter(gantry).context_provider(
        top_k=3,
        query_strategy="per_call",
    )

    agent = Agent(
        OpenAIChatClient(),
        name="ToolRouter",
        instructions=(
            "You are a helpful assistant. Use the provided tools to complete the "
            "user's request, one step at a time."
        ),
    )

    # One call wires up the context provider, the per-call retrieval middleware,
    # AND the console trace middleware (trace=True) — no hand-rolled glue.
    provider.attach_to(agent, trace=True)

    task = (
        "Generate a secure 20-character password, then compute its SHA-256 hash."
    )
    print(f"=== task: {task} ===\n")
    response = await agent.run(task)

    print("\n=== final answer ===")
    print(render_result(response.text, limit=500))

    # Per-round introspection: every retrieval decision, oldest first. This is
    # the history `last_selection` can't give you — see what was offered at
    # each step, not just the final round.
    print("\n=== router selections, per round ===")
    for i, decision in enumerate(provider.selections, start=1):
        print(f"  round {i}: {decision.summary()}")


if __name__ == "__main__":
    asyncio.run(main())
