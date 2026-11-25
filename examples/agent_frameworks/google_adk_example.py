"""Google ADK + dotfetch, through ADK's own mechanisms.

1. **Per-turn selection** -- ``adapter.agent(...)`` builds an ADK ``Agent`` whose
   ``before_model_callback`` picks each model request's tools. After the CRM
   lookup returns, the email tool appears because the next selection reads
   the lookup's output.
2. **Readable errors** -- a failing tool returns ``{"error": ...}``, ADK's way of
   telling the model a call failed, instead of ending the run.
3. **Approval** -- a ``requires_confirmation`` tool emits ADK's
   ``adk_request_confirmation`` call; confirming it runs the tool exactly once.
4. **Run context** -- a dotfetch tool reads the ADK session state through
   ``current_tool_context()``.

Everything runs **offline**: a scripted ``BaseLlm`` stands in for Gemini, so no
API key is needed. Pass ``model="gemini-2.5-flash"`` (with ``GOOGLE_API_KEY``
set) to use it for real.

Install and run::

    pip install 'dotfetch[google-adk]'
    python examples/agent_frameworks/google_adk_example.py
"""

from __future__ import annotations

import asyncio
from typing import Any

from dotfetch import DotFetch
from dotfetch.adapters.embedders.simple import SimpleEmbedder


def build_fetcher() -> DotFetch:
    fetcher = DotFetch(embedder=SimpleEmbedder(dimension=256))

    @fetcher.register(tags=["crm"], examples=["find the customer record for Jo"])
    def lookup_customer(name: str) -> str:
        """Look up a customer record by name."""
        return f"Customer {name} found. Next step: send an email message to jo@example.com"

    @fetcher.register(tags=["email"], examples=["email the team"], requires_confirmation=True)
    def send_email(to: str) -> str:
        """Send an email message to a recipient."""
        return f"Email sent to {to}."

    @fetcher.register(tags=["math"], examples=["divide 10 by 2"])
    def divide(a: int, b: int) -> float:
        """Divide integer a by integer b."""
        return a / b

    @fetcher.register(tags=["identity"], examples=["who am I signed in as"])
    def whoami() -> str:
        """Return the name of the signed-in user."""
        from dotfetch.google_adk import current_tool_context

        ctx = current_tool_context()
        return f"You are signed in as {ctx.state['user']}." if ctx else "Unknown user."

    return fetcher


def scripted(*steps: list[tuple[str, dict[str, Any]]]) -> Any:
    """An ADK model that issues ``steps`` of function calls, then answers."""
    from google.adk.models.base_llm import BaseLlm
    from google.adk.models.llm_response import LlmResponse
    from google.genai import types

    class Scripted(BaseLlm):
        model: str = "scripted"
        script: list[Any] = []
        offered: list[list[str]] = []

        async def generate_content_async(self, llm_request: Any, stream: bool = False):
            self.offered.append(sorted(llm_request.tools_dict))
            step = self.script.pop(0) if self.script else None
            parts = (
                [
                    types.Part(function_call=types.FunctionCall(name=n, args=a, id=f"c{i}"))
                    for i, (n, a) in enumerate(step)
                ]
                if step
                else [types.Part(text="done")]
            )
            yield LlmResponse(content=types.Content(role="model", parts=parts))

    return Scripted(script=list(steps), offered=[])


async def run(agent: Any, text: str, state: dict[str, Any] | None = None) -> dict[str, Any]:
    """Run one turn; return the function responses and any confirmation request."""
    from google.adk.runners import InMemoryRunner
    from google.genai import types

    runner = InMemoryRunner(agent=agent, app_name="demo")
    session = await runner.session_service.create_session(app_name="demo", user_id="u", state=state)
    out: dict[str, Any] = {
        "runner": runner,
        "session": session.id,
        "responses": [],
        "pending": None,
    }

    async def drive(message: Any) -> None:
        async for event in runner.run_async(
            user_id="u", session_id=session.id, new_message=message
        ):
            for part in (event.content.parts if event.content else None) or []:
                call = part.function_call
                if call and call.name == "adk_request_confirmation":
                    out["pending"] = call
                if part.function_response:
                    out["responses"].append(part.function_response.response)

    out["drive"] = drive
    await drive(types.Content(role="user", parts=[types.Part(text=text)]))
    return out


async def main() -> None:
    fetcher = build_fetcher()
    try:
        await fetcher.sync()
        try:
            from google.adk.agents import LlmAgent
            from google.genai import types
        except ImportError as exc:
            print(
                f"The Google ADK demo needs google-adk ({exc}).\n"
                "Install it with `pip install 'dotfetch[google-adk]'`."
            )
            return

        from dotfetch.google_adk import GoogleADKAdapter

        adapter = GoogleADKAdapter(fetcher)

        # 1. Per-turn selection ------------------------------------------- #
        model = scripted([("lookup_customer", {"name": "Jo"})])
        await run(
            adapter.agent(model=model, name="crm", limit=1), "find the customer record for Jo"
        )
        print(f"[per-turn] tools offered at turn 1: {model.offered[0]}")
        print(f"[per-turn] tools offered at turn 2: {model.offered[1]}  <- after the lookup\n")

        # 2. Readable errors ---------------------------------------------- #
        model = scripted([("divide", {"a": 1, "b": 0})])
        result = await run(adapter.agent(model=model, name="maths", limit=1), "divide 1 by 0")
        print(f"[errors] the model sees: {result['responses'][0]}\n")

        # 3. Approval ----------------------------------------------------- #
        tools = await adapter.select("send an email message", limit=1)
        model = scripted([("send_email", {"to": "jo@example.com"})])
        result = await run(LlmAgent(name="mail", model=model, tools=tools), "email Jo")
        pending = result["pending"]
        print(f"[approval] paused for {pending.args['originalFunctionCall']['name']}")
        reply = types.FunctionResponse(
            name="adk_request_confirmation", id=pending.id, response={"confirmed": True}
        )
        await result["drive"](
            types.Content(role="user", parts=[types.Part(function_response=reply)])
        )
        print(f"[approval] confirmed -> {result['responses'][-1]}\n")

        # 4. Run context -------------------------------------------------- #
        tools = await adapter.select("who am I signed in as", limit=1)
        model = scripted([("whoami", {})])
        result = await run(
            LlmAgent(name="id", model=model, tools=tools), "who am I?", {"user": "Dan"}
        )
        print(f"[context] {result['responses'][0]}")

        print(
            "\nEvery tool call above went through fetcher.execute, so the security "
            "policy, retries, timeouts and circuit breakers all applied."
        )
    finally:
        await fetcher.close()


if __name__ == "__main__":
    asyncio.run(main())
