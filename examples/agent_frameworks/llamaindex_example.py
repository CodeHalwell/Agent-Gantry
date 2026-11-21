"""LlamaIndex + Agent-Gantry, through LlamaIndex's own mechanisms.

1. **Per-step selection** -- ``adapter.function_agent(llm)`` builds a
   ``FunctionAgent`` whose tool retriever re-selects Gantry tools every step.
2. **Readable errors** -- a failing tool comes back as
   ``ToolOutput(is_error=True)`` the model can react to.
3. **Approval** -- a ``requires_confirmation`` tool waits for a
   ``HumanResponseEvent``; answering "yes" runs it exactly once.
4. **Run context** -- a Gantry tool reads the workflow state through
   ``current_workflow_context()``.

Everything runs **offline**: a scripted function-calling LLM stands in for a
real one, so no API key is needed. Swap in ``OpenAI(model="gpt-5.5")`` (from
``llama-index-llms-openai``) to use it for real.

Install and run::

    pip install 'agent-gantry[llamaindex]'
    python examples/agent_frameworks/llamaindex_example.py
"""

from __future__ import annotations

import asyncio
from typing import Any

from agent_gantry import AgentGantry
from agent_gantry.adapters.embedders.simple import SimpleEmbedder


def build_gantry() -> AgentGantry:
    gantry = AgentGantry(embedder=SimpleEmbedder(dimension=256))

    @gantry.register(tags=["prefs"], examples=["what are the preferences for user dev_123"])
    def get_preferences(user_id: str) -> str:
        """Look up a user's saved preferences."""
        return f"{user_id} prefers dark mode and email updates."

    @gantry.register(tags=["math"], examples=["divide 10 by 2"])
    def divide(a: int, b: int) -> float:
        """Divide integer a by integer b."""
        return a / b

    @gantry.register(tags=["email"], examples=["email the team"], requires_confirmation=True)
    def send_email(to: str) -> str:
        """Send an email message to a recipient."""
        return f"Email sent to {to}."

    @gantry.register(tags=["identity"], examples=["who am I signed in as"])
    async def whoami() -> str:
        """Return the signed-in user from the workflow state."""
        from agent_gantry.llamaindex import current_workflow_context

        ctx = current_workflow_context()
        return f"You are signed in as {await ctx.store.get('user')}." if ctx else "Unknown user."

    return gantry


def scripted(*steps: list[tuple[str, dict[str, Any]]]) -> Any:
    """A function-calling LLM that issues ``steps`` of tool calls, then answers."""
    from llama_index.core.llms import ChatMessage, ChatResponse, LLMMetadata, MessageRole
    from llama_index.core.llms.function_calling import FunctionCallingLLM
    from llama_index.core.llms.llm import ToolSelection

    class Scripted(FunctionCallingLLM):
        script: list[Any] = []
        offered: list[list[str]] = []

        @property
        def metadata(self) -> LLMMetadata:
            return LLMMetadata(is_function_calling_model=True, model_name="scripted")

        def _prepare_chat_with_tools(self, tools: Any, *args: Any, **kwargs: Any) -> dict:
            self.offered.append(sorted(t.metadata.name for t in tools))
            return {"messages": list(kwargs.get("chat_history") or [])}

        def _reply(self) -> ChatResponse:
            step = self.script.pop(0) if self.script else None
            calls = [{"id": f"c{i}", "name": n, "args": a} for i, (n, a) in enumerate(step or [])]
            message = ChatMessage(
                role=MessageRole.ASSISTANT,
                content="" if calls else "done",
                additional_kwargs={"tool_calls": calls},
            )
            return ChatResponse(message=message, delta=message.content)

        def chat(self, messages: Any, **kwargs: Any) -> ChatResponse:
            return self._reply()

        async def achat(self, messages: Any, **kwargs: Any) -> ChatResponse:
            return self._reply()

        def stream_chat(self, messages: Any, **kwargs: Any) -> Any:
            yield self._reply()

        async def astream_chat(self, messages: Any, **kwargs: Any) -> Any:
            reply = self._reply()

            async def gen() -> Any:
                yield reply

            return gen()

        def complete(self, *args: Any, **kwargs: Any) -> Any:
            raise NotImplementedError

        async def acomplete(self, *args: Any, **kwargs: Any) -> Any:
            raise NotImplementedError

        def stream_complete(self, *args: Any, **kwargs: Any) -> Any:
            raise NotImplementedError

        async def astream_complete(self, *args: Any, **kwargs: Any) -> Any:
            raise NotImplementedError

        def get_tool_calls_from_response(self, response: Any, *args: Any, **kwargs: Any) -> list:
            return [
                ToolSelection(tool_id=c["id"], tool_name=c["name"], tool_kwargs=c["args"])
                for c in response.message.additional_kwargs.get("tool_calls", [])
            ]

    return Scripted(script=list(steps), offered=[])


async def run(agent: Any, prompt: str, ctx: Any = None) -> list[Any]:
    """Run ``agent``, approving any gated call; return the tool outputs."""
    from llama_index.core.workflow import HumanResponseEvent, InputRequiredEvent

    handler = agent.run(user_msg=prompt, ctx=ctx)
    outputs = []
    async for event in handler.stream_events():
        if type(event).__name__ == "ToolCallResult":
            outputs.append(event.tool_output)
        if isinstance(event, InputRequiredEvent):
            print(f"[approval] {event.prefix} {event.gantry_arguments} -> answering yes")
            handler.ctx.send_event(
                HumanResponseEvent(response="yes", gantry_approval_id=event.gantry_approval_id)
            )
    await handler
    return outputs


async def main() -> None:
    gantry = build_gantry()
    try:
        await gantry.sync()
        try:
            from llama_index.core.agent.workflow import FunctionAgent
            from llama_index.core.workflow import Context
        except ImportError as exc:
            print(
                f"The LlamaIndex demo needs llama-index-core ({exc}).\n"
                "Install it with `pip install 'agent-gantry[llamaindex]'`."
            )
            return

        from agent_gantry.llamaindex import LlamaIndexAdapter

        adapter = LlamaIndexAdapter(gantry)

        # 1. Per-step selection ------------------------------------------- #
        llm = scripted([("get_preferences", {"user_id": "dev_123"})])
        outputs = await run(
            adapter.function_agent(llm, limit=1), "What are the preferences for user dev_123?"
        )
        print(f"[per-step] tools offered: {llm.offered[0]} -> {outputs[0].content}\n")

        # 2. Readable errors ---------------------------------------------- #
        tools = await adapter.select("divide 10 by 2", limit=1)
        outputs = await run(
            FunctionAgent(tools=tools, llm=scripted([("divide", {"a": 1, "b": 0})])),
            "divide 1 by 0",
        )
        print(f"[errors] is_error={outputs[0].is_error} content={outputs[0].content!r}\n")

        # 3. Approval ----------------------------------------------------- #
        tools = await adapter.select("send an email message", limit=1)
        agent = FunctionAgent(tools=tools, llm=scripted([("send_email", {"to": "jo@example.com"})]))
        outputs = await run(agent, "email Jo")
        print(f"[approval] approved -> {outputs[0].content}\n")

        # 4. Run context -------------------------------------------------- #
        tools = await adapter.select("who am I signed in as", limit=1)
        agent = FunctionAgent(tools=tools, llm=scripted([("whoami", {})]))
        ctx = Context(agent)
        await ctx.store.set("user", "Dan")
        outputs = await run(agent, "who am I?", ctx)
        print(f"[context] {outputs[0].content}")

        print(
            "\nEvery tool call above went through gantry.execute, so the security "
            "policy, retries, timeouts and circuit breakers all applied."
        )
    finally:
        await gantry.close()


if __name__ == "__main__":
    asyncio.run(main())
