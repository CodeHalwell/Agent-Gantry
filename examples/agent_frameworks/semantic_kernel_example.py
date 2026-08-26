"""
Semantic Kernel + Agent-Gantry integration example.

Uses ``SemanticKernelAdapter`` — the native integration — instead of
hand-wiring a plugin per tool:

- ``adapter.refresh(kernel, query)`` re-selects the relevant Gantry tools for
  the query and (re)builds the kernel's ``gantry`` plugin with exactly that
  slice, each function routing execution back through ``gantry.execute``
  (retries, timeouts, circuit breakers, security policy all apply).
- For a chat loop, hold a ``adapter.function_provider(kernel)`` and call
  ``await provider.refresh(history)`` before each turn instead — the
  function surface then tracks the conversation turn by turn.
"""

import asyncio
import os

from dotenv import load_dotenv
from semantic_kernel import Kernel
from semantic_kernel.connectors.ai.function_choice_behavior import FunctionChoiceBehavior
from semantic_kernel.connectors.ai.open_ai import (
    OpenAIChatCompletion,
    OpenAIChatPromptExecutionSettings,
)

from agent_gantry import AgentGantry
from agent_gantry.semantic_kernel import SemanticKernelAdapter

load_dotenv()


async def main():
    # 1. Initialize Agent-Gantry
    gantry = AgentGantry()

    @gantry.register(tags=["finance"])
    def calculate_roi(investment: float, return_amount: float) -> float:
        """Calculate the Return on Investment (ROI) percentage.

        Args:
            investment: The initial amount invested.
            return_amount: The total amount returned.
        """
        return ((return_amount - investment) / investment) * 100

    @gantry.register(tags=["finance"])
    def compound_interest(principal: float, rate: float, years: int) -> float:
        """Compute compound interest on a principal.

        Args:
            principal: Starting amount.
            rate: Annual interest rate (e.g. 0.05 for 5%).
            years: Number of years.
        """
        return principal * ((1 + rate) ** years)

    await gantry.sync()

    # 2. Initialize Semantic Kernel with a chat service
    kernel = Kernel()
    service_id = "chat-gpt"
    kernel.add_service(
        OpenAIChatCompletion(
            service_id=service_id,
            ai_model_id="gpt-5.5",
            api_key=os.getenv("OPENAI_API_KEY"),
        )
    )

    # 3. Let Gantry select the relevant tools for this query and register
    #    them as the kernel's "gantry" plugin in one call. Per-parameter
    #    descriptions from the tool docstrings flow through to the
    #    KernelFunction metadata the model sees.
    user_query = "What is the ROI for a $1000 investment that returned $1200?"
    # Lowering threshold for SimpleEmbedder compatibility in this example
    functions = await SemanticKernelAdapter(gantry).refresh(
        kernel, user_query, limit=1, score_threshold=0.1
    )
    print(f"Gantry registered functions: {sorted(functions)}")

    # 4. Run the kernel with automatic function choice
    print("--- Running Semantic Kernel with Agent-Gantry ---")
    settings = OpenAIChatPromptExecutionSettings(service_id=service_id)
    settings.function_choice_behavior = FunctionChoiceBehavior.Auto()

    result = await kernel.invoke_prompt(prompt=user_query, settings=settings)

    print(f"\nResult: {result}")


if __name__ == "__main__":
    asyncio.run(main())
