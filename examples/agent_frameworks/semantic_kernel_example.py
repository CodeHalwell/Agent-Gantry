import asyncio
import os
from dotenv import load_dotenv
from semantic_kernel import Kernel
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion, OpenAIChatCompletion
from semantic_kernel.functions import kernel_function

from agent_gantry import AgentGantry
from agent_gantry.schema.execution import ToolCall

load_dotenv()

async def main():
    # 1. Initialize Agent-Gantry
    gantry = AgentGantry()

    @gantry.register
    def calculate_roi(investment: float, return_amount: float) -> float:
        """Calculate the Return on Investment (ROI) percentage."""
        return ((return_amount - investment) / investment) * 100

    await gantry.sync()

    # 2. Initialize Semantic Kernel
    kernel = Kernel()
    
    # Add AI Service
    service_id = "chat-gpt"
    kernel.add_service(
        OpenAIChatCompletion(
            service_id=service_id,
            ai_model_id="gpt-4o",
            api_key=os.getenv("OPENAI_API_KEY"),
        )
    )

    # 3. Retrieve tools from Gantry
    user_query = "What is the ROI for a $1000 investment that returned $1200?"
    # Lowering threshold for SimpleEmbedder compatibility in this example
    retrieved_tools = await gantry.retrieve_tools(user_query, limit=1, score_threshold=0.1)

    # 4. Register Gantry tools as Semantic Kernel functions
    class GantryPlugin:
        @kernel_function(
            name="calculate_roi",
            description="Calculate the Return on Investment (ROI) percentage."
        )
        async def calculate_roi(self, investment: float, return_amount: float) -> float:
            result = await gantry.execute(ToolCall(
                tool_name="calculate_roi",
                arguments={"investment": investment, "return_amount": return_amount}
            ))
            return result.result

    kernel.add_plugin(GantryPlugin(), plugin_name="Gantry")

    # 5. Run the kernel
    print("--- Running Semantic Kernel with Agent-Gantry ---")
    
    from semantic_kernel.connectors.ai.function_choice_behavior import FunctionChoiceBehavior
    from semantic_kernel.connectors.ai.open_ai import OpenAIChatPromptExecutionSettings

    settings = OpenAIChatPromptExecutionSettings(service_id=service_id)
    settings.function_choice_behavior = FunctionChoiceBehavior.Auto()

    result = await kernel.invoke_prompt(
        prompt=user_query,
        settings=settings
    )
    
    print(f"\nResult: {result}")

if __name__ == "__main__":
    asyncio.run(main())
