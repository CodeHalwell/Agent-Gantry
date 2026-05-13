import asyncio
import time

from agent_gantry.integrations.anthropic_skills import Skill, SkillsClient


class MockClient:
    class Messages:
        async def create(self, **kwargs):
            return "mocked"
    messages = Messages()

class MockSkillsClient(SkillsClient):
    def __init__(self):
        self._skills = type("MockRegistry", (), {"list_skills": lambda self: [Skill(
            name=f"skill_{i}",
            description="a description " * 10,
            instructions="some instructions\n" * 10,
            tools=[f"tool_{i}_1", f"tool_{i}_2"],
            examples=[{"input": "in", "output": "out", "steps": ["s1", "s2"]}] * 5
        ) for i in range(100)]})()
        self._client = MockClient()
        self._gantry = None

async def run_bench():
    client = MockSkillsClient()
    start_time = time.perf_counter()
    iterations = 1000
    for _ in range(iterations):
        await client.create_message(
            model="mock",
            messages=[{"role": "user", "content": "hi"}],
            skills="all",
            auto_retrieve_tools=False
        )
    end_time = time.perf_counter()
    print(f"Time taken for {iterations} iterations: {end_time - start_time:.4f} seconds")

asyncio.run(run_bench())
