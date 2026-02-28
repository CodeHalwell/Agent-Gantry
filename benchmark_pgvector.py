import asyncio
import time
from unittest.mock import AsyncMock, MagicMock

from agent_gantry.adapters.vector_stores.remote import PGVectorStore
from agent_gantry.schema.tool import ToolDefinition

async def main():
    store = PGVectorStore("postgresql://dummy:dummy@localhost/dummy", table_name="test_tools")
    store._initialized = True

    mock_pool = MagicMock()
    mock_conn = AsyncMock()
    mock_pool.acquire.return_value.__aenter__.return_value = mock_conn
    store._pool = mock_pool

    # Generate mock tools
    tools = []
    embeddings = []
    for i in range(100):
        tools.append(ToolDefinition(
            name=f"tool_{i}",
            namespace="test",
            description=f"Test tool {i}",
            parameters_schema={"type": "object", "properties": {}},
        ))
        embeddings.append([0.1] * 1536)

    # Simulate a small latency for db execution
    async def mock_execute(*args, **kwargs):
        await asyncio.sleep(0.001)

    mock_conn.execute.side_effect = mock_execute
    mock_conn.executemany.side_effect = mock_execute

    start_time = time.time()
    await store.add_tools(tools, embeddings, upsert=True)
    end_time = time.time()

    print(f"Time taken for 100 tools: {end_time - start_time:.4f} seconds")
    print(f"Number of conn.execute calls: {mock_conn.execute.call_count}")
    print(f"Number of conn.executemany calls: {mock_conn.executemany.call_count}")

if __name__ == "__main__":
    asyncio.run(main())
