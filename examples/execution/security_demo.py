import asyncio

from agent_gantry import AgentGantry
from agent_gantry.core.security import SecurityPolicy


async def main():
    # 1. Define a Security Policy
    # We explicitly require confirmation for any tool starting with "delete_"
    policy = SecurityPolicy(require_confirmation=["delete_*"])

    # 2. Initialize AgentGantry with the policy
    gantry = AgentGantry(security_policy=policy)

    # 3. Register a sensitive tool
    @gantry.register
    def delete_database(db_name: str) -> str:
        """Deletes a database. VERY DANGEROUS."""
        return f"Database {db_name} deleted!"

    # 4. Register a safe tool
    @gantry.register
    def read_database(db_name: str) -> str:
        """Reads a database."""
        return f"Reading from {db_name}..."

    await gantry.sync()

    # 5. Try to execute the safe tool
    print("\n--- Executing Safe Tool ---")
    try:
        # We manually construct the call for demonstration
        from agent_gantry.schema.execution import ToolCall
        call = ToolCall(tool_name="read_database", arguments={"db_name": "users"})
        result = await gantry.execute(call)
        print(f"Result: {result.result}")
    except Exception as e:
        print(f"Error: {e}")

    # 6. Try to execute the sensitive tool
    print("\n--- Executing Sensitive Tool ---")
    call = ToolCall(tool_name="delete_database", arguments={"db_name": "users"})
    result = await gantry.execute(call)

    if result.status == "pending_confirmation":
         print("Security Policy Triggered: Confirmation required for 'delete_database'!")
    elif result.status == "success":
         print(f"Result: {result.result}")
    else:
         print(f"Execution Failed: {result.error}")

if __name__ == "__main__":
    asyncio.run(main())
