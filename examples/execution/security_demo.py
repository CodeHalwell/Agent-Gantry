"""
Gating execution with a ``SecurityPolicy``.

Tools whose names match ``delete_*`` require confirmation: executing one
returns ``status == "pending_confirmation"`` instead of running it, while an
unmatched tool executes normally. No API key needed.

Run with::

    python examples/execution/security_demo.py
"""

import asyncio

from dotfetch import DotFetch
from dotfetch.core.security import SecurityPolicy


async def main():
    # 1. Define a Security Policy
    # We explicitly require confirmation for any tool starting with "delete_"
    policy = SecurityPolicy(require_confirmation=["delete_*"])

    # 2. Initialize DotFetch with the policy
    fetcher = DotFetch(security_policy=policy)
    try:
        # 3. Register a sensitive tool
        @fetcher.register(examples=["drop the users database", "delete the staging database"])
        def delete_database(db_name: str) -> str:
            """Deletes a database. VERY DANGEROUS."""
            return f"Database {db_name} deleted!"

        # 4. Register a safe tool
        @fetcher.register(
            examples=["read from the users database", "show me what's in the orders db"]
        )
        def read_database(db_name: str) -> str:
            """Reads a database."""
            return f"Reading from {db_name}..."

        await fetcher.sync()

        # 5. Try to execute the safe tool
        print("\n--- Executing Safe Tool ---")
        try:
            # We manually construct the call for demonstration
            from dotfetch.schema.execution import ToolCall

            call = ToolCall(tool_name="read_database", arguments={"db_name": "users"})
            result = await fetcher.execute(call)
            print(f"Result: {result.result}")
        except Exception as e:
            print(f"Error: {e}")

        # 6. Try to execute the sensitive tool
        print("\n--- Executing Sensitive Tool ---")
        call = ToolCall(tool_name="delete_database", arguments={"db_name": "users"})
        result = await fetcher.execute(call)

        if result.status == "pending_confirmation":
            print("Security Policy Triggered: Confirmation required for 'delete_database'!")
        elif result.status == "success":
            print(f"Result: {result.result}")
        else:
            print(f"Execution Failed: {result.error}")
    finally:
        await fetcher.close()


if __name__ == "__main__":
    asyncio.run(main())
