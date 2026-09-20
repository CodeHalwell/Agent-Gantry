import asyncio

from agent_gantry import AgentGantry


async def main():
    gantry = AgentGantry()
    try:
        # 1. Register tools with different namespaces and tags
        @gantry.register(
            namespace="admin",
            tags=["sensitive", "db"],
            examples=["delete every user account", "wipe all users from the system"],
        )
        def delete_users() -> str:
            """Delete all users."""
            return "Deleted"

        @gantry.register(
            namespace="public",
            tags=["read-only"],
            examples=["show me the product catalogue", "what products do you sell"],
        )
        def list_products() -> str:
            """List all products."""
            return "Products..."

        @gantry.register(
            namespace="admin",
            tags=["read-only", "reporting"],
            examples=["who changed what last week", "show me the audit trail"],
        )
        def view_audit_log() -> str:
            """View audit logs."""
            return "Logs..."

        await gantry.sync()

        print("--- Filtering Demo ---")

        # 2. Filter by Namespace
        print("\n1. Query with namespace='admin':")
        # Note: We pass 'namespaces' (plural) to the underlying query via kwargs
        tools = await gantry.retrieve_tools("users logs", namespaces=["admin"])
        for t in tools:
            print(f" - {t['function']['name']}")
        # Expected: delete_users, view_audit_log

        # 3. Filter by Tags (requires custom query construction or support in retrieve_tools)

        print("\n2. Query with namespace='public':")
        tools = await gantry.retrieve_tools("products", namespaces=["public"])
        for t in tools:
            print(f" - {t['function']['name']}")
        # Expected: list_products
    finally:
        await gantry.close()


if __name__ == "__main__":
    asyncio.run(main())
