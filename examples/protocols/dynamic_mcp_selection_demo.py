"""
Dynamic MCP Server Selection Demo for Agent-Gantry.

Registers four MCP servers by metadata only (``register_mcp_server`` does not
connect), syncs that metadata into the vector store, and uses
``retrieve_mcp_servers`` to pick the server a prompt needs. Connecting to the
chosen server (``discover_tools_from_server``) is shown but not run, since the
servers here are not installed. No API key; needs ``agent-gantry[mcp]``.

Run with::

    pip install agent-gantry[mcp]
    python examples/protocols/dynamic_mcp_selection_demo.py
"""

import asyncio

from agent_gantry import AgentGantry


async def demo_dynamic_mcp_selection():
    """
    Demonstrate dynamic MCP server selection using semantic search.

    Instead of connecting to all MCP servers upfront, this approach:
    1. Registers server metadata (descriptions, tags, capabilities)
    2. Uses semantic search to find relevant servers for a query
    3. Connects to and loads tools only from selected servers
    """
    print("=" * 70)
    print("Dynamic MCP Server Selection Demo")
    print("=" * 70)

    # Initialize Agent-Gantry
    gantry = AgentGantry()
    if gantry._mcp_registry is None:
        print("MCP support is not installed: pip install agent-gantry[mcp]")
        await gantry.close()
        return

    try:
        # =========================================================================
        # Step 1: Register MCP servers with metadata (no immediate connection)
        # =========================================================================
        print("\n📝 Step 1: Registering MCP servers with metadata...\n")

        # Register filesystem server
        gantry.register_mcp_server(
            name="filesystem",
            command=["npx", "-y", "@modelcontextprotocol/server-filesystem"],
            description="Provides comprehensive tools for reading and writing files on the local filesystem. Supports file operations, directory listing, and path management.",
            args=["/tmp"],  # the directories the server may access, as positionals
            tags=["filesystem", "files", "io", "local"],
            examples=[
                "read a file",
                "write to a file",
                "list directory contents",
                "check if file exists",
            ],
            capabilities=["read_files", "write_files", "list_directory"],
        )
        print("✅ Registered: filesystem (file operations)")

        # Register database server
        gantry.register_mcp_server(
            name="postgresql",
            command=["python", "-m", "mcp_postgresql"],
            description="Access PostgreSQL databases for querying, inserting, updating, and managing data. Supports SQL operations and transaction management.",
            args=["--connection-string", "postgresql://localhost/mydb"],
            tags=["database", "sql", "data", "postgres"],
            examples=[
                "query database",
                "insert record",
                "update data",
                "create table",
            ],
            capabilities=["read_data", "write_data", "execute_sql"],
        )
        print("✅ Registered: postgresql (database operations)")

        # Register web API server
        gantry.register_mcp_server(
            name="rest_api",
            command=["node", "mcp-rest-api-server.js"],
            description="Provides tools for making REST API calls to external services. Supports GET, POST, PUT, DELETE with authentication and error handling.",
            args=["--base-url", "https://api.example.com"],
            env={"API_KEY": "demo-key"},
            tags=["api", "http", "rest", "external"],
            examples=[
                "make GET request",
                "send POST data",
                "call external API",
                "fetch remote data",
            ],
            capabilities=["network_access", "external_api"],
        )
        print("✅ Registered: rest_api (HTTP/REST operations)")

        # Register email server
        gantry.register_mcp_server(
            name="email",
            command=["python", "-m", "mcp_email"],
            description="Send and manage emails through various providers. Supports SMTP, templates, attachments, and scheduling.",
            args=["--provider", "smtp"],
            tags=["email", "communication", "messaging"],
            examples=[
                "send email",
                "read inbox",
                "send notification",
                "email with attachment",
            ],
            capabilities=["send_email", "read_email"],
        )
        print("✅ Registered: email (email operations)")

        print(f"\n📊 Total servers registered: {gantry._mcp_registry.server_count}")

        # =========================================================================
        # Step 2: Sync server metadata to vector store
        # =========================================================================
        print("\n🔄 Step 2: Syncing server metadata to vector store...\n")

        # Sync servers for semantic search (embeds descriptions, tags, examples)
        synced_count = await gantry.sync_mcp_servers()
        print(f"✅ Synced {synced_count} servers to vector store")

        # =========================================================================
        # Step 3: Semantic server retrieval based on queries
        # =========================================================================
        print("\n🔍 Step 3: Finding relevant servers via semantic search...\n")

        queries = [
            "I need to read a configuration file",
            "Query customer data from the database",
            "Send a notification email to users",
            "Call an external weather API",
        ]

        for query in queries:
            print(f"\n🎯 Query: '{query}'")

            # Semantic search over the synced metadata: description, tags,
            # examples. Nothing is connected yet.
            servers = await gantry.retrieve_mcp_servers(query, limit=2)
            for server in servers:
                print(f"   ✓ {server.name}: {server.description[:50]}...")

        # =========================================================================
        # Step 4: On-demand tool discovery from selected servers
        # =========================================================================
        print("\n\n🔧 Step 4: Discovering tools on-demand from selected server...\n")

        # Take the top server for the first query and connect to that one only.
        chosen = (await gantry.retrieve_mcp_servers(queries[0], limit=1))[0]
        print(f"💭 '{queries[0]}' → Selected: {chosen.name} server")

        # This is the call that spawns the server process and imports its tools.
        # It is not run here because `npx @modelcontextprotocol/server-filesystem`
        # may not be installed; uncomment it on a machine that has it.
        #
        # count = await gantry.discover_tools_from_server(chosen.name)
        # print(f"✅ Discovered {count} tools from {chosen.name} server")

        print("🔌 discover_tools_from_server() would connect to it now (not run in this demo)\n")

        # =========================================================================
        # Benefits of Dynamic Selection
        # =========================================================================
        print("\n" + "=" * 70)
        print("✨ Benefits of Dynamic MCP Server Selection")
        print("=" * 70 + "\n")

        print("🎯 Semantic Routing:")
        print("   • Automatically finds relevant servers based on query context")
        print("   • Uses vector embeddings for intelligent matching")
        print("   • Considers tags, examples, and capabilities\n")

        print("⚡ Performance:")
        print("   • No upfront connection to all servers")
        print("   • Tools loaded only when needed (lazy loading)")
        print("   • Reduces initialization time\n")

        print("🔒 Security & Resource Management:")
        print("   • Only connects to necessary servers")
        print("   • Minimizes attack surface")
        print("   • Tracks server health and availability\n")

        print("🧩 Plug & Play:")
        print("   • Simple registration with metadata")
        print("   • Works alongside existing add_mcp_server() method")
        print("   • Fully backward compatible\n")

        # =========================================================================
        # Comparison: Old vs New Approach
        # =========================================================================
        print("=" * 70)
        print("📊 Comparison: Traditional vs Dynamic Selection")
        print("=" * 70 + "\n")

        print("❌ Traditional approach (add_mcp_server):")
        print("   1. Connect to ALL servers at startup")
        print("   2. Import ALL tools from each server")
        print("   3. All tools available, but high initialization cost")
        print("   4. No semantic selection of servers\n")

        print("✅ New dynamic approach (register_mcp_server):")
        print("   1. Register server metadata (no connection)")
        print("   2. Use semantic search to find relevant servers")
        print("   3. Connect and load tools only from selected servers")
        print("   4. Intelligent, on-demand server selection\n")

        print("=" * 70)
        print("Demo complete! 🎉")
        print("=" * 70)
    finally:
        await gantry.close()


async def demo_workflow_example():
    """
    Show a complete workflow with code examples.
    """
    print("\n\n")
    print("=" * 70)
    print("💡 Complete Workflow Example")
    print("=" * 70 + "\n")

    # Create gantry
    gantry = AgentGantry()
    if gantry._mcp_registry is None:
        print("MCP support is not installed: pip install agent-gantry[mcp]")
        await gantry.close()
        return

    try:
        # Register servers with rich metadata
        gantry.register_mcp_server(
            name="files",
            command=["npx", "-y", "@modelcontextprotocol/server-filesystem"],
            description="Local filesystem operations for reading and writing files",
            args=["/home/user/documents"],
            tags=["files", "io"],
            examples=["read file", "write file"],
            capabilities=["read_files", "write_files"],
        )

        # Sync to enable semantic search
        await gantry.sync_mcp_servers()

        # Find relevant servers
        servers = await gantry.retrieve_mcp_servers("I need to read a log file", limit=2)
        print(f"Selected for 'I need to read a log file': {[s.name for s in servers]}")

        # The remaining steps need the server installed, so they are shown only:
        # count = await gantry.discover_tools_from_server("files")   # connect, import tools
        # tools = await gantry.retrieve_tools("read my config.yaml")  # route as usual
        # result = await gantry.execute(...)

        print("✅ Remaining steps (connect, route, execute) shown in the comments above")
    finally:
        await gantry.close()


async def main():
    """Run all demos."""
    await demo_dynamic_mcp_selection()
    await demo_workflow_example()

    print("\n\nFor more examples, see:")
    print("• examples/protocols/mcp_integration_demo.py")
    print("• examples/protocols/jev_mcp_selection_demo.py  (same selection via a decision model)")
    print("• tests/test_dynamic_mcp_selection.py")


if __name__ == "__main__":
    asyncio.run(main())
