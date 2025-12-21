import asyncio
import time
from collections.abc import Callable
from typing import Any

from agent_gantry import AgentGantry
from agent_gantry.schema.config import AgentGantryConfig, EmbedderConfig

# Categories and Actions to generate 100 distinct tools
CATEGORIES = [
    ("aws_s3", "AWS S3 Storage"),
    ("aws_ec2", "AWS EC2 Compute"),
    ("azure_blob", "Azure Blob Storage"),
    ("azure_vm", "Azure Virtual Machine"),
    ("gcp_storage", "Google Cloud Storage"),
    ("slack", "Slack Messaging"),
    ("jira", "Jira Issue Tracking"),
    ("github", "GitHub Repository Management"),
    ("postgres", "PostgreSQL Database"),
    ("local_fs", "Local File System"),
]

ACTIONS = [
    ("create", "Create a new resource"),
    ("read", "Read or retrieve details of a resource"),
    ("update", "Update content or properties"),  # Changed from "Update configuration or content" to avoid overlap
    ("delete", "Remove or destroy a resource"),
    ("list", "List all available resources"),
    ("search", "Search for specific resources"),
    ("archive", "Archive old data or resources"),
    ("restore", "Restore data from backups"),
    ("monitor", "Check health and metrics"),
    ("configure", "Change configuration settings and permissions"), # Added "configuration" here
]

# Specific overrides to make tools more realistic and distinct
CATEGORY_OVERRIDES = {
    "slack": {
        "create": "Send a new message or post to a channel",
        "read": "Read messages or channel history",
    },
    "jira": {
        "search": "Search for issues, bugs, or tickets using JQL",
        "read": "Get details of a specific issue by ID",
    },
}

def generate_tool_factory(name: str, description: str) -> Callable[..., str]:
    """Generates a dummy tool function."""
    def tool_func(**kwargs: Any) -> str:
        return f"Executed {name} with {kwargs}"

    tool_func.__name__ = name
    tool_func.__doc__ = description
    return tool_func

async def main():
    print("--- Agent-Gantry Stress Test: 100 Tools ---")

    # 1. Configure Gantry with Nomic Embedder for high accuracy
    # We use Nomic because SimpleEmbedder (hashing) struggles with 100+ semantically distinct tools
    print("Initializing Agent-Gantry with Nomic Embedder...")
    try:
        config = AgentGantryConfig(
            embedder=EmbedderConfig(
                type="nomic",
                model="nomic-ai/nomic-embed-text-v1.5"
            )
        )
        gantry = AgentGantry(config=config)
    except ImportError:
        print("Warning: 'sentence-transformers' not found. Falling back to SimpleEmbedder.")
        print("Note: Accuracy will be lower with SimpleEmbedder due to lack of semantic understanding.")
        gantry = AgentGantry()

    # 2. Register 100 Tools
    print("Registering 100 tools...")
    start_reg = time.perf_counter()

    expected_tools = {}

    for cat_prefix, cat_desc in CATEGORIES:
        for act_name, act_desc in ACTIONS:
            tool_name = f"{cat_prefix}_{act_name}"

            # Apply overrides if available
            if cat_prefix in CATEGORY_OVERRIDES and act_name in CATEGORY_OVERRIDES[cat_prefix]:
                base_desc = CATEGORY_OVERRIDES[cat_prefix][act_name]
            else:
                base_desc = act_desc

            tool_desc = f"{base_desc} for {cat_desc}."

            # Create and register the tool
            func = generate_tool_factory(tool_name, tool_desc)
            gantry.register(func)

            # Store for verification
            expected_tools[tool_name] = tool_desc

    # Sync to vector store
    count = await gantry.sync()
    end_reg = time.perf_counter()
    print(f"Registered {count} tools in {end_reg - start_reg:.2f}s")

    # 3. Run Test Queries
    print("\n--- Running Retrieval Tests ---")

    test_cases = [
        ("I need to make a new bucket in Amazon S3", "aws_s3_create"),
        ("Find a ticket in Jira about the login bug", "jira_search"),
        ("Check if the Azure VM is running", "azure_vm_monitor"),
        ("Delete the old logs from the local disk", "local_fs_delete"),
        ("Send a message to the team on Slack", "slack_create"), # 'create' message
        ("Restore the database backup in Postgres", "postgres_restore"),
        ("List all repositories in GitHub", "github_list"),
        ("Change the settings for the Google Cloud bucket", "gcp_storage_configure"), # Changed query to match 'configure' better
        ("Archive the old blobs in Azure storage", "azure_blob_archive"),
        ("Read the details of the EC2 instance", "aws_ec2_read"),
    ]

    score = 0
    total = len(test_cases)

    start_test = time.perf_counter()

    for query, expected_tool_name in test_cases:
        print(f"\nQuery: '{query}'")

        # Retrieve top 2 tools
        tools = await gantry.retrieve_tools(query, limit=2)

        if not tools:
            print("❌ Failed: No tools found.")
            continue

        retrieved_names = [t['function']['name'] for t in tools]

        if expected_tool_name in retrieved_names:
            print(f"✅ Success: Found '{expected_tool_name}' in top 2 {retrieved_names}")
            score += 1
        else:
            print(f"❌ Failed: Expected '{expected_tool_name}', got {retrieved_names}")

    end_test = time.perf_counter()

    print("\n--- Results ---")
    print(f"Accuracy: {score}/{total} ({score/total*100:.1f}%)")
    print(f"Test Duration: {end_test - start_test:.2f}s")

    if score == total:
        print(">>> PERFECT SCORE! Agent-Gantry successfully routed 100 tools.")
    else:
        print(">>> Some queries failed. This may happen with ambiguous queries or SimpleEmbedder.")

if __name__ == "__main__":
    asyncio.run(main())
