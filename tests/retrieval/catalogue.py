"""A synthetic catalogue and labelled queries for the retrieval eval.

Nothing here is a real product. The catalogue is generated so that it is large
enough to be hard (about 345 tools, most of them near-duplicates of one
another) without anyone hand-writing 345 descriptions:

* 264 CRUD-shaped tools across eleven business systems, which differ from
  their neighbours by one verb or one noun;
* 24 one-off utilities;
* 57 tools converted from three invented MCP servers through the real
  ``MCPClient._convert_tool``, so whatever discovery does to an MCP tool
  (truncation, tagging) is what gets measured.

No tool carries ``examples``. A query that paraphrases a tool's own example is
a leak, and it makes any retriever look better than it is.

Each case has a ``kind`` saying what it tests:

``paraphrase``
    Shares meaning with its answer but few words. The ordinary case.
``identifier``
    Names tokens from the tool's own name.
``parameter``
    Names a parameter of the tool that its description never mentions.
``server``
    Only the MCP server's name tells the answer from its twins on other
    servers.
``tail``
    The sentence that answers it sits past character 2000 of a long MCP
    description.
``body``
    (skills) The answer is in the skill's body, not its description.
``derived``
    (MCP servers) The server's own description is vague; only its tools say
    what it does.
``decline``
    Nothing in the catalogue fits.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from agent_gantry.evaluation import RetrievalCase
from agent_gantry.schema.skill import Skill, SkillCategory
from agent_gantry.schema.tool import ToolDefinition

# --------------------------------------------------------------------------- #
# Native tools
# --------------------------------------------------------------------------- #

#: (namespace, thing, how a person would say it)
_THINGS: list[tuple[str, str, str]] = [
    ("billing", "invoice", "customer invoice"),
    ("billing", "payment", "card payment"),
    ("billing", "refund", "refund to a customer"),
    ("billing", "subscription", "recurring subscription"),
    ("crm", "contact", "contact in the address book"),
    ("crm", "company", "company account"),
    ("crm", "deal", "sales deal"),
    ("crm", "lead", "sales lead"),
    ("support", "ticket", "customer support ticket"),
    ("support", "macro", "canned support reply"),
    ("hr", "employee", "employee record"),
    ("hr", "timesheet", "weekly timesheet"),
    ("hr", "leave_request", "request for time off"),
    ("inventory", "product", "stock item"),
    ("inventory", "warehouse", "warehouse location"),
    ("inventory", "shipment", "outbound shipment"),
    ("inventory", "purchase_order", "purchase order to a supplier"),
    ("calendar", "meeting", "calendar meeting"),
    ("calendar", "room_booking", "meeting room booking"),
    ("docs", "document", "shared document"),
    ("docs", "folder", "document folder"),
    ("docs", "comment", "comment on a document"),
    ("dev", "repository", "source code repository"),
    ("dev", "pull_request", "pull request for code review"),
    ("dev", "pipeline", "build pipeline"),
    ("dev", "release", "published software release"),
    ("marketing", "campaign", "marketing campaign"),
    ("marketing", "coupon", "discount coupon"),
    ("marketing", "newsletter", "email newsletter"),
    ("it", "device", "company laptop or phone"),
    ("it", "license", "software licence seat"),
    ("finance", "expense_report", "employee expense report"),
    ("finance", "budget", "departmental budget"),
]

_STRING = {"type": "string"}
_INTEGER = {"type": "integer"}

#: verb -> (description template, parameter names; ``{id}`` is the thing's id)
_OPERATIONS: dict[str, tuple[str, dict[str, dict[str, str]]]] = {
    "create": ("Create a new {label}.", {"name": _STRING, "details": _STRING}),
    "get": ("Fetch one {label} by its identifier.", {"{id}": _STRING}),
    "list": ("List every {label}, newest first.", {"page": _INTEGER, "page_size": _INTEGER}),
    "update": ("Change the fields of an existing {label}.", {"{id}": _STRING, "changes": _STRING}),
    "delete": ("Permanently remove a {label}.", {"{id}": _STRING}),
    "search": ("Find each {label} matching some text.", {"text": _STRING, "limit": _INTEGER}),
    "export": ("Export every {label} to a CSV file.", {"since": _STRING}),
    "archive": ("Archive a {label} so it drops out of the active list.", {"{id}": _STRING}),
}

#: (name, description, parameter names)
_UTILITIES: list[tuple[str, str, list[str]]] = [
    ("add_numbers", "Add two numbers and return the sum.", ["a", "b"]),
    ("multiply_numbers", "Multiply two numbers and return the product.", ["a", "b"]),
    ("reverse_string", "Reverse the characters of a string.", ["text"]),
    ("count_words", "Count the words in a piece of text.", ["text"]),
    ("convert_currency", "Convert an amount of money from one currency to another.",
     ["amount", "from_currency", "to_currency"]),
    ("convert_units", "Convert a measurement between units, such as miles to kilometres.",
     ["value", "from_unit", "to_unit"]),
    ("get_weather_forecast", "Get the weather forecast for a place over the coming days.",
     ["place", "days"]),
    ("get_current_time", "Get the current date and time in a time zone.", ["time_zone"]),
    ("translate_text", "Translate text from one language into another.",
     ["text", "target_language"]),
    ("summarise_text", "Shorten a long piece of text to its main points.", ["text"]),
    ("detect_language", "Work out which language a piece of text is written in.", ["text"]),
    ("send_email", "Send an email message to one or more recipients.",
     ["to", "subject", "body"]),
    ("send_sms", "Send a text message to a mobile phone number.", ["phone_number", "message"]),
    ("generate_password", "Generate a random, hard-to-guess password.", ["length"]),
    ("hash_text", "Compute a SHA-256 digest of some text.", ["text"]),
    ("resize_image", "Scale an image to a given width and height.", ["image", "width", "height"]),
    ("extract_text_from_pdf", "Read the text out of a PDF document, including scanned pages.",
     ["file"]),
    ("geocode_address", "Find the latitude and longitude of a street address.", ["address"]),
    ("calculate_route", "Work out driving directions and travel time between two places.",
     ["origin", "destination"]),
    ("shorten_url", "Turn a long web address into a short link.", ["url"]),
    ("validate_email_address", "Check whether an email address is well formed and deliverable.",
     ["email"]),
    ("generate_qr_code", "Make a QR code image that encodes a link or some text.", ["content"]),
    ("lookup_stock_price", "Look up the latest share price for a ticker symbol.", ["ticker"]),
    ("calculate_loan_repayment",
     "Work out the monthly repayment on a loan from its amount, rate and term.",
     ["principal", "annual_rate", "years"]),
]


def _schema(properties: dict[str, dict[str, str]]) -> dict[str, Any]:
    return {"type": "object", "properties": properties, "required": list(properties)}


def native_tools() -> list[ToolDefinition]:
    """The 288 tools that are registered directly, not discovered over MCP."""
    tools: list[ToolDefinition] = []
    for namespace, thing, label in _THINGS:
        for verb, (template, parameters) in _OPERATIONS.items():
            tools.append(
                ToolDefinition(
                    name=f"{verb}_{thing}",
                    namespace=namespace,
                    description=template.format(label=label),
                    parameters_schema=_schema(
                        {key.format(id=f"{thing}_id"): value for key, value in parameters.items()}
                    ),
                    tags=[namespace],
                )
            )
    for name, description, parameters in _UTILITIES:
        tools.append(
            ToolDefinition(
                name=name,
                description=description,
                parameters_schema=_schema({p: _STRING for p in parameters}),
            )
        )
    return tools


# --------------------------------------------------------------------------- #
# MCP tools
# --------------------------------------------------------------------------- #

_PADDING = (
    "Results are paged; pass the cursor from one response to fetch the next. "
    "Times are in UTC unless a zone is given. Requests are rate limited per workspace, "
    "and a request over the limit is rejected, not queued. "
)


def _long(head: str, tail: str) -> str:
    """A description whose last sentence sits past character 2000."""
    return head + " " + _PADDING * 11 + tail


#: server -> [(tool name, description, parameter names)]
MCP_TOOLS: dict[str, list[tuple[str, str, list[str]]]] = {
    "logwatch": [
        ("search_logs",
         _long("Search application logs.",
               "Use this to investigate checkout failures and payment gateway timeouts."),
         ["facet_query", "time_range"]),
        ("tail_logs", "Stream the newest log lines from a service as they arrive.", ["service"]),
        ("query_metrics", "Read a time series such as CPU, memory or request latency.",
         ["metric_name", "rollup"]),
        ("list_monitors", "List the alerting monitors and whether each is firing.", []),
        ("create_monitor", "Create an alerting monitor that fires when a threshold is crossed.",
         ["metric_name", "threshold"]),
        ("mute_monitor", "Silence an alerting monitor for a while so it stops paging people.",
         ["monitor_id", "downtime_minutes"]),
        ("get_trace", "Fetch one distributed trace with all of its spans.", ["trace_id"]),
        ("search_traces", "Find slow or failed requests across services.", ["span_filter"]),
        ("list_incidents", "List open and recently resolved incidents.", []),
        ("declare_incident", "Open a new incident and page the responders.",
         ["title", "severity_level"]),
        ("get_container_restarts",
         "Report how often each container has restarted and why it last stopped.",
         ["cluster"]),
        ("get_service_map", "Show which services call which, with error rates on each edge.", []),
        ("list_slos", "List the service level objectives and how each is tracking.", []),
        ("get_error_budget", "Show how much error budget a service has left this period.",
         ["service"]),
        ("list_users", "List the people who have access to this account.", []),
        ("create_webhook", "Register a URL to be called when an event happens.", ["url"]),
        ("get_audit_log", "Show who changed what in this account, and when.", ["since"]),
        ("get_usage_summary", "Show how much of the plan this account has used.", []),
        ("list_api_keys", "List the API credentials issued for this account.", []),
    ],
    "skyhost": [
        ("list_virtual_machines", "List the virtual machines and whether each is running.", []),
        ("start_virtual_machine", "Power on a stopped virtual machine.", ["machine_id"]),
        ("stop_virtual_machine", "Shut down a running virtual machine.", ["machine_id"]),
        ("resize_virtual_machine", "Give a virtual machine more or less CPU and memory.",
         ["machine_id", "sku_name"]),
        ("list_storage_accounts", "List the object storage accounts.", []),
        ("get_storage_account_details",
         "Show the size, redundancy and access tier of one storage account.",
         ["account_name", "resource_group"]),
        ("create_storage_bucket", "Create a bucket for storing files.", ["bucket_name"]),
        ("list_databases", "List the managed databases and their sizes.", []),
        ("restore_database_backup", "Bring a managed database back to how it was at an earlier time.",
         ["database_id", "point_in_time"]),
        ("get_advisor_recommendations",
         "List suggestions for improving security, reliability and performance.", []),
        ("list_dns_records", "List the DNS records in a hosted zone.", ["zone"]),
        ("update_dns_record", "Point a hostname at a different address.",
         ["zone", "hostname", "ttl_seconds"]),
        ("list_kubernetes_clusters", "List the managed Kubernetes clusters.", []),
        ("scale_node_pool", "Add or remove worker machines in a Kubernetes cluster.",
         ["cluster", "node_count"]),
        ("list_principals", "List the people who have access to this account.", []),
        ("add_event_hook", "Register a URL to be called when an event happens.", ["url"]),
        ("get_activity_log", "Show who changed what in this account, and when.", ["since"]),
        ("get_cost_summary", "Show how much of the plan this account has used.", []),
        ("list_service_keys", "List the API credentials issued for this account.", []),
    ],
    "tasktrack": [
        ("create_task", "Add a piece of work to a list.", ["title", "list_id"]),
        ("update_task_status", "Move a piece of work to to-do, in progress or done.",
         ["task_id", "status"]),
        ("assign_task", "Give a piece of work to a person.", ["task_id", "assignee_id"]),
        ("list_tasks_in_sprint", "List the work still open in the current iteration.",
         ["sprint_id"]),
        ("get_workspace_hierarchy",
         _long("Show the spaces, folders and lists in the workspace.",
               "Use this first to find out which team owns a folder."),
         []),
        ("create_sprint", "Start a new iteration with a start and end date.", ["name"]),
        ("close_sprint", "End the current iteration and roll unfinished work over.", ["sprint_id"]),
        ("add_task_comment", "Leave a note on a piece of work.", ["task_id", "text"]),
        ("list_overdue_tasks", "List the work that has gone past its due date.", []),
        ("link_task_dependency", "Record that one piece of work cannot start until another ends.",
         ["task_id", "blocked_by"]),
        ("get_burndown_chart", "Show remaining work against time for an iteration.",
         ["sprint_id"]),
        ("log_time_entry", "Record time spent on a piece of work.", ["task_id", "billable_hours"]),
        ("create_checklist", "Add a list of sub-steps to a piece of work.", ["task_id"]),
        ("move_task_to_list", "Move a piece of work from one list to another.",
         ["task_id", "list_id"]),
        ("list_members", "List the people who have access to this account.", []),
        ("register_webhook", "Register a URL to be called when an event happens.", ["url"]),
        ("list_audit_events", "Show who changed what in this account, and when.", ["since"]),
        ("get_plan_usage", "Show how much of the plan this account has used.", []),
        ("list_access_tokens", "List the API credentials issued for this account.", []),
    ],
}


def mcp_tools() -> list[ToolDefinition]:
    """The 57 MCP tools, converted the way discovery converts them.

    Raises:
        ImportError: If the optional ``mcp`` package is not installed.
    """
    from agent_gantry.adapters.executors.mcp_client import MCPClient
    from agent_gantry.schema.config import MCPServerConfig

    tools: list[ToolDefinition] = []
    for server, specs in MCP_TOOLS.items():
        client = MCPClient(MCPServerConfig(name=server, url=f"https://{server}.example/mcp"))
        for name, description, parameters in specs:
            tools.append(
                client._convert_tool(
                    SimpleNamespace(
                        name=name,
                        description=description,
                        inputSchema=_schema({p: _STRING for p in parameters}),
                    )
                )
            )
    return tools


def _case(query: str, *expected: str, kind: str = "paraphrase") -> RetrievalCase:
    return RetrievalCase(query, expected, kind=kind)


def _decline(query: str) -> RetrievalCase:
    return RetrievalCase(query, expect_decline=True, kind="decline")


TOOL_CASES: list[RetrievalCase] = [
    _case("What is 17 plus 25?", "default.add_numbers"),
    _case("Spell this backwards", "default.reverse_string"),
    _case("How many pounds is 80 dollars?", "default.convert_currency"),
    _case("Will it rain in Leeds on Saturday?", "default.get_weather_forecast"),
    _case("Put this paragraph into French", "default.translate_text"),
    _case("Give me the gist of this long article", "default.summarise_text"),
    _case("Text my landlord that I'll be late", "default.send_sms"),
    _case("Make me a strong login secret", "default.generate_password"),
    _case("Pull the words out of this scanned contract", "default.extract_text_from_pdf"),
    _case("Where on the map is 10 Downing Street?", "default.geocode_address"),
    _case("What will a 200k mortgage cost me each month?", "default.calculate_loan_repayment"),
    _case("Bill Acme for last month's consulting", "billing.create_invoice"),
    _case("Give the customer their money back for order 1182", "billing.create_refund"),
    _case("Find the person called Priya in our address book", "crm.search_contact"),
    _case("Stop the monthly plan on this account",
          "billing.delete_subscription", "billing.archive_subscription"),
    _case("Log a new complaint from a customer", "support.create_ticket"),
    _case("Reserve the boardroom for Tuesday at ten", "calendar.create_room_booking"),
    _case("How many days off has Sam asked for?",
          "hr.list_leave_request", "hr.search_leave_request"),
    _case("Dump all our stock items to a spreadsheet", "inventory.export_product"),
    _case("Open a change for review on the web repo", "dev.create_pull_request"),
    _case("Show the versions we have shipped", "dev.list_release"),
    _case("Set up a money-off voucher for the spring sale", "marketing.create_coupon"),
    _case("Which laptops have we handed out to staff?", "it.list_device", "it.search_device"),
    _case("File my travel costs for approval", "finance.create_expense_report"),
    _case("Why does my pod keep crashlooping?", "default.get_container_restarts"),
    _case("Stop the disk alert paging me for an hour", "default.mute_monitor"),
    _case("Spin the web server back up", "default.start_virtual_machine"),
    _case("Roll the orders database back to yesterday", "default.restore_database_backup"),
    _case("What is still to do before the iteration ends?", "default.list_tasks_in_sprint"),
    _case("Which jobs are past their deadline?", "default.list_overdue_tasks"),
    _case("get_storage_account_details for prod", "default.get_storage_account_details",
          kind="identifier"),
    _case("get workspace hierarchy", "default.get_workspace_hierarchy", kind="identifier"),
    _case("restore_database_backup", "default.restore_database_backup", kind="identifier"),
    _case("export purchase order", "inventory.export_purchase_order", kind="identifier"),
    _case("list kubernetes clusters", "default.list_kubernetes_clusters", kind="identifier"),
    _case("declare_incident now", "default.declare_incident", kind="identifier"),
    _case("update dns record for www", "default.update_dns_record", kind="identifier"),
    _case("get burndown chart", "default.get_burndown_chart", kind="identifier"),
    _case("run a facet_query for service:checkout env:prod", "default.search_logs",
          kind="parameter"),
    _case("set ttl_seconds to 300", "default.update_dns_record", kind="parameter"),
    _case("raise node_count to 5", "default.scale_node_pool", kind="parameter"),
    _case("what is this blocked_by?", "default.link_task_dependency", kind="parameter"),
    _case("put down 3 billable_hours", "default.log_time_entry", kind="parameter"),
    _case("use sku_name D4 for the box", "default.resize_virtual_machine", kind="parameter"),
    _case("who has access in tasktrack?", "default.list_members", kind="server"),
    _case("logwatch API credentials", "default.list_api_keys", kind="server"),
    _case("who changed what in skyhost last week?", "default.get_activity_log", kind="server"),
    _case("how much of our tasktrack plan have we used?", "default.get_plan_usage",
          kind="server"),
    _case("register a URL for events in logwatch", "default.create_webhook", kind="server"),
    _case("payment gateway timeouts at checkout", "default.search_logs", kind="tail"),
    _case("which team owns this folder?", "default.get_workspace_hierarchy", kind="tail"),
    _decline("Book me a flight to Tokyo"),
    _decline("Order a large pepperoni pizza"),
    _decline("Play some jazz in the kitchen"),
    _decline("What is the capital of Australia?"),
    _decline("Recommend a film for tonight"),
    _decline("Turn the living room lights off"),
    _decline("Write me a poem about autumn"),
    _decline("Is this mole something to worry about?"),
]


# --------------------------------------------------------------------------- #
# Skills
# --------------------------------------------------------------------------- #

#: (name, description, a detail that appears only in the body)
_SKILLS: list[tuple[str, str, str]] = [
    ("writing_release_notes", "How to write release notes for a new version of a library.",
     "Group entries under Added, Changed and Fixed."),
    ("incident_postmortem", "Template and process for a blameless postmortem after an outage.",
     "Include a timeline in UTC and ask why five times."),
    ("log_search_syntax", "Logwatch query syntax reference.",
     "Filter logs by service with service:checkout and by environment with env:prod."),
    ("sql_query_review", "Checklist for reviewing a SQL query before it runs in production.",
     "Look for missing indexes and run EXPLAIN ANALYZE first."),
    ("customer_refund_policy", "Rules for when a customer refund can be approved.",
     "Anything over 500 pounds needs a manager to sign it off."),
    ("onboarding_new_engineer", "Steps for onboarding a new engineer in their first week.",
     "Order the laptop ten days ahead and add them to the shadow rota."),
    ("api_error_handling", "Best practice for handling failures from third-party services.",
     "When an API returns 429 or 503, retry with exponential backoff and jitter."),
    ("git_branching", "The team's git branching and pull request conventions.",
     "Squash on merge; branch names start with feat/ or fix/."),
    ("expense_claims", "How to submit an expense claim and what is reimbursable.",
     "Mileage is paid at 45p per mile."),
    ("secret_rotation", "Procedure for rotating service credentials and secrets.",
     "Rotate every 90 days and revoke the old key after a day."),
    ("meeting_notes", "Template for taking meeting notes with actions and owners.",
     "Write each action as a verb with a name and a date."),
    ("accessibility_review", "Guidelines for checking a web page for accessibility problems.",
     "Colour contrast must be at least 4.5 to 1."),
    ("data_retention", "Guideline on how long customer data may be kept.",
     "Delete inactive accounts after 24 months."),
    ("code_review_etiquette", "How to give and receive code review feedback.",
     "Comment on the code, never the author."),
    ("kubernetes_rollback", "Procedure for rolling back a bad Kubernetes deployment.",
     "Run kubectl rollout undo and watch the readiness probes."),
    ("pricing_quotes", "How to prepare a price quote for an enterprise customer.",
     "A discount above 15 percent needs finance approval."),
    ("on_call_handover", "Checklist for handing over the on-call rota at the end of a shift.",
     "List open pages and anything you silenced."),
    ("unit_test_style", "Conventions for writing unit tests in Python.",
     "Use pytest fixtures and name each test after the behaviour."),
    ("subject_access_request", "Workflow for answering a data subject access request.",
     "Reply within one calendar month."),
    ("capacity_planning", "Pattern for forecasting infrastructure capacity needs.",
     "Plan for 30 percent headroom above peak."),
]


def skills() -> list[Skill]:
    """Twenty skills whose bodies say more than their descriptions do."""
    return [
        Skill(
            name=name,
            description=description,
            content=f"# {name.replace('_', ' ').title()}\n\n{description}\n\n" + _PADDING * 4 + body,
            category=SkillCategory.HOW_TO,
        )
        for name, description, body in _SKILLS
    ]


SKILL_CASES: list[RetrievalCase] = [
    _case("The site went down last night, how do I write it up?", "default.incident_postmortem"),
    _case("A client wants their money back", "default.customer_refund_policy"),
    _case("What should I check before running someone's query on the live database?",
          "default.sql_query_review"),
    _case("A new starter joins on Monday", "default.onboarding_new_engineer"),
    _case("How do I claim back my train fare?", "default.expense_claims"),
    _case("Our deploy broke production, how do I undo it?", "default.kubernetes_rollback"),
    _case("Someone asked for a copy of everything we hold on them",
          "default.subject_access_request"),
    _case("What goes in the changelog for 2.0?", "default.writing_release_notes"),
    _case("how do I filter logs by service?", "default.log_search_syntax", kind="body"),
    _case("what do I do when an API returns 429?", "default.api_error_handling", kind="body"),
    _case("how much is mileage paid per mile?", "default.expense_claims", kind="body"),
    _case("what colour contrast do we need?", "default.accessibility_review", kind="body"),
    _decline("What is the best pizza topping?"),
    _decline("How do I bake sourdough bread?"),
]


# --------------------------------------------------------------------------- #
# MCP servers
# --------------------------------------------------------------------------- #

#: name -> description. The three that own ``MCP_TOOLS`` are described as
#: vaguely as real registrations often are.
MCP_SERVERS: dict[str, str] = {
    "logwatch": "Observability platform.",
    "skyhost": "Cloud hosting provider.",
    "tasktrack": "Work management for teams.",
    "codehost": "Source code hosting: repositories, pull requests, issues and code review.",
    "mailroom": "Send and search email, and manage mailboxes and labels.",
    "calendarly": "Calendars: create events, find free slots and invite attendees.",
    "paydesk": "Payments: charge cards, issue refunds, and manage payouts and disputes.",
    "docstore": "Document storage: upload, share and search files and folders.",
    "chatline": "Team chat: post messages to channels, read threads and set reminders.",
    "mapkit": "Maps and geocoding: directions, distances and place search.",
    "sqlgate": "Run read-only SQL against the analytics warehouse and describe its tables.",
    "designboard": "Design files: read frames, export assets and comment on mock-ups.",
}

MCP_SERVER_CASES: list[RetrievalCase] = [
    _case("Charge the customer's card again", "default.paydesk"),
    _case("Find a time when everyone is free next week", "default.calendarly"),
    _case("Post an update in the team channel", "default.chatline"),
    _case("How far is it from Leeds to York?", "default.mapkit"),
    _case("How many orders did we take last quarter?", "default.sqlgate"),
    _case("Review the open change on the API repo", "default.codehost"),
    _case("Export the logo from the mock-up", "default.designboard"),
    _case("Share the contract with legal", "default.docstore"),
    _case("Why does my pod keep crashlooping?", "default.logwatch", kind="derived"),
    _case("Roll the orders database back to yesterday", "default.skyhost", kind="derived"),
    _case("What is still to do before the iteration ends?", "default.tasktrack", kind="derived"),
    _decline("Order a large pepperoni pizza"),
    _decline("Play some jazz in the kitchen"),
]
