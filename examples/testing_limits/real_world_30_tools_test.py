import asyncio
import json
import math
import os
import re
from datetime import datetime, timedelta
from typing import Any

from dotenv import load_dotenv

from agent_gantry import AgentGantry
from agent_gantry.schema.config import AgentGantryConfig, EmbedderConfig
from agent_gantry.schema.execution import ToolCall

# Load environment variables for OPENAI_API_KEY
load_dotenv()

# --- 1. Define 30 Tangible Tools ---

# Domain: Math
def calculate_hypotenuse(a: float, b: float) -> float:
    """Calculate the length of the hypotenuse of a right-angled triangle."""
    return math.sqrt(a**2 + b**2)

def calculate_circle_area(radius: float) -> float:
    """Calculate the area of a circle given its radius."""
    return math.pi * radius ** 2

def calculate_compound_interest(principal: float, rate: float, years: int) -> float:
    """Calculate compound interest."""
    return principal * ((1 + rate) ** years - 1)

def convert_celsius_to_fahrenheit(celsius: float) -> float:
    """Convert a temperature from Celsius to Fahrenheit."""
    return (celsius * 9/5) + 32

def solve_quadratic(a: float, b: float, c: float) -> str:
    """Solve a quadratic equation ax^2 + bx + c = 0."""
    d = b**2 - 4*a*c
    if d < 0:
        return "No real solutions"
    elif d == 0:
        x = -b / (2*a)
        return f"One solution: {x}"
    else:
        x1 = (-b + math.sqrt(d)) / (2*a)
        x2 = (-b - math.sqrt(d)) / (2*a)
        return f"Two solutions: {x1}, {x2}"

# Domain: Text Processing
def reverse_string(text: str) -> str:
    """Reverse the provided text string."""
    return text[::-1]

def count_vowels(text: str) -> int:
    """Count the number of vowels in a string."""
    return sum(1 for char in text.lower() if char in "aeiou")

def to_snake_case(text: str) -> str:
    """Convert a string to snake_case."""
    text = re.sub(r'(?<!^)(?=[A-Z])', '_', text).lower()
    return text.replace(" ", "_")

def extract_emails(text: str) -> list[str]:
    """Extract all email addresses from a text."""
    return re.findall(r'[\w\.-]+@[\w\.-]+', text)

def summarize_text_stats(text: str) -> dict[str, int]:
    """Get statistics about a text (word count, char count)."""
    return {
        "word_count": len(text.split()),
        "char_count": len(text),
        "line_count": len(text.splitlines())
    }

# Domain: Date & Time
def get_current_utc_time() -> str:
    """Get the current UTC time in ISO format."""
    return datetime.utcnow().isoformat()

def days_between_dates(date1: str, date2: str) -> int:
    """Calculate the number of days between two dates (YYYY-MM-DD)."""
    d1 = datetime.strptime(date1, "%Y-%m-%d")
    d2 = datetime.strptime(date2, "%Y-%m-%d")
    return abs((d2 - d1).days)

def get_day_of_week(date_str: str) -> str:
    """Get the day of the week for a given date (YYYY-MM-DD)."""
    d = datetime.strptime(date_str, "%Y-%m-%d")
    return d.strftime("%A")

def add_business_days(start_date: str, days: int) -> str:
    """Add business days (skipping weekends) to a start date."""
    # Simplified implementation
    d = datetime.strptime(start_date, "%Y-%m-%d")
    added = 0
    while added < days:
        d += timedelta(days=1)
        if d.weekday() < 5: # Mon-Fri
            added += 1
    return d.strftime("%Y-%m-%d")

def is_leap_year(year: int) -> bool:
    """Check if a year is a leap year."""
    return (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0)

# Domain: File System (Simulated)
def list_files_in_directory(path: str) -> list[str]:
    """List files in a directory (Simulated)."""
    return [f"{path}/file1.txt", f"{path}/image.png", f"{path}/data.csv"]

def read_file_content(path: str) -> str:
    """Read content of a file (Simulated)."""
    return f"Content of {path}: [Binary Data]"

def get_file_size(path: str) -> str:
    """Get the size of a file (Simulated)."""
    return "1024 KB"

def check_file_exists(path: str) -> bool:
    """Check if a file exists (Simulated)."""
    return True

def get_file_extension(filename: str) -> str:
    """Get the extension of a filename."""
    return os.path.splitext(filename)[1]

# Domain: Network (Simulated)
def ping_host(hostname: str) -> str:
    """Ping a host to check reachability (Simulated)."""
    return f"Reply from {hostname}: bytes=32 time=20ms TTL=54"

def get_ip_address(hostname: str) -> str:
    """Get IP address of a hostname (Simulated)."""
    return "192.168.1.100"

def check_port_open(host: str, port: int) -> bool:
    """Check if a network port is open (Simulated)."""
    return port in [80, 443, 22, 8080]

def validate_url(url: str) -> bool:
    """Validate if a string is a valid URL."""
    regex = re.compile(
        r'^(?:http|ftp)s?://' # http:// or https://
        r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+(?:[A-Z]{2,6}\.?|[A-Z0-9-]{2,}\.?)|' # domain...
        r'localhost|' # localhost...
        r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})' # ...or ip
        r'(?::\d+)?' # optional port
        r'(?:/?|[/?]\S+)$', re.IGNORECASE)
    return re.match(regex, url) is not None

def get_http_status(url: str) -> int:
    """Get HTTP status code for a URL (Simulated)."""
    return 200

# Domain: Data Processing
def sort_list_of_numbers(numbers: list[float]) -> list[float]:
    """Sort a list of numbers in ascending order."""
    if isinstance(numbers, str):
        # Handle case where LLM passes a comma-separated string
        try:
            numbers = [float(n.strip()) for n in numbers.split(',')]
        except ValueError:
            return []
    return sorted(numbers)

def filter_even_numbers(numbers: list[int]) -> list[int]:
    """Filter a list to return only even numbers."""
    return [n for n in numbers if n % 2 == 0]

def calculate_average(numbers: list[float]) -> float:
    """Calculate the average of a list of numbers."""
    if not numbers: return 0.0
    return sum(numbers) / len(numbers)

def find_max_value(numbers: list[float]) -> float:
    """Find the maximum value in a list of numbers."""
    if not numbers: return 0.0
    return max(numbers)

def merge_dictionaries(dict1: dict[str, Any], dict2: dict[str, Any]) -> dict[str, Any]:
    """Merge two dictionaries."""
    return {**dict1, **dict2}


# --- 2. Main Test Logic ---

async def main():
    print("=== Agent-Gantry Real-World 30 Tools Test ===\n")

    # Check for OpenAI API Key
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("⚠️  WARNING: OPENAI_API_KEY not found in environment.")
        print("   The test will perform retrieval but cannot call the actual LLM.")
        print("   Please set OPENAI_API_KEY in .env or environment variables.\n")
    else:
        print("✅ OPENAI_API_KEY found. Will perform actual LLM calls.\n")

    # Initialize Gantry
    print("1. Initializing AgentGantry with Nomic Embedder...")
    try:
        config = AgentGantryConfig(
            embedder=EmbedderConfig(
                type="nomic",
                model="nomic-ai/nomic-embed-text-v1.5"
            )
        )
        gantry = AgentGantry(config=config)
    except ImportError:
        print("   Nomic dependencies missing. Falling back to SimpleEmbedder.")
        gantry = AgentGantry()

    # Register Tools
    print("2. Registering 30 tangible tools...")
    tools_to_register = [
        calculate_hypotenuse, calculate_circle_area, calculate_compound_interest, convert_celsius_to_fahrenheit, solve_quadratic,
        reverse_string, count_vowels, to_snake_case, extract_emails, summarize_text_stats,
        get_current_utc_time, days_between_dates, get_day_of_week, add_business_days, is_leap_year,
        list_files_in_directory, read_file_content, get_file_size, check_file_exists, get_file_extension,
        ping_host, get_ip_address, check_port_open, validate_url, get_http_status,
        sort_list_of_numbers, filter_even_numbers, calculate_average, find_max_value, merge_dictionaries
    ]

    for func in tools_to_register:
        gantry.register(func)

    await gantry.sync()
    print(f"   Registered {gantry.tool_count} tools.\n")

    # Test Cases
    test_queries = [
        ("What is the area of a circle with radius 5?", "calculate_circle_area"),
        ("How many days are there between 2023-01-01 and 2023-12-31?", "days_between_dates"),
        ("Reverse the string 'Agent Gantry is awesome'", "reverse_string"),
        ("Is 2024 a leap year?", "is_leap_year"),
        ("Sort this list of numbers: 5, 2, 9, 1, 100, 42", "sort_list_of_numbers"),
        ("Convert 25 degrees Celsius to Fahrenheit", "convert_celsius_to_fahrenheit"),
        ("Extract emails from: Contact us at support@example.com or sales@example.org", "extract_emails")
    ]

    # Setup OpenAI Client
    client = None
    if api_key:
        from openai import AsyncOpenAI
        client = AsyncOpenAI(api_key=api_key)

    print("3. Running End-to-End Tests (Retrieval -> LLM -> Execution)\n")

    for query, expected_tool in test_queries:
        print(f"🔹 Query: '{query}'")

        # A. Retrieval
        retrieved_tools = await gantry.retrieve_tools(query, limit=3)
        retrieved_names = [t['function']['name'] for t in retrieved_tools]

        print(f"   [Retrieval] Top 3: {retrieved_names}")

        if expected_tool in retrieved_names:
            print(f"   ✅ Retrieval Success: Found '{expected_tool}'")
        else:
            print(f"   ❌ Retrieval Failed: Expected '{expected_tool}'")
            print("-" * 60)
            continue

        # B. LLM Call
        if client:
            print("   [LLM] Calling GPT-4o...")
            try:
                response = await client.chat.completions.create(
                    model="gpt-4o",
                    messages=[{"role": "user", "content": query}],
                    tools=retrieved_tools,
                    tool_choice="auto"
                )

                msg = response.choices[0].message
                tool_calls = msg.tool_calls

                if tool_calls:
                    for tc in tool_calls:
                        fn_name = tc.function.name
                        fn_args = json.loads(tc.function.arguments)
                        print(f"   [LLM] Selected Tool: {fn_name}")
                        print(f"   [LLM] Arguments: {fn_args}")

                        if fn_name == expected_tool:
                            print("   ✅ LLM Selection Success")
                        else:
                            print(f"   ⚠️  LLM selected '{fn_name}' instead of '{expected_tool}' (might be valid)")

                        # C. Execution
                        print("   [Execution] Running tool...")
                        result = await gantry.execute(ToolCall(
                            tool_name=fn_name,
                            arguments=fn_args
                        ))
                        print(f"   [Result] Output: {result.result}")
                else:
                    print("   ❌ LLM did not call any tool.")
                    print(f"   [LLM Response] {msg.content}")

            except Exception as e:
                print(f"   ❌ LLM Call Error: {e}")
        else:
            print("   [LLM] Skipped (No API Key)")

        print("-" * 60)

if __name__ == "__main__":
    asyncio.run(main())
