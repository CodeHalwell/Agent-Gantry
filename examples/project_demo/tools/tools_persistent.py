"""
Persistent tool registry for massive toolsets.

This module demonstrates how to handle 200+ tools efficiently by:
1. Using LanceDB for persistent vector storage (embeddings saved to disk)
2. Using lazy imports for heavy dependencies (only loaded when tool is executed)
3. Separating one-time sync from runtime retrieval

Usage:
    # First time: Run sync to embed all tools (do this once)
    python -m examples.project_demo.tools.tools_persistent --sync

    # Runtime: Tools are retrieved from persistent storage (fast!)
    from examples.project_demo.tools.tools_persistent import tools
"""

from __future__ import annotations

import asyncio
import datetime
import hashlib
import json
import math
import os
import pathlib
import platform
import random
import re
import shutil
import socket
import statistics
import time
import uuid
from typing import Any

from agent_gantry import AgentGantry
from agent_gantry.adapters.vector_stores.lancedb import LanceDBVectorStore
from agent_gantry.schema.config import AgentGantryConfig, EmbedderConfig, VectorStoreConfig

# =============================================================================
# LAZY IMPORT HELPERS - Heavy dependencies loaded only when needed
# =============================================================================


def _lazy_import(module_name: str, package: str | None = None) -> Any:
    """Lazily import a module only when first accessed."""
    import importlib
    return importlib.import_module(module_name, package)


def _get_rdkit():
    """Lazy load RDKit (chemistry library - very memory intensive)."""
    from rdkit import Chem
    from rdkit.Chem import Descriptors
    return Chem, Descriptors


def _get_pubchempy():
    """Lazy load PubChemPy."""
    import pubchempy as pcp
    return pcp


def _get_pint():
    """Lazy load Pint unit conversion."""
    import pint
    return pint


def _get_sympy():
    """Lazy load SymPy for symbolic math."""
    from sympy import solve, symbols, sympify
    return solve, symbols, sympify


def _get_requests():
    """Lazy load requests."""
    import requests
    return requests


def _get_pandas():
    """Lazy load pandas."""
    import pandas as pd
    return pd


def _get_numpy():
    """Lazy load numpy."""
    import numpy as np
    return np


def _get_sklearn():
    """Lazy load scikit-learn."""
    import sklearn
    return sklearn


def _get_pillow():
    """Lazy load Pillow."""
    from PIL import Image
    return Image


def _get_matplotlib():
    """Lazy load matplotlib."""
    import matplotlib.pyplot as plt
    return plt


def _get_psutil():
    """Lazy load psutil."""
    import psutil
    return psutil


# =============================================================================
# PERSISTENT GANTRY CONFIGURATION
# =============================================================================

# Database path for persistent storage
DB_PATH = pathlib.Path(__file__).parent / ".tool_cache" / "lancedb"


def create_persistent_gantry(dimension: int = 256) -> AgentGantry:
    """
    Create an AgentGantry with persistent LanceDB storage.
    
    Embeddings are stored on disk, so tools only need to be embedded once.
    Subsequent runs load from disk instantly.
    
    Args:
        dimension: Embedding dimension (256 for Nomic with Matryoshka)
        
    Returns:
        AgentGantry configured with persistent storage
    """
    import warnings

    # Try to use NomicEmbedder for quality semantic search
    try:
        import sentence_transformers  # noqa: F401

        from agent_gantry.adapters.embedders.nomic import NomicEmbedder
        embedder = NomicEmbedder(dimension=dimension)
        embedder_config = EmbedderConfig(type="nomic", dimension=dimension)
    except ImportError:
        warnings.warn(
            "Nomic embedder not available. Using SimpleEmbedder. "
            "For better semantic search: pip install agent-gantry[nomic]",
            UserWarning,
            stacklevel=2,
        )
        from agent_gantry.adapters.embedders.simple import SimpleEmbedder
        embedder = SimpleEmbedder()
        embedder_config = EmbedderConfig(type="sentence_transformers")

    # Configure persistent vector store
    vector_store_config = VectorStoreConfig(
        type="lancedb",
        db_path=str(DB_PATH),
        collection_name="project_demo_tools",
        dimension=dimension,
    )

    # Create config with auto_sync disabled for explicit control
    config = AgentGantryConfig(
        vector_store=vector_store_config,
        embedder=embedder_config,
        auto_sync=False,  # We control when to sync
    )

    # Create vector store
    vector_store = LanceDBVectorStore(
        db_path=str(DB_PATH),
        tools_table="project_demo_tools",
        dimension=dimension,
    )

    return AgentGantry(
        config=config,
        vector_store=vector_store,
        embedder=embedder,
    )


# Create the global instance
tools = create_persistent_gantry()


# =============================================================================
# TOOL DEFINITIONS - Using lazy imports for heavy dependencies
# =============================================================================

# --- Chemistry Tools (Heavy - uses RDKit and PubChem) ---

@tools.register(tags=["chemistry", "molecular"])
def get_molecular_weight(smiles: str) -> float:
    """Calculate the molecular weight of a compound given its SMILES representation."""
    Chem, Descriptors = _get_rdkit()
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("Invalid SMILES string.")
    return Descriptors.MolWt(mol)


@tools.register(tags=["chemistry", "compound"])
def get_compound_info(name: str) -> dict[str, Any]:
    """Fetch compound information from PubChem given its name."""
    pcp = _get_pubchempy()
    compounds = pcp.get_compounds(name, 'name')
    if not compounds:
        raise ValueError(f"No compound found for name: {name}")
    compound = compounds[0]
    return {
        "molecular_formula": compound.molecular_formula,
        "molecular_weight": compound.molecular_weight,
        "iupac_name": compound.iupac_name,
        "synonyms": compound.synonyms,
    }


@tools.register(tags=["chemistry"])
def get_smiles_from_name(name: str) -> str:
    """Get the SMILES string for a compound name using PubChem."""
    pcp = _get_pubchempy()
    compounds = pcp.get_compounds(name, 'name')
    if not compounds:
        raise ValueError(f"No compound found for name: {name}")
    return compounds[0].isomeric_smiles


@tools.register(tags=["chemistry"])
def calculate_logp(smiles: str) -> float:
    """Calculate the Octanol-Water Partition Coefficient (LogP) from SMILES."""
    Chem, Descriptors = _get_rdkit()
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("Invalid SMILES string.")
    return Descriptors.MolLogP(mol)


@tools.register(tags=["chemistry"])
def is_valid_smiles(smiles: str) -> bool:
    """Check if a SMILES string is valid."""
    Chem, _ = _get_rdkit()
    return Chem.MolFromSmiles(smiles) is not None


# --- Unit Conversion (uses Pint) ---

@tools.register(tags=["unit_conversion"])
def convert_units(value: float, from_unit: str, to_unit: str) -> float:
    """Convert a value from one unit to another."""
    pint = _get_pint()
    ureg = pint.UnitRegistry()
    quantity = value * ureg(from_unit)
    converted = quantity.to(to_unit)
    return converted.magnitude


# --- Math & Algebra (uses SymPy) ---

@tools.register(tags=["math", "algebra"])
def solve_equation(equation: str, variable: str) -> Any:
    """Solve a simple algebraic equation for the given variable."""
    solve, symbols, sympify = _get_sympy()
    var = symbols(variable)
    expr = sympify(equation)
    solution = solve(expr, var)
    return solution


# --- Date & Time Tools (stdlib only - lightweight) ---

@tools.register(tags=["date_calculation"])
def calculate_date_difference(date1: str, date2: str) -> int:
    """Calculate the difference in days between two dates (YYYY-MM-DD)."""
    d1 = datetime.datetime.strptime(date1, "%Y-%m-%d")
    d2 = datetime.datetime.strptime(date2, "%Y-%m-%d")
    return abs((d2 - d1).days)


@tools.register(tags=["datetime"])
def get_current_utc_time() -> str:
    """Get the current UTC time as an ISO formatted string."""
    return datetime.datetime.now(datetime.timezone.utc).isoformat().replace("+00:00", "Z")


@tools.register(tags=["time"])
def get_current_timestamp() -> float:
    """Get the current Unix timestamp."""
    return time.time()


@tools.register(tags=["time"])
def format_timestamp(timestamp: float, format_str: str = "%Y-%m-%d %H:%M:%S") -> str:
    """Format a Unix timestamp into a human-readable string."""
    return datetime.datetime.fromtimestamp(timestamp).strftime(format_str)


@tools.register(tags=["time"])
def get_days_until(target_date: str) -> int:
    """Calculate the number of days from today until the target date (YYYY-MM-DD)."""
    today = datetime.date.today()
    target = datetime.datetime.strptime(target_date, "%Y-%m-%d").date()
    return (target - today).days


@tools.register(tags=["time"])
def is_leap_year(year: int) -> bool:
    """Check if a year is a leap year."""
    import calendar
    return calendar.isleap(year)


@tools.register(tags=["time"])
def get_weekday(date_str: str) -> str:
    """Get the day of the week for a given date (YYYY-MM-DD)."""
    d = datetime.datetime.strptime(date_str, "%Y-%m-%d")
    return d.strftime("%A")


# --- File System Tools (stdlib only) ---

@tools.register(tags=["fs", "file"])
def list_directory(path: str = ".") -> list[str]:
    """List the contents of a directory."""
    return os.listdir(path)


@tools.register(tags=["fs", "file"])
def read_text_file(path: str) -> str:
    """Read the contents of a text file."""
    with open(path, encoding='utf-8') as f:
        return f.read()


@tools.register(tags=["fs", "file"])
def write_text_file(path: str, content: str) -> str:
    """Write content to a text file."""
    with open(path, 'w', encoding='utf-8') as f:
        f.write(content)
    return f"File written to {path}"


@tools.register(tags=["fs", "file"])
def file_exists(path: str) -> bool:
    """Check if a file or directory exists."""
    return os.path.exists(path)


@tools.register(tags=["fs", "file"])
def get_file_size(path: str) -> int:
    """Get the size of a file in bytes."""
    return os.path.getsize(path)


@tools.register(tags=["fs", "search"])
def search_files(pattern: str, root_dir: str = ".") -> list[str]:
    """Search for files matching a glob pattern."""
    return [str(p) for p in pathlib.Path(root_dir).rglob(pattern)]


# --- Math & Statistics Tools (stdlib) ---

@tools.register(tags=["math", "stats"])
def calculate_mean(numbers: list[float]) -> float:
    """Calculate the arithmetic mean of a list of numbers."""
    return statistics.mean(numbers)


@tools.register(tags=["math", "stats"])
def calculate_median(numbers: list[float]) -> float:
    """Calculate the median of a list of numbers."""
    return statistics.median(numbers)


@tools.register(tags=["math", "stats"])
def calculate_stdev(numbers: list[float]) -> float:
    """Calculate the standard deviation of a list of numbers."""
    return statistics.stdev(numbers)


@tools.register(tags=["math"])
def calculate_factorial(n: int) -> int:
    """Calculate the factorial of a non-negative integer."""
    return math.factorial(n)


@tools.register(tags=["math"])
def is_prime(n: int) -> bool:
    """Check if a number is prime."""
    if n < 2:
        return False
    for i in range(2, int(math.sqrt(n)) + 1):
        if n % i == 0:
            return False
    return True


@tools.register(tags=["math", "random"])
def get_random_int(min_val: int, max_val: int) -> int:
    """Generate a random integer between min_val and max_val (inclusive)."""
    return random.randint(min_val, max_val)


@tools.register(tags=["math"])
def calculate_percentage(part: float, whole: float) -> float:
    """Calculate what percentage part is of whole."""
    if whole == 0:
        return 0.0
    return (part / whole) * 100


@tools.register(tags=["math", "geometry"])
def calculate_triangle_area(base: float, height: float) -> float:
    """Calculate the area of a triangle."""
    return 0.5 * base * height


@tools.register(tags=["math", "geometry"])
def calculate_sphere_volume(radius: float) -> float:
    """Calculate the volume of a sphere."""
    return (4 / 3) * math.pi * (radius ** 3)


@tools.register(tags=["math", "geometry"])
def calculate_distance_2d(x1: float, y1: float, x2: float, y2: float) -> float:
    """Calculate the Euclidean distance between two points in 2D space."""
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


@tools.register(tags=["math"])
def get_gcd(a: int, b: int) -> int:
    """Calculate the Greatest Common Divisor of two numbers."""
    return math.gcd(a, b)


@tools.register(tags=["math"])
def get_lcm(a: int, b: int) -> int:
    """Calculate the Least Common Multiple of two numbers."""
    if a == 0 or b == 0:
        return 0
    return abs(a * b) // math.gcd(a, b)


# --- Text Tools (stdlib) ---

@tools.register(tags=["text", "regex"])
def regex_search(pattern: str, text: str) -> list[str]:
    """Search for all occurrences of a regex pattern in text."""
    return re.findall(pattern, text)


@tools.register(tags=["text", "regex"])
def regex_replace(pattern: str, replacement: str, text: str) -> str:
    """Replace occurrences of a regex pattern in text."""
    return re.sub(pattern, replacement, text)


@tools.register(tags=["text"])
def count_words(text: str) -> int:
    """Count the number of words in a string."""
    return len(text.split())


@tools.register(tags=["text"])
def count_characters(text: str, include_whitespace: bool = True) -> int:
    """Count the number of characters in a string."""
    if include_whitespace:
        return len(text)
    return len(text.replace(" ", "").replace("\n", "").replace("\t", ""))


@tools.register(tags=["text"])
def reverse_string(text: str) -> str:
    """Reverse a string."""
    return text[::-1]


@tools.register(tags=["text"])
def extract_emails(text: str) -> list[str]:
    """Extract all email addresses from text."""
    email_pattern = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
    return re.findall(email_pattern, text)


@tools.register(tags=["text"])
def extract_urls(text: str) -> list[str]:
    """Extract all URLs from text."""
    url_pattern = r'https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+'
    return re.findall(url_pattern, text)


@tools.register(tags=["text"])
def strip_html_tags(html: str) -> str:
    """Remove HTML tags from a string."""
    return re.sub(r'<[^>]*>', '', html)


@tools.register(tags=["text"])
def is_palindrome(text: str) -> bool:
    """Check if a string is a palindrome."""
    clean_text = re.sub(r'[^a-zA-Z0-9]', '', text).lower()
    return clean_text == clean_text[::-1]


@tools.register(tags=["text"])
def camel_to_snake(text: str) -> str:
    """Convert CamelCase to snake_case."""
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', text)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()


@tools.register(tags=["text"])
def snake_to_camel(text: str) -> str:
    """Convert snake_case to CamelCase."""
    return ''.join(word.title() for word in text.split('_'))


# --- Data & Serialization Tools (stdlib) ---

@tools.register(tags=["data", "json"])
def json_to_dict(json_str: str) -> dict[str, Any]:
    """Convert a JSON string to a dictionary."""
    return json.loads(json_str)


@tools.register(tags=["data", "json"])
def dict_to_json(data: dict[str, Any], indent: int = 4) -> str:
    """Convert a dictionary to a JSON string."""
    return json.dumps(data, indent=indent)


@tools.register(tags=["data", "base64"])
def base64_encode(text: str) -> str:
    """Encode a string to Base64."""
    import base64
    return base64.b64encode(text.encode()).decode()


@tools.register(tags=["data", "base64"])
def base64_decode(encoded_str: str) -> str:
    """Decode a Base64 string."""
    import base64
    return base64.b64decode(encoded_str.encode()).decode()


@tools.register(tags=["data", "uuid"])
def generate_uuid() -> str:
    """Generate a random UUID (v4)."""
    return str(uuid.uuid4())


@tools.register(tags=["data", "hash"])
def get_hash(text: str, algorithm: str = "sha256") -> str:
    """Calculate the hash of a string."""
    h = hashlib.new(algorithm)
    h.update(text.encode())
    return h.hexdigest()


# --- Network Tools (stdlib) ---

@tools.register(tags=["network", "dns"])
def resolve_dns(hostname: str) -> str:
    """Resolve a hostname to an IP address."""
    return socket.gethostbyname(hostname)


@tools.register(tags=["network", "ip"])
def get_local_ip() -> str:
    """Get the local IP address of the machine."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(('10.255.255.255', 1))
        ip = s.getsockname()[0]
    except Exception:
        ip = '127.0.0.1'
    finally:
        s.close()
    return ip


@tools.register(tags=["network", "port"])
def is_port_open(host: str, port: int, timeout: float = 1.0) -> bool:
    """Check if a specific port is open on a host."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(timeout)
        return s.connect_ex((host, port)) == 0


@tools.register(tags=["network"])
def is_ipv4(ip: str) -> bool:
    """Check if a string is a valid IPv4 address."""
    try:
        socket.inet_aton(ip)
        return True
    except OSError:
        return False


# --- Security Tools (stdlib) ---

@tools.register(tags=["security"])
def generate_password(length: int = 12, include_special: bool = True) -> str:
    """Generate a random secure password."""
    import secrets
    import string
    alphabet = string.ascii_letters + string.digits
    if include_special:
        alphabet += string.punctuation
    return ''.join(secrets.choice(alphabet) for _ in range(length))


@tools.register(tags=["security"])
def is_strong_password(password: str) -> bool:
    """Check if a password meets basic strength requirements."""
    if len(password) < 8:
        return False
    if not any(c.isupper() for c in password):
        return False
    if not any(c.islower() for c in password):
        return False
    if not any(c.isdigit() for c in password):
        return False
    return True


@tools.register(tags=["security"])
def mask_sensitive_data(text: str, visible_chars: int = 4) -> str:
    """Mask sensitive data, leaving only the last few characters visible."""
    if len(text) <= visible_chars:
        return "*" * len(text)
    return "*" * (len(text) - visible_chars) + text[-visible_chars:]


# --- System Tools (stdlib) ---

@tools.register(tags=["system", "os"])
def get_os_name() -> str:
    """Get the name of the operating system."""
    return os.name


@tools.register(tags=["system", "os"])
def get_platform_info() -> str:
    """Get detailed platform information."""
    return platform.platform()


@tools.register(tags=["system", "cpu"])
def get_cpu_count() -> int:
    """Get the number of logical CPUs in the system."""
    return os.cpu_count() or 0


@tools.register(tags=["system", "os"])
def get_current_working_directory() -> str:
    """Get the current working directory."""
    return os.getcwd()


@tools.register(tags=["system", "network"])
def get_hostname() -> str:
    """Get the hostname of the machine."""
    return socket.gethostname()


@tools.register(tags=["system"])
def get_process_id() -> int:
    """Get the current process ID."""
    return os.getpid()


@tools.register(tags=["system"])
def get_disk_usage(path: str = ".") -> dict[str, int]:
    """Get disk usage statistics for a path."""
    usage = shutil.disk_usage(path)
    return {
        "total": usage.total,
        "used": usage.used,
        "free": usage.free
    }


@tools.register(tags=["misc", "system"])
def get_python_version() -> str:
    """Get the current Python version."""
    return platform.python_version()


# --- Web Tools (lazy loading requests) ---

@tools.register(tags=["web"])
def fetch_web_content(url: str) -> str:
    """Fetch the content of a web page given its URL."""
    requests = _get_requests()
    try:
        max_bytes = 10_000_000  # 10MB limit
        with requests.get(
            url,
            timeout=30,
            headers={"User-Agent": "Agent-Gantry/0.1.0"},
            stream=True,
        ) as response:
            response.raise_for_status()
            content_length = response.headers.get("content-length")
            if content_length is not None:
                if int(content_length) > max_bytes:
                    raise ValueError(f"Content too large: {content_length} bytes")
            chunks: list[bytes] = []
            bytes_read = 0
            for chunk in response.iter_content(chunk_size=8192):
                if not chunk:
                    continue
                chunks.append(chunk)
                bytes_read += len(chunk)
                if bytes_read > max_bytes:
                    raise ValueError(f"Content too large: {bytes_read} bytes")
            encoding = response.encoding or "utf-8"
            return b"".join(chunks).decode(encoding, errors="replace")
    except Exception as e:
        raise ValueError(f"Error fetching {url}: {e}")


@tools.register(tags=["network", "http"])
def http_get_status(url: str) -> int:
    """Get the HTTP status code of a URL."""
    requests = _get_requests()
    response = requests.get(url, timeout=10)
    return response.status_code


@tools.register(tags=["web", "http"])
def get_http_headers(url: str) -> dict[str, str]:
    """Get the HTTP headers of a URL."""
    requests = _get_requests()
    response = requests.head(url, timeout=10)
    return dict(response.headers)


# --- URL Tools (stdlib) ---

@tools.register(tags=["network", "url"])
def url_encode(text: str) -> str:
    """URL-encode a string."""
    import urllib.parse
    return urllib.parse.quote(text)


@tools.register(tags=["network", "url"])
def url_decode(text: str) -> str:
    """URL-decode a string."""
    import urllib.parse
    return urllib.parse.unquote(text)


@tools.register(tags=["web"])
def get_domain_from_url(url: str) -> str:
    """Extract the domain name from a URL."""
    from urllib.parse import urlparse
    return urlparse(url).netloc


@tools.register(tags=["web"])
def is_valid_url(url: str) -> bool:
    """Check if a string is a valid URL."""
    from urllib.parse import urlparse
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except ValueError:
        return False


# --- Misc Tools (stdlib) ---

@tools.register(tags=["misc"])
def get_random_color() -> str:
    """Generate a random hex color code."""
    return f"#{random.randint(0, 0xFFFFFF):06x}"


@tools.register(tags=["misc", "math"])
def convert_temperature(value: float, from_unit: str, to_unit: str) -> float:
    """Convert temperature between Celsius, Fahrenheit, and Kelvin."""
    from_unit = from_unit.upper()[0]
    to_unit = to_unit.upper()[0]
    if from_unit == 'F':
        c = (value - 32) * 5 / 9
    elif from_unit == 'K':
        c = value - 273.15
    else:
        c = value
    if to_unit == 'F':
        return (c * 9 / 5) + 32
    elif to_unit == 'K':
        return c + 273.15
    else:
        return c


@tools.register(tags=["misc", "health"])
def calculate_bmi(weight_kg: float, height_m: float) -> float:
    """Calculate Body Mass Index (BMI)."""
    if height_m == 0:
        return 0.0
    return weight_kg / (height_m ** 2)


@tools.register(tags=["misc", "game"])
def roll_dice(sides: int = 6, count: int = 1) -> list[int]:
    """Roll a specified number of dice with a given number of sides."""
    return [random.randint(1, sides) for _ in range(count)]


@tools.register(tags=["misc", "game"])
def flip_coin() -> str:
    """Flip a coin and return 'Heads' or 'Tails'."""
    return random.choice(["Heads", "Tails"])


@tools.register(tags=["misc"])
def get_random_quote() -> str:
    """Get a random inspirational quote."""
    quotes = [
        "The only way to do great work is to love what you do. - Steve Jobs",
        "Innovation distinguishes between a leader and a follower. - Steve Jobs",
        "Your time is limited, so don't waste it living someone else's life. - Steve Jobs",
        "Stay hungry, stay foolish. - Steve Jobs",
        "The future belongs to those who believe in the beauty of their dreams. - Eleanor Roosevelt"
    ]
    return random.choice(quotes)


@tools.register(tags=["finance"])
def calculate_compound_interest(principal: float, rate: float, time: float, n: int = 1) -> float:
    """Calculate compound interest: A = P(1 + r/n)^(nt)."""
    return principal * (1 + rate / n) ** (n * time)


@tools.register(tags=["finance"])
def calculate_loan_payment(principal: float, annual_rate: float, years: int) -> float:
    """Calculate monthly loan payment."""
    monthly_rate = annual_rate / 12 / 100
    n_payments = years * 12
    if monthly_rate == 0:
        return principal / n_payments
    return (principal * monthly_rate) / (1 - (1 + monthly_rate) ** -n_payments)


@tools.register(tags=["finance"])
def calculate_roi(gain: float, cost: float) -> float:
    """Calculate Return on Investment (ROI) percentage."""
    if cost == 0:
        return 0.0
    return ((gain - cost) / cost) * 100


# --- Conversion Tools (stdlib) ---

@tools.register(tags=["conversion"])
def meters_to_feet(meters: float) -> float:
    """Convert meters to feet."""
    return meters * 3.28084


@tools.register(tags=["conversion"])
def feet_to_meters(feet: float) -> float:
    """Convert feet to meters."""
    return feet / 3.28084


@tools.register(tags=["conversion"])
def kilograms_to_pounds(kg: float) -> float:
    """Convert kilograms to pounds."""
    return kg * 2.20462


@tools.register(tags=["conversion"])
def pounds_to_kilograms(lbs: float) -> float:
    """Convert pounds to kilograms."""
    return lbs / 2.20462


@tools.register(tags=["conversion"])
def km_to_miles(km: float) -> float:
    """Convert kilometers to miles."""
    return km * 0.621371


@tools.register(tags=["conversion"])
def miles_to_km(miles: float) -> float:
    """Convert miles to kilometers."""
    return miles / 0.621371


# --- Data Science Tools (lazy loading) ---

@tools.register(tags=["data", "pandas"])
def create_dataframe_summary(data: list[dict[str, Any]]) -> dict[str, Any]:
    """Create a statistical summary of a list of dictionaries using Pandas."""
    pd = _get_pandas()
    df = pd.DataFrame(data)
    return df.describe().to_dict()


@tools.register(tags=["data", "numpy"])
def calculate_matrix_inverse(matrix: list[list[float]]) -> list[list[float]]:
    """Calculate the inverse of a square matrix using NumPy."""
    np = _get_numpy()
    arr = np.array(matrix)
    inv = np.linalg.inv(arr)
    return inv.tolist()


@tools.register(tags=["data", "numpy"])
def generate_normal_distribution(mean: float, std: float, size: int) -> list[float]:
    """Generate a list of numbers following a normal distribution."""
    np = _get_numpy()
    return np.random.normal(mean, std, size).tolist()


# --- System Monitoring (lazy loading psutil) ---

@tools.register(tags=["system", "psutil"])
def get_cpu_usage_percent(interval: float = 1.0) -> float:
    """Get the current CPU usage percentage."""
    psutil = _get_psutil()
    return psutil.cpu_percent(interval=interval)


@tools.register(tags=["system", "psutil"])
def get_memory_info() -> dict[str, Any]:
    """Get detailed system memory usage statistics."""
    psutil = _get_psutil()
    mem = psutil.virtual_memory()
    return {
        "total": mem.total,
        "available": mem.available,
        "percent": mem.percent,
        "used": mem.used,
        "free": mem.free
    }


# =============================================================================
# SYNC UTILITIES
# =============================================================================


async def sync_tools() -> int:
    """
    Sync all registered tools to the persistent vector store.
    
    This embeds all tools and saves them to disk. Only needs to be run once,
    or when tools are added/modified.
    
    Returns:
        Number of tools synced
    """
    print(f"Syncing {tools.tool_count} tools to {DB_PATH}...")
    count = await tools.sync()
    print(f"Successfully synced {count} tools to persistent storage.")
    return count


async def check_sync_status() -> dict[str, Any]:
    """
    Check if tools are already synced to the persistent store.
    
    Returns:
        Dict with sync status information
    """
    try:
        stored_tools = await tools.list_tools()
        pending = tools.tool_count
        return {
            "stored": len(stored_tools),
            "pending": pending,
            "needs_sync": len(stored_tools) == 0 and pending > 0,
            "db_path": str(DB_PATH),
        }
    except Exception as e:
        return {
            "stored": 0,
            "pending": tools.tool_count,
            "needs_sync": True,
            "db_path": str(DB_PATH),
            "error": str(e),
        }


# =============================================================================
# CLI ENTRY POINT
# =============================================================================


def main():
    """CLI entry point for syncing tools."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Manage persistent tool storage for Agent-Gantry"
    )
    parser.add_argument(
        "--sync",
        action="store_true",
        help="Sync all tools to persistent storage (run this first time)",
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="Check sync status",
    )
    parser.add_argument(
        "--clear",
        action="store_true",
        help="Clear the persistent storage",
    )

    args = parser.parse_args()

    if args.sync:
        print("Initializing embedder and syncing tools...")
        print("This may take a minute on first run (loading embedding model)...")
        count = asyncio.run(sync_tools())
        print(f"\nDone! {count} tools are now persisted and ready for fast retrieval.")

    elif args.status:
        status = asyncio.run(check_sync_status())
        print(f"Database path: {status['db_path']}")
        print(f"Tools in storage: {status['stored']}")
        print(f"Pending tools: {status['pending']}")
        if status.get('needs_sync'):
            print("\n⚠️  Run with --sync to persist tools")
        else:
            print("\n✓ Tools are synced and ready!")

    elif args.clear:
        if DB_PATH.exists():
            shutil.rmtree(DB_PATH)
            print(f"Cleared persistent storage at {DB_PATH}")
        else:
            print("No persistent storage to clear.")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
