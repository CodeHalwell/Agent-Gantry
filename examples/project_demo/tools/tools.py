from typing import Any

from agent_gantry import AgentGantry
from agent_gantry.adapters.embedders.nomic import NomicEmbedder
embedder = NomicEmbedder(dimension=256)

# tool libraries
import requests
from rdkit import Chem
from rdkit.Chem import Descriptors
import pubchempy as pcp
import pint
import datetime
from sympy import symbols, solve, sympify

tools = AgentGantry(embedder=embedder)

@tools.register(tags=["chemistry", "molecular"])
def get_molecular_weight(smiles: str) -> float:
    """Calculate the molecular weight of a compound given its SMILES representation."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("Invalid SMILES string.")
    return Descriptors.MolWt(mol)

@tools.register(tags=["chemistry", "compound"])
def get_compound_info(name: str) -> dict[str, Any]:
    """Fetch compound information from PubChem given its name."""
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

@tools.register(tags=["unit_conversion"])
def convert_units(value: float, from_unit: str, to_unit: str) -> float:
    """Convert a value from one unit to another."""
    ureg = pint.UnitRegistry()
    quantity = value * ureg(from_unit)
    converted = quantity.to(to_unit)
    return converted.magnitude

@tools.register(tags=["date_calculation"])
def calculate_date_difference(date1: str, date2: str) -> int:
    """Calculate the difference in days between two dates (YYYY-MM-DD)."""
    d1 = datetime.datetime.strptime(date1, "%Y-%m-%d")
    d2 = datetime.datetime.strptime(date2, "%Y-%m-%d")
    return abs((d2 - d1).days)

@tools.register(tags=["math", "algebra"])
def solve_equation(equation: str, variable: str) -> Any:
    """Solve a simple algebraic equation for the given variable."""
    var = symbols(variable)
    expr = sympify(equation)
    solution = solve(expr, var)
    return solution

@tools.register(tags=["web"])
def fetch_web_content(url: str) -> str:
    """Fetch the content of a web page given its URL."""
    try:
        response = requests.get(
            url,
            timeout=30,
            headers={"User-Agent": "Agent-Gantry/0.1.0"},
        )
        response.raise_for_status()
        
        # Validate content length to prevent excessive memory usage
        content_length = response.headers.get('content-length')
        if content_length and int(content_length) > 10_000_000:  # 10MB limit
            raise ValueError(f"Content too large: {content_length} bytes")
        
        return response.text
    except requests.exceptions.Timeout:
        raise ValueError(f"Request timed out while fetching: {url}")
    except requests.exceptions.ConnectionError:
        raise ValueError(f"Connection error while fetching: {url}")
    except requests.exceptions.HTTPError as e:
        raise ValueError(f"HTTP error while fetching {url}: {e}")
    except requests.exceptions.RequestException as e:
        raise ValueError(f"Error fetching {url}: {e}")

@tools.register(tags=["datetime"])
def get_current_utc_time() -> str:
    """Get the current UTC time as an ISO formatted string."""
    return datetime.datetime.now(datetime.timezone.utc).isoformat().replace("+00:00", "Z")

