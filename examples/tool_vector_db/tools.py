"""
50 simple tools for demonstrating Agent-Gantry semantic routing with Nomic embeddings.

All tools use only Python stdlib - no heavy dependencies.
"""

from __future__ import annotations

import datetime
import hashlib
import json
import math
import os
import random
import re
import statistics
import uuid
from typing import Any

from agent_gantry import AgentGantry
from agent_gantry.adapters.embedders.nomic import NomicEmbedder
from agent_gantry.adapters.vector_stores.lancedb import LanceDBVectorStore

# Create gantry instance with Nomic embeddings (256 dimensions)
embedder = NomicEmbedder(dimension=256)
vector_store = LanceDBVectorStore(db_path="gantry_tools.lancedb", dimension=256)
tools = AgentGantry(embedder=embedder, vector_store=vector_store)


# =============================================================================
# MATH TOOLS (10 tools)
# =============================================================================


@tools.register(tags=["math"])
def add(a: float, b: float) -> float:
    """Add two numbers together and return their sum.
    
    Performs basic arithmetic addition of two numerical values.
    Useful for calculating totals, combining quantities, or any additive operation.
    """
    return a + b


@tools.register(tags=["math"])
def subtract(a: float, b: float) -> float:
    """Subtract the second number from the first and return the difference.
    
    Performs basic arithmetic subtraction. Returns a - b.
    Useful for finding differences, calculating remaining amounts, or decrements.
    """
    return a - b


@tools.register(tags=["math"])
def multiply(a: float, b: float) -> float:
    """Multiply two numbers together and return their product.
    
    Performs basic arithmetic multiplication of two numerical values.
    Useful for scaling, area calculations, or repeated addition scenarios.
    """
    return a * b


@tools.register(tags=["math"])
def divide(a: float, b: float) -> float:
    """Divide the first number by the second and return the quotient.
    
    Performs basic arithmetic division. Returns a / b.
    Raises ValueError if attempting to divide by zero.
    Useful for ratios, averages, or splitting quantities.
    """
    if b == 0:
        raise ValueError("Cannot divide by zero")
    return a / b


@tools.register(tags=["math"])
def calculate_mean(numbers: list[float]) -> float:
    """Calculate the arithmetic mean (average) of a list of numbers.
    
    Computes the sum of all numbers divided by the count of numbers.
    This is the most common measure of central tendency.
    Useful for finding average values in datasets, scores, or measurements.
    """
    return statistics.mean(numbers)


@tools.register(tags=["math"])
def calculate_median(numbers: list[float]) -> float:
    """Calculate the median (middle value) of a list of numbers.
    
    Returns the middle value when numbers are sorted. For even-length lists,
    returns the average of the two middle values.
    Useful when you need a measure of central tendency that's resistant to outliers.
    """
    return statistics.median(numbers)


@tools.register(tags=["math"])
def calculate_stdev(numbers: list[float]) -> float:
    """Calculate the sample standard deviation of a list of numbers.
    
    Measures the amount of variation or dispersion in a dataset.
    Higher values indicate data points are spread out over a wider range.
    Useful for understanding data variability, quality control, or risk assessment.
    """
    return statistics.stdev(numbers)


@tools.register(tags=["math"])
def calculate_factorial(n: int) -> int:
    """Calculate the factorial of a non-negative integer (n!).
    
    Returns n! = n × (n-1) × (n-2) × ... × 2 × 1.
    For example: 5! = 120.
    Useful for permutations, combinations, and probability calculations.
    """
    return math.factorial(n)


@tools.register(tags=["math"])
def calculate_power(base: float, exponent: float) -> float:
    """Calculate base raised to the power of exponent (base^exponent).
    
    Returns the result of multiplying base by itself exponent times.
    Supports fractional exponents for roots (e.g., x^0.5 = √x).
    Useful for exponential growth, compound interest, or scientific calculations.
    """
    return math.pow(base, exponent)


@tools.register(tags=["math"])
def calculate_sqrt(n: float) -> float:
    """Calculate the square root of a non-negative number.
    
    Returns the value that, when multiplied by itself, equals n.
    For example: √16 = 4.
    Useful for distance calculations, geometry, or root-finding operations.
    """
    return math.sqrt(n)


# =============================================================================
# TEXT TOOLS (10 tools)
# =============================================================================


@tools.register(tags=["text"])
def count_words(text: str) -> int:
    """Count the total number of words in a text string.
    
    Splits the text by whitespace and counts the resulting segments.
    Useful for word count limits, content analysis, or document statistics.
    """
    return len(text.split())


@tools.register(tags=["text"])
def count_characters(text: str) -> int:
    """Count the total number of characters in a text string.
    
    Returns the length of the string including spaces and punctuation.
    Useful for character limits (tweets, SMS), validation, or text analysis.
    """
    return len(text)


@tools.register(tags=["text"])
def reverse_string(text: str) -> str:
    """Reverse a string, returning characters in opposite order.
    
    Flips the string so the last character becomes first, etc.
    Useful for palindrome checks, creative text effects, or data transformation.
    """
    return text[::-1]


@tools.register(tags=["text"])
def to_uppercase(text: str) -> str:
    """Convert all characters in a text string to uppercase letters.
    
    Transforms lowercase letters to uppercase; non-letters unchanged.
    Useful for headers, emphasis, or case-insensitive comparisons.
    """
    return text.upper()


@tools.register(tags=["text"])
def to_lowercase(text: str) -> str:
    """Convert all characters in a text string to lowercase letters.
    
    Transforms uppercase letters to lowercase; non-letters unchanged.
    Useful for normalization, case-insensitive matching, or data cleaning.
    """
    return text.lower()


@tools.register(tags=["text"])
def to_title_case(text: str) -> str:
    """Convert text to title case, capitalizing the first letter of each word.
    
    Makes the first character of each word uppercase and the rest lowercase.
    Useful for formatting names, titles, headings, or proper nouns.
    """
    return text.title()


@tools.register(tags=["text"])
def strip_whitespace(text: str) -> str:
    """Remove leading and trailing whitespace from a text string.
    
    Strips spaces, tabs, and newlines from the beginning and end of text.
    Useful for cleaning user input, data normalization, or text processing.
    """
    return text.strip()


@tools.register(tags=["text"])
def replace_text(text: str, old: str, new: str) -> str:
    """Replace all occurrences of a substring with another string.
    
    Finds every instance of 'old' in the text and replaces it with 'new'.
    Useful for find-and-replace operations, text correction, or templating.
    """
    return text.replace(old, new)


@tools.register(tags=["text"])
def split_text(text: str, delimiter: str = " ") -> list[str]:
    """Split a text string into a list of substrings using a delimiter.
    
    Divides the text at each occurrence of the delimiter character/string.
    Default delimiter is a space. Useful for parsing CSV, tokenization, or word extraction.
    """
    return text.split(delimiter)


@tools.register(tags=["text"])
def join_text(parts: list[str], delimiter: str = " ") -> str:
    """Join a list of strings into a single string using a delimiter.
    
    Concatenates all strings in the list, inserting the delimiter between each.
    Default delimiter is a space. Useful for building sentences, CSV rows, or paths.
    """
    return delimiter.join(parts)


# =============================================================================
# DATE/TIME TOOLS (10 tools)
# =============================================================================


@tools.register(tags=["datetime"])
def get_current_date() -> str:
    """Get today's date in ISO format (YYYY-MM-DD).
    
    Returns the current local date as a string in standard ISO 8601 format.
    Useful for timestamps, logging, date comparisons, or displaying today's date.
    """
    return datetime.date.today().isoformat()


@tools.register(tags=["datetime"])
def get_current_time() -> str:
    """Get the current local time in HH:MM:SS format.
    
    Returns the current time as a 24-hour formatted string.
    Useful for timestamps, logging, scheduling, or displaying current time.
    """
    return datetime.datetime.now().strftime("%H:%M:%S")


@tools.register(tags=["datetime"])
def get_current_datetime() -> str:
    """Get the current local date and time as an ISO formatted string.
    
    Returns full datetime in ISO 8601 format (YYYY-MM-DDTHH:MM:SS.ffffff).
    Useful for precise timestamps, audit logs, or datetime comparisons.
    """
    return datetime.datetime.now().isoformat()


@tools.register(tags=["datetime"])
def get_weekday(date_str: str) -> str:
    """Get the day of the week (e.g., Monday, Tuesday) for a given date.
    
    Takes a date string in YYYY-MM-DD format and returns the weekday name.
    Useful for scheduling, planning, or determining business days.
    """
    d = datetime.datetime.strptime(date_str, "%Y-%m-%d")
    return d.strftime("%A")


@tools.register(tags=["datetime"])
def days_between(date1: str, date2: str) -> int:
    """Calculate the number of days between two dates.
    
    Takes two dates in YYYY-MM-DD format and returns the absolute difference in days.
    Useful for age calculations, deadline tracking, or duration computations.
    """
    d1 = datetime.datetime.strptime(date1, "%Y-%m-%d")
    d2 = datetime.datetime.strptime(date2, "%Y-%m-%d")
    return abs((d2 - d1).days)


@tools.register(tags=["datetime"])
def add_days(date_str: str, days: int) -> str:
    """Add a specified number of days to a date and return the new date.
    
    Takes a date in YYYY-MM-DD format and adds (or subtracts if negative) days.
    Returns the resulting date in YYYY-MM-DD format.
    Useful for calculating due dates, expiration dates, or future/past dates.
    """
    d = datetime.datetime.strptime(date_str, "%Y-%m-%d")
    result = d + datetime.timedelta(days=days)
    return result.strftime("%Y-%m-%d")


@tools.register(tags=["datetime"])
def is_leap_year(year: int) -> bool:
    """Check if a given year is a leap year (has 366 days).
    
    Returns True if the year is divisible by 4, except for century years
    which must be divisible by 400. Useful for calendar calculations or validation.
    """
    import calendar

    return calendar.isleap(year)


@tools.register(tags=["datetime"])
def get_days_in_month(year: int, month: int) -> int:
    """Get the number of days in a specific month of a given year.
    
    Takes a year and month (1-12) and returns the day count (28-31).
    Accounts for leap years in February. Useful for calendar displays or date validation.
    """
    import calendar

    return calendar.monthrange(year, month)[1]


@tools.register(tags=["datetime"])
def format_date(date_str: str, output_format: str) -> str:
    """Format a date string into a different date format.
    
    Takes a date in YYYY-MM-DD format and reformats it using strftime codes.
    Example: format_date('2024-01-15', '%B %d, %Y') returns 'January 15, 2024'.
    Useful for displaying dates in user-friendly formats or locale-specific styles.
    """
    d = datetime.datetime.strptime(date_str, "%Y-%m-%d")
    return d.strftime(output_format)


@tools.register(tags=["datetime"])
def parse_timestamp(timestamp: float) -> str:
    """Convert a Unix timestamp (seconds since epoch) to a readable datetime string.
    
    Takes a numeric Unix timestamp and returns an ISO formatted datetime string.
    Useful for converting system timestamps, log entries, or API responses to human-readable format.
    """
    return datetime.datetime.fromtimestamp(timestamp).isoformat()


# =============================================================================
# UTILITY TOOLS (10 tools)
# =============================================================================


@tools.register(tags=["utility"])
def generate_uuid() -> str:
    """Generate a random UUID version 4 (universally unique identifier).
    
    Creates a 128-bit identifier that is virtually guaranteed to be unique.
    Format: xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx.
    Useful for database primary keys, session IDs, or unique resource identifiers.
    """
    return str(uuid.uuid4())


@tools.register(tags=["utility"])
def generate_random_number(min_val: int, max_val: int) -> int:
    """Generate a random integer within a specified range (inclusive).
    
    Returns a random integer N such that min_val <= N <= max_val.
    Useful for games, sampling, randomized testing, or simulations.
    """
    return random.randint(min_val, max_val)


@tools.register(tags=["utility"])
def hash_text(text: str, algorithm: str = "sha256") -> str:
    """Calculate a cryptographic hash of a text string.
    
    Generates a fixed-length hexadecimal digest using the specified algorithm.
    Supported algorithms: md5, sha1, sha256, sha512, etc.
    Useful for checksums, password hashing (with salt), or data integrity verification.
    """
    h = hashlib.new(algorithm)
    h.update(text.encode())
    return h.hexdigest()


@tools.register(tags=["utility"])
def base64_encode(text: str) -> str:
    """Encode a text string to Base64 format.
    
    Converts text to a Base64 encoded string for safe transmission.
    Useful for encoding binary data in JSON, email attachments, or URLs.
    """
    import base64

    return base64.b64encode(text.encode()).decode()


@tools.register(tags=["utility"])
def base64_decode(encoded: str) -> str:
    """Decode a Base64 encoded string back to plain text.
    
    Reverses Base64 encoding to recover the original text.
    Useful for decoding data received from APIs, emails, or encoded URLs.
    """
    import base64

    return base64.b64decode(encoded.encode()).decode()


@tools.register(tags=["utility"])
def json_stringify(data: dict[str, Any]) -> str:
    """Convert a Python dictionary to a formatted JSON string.
    
    Serializes a dictionary to a pretty-printed JSON string with 2-space indentation.
    Useful for API responses, configuration files, or data export.
    """
    return json.dumps(data, indent=2)


@tools.register(tags=["utility"])
def json_parse(json_str: str) -> dict[str, Any]:
    """Parse a JSON string into a Python dictionary.
    
    Deserializes a JSON formatted string into a Python dict object.
    Useful for processing API responses, reading config files, or data import.
    """
    return json.loads(json_str)


@tools.register(tags=["utility"])
def flip_coin() -> str:
    """Simulate a coin flip and return either 'Heads' or 'Tails'.
    
    Randomly selects between two outcomes with equal probability (50/50).
    Useful for making random binary decisions, games, or demonstrations.
    """
    return random.choice(["Heads", "Tails"])


@tools.register(tags=["utility"])
def roll_dice(sides: int = 6) -> int:
    """Simulate rolling a die with a specified number of sides.
    
    Returns a random integer from 1 to the number of sides (default: 6).
    Useful for games, random selection, or probability demonstrations.
    """
    return random.randint(1, sides)


@tools.register(tags=["utility"])
def get_env_var(name: str) -> str | None:
    """Get the value of an environment variable by name.
    
    Returns the value of the specified environment variable, or None if not set.
    Useful for accessing configuration, secrets, or system settings.
    """
    return os.environ.get(name)


# =============================================================================
# CONVERSION TOOLS (10 tools)
# =============================================================================


@tools.register(tags=["conversion"])
def celsius_to_fahrenheit(celsius: float) -> float:
    """Convert a temperature from Celsius to Fahrenheit scale.
    
    Uses the formula: F = (C × 9/5) + 32.
    Useful for weather data conversion or international temperature comparisons.
    """
    return (celsius * 9 / 5) + 32


@tools.register(tags=["conversion"])
def fahrenheit_to_celsius(fahrenheit: float) -> float:
    """Convert a temperature from Fahrenheit to Celsius scale.
    
    Uses the formula: C = (F - 32) × 5/9.
    Useful for weather data conversion or international temperature comparisons.
    """
    return (fahrenheit - 32) * 5 / 9


@tools.register(tags=["conversion"])
def meters_to_feet(meters: float) -> float:
    """Convert a distance from meters to feet.
    
    Uses the conversion factor: 1 meter = 3.28084 feet.
    Useful for construction, athletics, or converting between metric and imperial systems.
    """
    return meters * 3.28084


@tools.register(tags=["conversion"])
def feet_to_meters(feet: float) -> float:
    """Convert a distance from feet to meters.
    
    Uses the conversion factor: 1 foot = 0.3048 meters.
    Useful for construction, athletics, or converting between imperial and metric systems.
    """
    return feet / 3.28084


@tools.register(tags=["conversion"])
def kg_to_pounds(kg: float) -> float:
    """Convert a weight from kilograms to pounds.
    
    Uses the conversion factor: 1 kg = 2.20462 pounds.
    Useful for shipping, fitness tracking, or international weight comparisons.
    """
    return kg * 2.20462


@tools.register(tags=["conversion"])
def pounds_to_kg(pounds: float) -> float:
    """Convert a weight from pounds to kilograms.
    
    Uses the conversion factor: 1 pound = 0.453592 kg.
    Useful for shipping, fitness tracking, or international weight comparisons.
    """
    return pounds / 2.20462


@tools.register(tags=["conversion"])
def km_to_miles(km: float) -> float:
    """Convert a distance from kilometers to miles.
    
    Uses the conversion factor: 1 km = 0.621371 miles.
    Useful for travel planning, running distances, or map conversions.
    """
    return km * 0.621371


@tools.register(tags=["conversion"])
def miles_to_km(miles: float) -> float:
    """Convert a distance from miles to kilometers.
    
    Uses the conversion factor: 1 mile = 1.60934 km.
    Useful for travel planning, running distances, or map conversions.
    """
    return miles / 0.621371


@tools.register(tags=["conversion"])
def liters_to_gallons(liters: float) -> float:
    """Convert a volume from liters to US gallons.
    
    Uses the conversion factor: 1 liter = 0.264172 US gallons.
    Useful for fuel economy calculations, cooking, or liquid measurements.
    """
    return liters * 0.264172


@tools.register(tags=["conversion"])
def gallons_to_liters(gallons: float) -> float:
    """Convert a volume from US gallons to liters.
    
    Uses the conversion factor: 1 US gallon = 3.78541 liters.
    Useful for fuel economy calculations, cooking, or liquid measurements.
    """
    return gallons / 0.264172
