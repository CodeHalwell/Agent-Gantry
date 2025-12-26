"""
Reusable tool catalog for plug-and-play demos.

Expose an AgentGantry instance named ``tools`` so other scripts can import and
run semantic filtering without redefining tools.
"""

import datetime

from agent_gantry import create_default_gantry

tools = create_default_gantry()


@tools.register(tags=["conversion", "travel"], examples=["convert kilometers to miles"])
def convert_km_to_miles(kilometers: float) -> float:
    """Convert kilometers to miles."""
    return kilometers * 0.621371


@tools.register(tags=["time", "datetime"], examples=["what is the current UTC time?"])
def current_utc_time() -> str:
    """Get the current UTC time."""
    return datetime.datetime.utcnow().isoformat() + "Z"


@tools.register(tags=["finance", "payments"], examples=["format a $45 invoice"])
def format_invoice_amount(amount: float, currency: str = "USD") -> str:
    """Format an amount for an invoice."""
    return f"{currency} {amount:,.2f}"


@tools.register(tags=["productivity", "communication"], examples=["draft an email to Maria"])
def draft_email_summary(recipient: str, subject: str, body: str) -> str:
    """Draft a summary email."""
    return f"To: {recipient}\nSubject: {subject}\n\n{body}"


__all__ = ["tools"]
