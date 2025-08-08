from requests import get
from functools import lru_cache
from time import time
from langchain.tools import Tool

# Simple in-memory cache with TTL
_CACHE = {"rates": None, "base": None, "ts": 0}
_TTL = 3600  # 1 hour

API_URL = "https://api.exchangerate.host/latest"

@lru_cache(maxsize=64)
def convert_amount(amount: float, from_currency: str, to_currency: str = "INR") -> float:
    from_currency = (from_currency or "USD").upper()
    to_currency = (to_currency or "INR").upper()
    if from_currency == to_currency:
        return amount
    rates = _get_rates(base=from_currency)
    if not rates:
        return amount
    rate = rates.get(to_currency)
    if not rate:
        return amount
    return round(amount * rate, 2)

def _get_rates(base: str = "USD"):
    now = time()
    if _CACHE["rates"] and _CACHE["base"] == base and now - _CACHE["ts"] < _TTL:
        return _CACHE["rates"]
    try:
        resp = get(API_URL, params={"base": base}, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        rates = data.get("rates")
        if rates:
            _CACHE["rates"] = rates
            _CACHE["base"] = base
            _CACHE["ts"] = now
            return rates
    except Exception:
        return None
    return None

def _convert_wrapper(params: dict) -> str:
    if not isinstance(params, dict):
        return "Input must be dict with keys: amount, from_currency, to_currency"
    amount = params.get('amount'); fc = params.get('from_currency'); tc = params.get('to_currency')
    try:
        amount_f = float(amount)
    except Exception:
        return "Invalid amount"
    converted = convert_amount(amount_f, fc, tc)
    return f"{amount_f} {fc} ≈ {converted} {tc}"


def get_currency_tool():
    return Tool(
        name="Currency Conversion Tool",
        func=_convert_wrapper,
        description="Convert monetary amount between currencies. Input dict: {amount: float, from_currency: str, to_currency: str}.",
    )
