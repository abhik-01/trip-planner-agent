from hashlib import md5
from langchain.tools import Tool
from prompts import format_prompt, PromptType
from tools.currency import convert_amount
from utils.set_llm import get_llm


def _create_budget_cache_key(trip_details: dict) -> str:
    """Create a cache key for budget estimation based on core parameters"""
    key_params = {
        'destination': trip_details.get('destination', '').lower().strip(),
        'nights': trip_details.get('nights', 0),
        'travelers': trip_details.get('travelers', 1),
        'activities': sorted(trip_details.get('activities', [])),
        'has_flight': bool(trip_details.get('flight_cost'))
    }
    key_str = str(sorted(key_params.items()))
    return md5(key_str.encode()).hexdigest()[:8]


# Simple cache for budget estimates (not using lru_cache due to dict input)
_budget_cache = {}


def trip_budget_estimator(trip_details: dict) -> str:
    """
    Estimates a detailed trip budget using the LLM for all categories in a single call.
    Expects trip_details to include:
      - flight_cost (float)
      - destination (str)
      - nights (int)
      - activities (list)
      - days (int)
      - travelers (int)
      - flight_currency (str, optional)
    """
    # Input validation
    if not isinstance(trip_details, dict):
        return "Error: Trip details must be provided as a dictionary."
    
    destination = trip_details.get('destination')
    if not destination:
        return "Error: Destination is required for budget estimation."
    
    # Check cache first
    cache_key = _create_budget_cache_key(trip_details)
    if cache_key in _budget_cache:
        return _budget_cache[cache_key]
    
    llm = get_llm()
    flight_cost = trip_details.get('flight_cost')
    flight_currency = trip_details.get('flight_currency', 'USD')
    
    if not flight_cost:
        flight_line = "Flight cost unknown: exclude from total unless you must estimate (then be conservative)."
    else:
        # Convert flight cost to INR if needed
        if flight_currency.upper() != 'INR':
            try:
                flight_cost_inr = convert_amount(float(flight_cost), flight_currency, 'INR')
                flight_line = f"Flight cost: ₹{flight_cost_inr} INR (converted from {flight_currency} {flight_cost})"
            except Exception:
                # Fallback if conversion fails
                flight_line = f"Flight cost: {flight_currency} {flight_cost} (conversion to INR failed, please estimate)"
        else:
            flight_line = f"Flight cost: ₹{flight_cost} INR"
            
    prompt = format_prompt(
        PromptType.BUDGET_ESTIMATION,
        destination=destination,
        flight_line=flight_line,
        nights=trip_details.get('nights', 1),
        travelers=trip_details.get('travelers', 1),
        activities=', '.join(trip_details.get('activities', ['general tourism']))
    )

    try:
        res = llm.invoke(prompt)
        result = str(getattr(res, 'content', res))

        # Cache the result
        _budget_cache[cache_key] = result

        return result
    except Exception as e:
        return "Sorry, I couldn't generate a budget estimate at this time. Please try again."


def get_budget_tool():
    return Tool(
        name="Trip Budget Estimator",
        func=trip_budget_estimator,
        description="Estimates total trip budget in INR using LLM for all categories. Input should be a dict with keys: flight_cost, destination, nights, activities, days, travelers, flight_currency (optional)."
    )
