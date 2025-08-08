from langchain.tools import Tool
from utils.set_llm import get_llm


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
    """
    llm = get_llm()
    flight_cost = trip_details.get('flight_cost')
    if not flight_cost:
        flight_line = "Flight cost unknown: exclude from total unless you must estimate (then be conservative)."
    else:
        flight_line = f"Flight cost: ${flight_cost}"
    prompt = (
        f"Estimate a realistic (avoid overestimation) trip budget in USD for the following:\n"
        f"Destination: {trip_details.get('destination')}\n"
        f"{flight_line}\n"
        f"Number of nights: {trip_details.get('nights')}\n"
        f"Number of travelers: {trip_details.get('travelers', 1)}\n"
        f"Activities: {', '.join(trip_details.get('activities', []))}\n"
        f"Provide accommodation, activities, food, local transport, and a 10% miscellaneous buffer. If flight cost unknown, omit it. "
        f"Return a category breakdown plus total as plain text."
    )

    res = llm.invoke(prompt)
    return str(getattr(res, 'content', res))


def get_budget_tool():
    return Tool(
        name="Trip Budget Estimator",
        func=trip_budget_estimator,
        description="Estimates total trip budget using LLM for all categories. Input should be a dict with keys: flight_cost, destination, nights, activities, days, travelers."
    )
