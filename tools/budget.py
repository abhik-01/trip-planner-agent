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
    prompt = (
        f"Estimate a detailed trip budget in USD for the following:\n"
        f"Destination: {trip_details.get('destination')}\n"
        f"Flight cost: ${trip_details.get('flight_cost')}\n"
        f"Number of nights: {trip_details.get('nights')}\n"
        f"Number of travelers: {trip_details.get('travelers', 1)}\n"
        f"Activities: {', '.join(trip_details.get('activities', []))}\n"
        f"Include accommodation, activities, food, local transport, and a 10% miscellaneous buffer. "
        f"Return a breakdown and the total as plain text."
    )

    return llm.invoke(prompt)


def get_budget_tool():
    return Tool(
        name="Trip Budget Estimator",
        func=trip_budget_estimator,
        description="Estimates total trip budget using LLM for all categories. Input should be a dict with keys: flight_cost, destination, nights, activities, days, travelers."
    )
