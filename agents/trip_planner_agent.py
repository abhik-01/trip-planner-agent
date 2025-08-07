from agents.budget_agent import budget_agent_tool
from datetime import datetime
from json import loads
from re import split
from requests import get

from tools.activity import get_activity_tool
from tools.assembler import get_assembler_tool
from tools.destination import get_destination_tool
from tools.flight import format_flights_for_display, get_flight_tool
from tools.map import get_map_tool
from tools.weather import get_weather_tool

from utils.set_llm import get_llm        


llm = get_llm()


class TripPlannerAgent:
    def __init__(self):
        self.llm = get_llm()
        self.destination_tool = get_destination_tool()
        self.activity_tool = get_activity_tool()
        self.map_tool = get_map_tool()
        self.weather_tool = get_weather_tool()
        self.flight_tool = get_flight_tool()
        self.budget_tool = budget_agent_tool()
        self.assembler_tool = get_assembler_tool()

    def parse_user_query(self, query: str) -> dict:
        llm = get_llm()
        prompt = (
            f"Extract the following fields from this trip planning request: "
            f"destination, start date, end date, number of travelers, interests, user city. "
            f"Return as a JSON object. Query: {query}"
        )
        response = llm.invoke(prompt)

        try:
            return loads(response)
        except Exception:
            return {}


    def extract_flight_cost_and_currency(self, flight_info) -> tuple:
        """
        Given a list of flight dicts, returns (price, currency) of the cheapest flight.
        If no valid flight, returns (0.0, 'USD').
        """
        if not flight_info or not isinstance(flight_info, list):
            return 0.0, 'USD'

        if 'error' in flight_info[0]:
            return 0.0, 'USD'

        cheapest = min(flight_info, key=lambda f: f.get('price', float('inf')))

        return cheapest.get('price', 0.0), cheapest.get('currency', 'USD')

    def convert_currency(self, amount: float, from_currency: str, to_currency: str) -> float:
        """
        Converts amount from from_currency to to_currency using exchangerate.host API.
        Returns the converted amount as float. If conversion fails, returns the original amount.
        """
        try:
            url = f"https://api.exchangerate.host/convert?from={from_currency}&to={to_currency}&amount={amount}"
            resp = get(url, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            result = data.get("result")

            if result:
                return float(result)

        except Exception:
            pass

        return amount

    def plan_trip(self, user_query: str) -> str:
        trip_data = {}
        # 0. Greetings/small talk check
        greetings = ["hi", "hello", "hey", "greetings", "good morning", "good afternoon", "good evening"]
        thanks = ["thank you", "thanks", "thx", "appreciate it"]
        msg = user_query.strip().lower()
        if any(g in msg for g in greetings):
            prompt = (
                "You are a friendly, helpful AI travel assistant. The user greeted you. "
                "Reply with a warm, natural greeting and invite them to ask about trip planning. "
                "Keep it concise and engaging."
            )
            return self.llm.invoke(prompt)
        if any(t in msg for t in thanks):
            prompt = (
                "You are a friendly, helpful AI travel assistant. The user thanked you. "
                "Reply with a natural, gracious response and encourage them to ask more travel questions if they wish. "
                "Keep it concise and positive."
            )
            return self.llm.invoke(prompt)

        # 1. Intent detection step (LLM classification, not asking user)
        intent_prompt = (
            "Classify the following message as 'trip' if it is a trip planning or travel-related request, "
            "otherwise reply with 'other'. Only reply with 'trip' or 'other'.\nMessage: " + user_query
        )
        intent_response = self.llm.invoke(intent_prompt).strip().lower()

        if not intent_response.startswith('trip'):
            # LLM fallback for ambiguous/off-topic
            prompt = (
                "You are a friendly, helpful AI travel assistant. The user's message was not about trip planning. "
                "Gently steer the conversation back to travel or trip planning, inviting them to ask about destinations, flights, activities, or budgets. "
                "Be polite, concise, and engaging."
            )
            return self.llm.invoke(prompt)

        parsed = self.parse_user_query(user_query)

        # If parsed is empty or missing key info, ask for clarification or suggest best time to visit
        required_fields = ["destination", "start date", "end date"]
        missing = [f for f in required_fields if not parsed.get(f)]
        if not parsed or missing:
            destination = parsed.get("destination", "")

            if destination and (not parsed.get("start date") or not parsed.get("end date")):
                best_time_or_none_prompt = (
                    f"If the following message is asking for the best time to visit {destination}, reply with a concise, friendly summary of the best months or seasons to visit {destination}, mentioning weather, crowds, and any special events if relevant. "
                    f"If not, reply with ONLY the string '__NOT_BEST_TIME__'.\nMessage: {user_query}"
                )
                best_time_result = self.llm.invoke(best_time_or_none_prompt).strip()

                if not best_time_result.startswith("__NOT_BEST_TIME__"):
                    return best_time_result

            missing_str = ", ".join(missing) if missing else "trip details"
            prompt = (
                f"You are a helpful AI travel assistant. The user asked to plan a trip but did not provide all the required information. "
                f"The missing fields are: {missing_str}. "
                "Politely ask the user to provide the missing details so you can help plan their trip. Keep it concise and friendly."
            )

            return self.llm.invoke(prompt)

        # Collect parsed info
        trip_data["destination"] = parsed.get("destination", "")
        trip_data["dates"] = f"{parsed.get('start date', '')} to {parsed.get('end date', '')}"
        trip_data["travelers"] = parsed.get("number of travelers", 1)
        trip_data["interests"] = parsed.get("interests", "")

        user_city = parsed.get("user city", "")
        date_depart = parsed.get("start date", "")
        nights = 1

        try:
            d1 = datetime.strptime(parsed.get('start date', ''), "%Y-%m-%d")
            d2 = datetime.strptime(parsed.get('end date', ''), "%Y-%m-%d")
            nights = (d2 - d1).days
            if nights < 1:
                nights = 1
        except Exception:
            pass

        trip_data["nights"] = nights

        # 1. Suggest activities
        activities = self.activity_tool.func(trip_data["destination"])

        # Parse activities string into a list for budget tool (split by newlines or numbered list)
        activities_list = []
        if isinstance(activities, str):
            acts = split(r"\n|\d+\. |\- ", activities)
            acts = [a.strip() for a in acts if a.strip() and len(a.strip()) > 2]
            activities_list = acts
        elif isinstance(activities, list):
            activities_list = activities

        trip_data["activities"] = activities_list

        # 2. Find flights (returns list of dicts)
        flights = self.flight_tool.func(user_city, trip_data["destination"], date_depart, trip_data["travelers"])
        trip_data["flights"] = flights

        # 3. Get weather
        weather = self.weather_tool.func(trip_data["destination"], date_depart)
        trip_data["weather"] = weather

        # 4. Extract flight cost/currency (cheapest)
        flight_cost, flight_currency = self.extract_flight_cost_and_currency(flights)

        # 5. Convert to INR if needed
        if flight_cost > 0 and flight_currency != "INR":
            flight_cost_inr = self.convert_currency(flight_cost, flight_currency, "INR")

        else:
            flight_cost_inr = flight_cost

        trip_data["flight_cost"] = flight_cost_inr

        # 6. Budget estimation
        budget_input = {
            "flight_cost": flight_cost_inr,
            "destination": trip_data["destination"],
            "nights": nights,
            "activities": activities_list,
            "days": nights + 1,
            "travelers": trip_data["travelers"]
        }
        budget = self.budget_tool.func(budget_input)
        trip_data["budget"] = budget

        # 7. Assemble itinerary
        itinerary = self.assembler_tool.func(trip_data)

        # 8. Format flights for display
        flights_display = format_flights_for_display(flights)

        # 9. Return a user-friendly summary
        summary = (
            f"\n---\n**Flight Options:**\n{flights_display}\n\n"
            f"**Weather:**\n{weather}\n\n"
            f"**Budget Estimate (INR):**\n{budget}\n\n"
            f"**Itinerary:**\n{itinerary}\n---\n"
        )

        return summary


def get_trip_planner_agent():
    """
    Returns an instance of the TripPlannerAgent class, which orchestrates the entire planning pipeline.
    """
    return TripPlannerAgent()
