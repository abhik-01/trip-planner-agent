from amadeus import Client, ResponseError
from langchain.tools import Tool


AMADEUS_CLIENT_ID = ""
AMADEUS_CLIENT_SECRET = ""

amadeus = Client(
    client_id=AMADEUS_CLIENT_ID,
    client_secret=AMADEUS_CLIENT_SECRET
)


def get_nearest_airport(city: str) -> str:
    """
    Uses Amadeus API to find the nearest airport IATA code for a given city.
    """
    try:
        response = amadeus.reference_data.locations.get(
            keyword=city,
            subType='AIRPORT',
            page_limit=1
        )
        data = response.data

        if data and isinstance(data, list):
            return data[0]['iataCode']

        return None
    except ResponseError:
        return None


def search_flights_from_city(user_city: str, destination_city: str, date: str, adults: int) -> str:
    """
    Finds flights from the nearest airport to the user's city to the destination city.
    Returns a list of flight dicts (price, currency, airline, dep/arr airports, dep_time, etc.).
    If no airport is found, returns a dict with an 'error' key and message.
    """
    if not (user_city and destination_city and date):
        return [{"error": "Please provide your city, destination city, and date (YYYY-MM-DD)."}]

    origin_code = get_nearest_airport(user_city)
    dest_code = get_nearest_airport(destination_city)

    if not origin_code:
        return [{
            "error": f"Sorry, there is no airport in or near {user_city}. You may consider ground transport (bus, train, or car) to a nearby city with an airport. If you want more options, let me know—I'm working to add more transport modes in the future!"
        }]

    if not dest_code:
        return [{"error": f"Sorry, couldn't find an airport near {destination_city}."}]

    try:
        response = amadeus.shopping.flight_offers_search.get(
            originLocationCode=origin_code,
            destinationLocationCode=dest_code,
            departureDate=date,
            adults=adults,
            max=5
        )
        flights = response.data

        if flights:
            results = []
            for offer in flights:
                price = offer['price']['total']
                currency = offer['price'].get('currency', 'USD')
                itineraries = offer['itineraries'][0]['segments'][0]
                dep = itineraries['departure']['iataCode']
                arr = itineraries['arrival']['iataCode']
                dep_time = itineraries['departure']['at']
                airline = itineraries['carrierCode']
                results.append({
                    "price": float(price),
                    "currency": currency,
                    "airline": airline,
                    "departure_airport": dep,
                    "arrival_airport": arr,
                    "departure_time": dep_time
                })
            return results
        else:
            return [{
                "error": f"No flights found from {user_city} ({origin_code}) to {destination_city} ({dest_code}) on {date}. You may consider ground transport to a different airport or check other dates. I'm working to add more transport modes in the future!"
            }]

    except Exception as e:
        return [{"error": f"Error fetching flight data: {e}"}]


def format_flights_for_display(flights):
    """
    Helper to format a list of flight dicts for user display.
    """
    if not flights or not isinstance(flights, list):
        return "No flight data available."

    if 'error' in flights[0]:
        return flights[0]['error']

    lines = []

    for f in flights:
        lines.append(f"{f['departure_airport']} → {f['arrival_airport']} | {f['airline']} | {f['departure_time']} | {f['currency']} {f['price']}")

    return "\n".join(lines)


def get_flight_tool() -> Tool:
    return Tool(
        name="Flight Search Tool",
        func=search_flights_from_city,
        description=(
            "Finds real-time flights from the user's city to the destination city using the Amadeus API. "
            "Returns a list of flight dicts (price, currency, airline, dep/arr airports, dep_time, etc.). "
            "Input should be user city, destination city, date (YYYY-MM-DD), and number of adults."
        )
    )
