from amadeus import Client, ResponseError
from langchain.tools import Tool
from os import environ
from tools.currency import convert_amount


AMADEUS_CLIENT_ID = environ.get("AMADEUS_CLIENT_ID")
AMADEUS_CLIENT_SECRET = environ.get("AMADEUS_CLIENT_SECRET")

_amadeus_client = None

# Fallback IATA codes for major cities if API lookup fails or credentials missing
FALLBACK_AIRPORTS = {
    'kolkata': 'CCU', 'calcutta': 'CCU',
    'goa': 'GOI',
    'mumbai': 'BOM', 'bombay': 'BOM',
    'delhi': 'DEL', 'new delhi': 'DEL',
    'bengaluru': 'BLR', 'bangalore': 'BLR',
    'chennai': 'MAA',
    'hyderabad': 'HYD',
    'pune': 'PNQ',
    'ahmedabad': 'AMD',
    'jaipur': 'JAI',
    'kochi': 'COK', 'cochin': 'COK',
    'varanasi': 'VNS',
    'lucknow': 'LKO'
}


def _get_client():
    global _amadeus_client
    if _amadeus_client is None:
        if not (AMADEUS_CLIENT_ID and AMADEUS_CLIENT_SECRET):
            print("[WARN] Amadeus credentials missing; flight search disabled.")
            return None
        try:
            _amadeus_client = Client(
                client_id=AMADEUS_CLIENT_ID,
                client_secret=AMADEUS_CLIENT_SECRET
            )
        except Exception as e:
            print(f"[ERROR] Could not initialize Amadeus client: {e}")
            _amadeus_client = None
    return _amadeus_client


def get_nearest_airport(city: str) -> str:
    print(f"[API CALL] Amadeus: get_nearest_airport(city={city})")
    """
    Uses Amadeus API to find the nearest airport IATA code for a given city.
    """
    # Fallback first if no credentials
    client = _get_client()
    if client is None:
        return FALLBACK_AIRPORTS.get(city.lower())
    try:
        response = client.reference_data.locations.get(
            keyword=city,
            subType='AIRPORT',
            page_limit=1
        )
        data = response.data

        if data and isinstance(data, list):
            return data[0]['iataCode']
        # Try fallback mapping if API returns nothing
        return FALLBACK_AIRPORTS.get(city.lower())
    except ResponseError:
        return FALLBACK_AIRPORTS.get(city.lower())


def search_flights_from_city(user_city: str, destination_city: str, date: str, adults: int) -> str:
    print(f"[API CALL] Amadeus: search_flights_from_city(user_city={user_city}, destination_city={destination_city}, date={date}, adults={adults})")
    """
    Finds flights from the nearest airport to the user's city to the destination city.
    Returns a list of flight dicts (price, currency, airline, dep/arr airports, dep_time, etc.).
    If no airport is found, returns a dict with an 'error' key and message.
    """
    if not (user_city and destination_city and date):
        return [{"error": "Please provide your city, destination city, and date (YYYY-MM-DD)."}]

    # Ensure client exists
    client = _get_client()
    if client is None:
        return [{"error": "Flight search not configured: missing Amadeus credentials."}]

    origin_code = get_nearest_airport(user_city)
    dest_code = get_nearest_airport(destination_city)

    if not origin_code:
        return [{
            "error": f"Could not resolve an airport code for {user_city} (API lookup failed). You can still proceed; consider providing the nearest major airport name."
        }]

    if not dest_code:
        return [{"error": f"Sorry, couldn't find an airport near {destination_city}."}]

    try:
        response = client.shopping.flight_offers_search.get(
            originLocationCode=origin_code,
            destinationLocationCode=dest_code,
            departureDate=date,
            adults=adults,
            max=5
        )
        flights = getattr(response, 'data', []) or []

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
                price_float = float(price)
                inr_price = convert_amount(price_float, currency, "INR")
                results.append({
                    "price": price_float,
                    "currency": currency,
                    "price_in_inr": inr_price,
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
        base_line = f"{f['departure_airport']} → {f['arrival_airport']} | {f['airline']} | {f['departure_time']} | {f['currency']} {f['price']}"
        if 'price_in_inr' in f and f.get('currency') != 'INR':
            base_line += f" (≈ INR {f['price_in_inr']})"
        lines.append(base_line)

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
