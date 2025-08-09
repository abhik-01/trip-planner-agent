from amadeus import Client, ResponseError
from langchain.tools import Tool
from os import environ
from tools.currency import convert_amount


AMADEUS_CLIENT_ID = environ.get("AMADEUS_CLIENT_ID")
AMADEUS_CLIENT_SECRET = environ.get("AMADEUS_CLIENT_SECRET")

_amadeus_client = None

# Cache for airport lookups to avoid repeated API calls
_airport_cache = {}


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
    Uses caching to avoid repeated API calls for the same city.
    """
    # Check cache first
    city_lower = city.lower().strip()
    if city_lower in _airport_cache:
        print(f"[CACHE HIT] Using cached airport code for {city}")
        return _airport_cache[city_lower]
    
    # Try API lookup
    client = _get_client()
    if client is None:
        print(f"[ERROR] No Amadeus client available for airport lookup: {city}")
        return None
        
    try:
        # Try multiple search strategies for better coverage
        search_terms = [city, f"{city} airport", f"{city} city"]
        
        for search_term in search_terms:
            try:
                response = client.reference_data.locations.get(
                    keyword=search_term,
                    subType='AIRPORT',
                    page_limit=3  # Get top 3 results
                )
                data = response.data

                if data and isinstance(data, list):
                    # Find the best match (preferably with matching city name)
                    for airport in data:
                        airport_city = airport.get('address', {}).get('cityName', '').lower()
                        if city_lower in airport_city or airport_city in city_lower:
                            iata_code = airport['iataCode']
                            _airport_cache[city_lower] = iata_code
                            print(f"[SUCCESS] Found airport {iata_code} for {city}")
                            return iata_code
                    
                    # If no perfect match, use the first result
                    iata_code = data[0]['iataCode']
                    _airport_cache[city_lower] = iata_code
                    print(f"[SUCCESS] Found nearest airport {iata_code} for {city}")
                    return iata_code
                    
            except ResponseError as e:
                print(f"[DEBUG] Search term '{search_term}' failed: {e}")
                continue
                
        # Cache negative result to avoid repeated failures
        _airport_cache[city_lower] = None
        print(f"[ERROR] No airport found for {city} after trying multiple search terms")
        return None
        
    except Exception as e:
        print(f"[ERROR] Unexpected error in airport lookup for {city}: {e}")
        _airport_cache[city_lower] = None
        return None


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

    # Get airport codes with improved error handling
    origin_code = get_nearest_airport(user_city)
    dest_code = get_nearest_airport(destination_city)

    if not origin_code:
        return [{
            "error": f"Could not find an airport for '{user_city}'. Please try with a nearby major city name or provide the airport code directly (e.g., 'Mumbai' or 'BOM')."
        }]

    if not dest_code:
        return [{
            "error": f"Could not find an airport for '{destination_city}'. Please try with a major city name or provide the airport code directly."
        }]

    print(f"[INFO] Using airports: {user_city} -> {origin_code}, {destination_city} -> {dest_code}")

    try:
        # Add some flight search options for better results
        response = client.shopping.flight_offers_search.get(
            originLocationCode=origin_code,
            destinationLocationCode=dest_code,
            departureDate=date,
            adults=adults,
            max=10,  # Increased from 5 for more options
            currencyCode='USD'  # Consistent currency for conversion
        )
        flights = getattr(response, 'data', []) or []

        if flights:
            results = []
            for offer in flights:
                try:
                    price = float(offer['price']['total'])
                    currency = offer['price'].get('currency', 'USD')
                    
                    # Get first segment of first itinerary
                    first_segment = offer['itineraries'][0]['segments'][0]
                    dep = first_segment['departure']['iataCode']
                    arr = first_segment['arrival']['iataCode']
                    dep_time = first_segment['departure']['at']
                    airline = first_segment['carrierCode']
                    
                    # Convert to INR for consistent pricing
                    inr_price = convert_amount(price, currency, "INR")
                    
                    results.append({
                        "price": price,
                        "currency": currency,
                        "price_in_inr": inr_price,
                        "airline": airline,
                        "departure_airport": dep,
                        "arrival_airport": arr,
                        "departure_time": dep_time,
                        "duration": offer['itineraries'][0].get('duration', 'N/A')
                    })
                except (KeyError, ValueError, TypeError) as e:
                    print(f"[WARN] Skipping malformed flight offer: {e}")
                    continue
                    
            if results:
                # Sort by price (cheapest first)
                results.sort(key=lambda x: x['price'])
                print(f"[SUCCESS] Found {len(results)} flights from {origin_code} to {dest_code}")
                return results
            else:
                return [{"error": "Found flights but could not parse pricing information."}]
        else:
            return [{
                "error": f"No flights found from {user_city} ({origin_code}) to {destination_city} ({dest_code}) on {date}. Try checking different dates or nearby airports."
            }]

    except ResponseError as e:
        print(f"[ERROR] Amadeus API error: {e}")
        return [{"error": f"Flight search failed: {str(e)}. Please try again or check your travel details."}]
    except Exception as e:
        print(f"[ERROR] Unexpected error in flight search: {e}")
        return [{"error": f"Unexpected error during flight search: {str(e)}"}]


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
