from amadeus import Client, ResponseError
from langchain.tools import Tool
from os import environ
from tools.currency import convert_amount
from utils.set_llm import get_llm
from prompts import format_prompt, PromptType


AMADEUS_CLIENT_ID = environ.get("AMADEUS_CLIENT_ID")
AMADEUS_CLIENT_SECRET = environ.get("AMADEUS_CLIENT_SECRET")

_amadeus_client = None

# Cache for airport lookups to avoid repeated API calls
_airport_cache = {}


def _resolve_location_intelligently(location: str) -> str:
    """Use LLM to resolve ambiguous locations to major cities with airports"""
    llm = get_llm(temperature=0.2)
    
    prompt = format_prompt(
        PromptType.FLIGHT_LOCATION_RESOLUTION,
        location=location
    )

    try:
        response = llm.invoke(prompt)
        resolved = response.content.strip().strip('"').strip()
        if resolved and resolved.lower() != location.lower():
            return resolved
        return location
    except Exception:
        return location


def _get_client():
    global _amadeus_client
    if _amadeus_client is None:
        if not (AMADEUS_CLIENT_ID and AMADEUS_CLIENT_SECRET):
            return None
        try:
            _amadeus_client = Client(
                client_id=AMADEUS_CLIENT_ID,
                client_secret=AMADEUS_CLIENT_SECRET
            )
        except Exception:
            _amadeus_client = None

    return _amadeus_client


def get_nearest_airport(city: str) -> str:
    """
    Uses Amadeus API to find the nearest airport IATA code for a given city.
    Uses LLM to intelligently resolve ambiguous locations first.
    """
    # Check cache first
    city_lower = city.lower().strip()
    if city_lower in _airport_cache:
        return _airport_cache[city_lower]
    
    # Use LLM to resolve the location intelligently
    resolved_city = _resolve_location_intelligently(city)
    resolved_lower = resolved_city.lower().strip()
    
    # Check cache for resolved city
    if resolved_lower in _airport_cache:
        return _airport_cache[resolved_lower]
    
    # Try API lookup
    client = _get_client()
    if client is None:
        return

    try:
        # Try multiple search strategies for better coverage
        search_terms = [resolved_city, f"{resolved_city} airport", city, f"{city} airport"]
        
        for search_term in search_terms:
            try:
                response = client.reference_data.locations.get(
                    keyword=search_term,
                    subType='AIRPORT'
                )
                data = response.data

                if data and isinstance(data, list):
                    # Find the best match (preferably with matching city name)
                    for airport in data:
                        airport_city = airport.get('address', {}).get('cityName', '').lower()
                        if (resolved_lower in airport_city or airport_city in resolved_lower or 
                            city_lower in airport_city or airport_city in city_lower):
                            iata_code = airport['iataCode']
                            # Cache both original and resolved city
                            _airport_cache[city_lower] = iata_code
                            _airport_cache[resolved_lower] = iata_code

                            return iata_code
                    
                    # If no perfect match, use the first result
                    iata_code = data[0]['iataCode']
                    _airport_cache[city_lower] = iata_code
                    _airport_cache[resolved_lower] = iata_code

                    return iata_code
                    
            except ResponseError as e:
                continue
                
        # Cache negative result to avoid repeated failures
        _airport_cache[city_lower] = None
        _airport_cache[resolved_lower] = None

        return

    except Exception:
        _airport_cache[city_lower] = None

        return


def search_flights_from_city(user_city: str, destination_city: str, date: str, adults: int) -> str:
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
        error_msg = _generate_intelligent_error_message(user_city, is_origin=True)
        return [{"error": error_msg}]

    if not dest_code:
        error_msg = _generate_intelligent_error_message(destination_city, is_origin=False)
        return [{"error": error_msg}]

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
                    continue
                    
            if results:
                # Sort by price (cheapest first)
                results.sort(key=lambda x: x['price'])
                return results
            else:
                return [{"error": "Found flights but could not parse pricing information."}]
        else:
            return [{
                "error": f"No flights found from {user_city} ({origin_code}) to {destination_city} ({dest_code}) on {date}. Try checking different dates or nearby airports."
            }]

    except ResponseError as e:
        return [{"error": f"Flight search failed: {str(e)}. Please try again or check your travel details."}]
    except Exception:
        return [{"error": "Unexpected error during flight search."}]


def _generate_intelligent_error_message(location: str, is_origin: bool = True) -> str:
    """Generate helpful error messages with LLM suggestions"""
    llm = get_llm(temperature=0.3)
    
    location_type = "departure city" if is_origin else "destination"
    
    prompt = format_prompt(
        PromptType.FLIGHT_ERROR_MESSAGE,
        location=location,
        location_type=location_type
    )

    try:
        response = llm.invoke(prompt)
        return response.content.strip()
    except Exception:
        return f"I couldn't find an airport for '{location}'. Could you try a nearby major city or provide the airport code directly (like 'DEL' for Delhi)?"


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
