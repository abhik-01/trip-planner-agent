from langchain.tools import Tool
from utils.set_llm import get_llm
import re
import hashlib


def _create_itinerary_cache_key(trip_data: dict) -> str:
    """Create a cache key for itinerary based on core data"""
    key_params = {
        'destination': str(trip_data.get('destination', '')).lower().strip(),
        'start_date': trip_data.get('start_date', ''),
        'duration': trip_data.get('duration_days', 0),
        'travelers': trip_data.get('number_of_travelers', 1),
        'has_flights': bool(trip_data.get('flights')),
        'has_weather': bool(trip_data.get('weather')),
        'has_activities': bool(trip_data.get('activities')),
        'has_budget': bool(trip_data.get('budget'))
    }
    key_str = str(sorted(key_params.items()))
    return hashlib.md5(key_str.encode()).hexdigest()[:8]


# Simple cache for assembled itineraries
_itinerary_cache = {}


def _convert_usd_to_inr_in_text(text: str) -> str:
    """Convert any USD amounts found in text to INR and append the conversion"""
    try:
        from tools.currency import convert_amount
        
        # Find USD amounts in the text (patterns like $500, USD 500, $1,200, etc.)
        usd_patterns = [
            r'\$(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)',  # $500, $1,200, $1,200.50
            r'USD\s*(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)',  # USD 500, USD 1,200
            r'(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)\s*USD',  # 500 USD, 1,200 USD
        ]
        
        converted_text = text
        conversions_made = []
        
        for pattern in usd_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                amount_str = match.group(1).replace(',', '')  # Remove commas
                try:
                    amount = float(amount_str)
                    inr_amount = convert_amount(amount, 'USD', 'INR')
                    conversions_made.append(f"${amount_str} USD = ₹{inr_amount} INR")
                except Exception:
                    continue
        
        # If we made conversions, append them to the text
        if conversions_made:
            converted_text += "\n\n💱 Currency Conversions:\n"
            for conversion in conversions_made:
                converted_text += f"• {conversion}\n"
                
        return converted_text
        
    except Exception as e:
        print(f"[DEBUG] Currency conversion in text failed: {e}")
        return text


def assemble_itinerary(trip_data: dict) -> str:
    """
    Uses the LLM to assemble a detailed, readable itinerary from all trip components.
    Expects trip_data to include keys like: destination, activities, flights, weather, budget, dates, travelers, etc.
    """
    # Input validation
    if not isinstance(trip_data, dict):
        return "Error: Trip data must be provided as a dictionary."
    
    destination = trip_data.get('destination')
    if not destination:
        return "Error: Destination is required to assemble itinerary."
    
    # Check cache first
    cache_key = _create_itinerary_cache_key(trip_data)
    if cache_key in _itinerary_cache:
        print(f"[CACHE HIT] Using cached itinerary for key {cache_key}")
        cached_result = _itinerary_cache[cache_key]
        return _convert_usd_to_inr_in_text(cached_result)  # Always apply currency conversion
    
    llm = get_llm()
    
    # Create more structured prompt for better results
    data_summary = []
    if trip_data.get('flights'):
        data_summary.append(f"✈️ Flights: {len(trip_data['flights'])} options found")
    if trip_data.get('weather'):
        data_summary.append("🌤️ Weather information available")
    if trip_data.get('activities'):
        data_summary.append("🎯 Activity suggestions available")
    if trip_data.get('budget'):
        data_summary.append("💰 Budget estimate available")
    if trip_data.get('nearby'):
        data_summary.append("📍 Nearby places information available")
    
    prompt = (
        f"Create a comprehensive, engaging travel itinerary for {destination} using the following data:\n\n"
        f"📊 Available Data: {', '.join(data_summary)}\n\n"
        f"🗂️ Trip Details:\n{trip_data}\n\n"
        "📋 Requirements:\n"
        "- Create a day-by-day plan if duration is specified\n"
        "- Include all available information (flights, weather, activities, budget)\n"
        "- Make it engaging and user-friendly\n"
        "- Highlight important details with emojis\n"
        "- Include practical travel tips\n"
        "- Structure with clear headings and sections\n"
        "- All prices should be in INR (Indian Rupees)\n"
        "- Be enthusiastic but informative"
    )

    try:
        res = llm.invoke(prompt)
        itinerary_text = str(getattr(res, 'content', res))
        
        # Cache the raw result (before currency conversion)
        _itinerary_cache[cache_key] = itinerary_text
        print(f"[CACHE] Itinerary cached with key {cache_key}")
        
        # Convert any remaining USD amounts to INR for user clarity
        itinerary_with_conversions = _convert_usd_to_inr_in_text(itinerary_text)
        
        return itinerary_with_conversions
        
    except Exception as e:
        print(f"[ERROR] Itinerary assembly failed: {e}")
        return f"Sorry, I couldn't assemble the itinerary at this time. Here's what I have for {destination}: {str(trip_data)[:500]}..."


def get_assembler_tool():
    return Tool(
        name="Itinerary Assembler Tool",
        func=assemble_itinerary,
        description="Compiles all trip details into a cohesive, user-friendly itinerary. Input should be a dict with all trip components."
    )
