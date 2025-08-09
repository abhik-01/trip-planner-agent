from langchain.tools import Tool
from os import environ
from requests import get, RequestException
from functools import lru_cache


GEOAPIFY_API_KEY = environ.get("GEOAPIFY_API_KEY")


@lru_cache(maxsize=64)
def _geocode_place(query: str):
    if not GEOAPIFY_API_KEY:
        return None
    url = "https://api.geoapify.com/v1/geocode/search"
    params = {"text": query, "limit": 1, "apiKey": GEOAPIFY_API_KEY}
    try:
        r = get(url, params=params, timeout=10)
        r.raise_for_status()
        data = r.json()
        feats = data.get('features') or []
        if feats:
            props = feats[0]['properties']
            return props.get('lon'), props.get('lat')
    except Exception as e:
        print(f"[WARN] geocode failed: {e}")
    return None


def find_nearby_places(location: str, category: str = "tourism") -> str:
    print(f"[API CALL] Geoapify: find_nearby_places(location={location}, category={category})")
    if not location or not isinstance(location, str):
        return "Please provide a valid location."
    if not GEOAPIFY_API_KEY:
        return "Nearby places search not configured (missing GEOAPIFY_API_KEY)."

    # Geocode first
    coords = _geocode_place(location)
    if not coords:
        return "Could not geocode location for nearby search."
    lon, lat = coords

    # Use circle filter around coordinates
    url = "https://api.geoapify.com/v2/places"
    params = {
        "categories": category or 'tourism',
        "filter": f"circle:{lon},{lat},3000",
        "limit": 6,
        "apiKey": GEOAPIFY_API_KEY
    }
    try:
        response = get(url, params=params, timeout=15)
        response.raise_for_status()
        data = response.json()
        feats = data.get('features') or []
        if not feats:
            return "No nearby places found."
        lines = []
        for p in feats:
            props = p.get('properties', {})
            name = props.get('name') or 'Unnamed'
            cat = props.get('categories', [category])[0] if props.get('categories') else category
            addr = props.get('formatted', '')
            lines.append(f"{name} ({cat}) - {addr}")
        return "\n".join(lines)
    except RequestException as e:
        return f"Error fetching nearby places: {e}"


def get_map_tool() -> Tool:
    return Tool(
        name="Map Tool",
        func=find_nearby_places,
        description="Finds nearby places of interest. Input should be a string describing the location and an optional category (e.g., 'Paris', 'tourism')."
    )
