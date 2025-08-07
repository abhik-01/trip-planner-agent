from langchain.tools import Tool
from requests import get, RequestException


GEOAPIFY_API_KEY = ""


def find_nearby_places(location: str, category: str = "tourism") -> str:
    """
    Uses the Geoapify API to find nearby places based on a given location.
    """
    if not location or not isinstance(location, str):
        return "Please provide a valid location as a text description."

    url = "https://api.geoapify.com/v2/places"
    params = {
        "categories": category,
        "filter": f"place:{location}",
        "limit": 5,
        "apiKey": GEOAPIFY_API_KEY
    }

    try:
        response = get(url, params=params)
        response.raise_for_status()
        data = response.json()

        if not data.get('features'):
            return "No nearby places found."

        results = []

        for place in data["features"]:
            name = place["properties"].get("name", "Unknown")
            address = place["properties"].get("formatted", "")
            results.append(f"{name} - {address}")

        return "\n".join(results)

    except RequestException as e:
        return f"Error fetching nearby places: {str(e)}"


def get_map_tool() -> Tool:
    return Tool(
        name="Map Tool",
        func=find_nearby_places,
        description="Finds nearby places of interest. Input should be a string describing the location and an optional category (e.g., 'Paris', 'tourism')."
    )
