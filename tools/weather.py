from datetime import datetime
from langchain.tools import Tool
from requests import get
from functools import lru_cache

WEATHER_CODE_MAP = {
    0: "Clear sky",
    1: "Mainly clear",
    2: "Partly cloudy",
    3: "Overcast",
    45: "Fog",
    48: "Depositing rime fog",
    51: "Light drizzle",
    53: "Moderate drizzle",
    55: "Dense drizzle",
    61: "Slight rain",
    63: "Moderate rain",
    65: "Heavy rain",
    71: "Slight snow fall",
    73: "Moderate snow fall",
    75: "Heavy snow fall",
    80: "Rain showers",
    81: "Heavy rain showers",
    82: "Violent rain showers",
    95: "Thunderstorm",
    96: "Thunderstorm with slight hail",
    99: "Thunderstorm with heavy hail",
}


@lru_cache(maxsize=128)
def get_lat_lon(city: str):
    """
    Uses Open-Meteo's geocoding API to get latitude and longitude for a city.
    """
    url = "https://geocoding-api.open-meteo.com/v1/search"
    params = {"name": city, "count": 1}

    try:
        resp = get(url, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()

        if data.get("results"):
            lat = data["results"][0]["latitude"]
            lon = data["results"][0]["longitude"]

            return lat, lon

        return None, None
    except Exception:
        return None, None


def get_weather(city: str, date: str) -> str:
    """
    Gets weather for the given city and date using Open-Meteo API.
    If forecast is not available for the date, returns current weather.
    """
    lat, lon = get_lat_lon(city)

    if lat is None or lon is None:
        return f"Could not find location for '{city}'."

    today = datetime.utcnow().date()

    try:
        trip_date = datetime.strptime(date, "%Y-%m-%d").date()
    except Exception:
        return "Invalid date format. Please use YYYY-MM-DD."

    days_ahead = (trip_date - today).days

    if 0 <= days_ahead <= 15:
        url = "https://api.open-meteo.com/v1/forecast"
        params = {
            "latitude": lat,
            "longitude": lon,
            "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum,weathercode",
            "timezone": "auto",
            "start_date": date,
            "end_date": date
        }

        try:
            resp = get(url, params=params, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            daily = data.get("daily", {})

            if daily and daily.get("temperature_2m_max"):
                tmax = daily["temperature_2m_max"][0]
                tmin = daily["temperature_2m_min"][0]
                precip = daily["precipitation_sum"][0]

                return (
                    f"Weather forecast for {city} on {date}: Max {tmax}°C / Min {tmin}°C, Precip {precip} mm"
                )
        except Exception:
            pass

    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "current_weather": True,
        "timezone": "auto"
    }

    try:
        resp = get(url, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        current = data.get("current_weather", {})

        if current:
            temp = current.get("temperature")
            wind = current.get("windspeed")
            code = current.get("weathercode")
            desc = WEATHER_CODE_MAP.get(code, f"Code {code}")
            return (
                f"Current weather in {city}: {desc}, Temp {temp}°C, Wind {wind} km/h. "
                f"(Forecast for {date} not available; showing current.)"
            )
        else:
            return f"Could not retrieve weather data for {city}."

    except Exception:
        return f"Could not retrieve weather data for {city}."


def get_weather_tool() -> Tool:
    return Tool(
        name="Weather Tool",
        func=get_weather,
        description="Gets the weather for a city and date (YYYY-MM-DD). If forecast is not available, returns current weather."
    )
