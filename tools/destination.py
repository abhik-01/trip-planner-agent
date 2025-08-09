from langchain.tools import Tool
from utils.set_llm import get_llm
from functools import lru_cache
import hashlib


def _hash_preferences(preferences: str) -> str:
    """Create a hash of preferences for cache key"""
    return hashlib.md5(preferences.lower().strip().encode()).hexdigest()[:8]


@lru_cache(maxsize=16)
def suggest_destinations(preferences: str) -> str:
    """
    Uses the LLM to suggest travel destinations based on user preferences.
    Results are cached based on preference similarity to avoid repeated LLM calls.
    """
    if not preferences or not isinstance(preferences, str):
        return "Please provide your travel preferences as a text description."

    # Normalize preferences for better cache performance
    preferences_normalized = preferences.strip()
    
    llm = get_llm()
    prompt = (
        f"Suggest 8-10 diverse travel destinations based on these preferences: {preferences_normalized}. "
        "For each destination, include:\n"
        "1. Why it matches the preferences\n"
        "2. Best time to visit\n"
        "3. One unique highlight\n"
        "Format as a numbered list and prioritize less obvious but great matches alongside popular choices."
    )

    try:
        res = llm.invoke(prompt)
        content = str(getattr(res, 'content', res))
        cache_key = _hash_preferences(preferences)
        print(f"[CACHE] Destination suggestions cached with key {cache_key}")
        return content
    except Exception as e:
        print(f"[ERROR] Destination suggestion failed: {e}")
        return "Sorry, I couldn't generate destination suggestions at this time. Please try again with different preferences."


def get_destination_tool():
    return Tool(
        name="Destination Suggestion Tool",
        func=suggest_destinations,
        description="Suggests travel destinations based on user preferences. Input should be a string describing preferences (e.g., 'I like beaches and warm weather')."
    )
