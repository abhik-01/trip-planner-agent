from functools import lru_cache
from langchain.tools import Tool
from prompts import format_prompt, PromptType
from utils.set_llm import get_llm


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
    prompt = format_prompt(
        PromptType.DESTINATION_SUGGESTION,
        preferences=preferences_normalized
    )

    try:
        res = llm.invoke(prompt)
        content = str(getattr(res, 'content', res))

        return content
    except Exception:
        return "Sorry, I couldn't generate destination suggestions at this time. Please try again with different preferences."


def get_destination_tool():
    return Tool(
        name="Destination Suggestion Tool",
        func=suggest_destinations,
        description="Suggests travel destinations based on user preferences. Input should be a string describing preferences (e.g., 'I like beaches and warm weather')."
    )
