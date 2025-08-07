from langchain.tools import Tool
from utils.set_llm import get_llm


def suggest_destinations(preferences: str) -> str:
    """
    Uses the LLM to suggest travel destinations based on user preferences.
    """
    if not preferences or not isinstance(preferences, str):
        return "Please provide your travel preferences as a text description."

    llm = get_llm()
    prompt = (
        f"Suggest 10 travel destinations for the following preferences: {preferences}. "
        "Be diverse and explain briefly why each destination fits."
    )

    try:
        return llm.invoke(prompt)
    except Exception as e:
        return "Sorry, I couldn't generate destination suggestions at this time."


def get_destination_tool():
    return Tool(
        name="Destination Suggestion Tool",
        func=suggest_destinations,
        description="Suggests travel destinations based on user preferences. Input should be a string describing preferences (e.g., 'I like beaches and warm weather')."
    )
