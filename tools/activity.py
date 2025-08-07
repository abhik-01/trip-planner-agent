from langchain.tools import Tool
from utils.set_llm import get_llm


def suggest_activities(destination: str) -> str:
    """
    Uses the LLM to suggest activities based on the given destination.
    """
    if not destination or not isinstance(destination, str):
        return "Please provide a valid travel destination as a text description."

    llm = get_llm()
    prompt = (
        f"List around 8 popular and diverse activities that a traveler can enjoy in {destination}. "
        "Include a short description for each activity."
    )

    try:
        return llm.invoke(prompt)
    except Exception as e:
        return "Sorry, I couldn't generate activity suggestions at this time."


def get_activity_tool():
    return Tool(
        name="Activity Suggestion Tool",
        func=suggest_activities,
        description="Suggests activities based on the provided travel destination. Input should be a string describing the destination (e.g., 'Paris')."
    )
