from langchain.tools import Tool
from utils.set_llm import get_llm
from functools import lru_cache
from prompts import format_prompt, PromptType


@lru_cache(maxsize=32)
def suggest_activities(destination: str) -> str:
    """
    Uses the LLM to suggest activities based on the given destination.
    Results are cached to avoid repeated LLM calls for the same destination.
    """
    if not destination or not isinstance(destination, str):
        return "Please provide a valid travel destination as a text description."

    # Normalize destination for better cache hits
    destination_normalized = destination.strip().lower().title()
    
    llm = get_llm()
    prompt = format_prompt(
        PromptType.ACTIVITY_SUGGESTION,
        destination=destination_normalized
    )

    try:
        res = llm.invoke(prompt)
        content = str(getattr(res, 'content', res))
        return content
    except Exception:
        return "Sorry, I couldn't generate activity suggestions at this time. Please try again."


def get_activity_tool():
    return Tool(
        name="Activity Suggestion Tool",
        func=suggest_activities,
        description="Suggests activities based on the provided travel destination. Input should be a string describing the destination (e.g., 'Paris')."
    )
