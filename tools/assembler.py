from langchain.tools import Tool
from utils.set_llm import get_llm


def assemble_itinerary(trip_data: dict) -> str:
    """
    Uses the LLM to assemble a detailed, readable itinerary from all trip components.
    Expects trip_data to include keys like: destination, activities, flights, weather, budget, dates, travelers, etc.
    """
    llm = get_llm()
    prompt = (
        f"Given the following trip details, assemble a clear, friendly, and well-structured itinerary for the user:\n\n"
        f"{trip_data}\n\n"
        "Format the response as a day-by-day plan if possible, include travel tips, and make it engaging. "
        "Highlight important info like flights, weather, activities, and budget summary."
    )

    return llm.invoke(prompt)


def get_assembler_tool():
    return Tool(
        name="Itinerary Assembler Tool",
        func=assemble_itinerary,
        description="Compiles all trip details into a cohesive, user-friendly itinerary. Input should be a dict with all trip components."
    )
