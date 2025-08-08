from typing import TypedDict, List, Any


class TripContext(TypedDict, total=False):
    destination: str
    start_date: str
    end_date: str
    number_of_travelers: int
    interests: list[str]
    user_city: str
    budget: str
    proposed_start_date: str
    user_currency: str
    duration_days: int
    committed_start_date: str
    slot_status: dict  # slot -> unfilled|asked|committed


class TripPlannerState(TypedDict, total=False):
    user_input: str
    chat_history: List[Any]
    context: TripContext
    response: str
    handoff: bool
    missing_info: bool
    planning_stage: str  # 'smalltalk' | 'explore' | 'plan'
    plan: list  # ordered list of tool step dicts
    tool_cursor: int
    tool_results: dict
    errors: list
