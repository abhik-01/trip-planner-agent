from langchain.agents import initialize_agent, AgentType
from utils.set_llm import get_llm
from tools.budget import get_budget_tool


llm = get_llm()


def get_budget_agent():
    """
    Initializes the Budget Agent with its own tools.
    Handles cost estimation and budget feasibility.
    """
    tools = [
        get_budget_tool()
    ]
    agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True
    )

    return agent
