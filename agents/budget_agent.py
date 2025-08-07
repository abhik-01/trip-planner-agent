from langchain.agents import initialize_agent, AgentType
from langchain.tools import Tool
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


def budget_agent_tool():
    """
    Wraps the Budget Agent as a tool for use by other agents.
    """
    agent = get_budget_agent()

    def run_budget_agent(input):
        return agent.run(input)

    return Tool(
        name="Budget Agent",
        func=run_budget_agent,
        description="Estimates trip costs and checks budget feasibility based on trip details. Input should be a dict with keys: flight_cost, destination, nights, activities, days, travelers."
    )
