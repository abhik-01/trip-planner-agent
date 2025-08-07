from langchain_together import ChatTogether
from os import environ
from utils.config import LLM_MODEL, LLM_TEMPERATURE


_llm_instance = None


def get_llm():
    global _llm_instance

    if _llm_instance is None:
        _llm_instance = ChatTogether(
            api_key=environ.get("API_KEY"),
            temperature=LLM_TEMPERATURE,
            model=LLM_MODEL
        )

    return _llm_instance