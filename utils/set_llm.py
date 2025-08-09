from langchain_together import ChatTogether
from os import environ
from utils.config import LLM_MODEL, LLM_TEMPERATURE


_LLM_CACHE = {}


def get_llm(*, model: str | None = None, temperature: float | None = None):
    """Return a cached LLM client. Allows per-call overrides for model/temperature.
    Cache keyed by (model, temperature) to avoid rebuilding for each call.
    """
    m = model or LLM_MODEL
    t = LLM_TEMPERATURE if temperature is None else temperature
    key = (m, float(t))
    if key not in _LLM_CACHE:
        _LLM_CACHE[key] = ChatTogether(
            api_key=environ.get("API_KEY"),
            temperature=t,
            model=m,
        )
    return _LLM_CACHE[key]