
"""Groq chat models (via LangChain), created once and reused."""
from langchain_groq import ChatGroq

from app import config

_cache: dict[str, ChatGroq] = {}


def get_llm(fast: bool = False) -> ChatGroq:
    """Return a cached Groq chat model."""
    if not config.GROQ_API_KEY:
        raise RuntimeError(
            "GROQ_API_KEY is missing. Add it to your .env file."
        )

    model = config.GROQ_FAST_MODEL if fast else config.GROQ_MODEL

    if not model:
        raise RuntimeError("Groq model name is not configured.")

    if model not in _cache:
        _cache[model] = ChatGroq(
            model=model,
            api_key=config.GROQ_API_KEY,
            temperature=0,
            max_retries=3,
        )

    return _cache[model]
