"""Groq chat models (via LangChain), created once and reused."""
from langchain_groq import ChatGroq

from app import config

_cache: dict[str, ChatGroq] = {}


def get_llm(fast: bool = False) -> ChatGroq:
    """fast=True uses the helper model for lightweight tasks."""
    model = config.GROQ_FAST_MODEL if fast else config.GROQ_MODEL
    if model not in _cache:
        _cache[model] = ChatGroq(
            model=model,
            api_key=config.GROQ_API_KEY,
            temperature=0,
            max_retries=3,   # retries automatically on rate limits
        )
    return _cache[model]