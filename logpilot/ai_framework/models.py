"""Model builders for the optional LangChain extension."""

from __future__ import annotations

from langchain_openai import ChatOpenAI

from logpilot.ai_framework.config import (
    get_openai_api_key,
    get_openai_base_url,
    get_openai_model,
)


def build_chat_model(temperature: float = 0.0) -> ChatOpenAI:
    """Build an OpenAI-compatible LangChain chat model."""
    return ChatOpenAI(
        model=get_openai_model(),
        api_key=get_openai_api_key(),
        base_url=get_openai_base_url() or None,
        temperature=temperature,
    )

