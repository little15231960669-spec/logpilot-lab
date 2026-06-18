"""Configuration helpers for the optional LangChain extension."""

from __future__ import annotations

import os

from dotenv import load_dotenv


load_dotenv()


def get_openai_api_key() -> str:
    """Return the OpenAI-compatible API key, or an empty string if unset."""
    return os.getenv("OPENAI_API_KEY", "").strip()


def get_openai_base_url() -> str:
    """Return the OpenAI-compatible base URL, or an empty string if unset."""
    return os.getenv("OPENAI_BASE_URL", "").strip()


def get_openai_model() -> str:
    """Return the configured chat model name, or an empty string if unset."""
    return os.getenv("OPENAI_MODEL", "").strip()


def is_langchain_configured() -> bool:
    """Return whether the required LangChain/OpenAI-compatible settings exist."""
    api_key = get_openai_api_key()
    base_url = get_openai_base_url()
    model = get_openai_model()

    return bool(
        api_key
        and api_key != "your_api_key_here"
        and base_url
        and model
        and model != "your_model_name"
    )

