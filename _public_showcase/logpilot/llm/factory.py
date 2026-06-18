from logpilot.llm.mock_backend import MockBackend
from logpilot.llm.ollama_backend import OllamaBackend
from logpilot.llm.openai_compatible import OpenAICompatibleBackend


BACKEND_OPTIONS = [
    "MockBackend",
    "OllamaBackend",
    "OpenAICompatibleBackend",
]


def create_llm_backend(backend_name: str):
    if backend_name == "MockBackend":
        return MockBackend()

    if backend_name == "OllamaBackend":
        return OllamaBackend()

    if backend_name == "OpenAICompatibleBackend":
        return OpenAICompatibleBackend()

    raise ValueError(f"Unknown LLM backend: {backend_name}")