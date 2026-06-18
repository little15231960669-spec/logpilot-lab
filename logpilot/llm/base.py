from abc import ABC, abstractmethod
from typing import List, Dict, Any


class BaseLLMBackend(ABC):
    """
    Base interface for all LLM backends.

    Different backends can be implemented later:
    - MockBackend
    - OllamaBackend
    - OpenAICompatibleBackend
    """

    name: str = "base"

    @abstractmethod
    def parse_batch(
        self,
        records: List[Dict[str, Any]],
        text_field: str = "content",
    ) -> str:
        """
        Parse a batch of log records and return a raw JSON string.

        The returned string should be a JSON array:
        [
            {
                "line_id": 1,
                "template": "...",
                "variables": ["..."],
                "confidence": 0.95
            }
        ]
        """
        raise NotImplementedError