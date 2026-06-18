import json
import os
from typing import List, Dict, Any

import requests

from logpilot.llm.base import BaseLLMBackend


class OllamaBackend(BaseLLMBackend):
    """
    Local Ollama backend.

    Make sure Ollama is running locally before using this backend:
        ollama serve
        ollama run llama3:8b
    """

    name = "Ollama"

    def __init__(
        self,
        base_url: str | None = None,
        model: str | None = None,
        timeout: int = 120,
        temperature: float = 0.0,
    ):
        self.base_url = (base_url or os.getenv("OLLAMA_BASE_URL") or "http://localhost:11434").rstrip("/")
        self.model = model or os.getenv("OLLAMA_MODEL") or "llama3:8b"
        self.timeout = timeout
        self.temperature = temperature

    def _build_messages(
        self,
        records: List[Dict[str, Any]],
        text_field: str,
    ) -> List[Dict[str, str]]:
        input_items = []

        for record in records:
            text = record.get(text_field) or record.get("content") or record.get("raw_log") or ""
            input_items.append(
                {
                    "line_id": record["line_id"],
                    "log": text,
                }
            )

        system_prompt = (
            "You are a log parsing assistant. "
            "Return valid JSON only. Do not add markdown code fences or explanations."
        )

        user_prompt = f"""
Convert each log message into a log template.

Rules:
1. Replace variable parts such as IDs, IP addresses, paths, numbers, timestamps, ports, and block IDs with <*>.
2. Keep constant words unchanged.
3. Return a JSON array only.
4. Each output item must contain: line_id, template, variables, confidence.

Input logs:
{json.dumps(input_items, ensure_ascii=False, indent=2)}

Required JSON schema:
[
  {{
    "line_id": 1,
    "template": "Receiving block <*> src: <*> dest: <*>",
    "variables": ["blk_123", "/10.1.1.1:50010", "/10.1.1.2:50010"],
    "confidence": 0.95
  }}
]
""".strip()

        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

    def parse_batch(
        self,
        records: List[Dict[str, Any]],
        text_field: str = "content",
    ) -> str:
        url = f"{self.base_url}/api/chat"

        payload = {
            "model": self.model,
            "messages": self._build_messages(records, text_field),
            "stream": False,
            "options": {
                "temperature": self.temperature,
            },
        }

        response = requests.post(
            url,
            json=payload,
            timeout=self.timeout,
        )

        response.raise_for_status()
        data = response.json()

        return data["message"]["content"].strip()