import json
import os
from typing import List, Dict, Any
import time
import requests

from logpilot.llm.base import BaseLLMBackend


class OpenAICompatibleBackend(BaseLLMBackend):
    """
    OpenAI-compatible chat completion backend.

    It supports providers that follow the OpenAI /v1/chat/completions style API.
    """

    name = "OpenAI-compatible"

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        timeout: int = 120,
        temperature: float = 0.0,
        max_retries: int = 3,
        retry_backoff: float = 1.5,
    ):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.base_url = (base_url or os.getenv("OPENAI_BASE_URL") or "https://api.openai.com/v1").rstrip("/")
        self.model = model or os.getenv("OPENAI_MODEL") or "gpt-4o-mini"
        self.timeout = timeout
        self.temperature = temperature
        self.max_retries = max_retries
        self.retry_backoff = retry_backoff

        if not self.api_key:
            raise ValueError(
                "OPENAI_API_KEY is missing. Please create a local .env file based on .env.example."
            )

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
            "Your task is to convert raw log messages into log templates. "
            "Return valid JSON only. Do not add explanations."
        )

        user_prompt = f"""
Convert each log message into a log template.

Rules:
1. Replace variable parts such as IDs, IP addresses, paths, numbers, timestamps, ports, and block IDs with <*>.
2. Keep constant words unchanged.
3. Return a JSON array only.
4. Each output item must contain: line_id, template, variables, confidence.
5. variables should be a list of extracted variable values.
6. confidence should be a float between 0 and 1.

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
        url = f"{self.base_url}/chat/completions"
    
        payload = {
            "model": self.model,
            "messages": self._build_messages(records, text_field),
            "temperature": self.temperature,
        }
    
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
    
        last_error = None
    
        for attempt in range(self.max_retries + 1):
            try:
                response = requests.post(
                    url,
                    headers=headers,
                    json=payload,
                    timeout=self.timeout,
                )
                response.raise_for_status()
                data = response.json()
                return data["choices"][0]["message"]["content"].strip()
    
            except requests.RequestException as exc:
                last_error = exc
                if attempt < self.max_retries:
                    time.sleep(self.retry_backoff * (attempt + 1))
                    continue
                raise last_error