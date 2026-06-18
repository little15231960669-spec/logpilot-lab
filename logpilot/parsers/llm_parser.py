import json
import re
import time
from typing import List, Dict, Any

import pandas as pd

from logpilot.llm.base import BaseLLMBackend


class LLMLogParser:
    """
    LLM Direct parser.

    It sends log messages to an LLM backend and expects structured JSON output.
    """

    def __init__(
        self,
        backend: BaseLLMBackend,
        batch_size: int = 8,
    ):
        self.backend = backend
        self.batch_size = batch_size

    def _estimate_tokens(self, text: str) -> int:
        """
        A lightweight token estimation for product-level cost analysis.
        This is not exact, but enough for the demo.
        """
        if not text:
            return 0
        return max(1, len(text) // 4)

    def _extract_json_array_text(self, raw_response: str) -> str:
        """
        Real LLMs may return markdown code fences or explanations.
        This function tries to extract the first JSON array from the response.
        """
        if not raw_response:
            return ""

        text = raw_response.strip()

        # Remove markdown code fence if present.
        text = re.sub(r"^```(?:json)?", "", text.strip(), flags=re.IGNORECASE)
        text = re.sub(r"```$", "", text.strip())

        start = text.find("[")
        end = text.rfind("]")

        if start != -1 and end != -1 and end > start:
            return text[start:end + 1]

        return text

    def parse(
        self,
        records: List[Dict[str, Any]],
        text_field: str = "content",
    ) -> pd.DataFrame:
        rows = []

        for start_idx in range(0, len(records), self.batch_size):
            batch = records[start_idx:start_idx + self.batch_size]

            batch_input_text = "\n".join(
                str(record.get(text_field) or record.get("content") or "")
                for record in batch
            )

            input_tokens = self._estimate_tokens(batch_input_text)

            backend_error = ""

            start_time = time.perf_counter()

            try:
                raw_response = self.backend.parse_batch(batch, text_field=text_field)
            except Exception as exc:
                raw_response = ""
                backend_error = str(exc)

            batch_latency_ms = (time.perf_counter() - start_time) * 1000

            cleaned_response = self._extract_json_array_text(raw_response)
            output_tokens = self._estimate_tokens(raw_response)

            try:
                parsed_outputs = json.loads(cleaned_response)
                valid_json = isinstance(parsed_outputs, list) and backend_error == ""
            except json.JSONDecodeError:
                parsed_outputs = []
                valid_json = False

            output_map = {}

            if valid_json:
                for item in parsed_outputs:
                    if isinstance(item, dict) and "line_id" in item:
                        output_map[item["line_id"]] = item

            per_log_latency = batch_latency_ms / max(len(batch), 1)
            per_log_input_tokens = input_tokens / max(len(batch), 1)
            per_log_output_tokens = output_tokens / max(len(batch), 1)

            for record in batch:
                line_id = record["line_id"]
                raw_log = record.get("raw_log", "")
                content = record.get("content", "")
                masked_content = record.get("masked_content", "")
                parsed_text = record.get(text_field) or content or raw_log

                item = output_map.get(line_id, {})

                template = item.get("template", "")
                variables = item.get("variables", [])
                confidence = item.get("confidence", 0.0)

                rows.append(
                    {
                        "line_id": line_id,
                        "raw_log": raw_log,
                        "content": content,
                        "masked_content": masked_content,
                        "parsed_text": parsed_text,
                        "parser": f"LLM Direct ({self.backend.name})",
                        "template": template,
                        "variables": json.dumps(variables, ensure_ascii=False),
                        "confidence": confidence,
                        "valid_json": valid_json,
                        "input_tokens_est": round(per_log_input_tokens, 2),
                        "output_tokens_est": round(per_log_output_tokens, 2),
                        "latency_ms": round(per_log_latency, 4),
                        "raw_response": raw_response,
                        "cleaned_response": cleaned_response,
                        "backend_error": backend_error,
                    }
                )

        return pd.DataFrame(rows)