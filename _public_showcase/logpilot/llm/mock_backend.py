import json
import re
from typing import List, Dict, Any

from logpilot.data.preprocess import mask_variables


VARIABLE_PATTERN = re.compile(
    r"blk_-?\d+"
    r"|/?(?:\d{1,3}\.){3}\d{1,3}(?::\d+)?"
    r"|(?<!\w)/(?:[\w.\-]+/)*[\w.\-]+"
    r"|\b\d+\b"
)


class MockBackend:
    """
    A deterministic mock LLM backend.

    It does not call any real LLM API. Instead, it simulates an LLM that:
    - receives log messages
    - returns JSON
    - extracts templates
    - extracts variables

    This makes the open-source demo runnable without API keys.
    """

    name = "Mock LLM"

    def _extract_variables(self, text: str) -> List[str]:
        if not text:
            return []

        variables = []
        for match in VARIABLE_PATTERN.finditer(text):
            value = match.group(0)
            if value not in variables:
                variables.append(value)

        return variables

    def parse_batch(
        self,
        records: List[Dict[str, Any]],
        text_field: str = "content",
    ) -> str:
        outputs = []

        for record in records:
            line_id = record["line_id"]
            text = record.get(text_field) or record.get("content") or record.get("raw_log") or ""

            template = mask_variables(text)
            variables = self._extract_variables(text)

            outputs.append(
                {
                    "line_id": line_id,
                    "template": template,
                    "variables": variables,
                    "confidence": 0.95,
                }
            )

        return json.dumps(outputs, ensure_ascii=False)