"""Build lightweight template memory CSV files from raw logs."""

from __future__ import annotations

import csv
import random
import re
from collections import Counter
from datetime import datetime
from pathlib import Path


def simple_template_from_log(log: str) -> str:
    """Convert a raw log into a coarse template with obvious variables masked."""
    template = str(log or "")
    template = re.sub(r"\b\d{4}-\d{2}-\d{2}[T\s]\d{2}:\d{2}:\d{2}(?:\.\d+)?\b", "<*>", template)
    template = re.sub(r"\b\d{2}:\d{2}:\d{2}(?:\.\d+)?\b", "<*>", template)
    template = re.sub(r"\bblk_[A-Za-z0-9_]+\b", "<*>", template)
    template = re.sub(r"\b\d{1,3}(?:\.\d{1,3}){3}:\d+\b", "<*>", template)
    template = re.sub(r"/[^\s:]+(?:/[^\s:]+)*", "<*>", template)
    template = re.sub(r"\b\d+\b", "<*>", template)
    template = re.sub(r"\s+", " ", template).strip()
    return template


def _select_memory_logs(
    logs: list[str],
    method: str,
    max_memory_logs: int,
) -> list[str]:
    cleaned_logs = [str(log).strip() for log in logs if str(log).strip()]
    if not cleaned_logs:
        return []

    limit = max(1, min(max_memory_logs, len(cleaned_logs)))
    if method == "first_30_percent":
        count = max(1, min(limit, int(len(cleaned_logs) * 0.3) or 1))
        return cleaned_logs[:count]

    rng = random.Random(42)
    return rng.sample(cleaned_logs, k=limit)


def build_template_memory_from_logs(
    logs: list[str],
    method: str = "first_30_percent",
    max_memory_logs: int = 200,
    output_path: str | None = None,
) -> str:
    """Build a template memory CSV from logs and return the output path."""
    selected_logs = _select_memory_logs(logs, method, max_memory_logs)
    template_counts = Counter(simple_template_from_log(log) for log in selected_logs)

    if output_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"results/template_memory/generated_memory_{timestamp}.csv"

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["template", "description", "dataset", "source", "count"],
        )
        writer.writeheader()
        for template, count in sorted(template_counts.items()):
            writer.writerow(
                {
                    "template": template,
                    "description": "Generated from current logs by lightweight template extraction.",
                    "dataset": "generated",
                    "source": method,
                    "count": count,
                }
            )

    return str(path)
