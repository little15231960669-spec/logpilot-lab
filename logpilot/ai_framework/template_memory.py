"""Lightweight template memory retrieval for LogPilot RAG experiments."""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass


@dataclass
class TemplateRecord:
    template: str
    description: str


@dataclass
class RetrievedTemplate:
    template: str
    description: str
    score: float


def load_template_records(csv_path: str) -> list[TemplateRecord]:
    """Load template memory records from a CSV file."""
    try:
        with open(csv_path, newline="", encoding="utf-8") as csv_file:
            reader = csv.DictReader(csv_file)
            records = []
            for row in reader:
                template = (row.get("template") or "").strip()
                description = (row.get("description") or "").strip()
                if template:
                    records.append(
                        TemplateRecord(template=template, description=description)
                    )
            return records
    except FileNotFoundError:
        return []


def normalize_for_retrieval(text: str) -> list[str]:
    """Normalize log/template text into stable lexical retrieval tokens."""
    normalized = text.lower().replace("<*>", " wildcard ")
    return re.findall(r"[a-z0-9_.*]+", normalized)


def lexical_similarity(query: str, candidate: str) -> float:
    """Return a simple overlap score between query and candidate text."""
    query_tokens = set(normalize_for_retrieval(query))
    candidate_tokens = set(normalize_for_retrieval(candidate))

    if not query_tokens or not candidate_tokens:
        return 0.0

    overlap = query_tokens & candidate_tokens
    union = query_tokens | candidate_tokens
    overlap_ratio = len(overlap) / len(candidate_tokens)
    jaccard = len(overlap) / len(union)

    return round((overlap_ratio + jaccard) / 2, 6)


def retrieve_similar_templates(
    query_log: str,
    csv_path: str,
    k: int = 3,
) -> list[RetrievedTemplate]:
    """Retrieve top-k lexically similar historical templates."""
    records = load_template_records(csv_path)
    if not records:
        return []

    retrieved = [
        RetrievedTemplate(
            template=record.template,
            description=record.description,
            score=lexical_similarity(query_log, record.template),
        )
        for record in records
    ]
    retrieved.sort(key=lambda item: item.score, reverse=True)

    return retrieved[: max(k, 0)]


def build_template_memory_context(retrieved: list[RetrievedTemplate]) -> str:
    """Build human-readable memory context for LLM prompting."""
    if not retrieved:
        return "Similar historical templates: none"

    lines = ["Similar historical templates:"]
    for index, item in enumerate(retrieved, start=1):
        lines.extend(
            [
                f"{index}. template: {item.template}",
                f"   description: {item.description}",
                f"   score: {item.score:.3f}",
            ]
        )

    return "\n".join(lines)
