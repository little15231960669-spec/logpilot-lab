"""Python-controlled tools for lightweight log review workflows."""

from __future__ import annotations

import re
from typing import Any

from logpilot.ai_framework.config import get_openai_api_key
from logpilot.ai_framework.log_preprocess import extract_log_content, mask_variables
from logpilot.ai_framework.rag_enhanced_parser import parse_log_with_template_memory
from logpilot.ai_framework.template_memory import retrieve_similar_templates
from logpilot.parsers.drain_parser import DrainLogParser


def _safe_error_summary(error: Exception) -> str:
    summary = str(error)
    api_key = get_openai_api_key()
    if api_key:
        summary = summary.replace(api_key, "***")
    return summary


def retrieve_template_memory_tool(
    log: str,
    template_csv_path: str,
    k: int = 3,
) -> dict:
    """Retrieve similar templates and return a tool-style dictionary."""
    retrieved = retrieve_similar_templates(log, template_csv_path, k=k)
    return {
        "tool_name": "retrieve_template_memory",
        "log": log,
        "top_k": [
            {
                "template": item.template,
                "description": item.description,
                "score": item.score,
            }
            for item in retrieved
        ],
    }


def parse_with_template_memory_tool(
    log: str,
    template_csv_path: str,
    k: int = 3,
) -> dict:
    """Parse a log with template memory and return a tool-style dictionary."""
    try:
        result = parse_log_with_template_memory(log, template_csv_path, k=k)
    except Exception as exc:
        return {
            "tool_name": "parse_with_template_memory",
            "log": log,
            "error": _safe_error_summary(exc),
        }

    return {
        "tool_name": "parse_with_template_memory",
        "log": log,
        "result": {
            "template": result.template,
            "variables": result.variables,
            "confidence": result.confidence,
            "reason": result.reason,
        },
    }


def normalize_template_for_compare(template: str) -> list[str]:
    """Normalize a template into stable comparison tokens."""
    normalized = template.lower().replace("<*>", " wildcard ")
    return re.findall(r"[a-z0-9_.*]+", normalized)


def compare_template_strings(
    template_a: str,
    template_b: str,
) -> dict:
    """Compare two log template strings with deterministic token overlap."""
    tokens_a = normalize_template_for_compare(template_a)
    tokens_b = normalize_template_for_compare(template_b)
    set_a = set(tokens_a)
    set_b = set(tokens_b)

    exact_match = template_a.strip() == template_b.strip()
    if set_a or set_b:
        token_overlap = len(set_a & set_b) / len(set_a | set_b)
    else:
        token_overlap = 1.0 if exact_match else 0.0

    if exact_match:
        explanation = "Templates are exactly identical."
    elif token_overlap > 0:
        explanation = "Templates are similar but not identical."
    else:
        explanation = "Templates have no shared normalized tokens."

    return {
        "tool_name": "compare_template_strings",
        "exact_match": exact_match,
        "token_overlap": round(token_overlap, 6),
        "template_a_tokens": tokens_a,
        "template_b_tokens": tokens_b,
        "only_in_a": sorted(set_a - set_b),
        "only_in_b": sorted(set_b - set_a),
        "explanation": explanation,
    }


def _get_parsed_result(parsed_result: dict | None) -> dict[str, Any]:
    if not parsed_result:
        return {}
    if "result" in parsed_result and isinstance(parsed_result["result"], dict):
        return parsed_result["result"]
    return parsed_result


def _format_top_template(retrieved_templates: list[dict] | None) -> str:
    if not retrieved_templates:
        return "none"

    top = retrieved_templates[0]
    return (
        f"{top.get('template', '')} "
        f"(score={float(top.get('score', 0.0)):.3f}, "
        f"description={top.get('description', '')})"
    )


def build_review_summary(
    log: str,
    parsed_result: dict | None = None,
    retrieved_templates: list[dict] | None = None,
    comparison_result: dict | None = None,
) -> str:
    """Build a deterministic review summary without calling an LLM."""
    result = _get_parsed_result(parsed_result)
    template = result.get("template", "not generated")
    variables = result.get("variables", [])
    confidence = result.get("confidence", "unknown")
    parse_error = parsed_result.get("error") if parsed_result else None
    top_template = _format_top_template(retrieved_templates)

    comparison_text = "not executed"
    recommendation = "Add a parse result before making a final review decision."
    if comparison_result:
        if comparison_result.get("error"):
            comparison_text = f"not completed, error={comparison_result.get('error')}"
        else:
            exact_match = comparison_result.get("exact_match", False)
            token_overlap = comparison_result.get("token_overlap", 0.0)
            comparison_text = (
                f"exact_match={exact_match}, token_overlap={token_overlap}, "
                f"{comparison_result.get('explanation', '')}"
            )

    if parse_error:
        recommendation = (
            "Retry RAG parsing or perform manual review because the parse result is "
            "not available."
        )
    elif comparison_result and not comparison_result.get("error"):
        exact_match = comparison_result.get("exact_match", False)
        token_overlap = comparison_result.get("token_overlap", 0.0)
        if exact_match:
            recommendation = (
                "Accept the parse result because the candidate template exactly "
                "matches the RAG template."
            )
        elif token_overlap >= 0.6:
            recommendation = (
                "Do a quick manual check before accepting; the templates are highly "
                "similar."
            )
        else:
            recommendation = (
                "Manual review is recommended because the candidate template differs "
                "from the RAG template."
            )
    elif result:
        recommendation = "Accept the RAG parse result, with manual sampling if confidence is low."
    elif retrieved_templates:
        recommendation = "Continue with RAG parsing based on the retrieved templates."

    lines = [
        "Log review report:",
        f"- Input log: {log}",
        f"- RAG parsed template: {template}",
        f"- variables: {variables}",
        f"- confidence: {confidence}",
        f"- top retrieved template: {top_template}",
        f"- comparison result: {comparison_text}",
    ]
    if parse_error:
        lines.append(f"- parse error: {parse_error}")
    lines.append(f"- final recommendation: {recommendation}")

    return "\n".join(lines)


def _simple_template_fallback(log: str) -> str:
    return mask_variables(log)


def _preprocess_log(raw_log: str, dataset_name: str | None = None) -> dict:
    content = extract_log_content(raw_log, dataset_name=dataset_name)
    masked_content = mask_variables(content)
    return {
        "raw_log": raw_log,
        "content": content,
        "masked_content": masked_content,
    }


def drain_parse_logs_tool(
    logs: list[str],
    drain_config: dict | None = None,
    dataset_name: str | None = None,
) -> dict:
    """Parse logs with the existing Drain parser, falling back if needed."""
    preprocessed = [_preprocess_log(log, dataset_name=dataset_name) for log in logs]
    try:
        config = drain_config or {}
        records = [
            {
                "line_id": index + 1,
                "raw_log": item["raw_log"],
                "content": item["content"],
                "masked_content": item["masked_content"],
            }
            for index, item in enumerate(preprocessed)
        ]
        parser = DrainLogParser(**config)
        parsed_df = parser.parse(records, text_field="masked_content")
        results = [
            {
                "index": int(row["line_id"]) - 1,
                "log": row["content"],
                "raw_log": row["raw_log"],
                "content": row["content"],
                "masked_content": row["masked_content"],
                "template": row["template"],
                "event_id": str(row["cluster_id"]),
            }
            for _, row in parsed_df.iterrows()
        ]
        return {
            "tool_name": "drain_parse_logs",
            "drain_tool_mode": "existing",
            "results": results,
            "error": None,
        }
    except Exception as exc:
        return {
            "tool_name": "drain_parse_logs",
            "drain_tool_mode": "fallback",
            "results": [
                {
                    "index": index,
                    "log": item["content"],
                    "raw_log": item["raw_log"],
                    "content": item["content"],
                    "masked_content": item["masked_content"],
                    "template": _simple_template_fallback(item["content"]),
                    "event_id": f"fallback_{index}",
                }
                for index, item in enumerate(preprocessed)
            ],
            "error": _safe_error_summary(exc),
        }


def template_memory_parse_logs_tool(
    logs: list[str],
    template_csv_path: str,
    k: int = 3,
    offline: bool = True,
    dataset_name: str | None = None,
) -> dict:
    """Parse multiple logs with Template Memory retrieval/RAG."""
    mode = "offline" if offline else "online"
    results = []
    for index, log in enumerate(logs):
        item = _preprocess_log(log, dataset_name=dataset_name)
        content = item["content"]
        masked_content = item["masked_content"]
        if offline:
            retrieved = retrieve_similar_templates(content, template_csv_path, k=k)
            top = retrieved[0] if retrieved else None
            results.append(
                {
                    "index": index,
                    "log": content,
                    "raw_log": item["raw_log"],
                    "content": content,
                    "masked_content": masked_content,
                    "template": top.template if top else "",
                    "top_retrieved_template": top.template if top else "",
                    "score": top.score if top else 0.0,
                }
            )
        else:
            try:
                parsed = parse_log_with_template_memory(content, template_csv_path, k=k)
                results.append(
                    {
                        "index": index,
                        "log": content,
                        "raw_log": item["raw_log"],
                        "content": content,
                        "masked_content": masked_content,
                        "template": parsed.template,
                        "top_retrieved_template": None,
                        "score": None,
                    }
                )
            except Exception as exc:
                results.append(
                    {
                        "index": index,
                        "log": content,
                        "raw_log": item["raw_log"],
                        "content": content,
                        "masked_content": masked_content,
                        "template": "",
                        "top_retrieved_template": None,
                        "score": None,
                        "error": _safe_error_summary(exc),
                    }
                )
    return {
        "tool_name": "template_memory_parse_logs",
        "mode": mode,
        "results": results,
        "error": None,
    }


def compare_batch_parse_outputs_tool(
    drain_outputs: dict | None = None,
    rag_outputs: dict | None = None,
) -> dict:
    """Compare Drain and Template Memory/RAG batch parse outputs."""
    drain_results = {
        item.get("index"): item
        for item in (drain_outputs or {}).get("results", [])
    }
    rag_results = {
        item.get("index"): item
        for item in (rag_outputs or {}).get("results", [])
    }
    indexes = sorted(set(drain_results) | set(rag_results))
    items = []
    exact_match_count = 0
    for index in indexes:
        drain_template = drain_results.get(index, {}).get("template")
        rag_template = rag_results.get(index, {}).get("template")
        exact_match = bool(drain_template and rag_template and drain_template == rag_template)
        if exact_match:
            exact_match_count += 1
        items.append(
            {
                "index": index,
                "drain_template": drain_template,
                "rag_template": rag_template,
                "exact_match": exact_match,
            }
        )
    total = len(items)
    mismatch_count = total - exact_match_count
    return {
        "tool_name": "compare_batch_parse_outputs",
        "total": total,
        "exact_match_count": exact_match_count,
        "mismatch_count": mismatch_count,
        "match_rate": round(exact_match_count / total, 6) if total else 0.0,
        "items": items,
    }


def build_batch_parse_summary(
    logs: list[str],
    drain_outputs: dict | None = None,
    rag_outputs: dict | None = None,
    comparison_outputs: dict | None = None,
    task: str = "",
    tools_used: list[str] | None = None,
) -> str:
    """Build a deterministic batch parse review summary."""
    drain_count = len((drain_outputs or {}).get("results", []))
    rag_count = len((rag_outputs or {}).get("results", []))
    drain_templates = [
        item.get("template", "")
        for item in (drain_outputs or {}).get("results", [])
        if item.get("template")
    ]
    rag_templates = [
        item.get("template", "")
        for item in (rag_outputs or {}).get("results", [])
        if item.get("template")
    ]
    exact_count = (comparison_outputs or {}).get("exact_match_count", 0)
    mismatch_count = (comparison_outputs or {}).get("mismatch_count", 0)
    match_rate = (comparison_outputs or {}).get("match_rate", 0.0)
    risky_indexes = [
        item.get("index")
        for item in (comparison_outputs or {}).get("items", [])
        if not item.get("exact_match")
    ]
    recommendation = (
        "Templates are mostly aligned; use the agent trace for spot checks."
        if not comparison_outputs or mismatch_count == 0
        else "Review mismatched templates before accepting batch results."
    )
    return "\n".join(
        [
            "Agent parsing report:",
            f"- task: {task or 'current log parsing'}",
            f"- logs analyzed: {len(logs)}",
            f"- tools used: {', '.join(tools_used or []) or 'none'}",
            "- note: 本次解析基于日志正文 Content，已忽略时间戳、日志级别、logger 等头部字段。",
            f"- Drain parsed templates: {drain_count} ({len(set(drain_templates))} unique)",
            f"- Template Memory parsed templates: {rag_count} ({len(set(rag_templates))} unique)",
            f"- match_rate: {match_rate}",
            f"- mismatch_count: {mismatch_count}",
            f"- risky indexes: {risky_indexes}",
            f"- exact matches: {exact_count}",
            f"- final recommendation: {recommendation}",
        ]
    )
