"""Offline evaluation utilities for LogPilot agent workflows."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path

from logpilot.ai_framework.log_review_agent import (
    run_log_review_agent_offline_demo_with_trace,
)


@dataclass
class AgentEvalCase:
    case_id: str
    log: str
    user_query: str
    candidate_template: str
    expected_action: str
    expected_template: str
    expected_accept: bool
    notes: str = ""


@dataclass
class AgentEvalResult:
    case_id: str
    selected_action: str
    expected_action: str
    action_correct: bool
    predicted_template: str | None
    expected_template: str
    template_exact_match: bool
    retrieval_top1_template: str | None
    retrieval_top1_hit: bool
    predicted_accept: bool | None
    expected_accept: bool
    final_decision_correct: bool
    status: str
    tool_error_count: int
    markdown_report_path: str | None
    json_trace_path: str | None
    notes: str = ""


def parse_bool(value: str) -> bool:
    """Parse a CSV boolean value."""
    normalized = value.strip().lower()
    if normalized == "true":
        return True
    if normalized == "false":
        return False
    raise ValueError(f"Invalid boolean value: {value}")


def load_agent_eval_cases(csv_path: str) -> list[AgentEvalCase]:
    """Load offline agent evaluation cases from CSV."""
    with open(csv_path, newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        return [
            AgentEvalCase(
                case_id=(row.get("case_id") or "").strip(),
                log=(row.get("log") or "").strip(),
                user_query=(row.get("user_query") or "").strip(),
                candidate_template=(row.get("candidate_template") or "").strip(),
                expected_action=(row.get("expected_action") or "").strip(),
                expected_template=(row.get("expected_template") or "").strip(),
                expected_accept=parse_bool(row.get("expected_accept") or ""),
                notes=(row.get("notes") or "").strip(),
            )
            for row in reader
        ]


def infer_predicted_accept(
    predicted_template: str | None,
    candidate_template: str | None,
    comparison_exact_match: bool | None,
) -> bool | None:
    """Infer whether the agent accepted the candidate template."""
    if not predicted_template or not candidate_template:
        return None
    if comparison_exact_match is None:
        return None
    return bool(comparison_exact_match)


def _extract_predicted_template(tool_outputs: dict) -> str | None:
    parse_output = tool_outputs.get("offline_parse_with_template_memory", {})
    result = parse_output.get("result") if isinstance(parse_output, dict) else None
    if isinstance(result, dict):
        template = result.get("template")
        if isinstance(template, str):
            return template
    return None


def _extract_retrieval_top1_template(tool_outputs: dict) -> str | None:
    retrieve_output = tool_outputs.get("retrieve_template_memory", {})
    top_k = retrieve_output.get("top_k") if isinstance(retrieve_output, dict) else None
    if isinstance(top_k, list) and top_k:
        template = top_k[0].get("template")
        if isinstance(template, str):
            return template
    return None


def _extract_comparison_exact_match(tool_outputs: dict) -> bool | None:
    compare_output = tool_outputs.get("compare_template_strings", {})
    if isinstance(compare_output, dict) and "exact_match" in compare_output:
        return bool(compare_output["exact_match"])
    return None


def evaluate_agent_case_offline(
    case: AgentEvalCase,
    template_csv_path: str,
    save_outputs: bool = True,
) -> AgentEvalResult:
    """Evaluate one case with the offline demo agent."""
    result, trace, saved_paths = run_log_review_agent_offline_demo_with_trace(
        user_query=case.user_query,
        log=case.log,
        template_csv_path=template_csv_path,
        candidate_template=case.candidate_template,
        k=3,
        save_outputs=save_outputs,
    )

    predicted_template = _extract_predicted_template(result.tool_outputs)
    retrieval_top1_template = _extract_retrieval_top1_template(result.tool_outputs)
    comparison_exact_match = _extract_comparison_exact_match(result.tool_outputs)
    predicted_accept = infer_predicted_accept(
        predicted_template,
        case.candidate_template,
        comparison_exact_match,
    )

    return AgentEvalResult(
        case_id=case.case_id,
        selected_action=result.plan.action,
        expected_action=case.expected_action,
        action_correct=result.plan.action == case.expected_action,
        predicted_template=predicted_template,
        expected_template=case.expected_template,
        template_exact_match=predicted_template == case.expected_template,
        retrieval_top1_template=retrieval_top1_template,
        retrieval_top1_hit=retrieval_top1_template == case.expected_template,
        predicted_accept=predicted_accept,
        expected_accept=case.expected_accept,
        final_decision_correct=predicted_accept == case.expected_accept,
        status=trace.status,
        tool_error_count=trace.tool_error_count,
        markdown_report_path=saved_paths.get("markdown"),
        json_trace_path=saved_paths.get("json"),
        notes=case.notes,
    )


def evaluate_agent_cases_offline(
    eval_csv_path: str,
    template_csv_path: str,
    save_outputs: bool = True,
) -> list[AgentEvalResult]:
    """Evaluate all cases from a CSV file with the offline demo agent."""
    cases = load_agent_eval_cases(eval_csv_path)
    return [
        evaluate_agent_case_offline(case, template_csv_path, save_outputs=save_outputs)
        for case in cases
    ]


def _rate(count: int, total: int) -> float:
    if total == 0:
        return 0.0
    return round(count / total, 6)


def compute_agent_eval_summary(results: list[AgentEvalResult]) -> dict:
    """Compute aggregate metrics for offline agent evaluation."""
    total = len(results)
    return {
        "total_cases": total,
        "action_accuracy": _rate(sum(item.action_correct for item in results), total),
        "template_exact_match_rate": _rate(
            sum(item.template_exact_match for item in results),
            total,
        ),
        "retrieval_top1_hit_rate": _rate(
            sum(item.retrieval_top1_hit for item in results),
            total,
        ),
        "final_decision_accuracy": _rate(
            sum(item.final_decision_correct for item in results),
            total,
        ),
        "success_status_rate": _rate(
            sum(item.status == "success" for item in results),
            total,
        ),
        "partial_success_rate": _rate(
            sum(item.status == "partial_success" for item in results),
            total,
        ),
        "failed_rate": _rate(sum(item.status == "failed" for item in results), total),
        "average_tool_error_count": (
            round(sum(item.tool_error_count for item in results) / total, 6)
            if total
            else 0.0
        ),
    }


def save_agent_eval_results_csv(
    results: list[AgentEvalResult],
    output_path: str = "results/agent_eval/agent_eval_results.csv",
) -> str:
    """Save per-case evaluation results as CSV."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(AgentEvalResult.__dataclass_fields__.keys())
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            writer.writerow(asdict(result))
    return str(path)


def save_agent_eval_summary_json(
    summary: dict,
    output_path: str = "results/agent_eval/agent_eval_summary.json",
) -> str:
    """Save aggregate evaluation metrics as JSON."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return str(path)
