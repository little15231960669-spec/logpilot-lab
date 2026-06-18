from __future__ import annotations

import json
from pathlib import Path

import pytest

from logpilot.ai_framework.agent_eval import (
    AgentEvalCase,
    AgentEvalResult,
    compute_agent_eval_summary,
    infer_predicted_accept,
    load_agent_eval_cases,
    parse_bool,
    save_agent_eval_results_csv,
    save_agent_eval_summary_json,
)


def test_parse_bool_true_and_false() -> None:
    assert parse_bool("true") is True
    assert parse_bool("false") is False


def test_parse_bool_raises_for_invalid_value() -> None:
    with pytest.raises(ValueError, match="Invalid boolean value"):
        parse_bool("yes")


def test_load_agent_eval_cases_reads_temp_csv(tmp_path: Path) -> None:
    csv_path = tmp_path / "cases.csv"
    csv_path.write_text(
        "case_id,log,user_query,candidate_template,expected_action,"
        "expected_template,expected_accept,notes\n"
        'case_001,"log line","query","template","full_review",'
        '"template","true","note"\n',
        encoding="utf-8",
    )

    cases = load_agent_eval_cases(str(csv_path))

    assert cases == [
        AgentEvalCase(
            case_id="case_001",
            log="log line",
            user_query="query",
            candidate_template="template",
            expected_action="full_review",
            expected_template="template",
            expected_accept=True,
            notes="note",
        )
    ]


def test_infer_predicted_accept_from_exact_match() -> None:
    assert infer_predicted_accept("template", "template", True) is True
    assert infer_predicted_accept("template", "other", False) is False


def test_compute_agent_eval_summary_counts_cases_and_action_accuracy() -> None:
    results = [
        AgentEvalResult(
            case_id="case_001",
            selected_action="full_review",
            expected_action="full_review",
            action_correct=True,
            predicted_template="a",
            expected_template="a",
            template_exact_match=True,
            retrieval_top1_template="a",
            retrieval_top1_hit=True,
            predicted_accept=True,
            expected_accept=True,
            final_decision_correct=True,
            status="success",
            tool_error_count=0,
            markdown_report_path=None,
            json_trace_path=None,
        ),
        AgentEvalResult(
            case_id="case_002",
            selected_action="retrieve_only",
            expected_action="full_review",
            action_correct=False,
            predicted_template="b",
            expected_template="a",
            template_exact_match=False,
            retrieval_top1_template="b",
            retrieval_top1_hit=False,
            predicted_accept=False,
            expected_accept=True,
            final_decision_correct=False,
            status="partial_success",
            tool_error_count=1,
            markdown_report_path=None,
            json_trace_path=None,
        ),
    ]

    summary = compute_agent_eval_summary(results)

    assert summary["total_cases"] == 2
    assert summary["action_accuracy"] == 0.5
    assert summary["success_status_rate"] == 0.5
    assert summary["partial_success_rate"] == 0.5
    assert summary["average_tool_error_count"] == 0.5


def test_save_agent_eval_results_csv_writes_file(tmp_path: Path) -> None:
    result = AgentEvalResult(
        case_id="case_001",
        selected_action="full_review",
        expected_action="full_review",
        action_correct=True,
        predicted_template="template",
        expected_template="template",
        template_exact_match=True,
        retrieval_top1_template="template",
        retrieval_top1_hit=True,
        predicted_accept=True,
        expected_accept=True,
        final_decision_correct=True,
        status="success",
        tool_error_count=0,
        markdown_report_path=None,
        json_trace_path=None,
    )

    path = save_agent_eval_results_csv([result], str(tmp_path / "results.csv"))

    assert Path(path).exists()
    assert "case_001" in Path(path).read_text(encoding="utf-8")


def test_save_agent_eval_summary_json_writes_file(tmp_path: Path) -> None:
    path = save_agent_eval_summary_json(
        {"total_cases": 1},
        str(tmp_path / "summary.json"),
    )

    assert Path(path).exists()
    assert json.loads(Path(path).read_text(encoding="utf-8"))["total_cases"] == 1
