from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from logpilot.ai_framework.ui_helpers import (
    build_agent_log_preview,
    build_memory_dataset_warning,
    build_parsed_results_table,
    build_tool_trace_table,
    format_final_recommendation_zh,
    get_default_agent_demo_log,
    get_default_candidate_template,
    get_default_user_query,
    get_task_labels,
    get_task_key_from_label,
    infer_dataset_type_from_name_or_logs,
    infer_template_memory_type,
    load_eval_summary_if_exists,
    normalize_selected_logs_for_agent,
    summarize_agent_result_for_ui,
    task_requires_template_memory,
    translate_status,
    translate_tool_name,
    validate_template_memory_csv_text,
)


@dataclass
class DummyPlan:
    action: str
    reason: str


@dataclass
class DummyResult:
    plan: DummyPlan
    final_answer: str


@dataclass
class DummyTrace:
    status: str
    success: bool
    tool_error_count: int
    tool_calls: list | None = None


def test_default_values_are_non_empty() -> None:
    assert get_default_agent_demo_log()
    assert get_default_candidate_template()
    assert get_default_user_query()


def test_load_eval_summary_if_exists_returns_none_for_missing_file(tmp_path: Path) -> None:
    assert load_eval_summary_if_exists(str(tmp_path / "missing.json")) is None


def test_load_eval_summary_if_exists_reads_summary_json(tmp_path: Path) -> None:
    path = tmp_path / "summary.json"
    path.write_text(json.dumps({"total_cases": 5}), encoding="utf-8")

    assert load_eval_summary_if_exists(str(path)) == {"total_cases": 5}


def test_summarize_agent_result_for_ui_returns_display_dict() -> None:
    result = DummyResult(
        plan=DummyPlan(action="full_review", reason="test reason"),
        final_answer="final answer",
    )
    trace = DummyTrace(status="success", success=True, tool_error_count=0)

    summary = summarize_agent_result_for_ui(
        result,
        trace,
        {"json": "trace.json", "markdown": "report.md"},
    )

    assert summary["selected_action"] == "full_review"
    assert summary["status"] == "success"
    assert summary["final_answer"] == "final answer"


def test_normalize_selected_logs_for_agent_supports_list_of_strings() -> None:
    assert normalize_selected_logs_for_agent([" a ", "", "b"], max_logs=2) == ["a", "b"]


def test_normalize_selected_logs_for_agent_supports_list_of_dicts() -> None:
    logs = normalize_selected_logs_for_agent(
        [{"content": "log a"}, {"message": "log b"}],
        max_logs=10,
    )

    assert logs == ["log a", "log b"]


def test_normalize_selected_logs_for_agent_empty_input_returns_empty_list() -> None:
    assert normalize_selected_logs_for_agent(None) == []
    assert normalize_selected_logs_for_agent([]) == []


def test_build_agent_log_preview_outputs_numbered_lines() -> None:
    preview = build_agent_log_preview(["log a", "log b"])

    assert "1. log a" in preview
    assert "2. log b" in preview


def test_translate_status_returns_chinese_label() -> None:
    assert translate_status("success") == "成功"


def test_translate_tool_name_returns_chinese_label() -> None:
    assert translate_tool_name("drain_parse_logs") == "Drain 基线解析"


def test_build_tool_trace_table_handles_empty_trace() -> None:
    trace = DummyTrace(status="success", success=True, tool_error_count=0, tool_calls=[])

    assert build_tool_trace_table(trace) == []


def test_build_parsed_results_table_does_not_show_raw_none() -> None:
    rows = build_parsed_results_table(
        {
            "template_memory_parse_logs": {
                "results": [
                    {
                        "index": 0,
                        "content": "log a",
                        "template": None,
                        "score": None,
                    }
                ]
            }
        }
    )

    assert rows == [
        {
            "序号": 0,
            "日志正文": "log a",
            "模板记忆模板": "-",
            "Top-1 检索分数": "-",
            "说明": "当前任务未执行对比",
        }
    ]


def test_build_parsed_results_table_adds_mismatch_explanation() -> None:
    rows = build_parsed_results_table(
        {
            "drain_parse_logs": {"results": [{"index": 0, "content": "log a", "template": "a"}]},
            "template_memory_parse_logs": {
                "results": [{"index": 0, "content": "log a", "template": "b"}]
            },
            "compare_batch_parse_outputs": {
                "items": [
                    {
                        "index": 0,
                        "drain_template": "a",
                        "rag_template": "b",
                        "exact_match": False,
                    }
                ]
            },
        }
    )

    assert rows[0]["是否一致"] == "不一致"
    assert rows[0]["说明"] == "两个工具输出不同，建议复核"


def test_infer_dataset_type_from_name_detects_bgl() -> None:
    assert infer_dataset_type_from_name_or_logs("BGL_2k.log", []) == "BGL"


def test_infer_dataset_type_from_name_detects_hdfs() -> None:
    assert infer_dataset_type_from_name_or_logs("hdfs_sample.log", []) == "HDFS"


def test_infer_dataset_type_from_logs_detects_bgl() -> None:
    logs = [
        "2005-06-03 R01-M0-N0 RAS KERNEL INFO instruction cache parity error corrected",
        "APPREAD node RAS APP FATAL ciod: failed to read message prefix on control stream",
    ]

    assert infer_dataset_type_from_name_or_logs("uploaded.log", logs) == "BGL"


def test_infer_dataset_type_from_logs_detects_hdfs() -> None:
    logs = [
        "Receiving block blk_38865049064139660 src: /10.250.19.102:54106",
        "DataNode NameSystem addStoredBlock blk_111",
    ]

    assert infer_dataset_type_from_name_or_logs("uploaded.log", logs) == "HDFS"


def test_infer_template_memory_type_detects_hdfs() -> None:
    assert infer_template_memory_type("data/template_memory/hdfs_templates.csv") == "HDFS"


def test_infer_template_memory_type_detects_bgl() -> None:
    assert infer_template_memory_type("data/template_memory/bgl_templates.csv") == "BGL"


def test_build_memory_dataset_warning_returns_warning_for_mismatch() -> None:
    warning = build_memory_dataset_warning("BGL", "HDFS")

    assert warning
    assert "BGL" in warning
    assert "HDFS" in warning


def test_build_memory_dataset_warning_returns_none_for_match() -> None:
    assert build_memory_dataset_warning("BGL", "BGL") is None


def test_validate_template_memory_csv_text_rejects_missing_template_column() -> None:
    is_valid, message, normalized_csv = validate_template_memory_csv_text(
        "description\nmissing template\n"
    )

    assert not is_valid
    assert "template" in message
    assert normalized_csv == ""


def test_validate_template_memory_csv_text_adds_missing_description_column() -> None:
    is_valid, message, normalized_csv = validate_template_memory_csv_text(
        "template\nReceiving block <*>\n"
    )

    assert is_valid
    assert message == ""
    assert "description" in normalized_csv.splitlines()[0]


def test_agent_lab_task_labels_are_simplified_to_three_items() -> None:
    assert get_task_labels() == ["快速解析当前日志", "复核解析结果", "查找风险日志"]


def test_get_task_key_from_label_maps_new_chinese_tasks() -> None:
    assert get_task_key_from_label("快速解析当前日志") == "Fast baseline parse"
    assert get_task_key_from_label("复核解析结果") == "High-confidence parse review"
    assert get_task_key_from_label("查找风险日志") == "Find risky templates"


def test_task_requires_template_memory_for_new_agent_lab_tasks() -> None:
    assert task_requires_template_memory("快速解析当前日志") is False
    assert task_requires_template_memory("复核解析结果") is True
    assert task_requires_template_memory("查找风险日志") is True


def test_task_requires_template_memory_keeps_legacy_compatibility() -> None:
    assert task_requires_template_memory("高置信度解析复核") is True
    assert task_requires_template_memory("解释解析结果") is True
    assert task_requires_template_memory("快速基线解析") is False
    assert task_requires_template_memory("Fast baseline parse") is False


def test_agent_app_does_not_write_template_widget_state_after_creation() -> None:
    text = Path("agent_app.py").read_text(encoding="utf-8")

    assert 'key="template_csv_path"' not in text
    assert 'st.session_state["template_csv_path"]' not in text


def test_format_final_recommendation_zh_does_not_include_heading() -> None:
    text = format_final_recommendation_zh(
        "final answer",
        plan={
            "log_count": 1,
            "tool_sequence": ["drain_parse_logs", "build_batch_parse_summary"],
        },
        intermediate_outputs={
            "drain_parse_logs": {"results": [{"index": 0, "template": "a"}]}
        },
    )

    assert "### 最终建议" not in text
    assert "建议" in text
