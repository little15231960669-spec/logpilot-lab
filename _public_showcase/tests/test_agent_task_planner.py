from __future__ import annotations

from logpilot.ai_framework.log_review_agent import (
    plan_tools_for_agent_task,
    run_agent_task_on_logs_offline_with_trace,
)


LOGS = [
    "Receiving block blk_38865049064139660 src: /10.250.19.102:54106 dest: /10.250.19.102:50010",
    "Served block blk_777 to /10.250.19.103:50010",
]

HDFS_RAW_LOG = (
    "081109 203518 143 INFO dfs.DataNode$DataXceiver: "
    "Receiving block blk_38865049064139660 src: /10.250.19.102:54106 "
    "dest: /10.250.19.102:50010"
)


def test_high_confidence_plan_contains_drain_and_template_memory() -> None:
    plan = plan_tools_for_agent_task("High-confidence parse review", LOGS)

    assert "drain_parse_logs" in plan["tool_sequence"]
    assert "template_memory_parse_logs" in plan["tool_sequence"]


def test_fast_baseline_plan_uses_drain_and_summary_only() -> None:
    plan = plan_tools_for_agent_task("Fast baseline parse", LOGS)

    assert plan["tool_sequence"] == ["drain_parse_logs", "build_batch_parse_summary"]


def test_compare_parser_tools_plan_contains_compare_tool() -> None:
    plan = plan_tools_for_agent_task("Compare parser tools", LOGS)

    assert "compare_batch_parse_outputs" in plan["tool_sequence"]


def test_unknown_task_falls_back_to_high_confidence() -> None:
    plan = plan_tools_for_agent_task("unknown", LOGS)

    assert plan["action"] == "high_confidence_parse_review"


def test_review_results_chinese_task_uses_full_review_toolchain() -> None:
    plan = plan_tools_for_agent_task("复核解析结果", LOGS)

    assert plan["action"] == "high_confidence_parse_review"
    assert plan["tool_sequence"] == [
        "drain_parse_logs",
        "template_memory_parse_logs",
        "compare_batch_parse_outputs",
        "build_batch_parse_summary",
    ]


def test_fast_parse_chinese_task_uses_drain_and_summary_only() -> None:
    plan = plan_tools_for_agent_task("快速解析当前日志", LOGS)

    assert plan["action"] == "fast_baseline_parse"
    assert plan["tool_sequence"] == ["drain_parse_logs", "build_batch_parse_summary"]


def test_find_risky_logs_chinese_task_contains_compare_tool() -> None:
    plan = plan_tools_for_agent_task("查找风险日志", LOGS)

    assert plan["action"] == "find_risky_templates"
    assert "compare_batch_parse_outputs" in plan["tool_sequence"]


def test_run_agent_task_on_logs_offline_with_trace_returns_trace(tmp_path) -> None:
    csv_path = tmp_path / "templates.csv"
    csv_path.write_text(
        "template,description\n"
        "Receiving block <*> src: <*> dest: <*>,receiving block\n"
        "Served block <*> to <*>,served block\n",
        encoding="utf-8",
    )

    _result, trace, _saved_paths, _plan, intermediate_outputs = (
        run_agent_task_on_logs_offline_with_trace(
            agent_task="High-confidence parse review",
            user_query="review logs",
            logs=LOGS,
            template_csv_path=str(csv_path),
            save_outputs=False,
        )
    )

    assert trace.status in {"success", "partial_success"}
    assert trace.tool_calls
    assert "drain_parse_logs" in intermediate_outputs
    assert "template_memory_parse_logs" in intermediate_outputs


def test_fast_baseline_run_does_not_require_template_csv_path() -> None:
    _result, trace, _saved_paths, plan, intermediate_outputs = (
        run_agent_task_on_logs_offline_with_trace(
            agent_task="Fast baseline parse",
            user_query="quick parse",
            logs=LOGS,
            template_csv_path=None,
            save_outputs=False,
        )
    )

    assert trace.status in {"success", "partial_success"}
    assert plan["tool_sequence"] == ["drain_parse_logs", "build_batch_parse_summary"]
    assert "drain_parse_logs" in intermediate_outputs
    assert "template_memory_parse_logs" not in intermediate_outputs


def test_run_agent_task_accepts_dataset_name_for_hdfs(tmp_path) -> None:
    csv_path = tmp_path / "templates.csv"
    csv_path.write_text(
        "template,description\n"
        "Receiving block <*> src: <*> dest: <*>,receiving block\n",
        encoding="utf-8",
    )

    _result, trace, _saved_paths, _plan, intermediate_outputs = (
        run_agent_task_on_logs_offline_with_trace(
            agent_task="High-confidence parse review",
            user_query="review hdfs log",
            logs=[HDFS_RAW_LOG],
            template_csv_path=str(csv_path),
            save_outputs=False,
            dataset_name="HDFS",
        )
    )

    template = intermediate_outputs["drain_parse_logs"]["results"][0]["template"]
    assert trace.status in {"success", "partial_success"}
    assert "081109" not in template
    assert "INFO" not in template
    assert "dfs.DataNode$DataXceiver" not in template
