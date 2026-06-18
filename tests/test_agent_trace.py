from __future__ import annotations

import json
from pathlib import Path

from logpilot.ai_framework.agent_trace import (
    ToolCallTrace,
    agent_trace_to_dict,
    build_agent_run_trace,
    generate_run_id,
    now_iso,
    render_agent_report_markdown,
    save_agent_report_markdown,
    save_agent_trace_json,
    summarize_for_trace,
)


def _sample_trace():
    started_at = now_iso()
    ended_at = now_iso()
    tool_call = ToolCallTrace(
        tool_name="retrieve_template_memory",
        input_summary={"log": "Receiving block blk_123"},
        output_summary={"top_k": [{"template": "Receiving block <*>"}]},
        error=None,
        started_at=started_at,
        ended_at=ended_at,
        duration_ms=0.0,
    )
    return build_agent_run_trace(
        user_query="review this log",
        log="Receiving block blk_123",
        selected_action="full_review",
        plan_reason="User asks for full review.",
        tool_calls=[tool_call],
        final_answer="Final answer text.",
        success=True,
        started_at=started_at,
        ended_at=ended_at,
    )


def _sample_partial_trace():
    started_at = now_iso()
    ended_at = now_iso()
    tool_call = ToolCallTrace(
        tool_name="parse_with_template_memory",
        input_summary={"log": "Receiving block blk_123"},
        output_summary={"error": "Connection error."},
        error="Connection error.",
        started_at=started_at,
        ended_at=ended_at,
        duration_ms=0.0,
    )
    return build_agent_run_trace(
        user_query="review this log",
        log="Receiving block blk_123",
        selected_action="full_review",
        plan_reason="User asks for full review.",
        tool_calls=[tool_call],
        final_answer="Partial answer text.",
        success=True,
        started_at=started_at,
        ended_at=ended_at,
    )


def test_generate_run_id_returns_non_empty_string() -> None:
    assert generate_run_id()


def test_summarize_for_trace_truncates_long_string() -> None:
    value = summarize_for_trace("x" * 600, max_length=20)

    assert value == "x" * 20 + "...[truncated]"


def test_agent_trace_to_dict_converts_dataclass() -> None:
    data = agent_trace_to_dict(_sample_trace())

    assert data["selected_action"] == "full_review"
    assert data["status"] == "success"
    assert data["tool_error_count"] == 0
    assert data["tool_calls"][0]["tool_name"] == "retrieve_template_memory"


def test_render_agent_report_markdown_contains_key_fields() -> None:
    markdown = render_agent_report_markdown(_sample_trace())

    assert "run_id" in markdown
    assert "selected action" in markdown
    assert "status" in markdown
    assert "tool_error_count" in markdown
    assert "retrieve_template_memory" in markdown
    assert "Final answer text." in markdown


def test_render_agent_report_markdown_shows_partial_success() -> None:
    markdown = render_agent_report_markdown(_sample_partial_trace())

    assert "- status: `partial_success`" in markdown
    assert "- success: True" in markdown
    assert "- tool_error_count: 1" in markdown


def test_summarize_for_trace_truncates_nested_long_field() -> None:
    data = summarize_for_trace({"long": "x" * 600}, max_length=30)

    assert data["long"] == "x" * 30 + "...[truncated]"


def test_save_agent_trace_json_writes_file(tmp_path: Path) -> None:
    trace = _sample_trace()
    path = save_agent_trace_json(trace, output_dir=str(tmp_path))

    assert Path(path).exists()
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    assert data["run_id"] == trace.run_id


def test_save_agent_report_markdown_writes_file(tmp_path: Path) -> None:
    trace = _sample_trace()
    path = save_agent_report_markdown(trace, output_dir=str(tmp_path))

    assert Path(path).exists()
    assert trace.run_id in Path(path).read_text(encoding="utf-8")
