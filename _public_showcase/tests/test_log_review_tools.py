from __future__ import annotations

from logpilot.ai_framework.log_review_agent import parse_agent_plan
from logpilot.ai_framework.log_review_agent import run_current_logs_agent_offline_with_trace
from logpilot.ai_framework.log_tools import (
    build_batch_parse_summary,
    build_review_summary,
    compare_batch_parse_outputs_tool,
    compare_template_strings,
    normalize_template_for_compare,
    template_memory_parse_logs_tool,
)


def test_normalize_template_for_compare_returns_tokens() -> None:
    tokens = normalize_template_for_compare("Receiving block <*> src: <*> dest: <*>")

    assert tokens
    assert "receiving" in tokens
    assert "block" in tokens
    assert "wildcard" in tokens


def test_compare_template_strings_exact_match() -> None:
    result = compare_template_strings(
        "Receiving block <*> src: <*> dest: <*>",
        "Receiving block <*> src: <*> dest: <*>",
    )

    assert result["exact_match"] is True
    assert result["token_overlap"] == 1.0


def test_compare_template_strings_similar_overlap() -> None:
    result = compare_template_strings(
        "Receiving block <*> src: <*> dest: <*>",
        "Received block <*> of size <*> from <*>",
    )

    assert result["exact_match"] is False
    assert result["token_overlap"] > 0


def test_build_review_summary_contains_key_information() -> None:
    summary = build_review_summary(
        "Receiving block blk_123 src: /a dest: /b",
        parsed_result={
            "result": {
                "template": "Receiving block <*> src: <*> dest: <*>",
                "variables": ["blk_123", "/a", "/b"],
                "confidence": 0.98,
                "reason": "test",
            }
        },
        retrieved_templates=[
            {
                "template": "Receiving block <*> src: <*> dest: <*>",
                "description": "DataNode receives a block.",
                "score": 0.9,
            }
        ],
        comparison_result={
            "exact_match": True,
            "token_overlap": 1.0,
            "explanation": "Templates are exactly identical.",
        },
    )

    assert "Receiving block <*> src: <*> dest: <*>" in summary
    assert "confidence" in summary
    assert "0.98" in summary
    assert "top retrieved template" in summary


def test_parse_agent_plan_reads_valid_json_action() -> None:
    plan = parse_agent_plan(
        '{"action": "retrieve_only", "reason": "User asks for retrieval."}'
    )

    assert plan.action == "retrieve_only"
    assert plan.reason == "User asks for retrieval."


def test_parse_agent_plan_invalid_action_falls_back_to_full_review() -> None:
    plan = parse_agent_plan('{"action": "delete_logs", "reason": "bad"}')

    assert plan.action == "full_review"
    assert plan.reason == "Invalid plan, fallback to full_review."


def test_template_memory_parse_logs_tool_offline_returns_results(tmp_path) -> None:
    csv_path = tmp_path / "templates.csv"
    csv_path.write_text(
        "template,description\n"
        "Receiving block <*> src: <*> dest: <*>,receiving block\n",
        encoding="utf-8",
    )

    output = template_memory_parse_logs_tool(
        ["Receiving block blk_1 src: /a dest: /b"],
        str(csv_path),
        offline=True,
    )

    assert output["mode"] == "offline"
    assert output["results"][0]["template"] == "Receiving block <*> src: <*> dest: <*>"


def test_compare_batch_parse_outputs_tool_computes_match_rate() -> None:
    output = compare_batch_parse_outputs_tool(
        {"results": [{"index": 0, "template": "a"}, {"index": 1, "template": "b"}]},
        {"results": [{"index": 0, "template": "a"}, {"index": 1, "template": "c"}]},
    )

    assert output["total"] == 2
    assert output["exact_match_count"] == 1
    assert output["match_rate"] == 0.5


def test_build_batch_parse_summary_contains_total_and_recommendation() -> None:
    summary = build_batch_parse_summary(
        ["log a"],
        drain_outputs={"results": [{"index": 0, "template": "a"}]},
        rag_outputs={"results": [{"index": 0, "template": "a"}]},
        comparison_outputs={"exact_match_count": 1, "mismatch_count": 0, "match_rate": 1.0},
    )

    assert "Agent parsing report:" in summary
    assert "logs analyzed: 1" in summary
    assert "final recommendation:" in summary


def test_run_current_logs_agent_offline_with_trace_returns_success(tmp_path) -> None:
    csv_path = tmp_path / "templates.csv"
    csv_path.write_text(
        "template,description\n"
        "Receiving block <*> src: <*> dest: <*>,receiving block\n",
        encoding="utf-8",
    )

    _result, trace, _saved_paths = run_current_logs_agent_offline_with_trace(
        user_query="review current logs",
        logs=["Receiving block blk_1 src: /a dest: /b"],
        template_csv_path=str(csv_path),
        save_outputs=False,
    )

    assert trace.success is True
    assert trace.selected_action == "parse_current_logs"
    assert trace.tool_calls
