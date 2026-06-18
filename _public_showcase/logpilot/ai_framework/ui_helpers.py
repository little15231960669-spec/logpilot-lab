"""UI helper functions for optional Agent UI surfaces."""

from __future__ import annotations

import csv
import io
import json
import re
from pathlib import Path
from typing import Any


TASK_LABEL_TO_KEY = {
    "快速解析当前日志": "Fast baseline parse",
    "复核解析结果": "High-confidence parse review",
    "查找风险日志": "Find risky templates",
}

TASK_KEY_TO_LABEL = {value: key for key, value in TASK_LABEL_TO_KEY.items()}
TASK_KEY_TO_LABEL.update(
    {
        "high_confidence_parse_review": "复核解析结果",
        "fast_baseline_parse": "快速解析当前日志",
        "find_risky_templates": "查找风险日志",
        "compare_parser_tools": "复核解析结果",
        "explain_parsing_results": "复核解析结果",
        "parse_current_logs": "快速解析当前日志",
        "full_review": "复核解析结果",
        "Compare parser tools": "复核解析结果",
        "Explain parsing results": "复核解析结果",
    }
)

TASK_DESCRIPTIONS_ZH = {
    "快速解析当前日志": "使用 Drain 快速生成基线模板，适合快速查看当前日志的基础解析结果。",
    "复核解析结果": "Agent 会同时调用 Drain 和模板记忆解析，并比较两者输出，适合获得更可靠的解析复核结果。",
    "查找风险日志": "Agent 会找出不同解析工具输出不一致的日志，帮助定位需要人工复核的风险样本。",
}

TOOL_LABELS = {
    "drain_parse_logs": "Drain 基线解析",
    "template_memory_parse_logs": "模板记忆解析",
    "compare_batch_parse_outputs": "解析结果对比",
    "build_batch_parse_summary": "生成解释与建议",
    "retrieve_template_memory": "模板记忆检索",
    "parse_with_template_memory": "RAG 模板解析",
    "offline_parse_with_template_memory": "离线模板解析",
}

TOOL_DESCRIPTIONS = {
    "drain_parse_logs": "调用传统 Drain 思路生成基线模板。",
    "template_memory_parse_logs": "从历史模板库检索相似模板并生成模板记忆解析结果。",
    "compare_batch_parse_outputs": "比较不同解析工具的模板输出是否一致。",
    "build_batch_parse_summary": "汇总工具结果并生成解释与建议。",
    "retrieve_template_memory": "检索与当前日志最相似的历史模板。",
    "parse_with_template_memory": "结合模板记忆进行 RAG 模板解析。",
    "offline_parse_with_template_memory": "在离线模式下用模板记忆生成解析结果。",
}


def translate_status(status: str) -> str:
    return {
        "success": "成功",
        "partial_success": "部分成功",
        "failed": "失败",
    }.get(status, status or "-")


def translate_tool_name(tool_name: str) -> str:
    return TOOL_LABELS.get(tool_name, tool_name or "-")


def translate_agent_task(task_key: str) -> str:
    return TASK_KEY_TO_LABEL.get(task_key, task_key or "-")


def get_tool_description_zh(tool_name: str) -> str:
    return TOOL_DESCRIPTIONS.get(tool_name, "执行 Agent 计划中的工具步骤。")


def get_task_description_zh(task_label_or_key: str) -> str:
    label = translate_agent_task(task_label_or_key)
    return TASK_DESCRIPTIONS_ZH.get(label, "")


def get_task_key_from_label(task_label: str) -> str:
    return TASK_LABEL_TO_KEY.get(task_label, task_label)


def get_task_labels() -> list[str]:
    return list(TASK_LABEL_TO_KEY.keys())


def task_requires_template_memory(task_label_or_key: str) -> bool:
    label = translate_agent_task(task_label_or_key)
    return label in {"复核解析结果", "查找风险日志"} or task_label_or_key in {
        "高置信度解析复核",
        "解析器结果对比",
        "发现风险模板",
        "解释解析结果",
        "High-confidence parse review",
        "Compare parser tools",
        "Find risky templates",
        "Explain parsing results",
    }


def validate_template_memory_csv_text(csv_text: str) -> tuple[bool, str, str]:
    """Validate and normalize an uploaded template memory CSV."""
    try:
        reader = csv.DictReader(io.StringIO(csv_text or ""))
        fieldnames = list(reader.fieldnames or [])
        if "template" not in fieldnames:
            return False, "模板记忆库 CSV 至少需要包含 template 列。", ""

        normalized_fieldnames = list(fieldnames)
        if "description" not in normalized_fieldnames:
            normalized_fieldnames.append("description")

        rows = []
        for row in reader:
            normalized_row = {field: row.get(field, "") for field in normalized_fieldnames}
            normalized_row["description"] = normalized_row.get("description", "") or ""
            if normalized_row.get("template", "").strip():
                rows.append(normalized_row)

        if not rows:
            return False, "模板记忆库 CSV 至少需要包含一条非空 template。", ""

        output = io.StringIO()
        writer = csv.DictWriter(output, fieldnames=normalized_fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
        return True, "", output.getvalue()
    except csv.Error as exc:
        return False, f"模板记忆库 CSV 解析失败：{exc}", ""


def infer_dataset_type_from_name_or_logs(source_name: str, logs: list[str]) -> str:
    """Infer a coarse dataset type from the source name or log content."""
    name = (source_name or "").lower()
    if "bgl" in name:
        return "BGL"
    if "hdfs" in name:
        return "HDFS"

    sample = "\n".join(str(log) for log in logs[:100])
    hdfs_hits = sum(sample.count(token) for token in ["blk_", "NameSystem", "DataNode"])
    bgl_hits = sum(sample.count(token) for token in ["RAS", "KERNEL", "APPREAD"])
    if re.search(r"R\d+-M\d+-N\d+", sample):
        bgl_hits += 2

    if hdfs_hits >= 2:
        return "HDFS"
    if bgl_hits >= 2:
        return "BGL"
    return "unknown"


def infer_template_memory_type(template_csv_path: str) -> str:
    path = (template_csv_path or "").lower()
    if "bgl" in path:
        return "BGL"
    if "hdfs" in path:
        return "HDFS"
    return "unknown"


def build_memory_dataset_warning(dataset_type: str, memory_type: str) -> str | None:
    if (
        dataset_type
        and memory_type
        and dataset_type != "unknown"
        and memory_type != "unknown"
        and dataset_type != memory_type
    ):
        return (
            f"检测到当前日志可能是 {dataset_type}，但模板记忆库为 {memory_type}。"
            "跨数据集检索会导致模板记忆解析结果不可靠，建议切换到匹配的数据集模板库。"
        )
    return None


def _display_value(value: Any) -> Any:
    if value is None or value == "":
        return "-"
    if isinstance(value, bool):
        return "一致" if value else "不一致"
    return value


def _short_log(log: str, max_length: int = 120) -> str:
    text = str(log or "").strip()
    if len(text) <= max_length:
        return text or "-"
    return text[:max_length] + "..."


def build_tool_trace_table(trace) -> list[dict]:
    """Build a readable Chinese table for an agent trace."""
    tool_calls = getattr(trace, "tool_calls", None) or []
    rows = []
    for index, call in enumerate(tool_calls, start=1):
        tool_name = getattr(call, "tool_name", "")
        error = getattr(call, "error", None)
        rows.append(
            {
                "步骤": index,
                "工具": translate_tool_name(tool_name),
                "作用": get_tool_description_zh(tool_name),
                "状态": "失败" if error else "成功",
                "耗时(ms)": _display_value(getattr(call, "duration_ms", None)),
                "错误": error or "-",
            }
        )
    return rows


def _results_by_index(output: dict | None) -> dict:
    return {
        item.get("index"): item
        for item in (output or {}).get("results", [])
        if isinstance(item, dict)
    }


def _comparison_by_index(output: dict | None) -> dict:
    return {
        item.get("index"): item
        for item in (output or {}).get("items", [])
        if isinstance(item, dict)
    }


def build_parsed_results_table(
    intermediate_outputs: dict,
    dataset_memory_warning: str | None = None,
) -> list[dict]:
    """Build a task-aware parsed result table without leaking raw None values."""
    outputs = intermediate_outputs or {}
    drain_rows = _results_by_index(outputs.get("drain_parse_logs"))
    memory_rows = _results_by_index(outputs.get("template_memory_parse_logs"))
    comparison_rows = _comparison_by_index(outputs.get("compare_batch_parse_outputs"))

    has_drain = bool(drain_rows)
    has_memory = bool(memory_rows)
    has_comparison = bool(comparison_rows)
    indexes = sorted(set(drain_rows) | set(memory_rows) | set(comparison_rows))
    rows = []

    for index in indexes:
        drain_item = drain_rows.get(index, {})
        memory_item = memory_rows.get(index, {})
        comparison_item = comparison_rows.get(index, {})
        log = (
            drain_item.get("content")
            or memory_item.get("content")
            or drain_item.get("log")
            or memory_item.get("log")
            or "-"
        )
        base = {
            "序号": index,
            "日志正文": _short_log(log),
        }

        exact_match = comparison_item.get("exact_match")
        if dataset_memory_warning and has_memory:
            explanation = "模板记忆库可能与数据集不匹配"
        elif has_comparison:
            explanation = "两个工具给出相同模板" if exact_match else "两个工具输出不同，建议复核"
        else:
            explanation = "当前任务未执行对比"

        if has_drain and has_memory and has_comparison:
            base.update(
                {
                    "Drain 模板": _display_value(
                        comparison_item.get("drain_template") or drain_item.get("template")
                    ),
                    "模板记忆模板": _display_value(
                        comparison_item.get("rag_template") or memory_item.get("template")
                    ),
                    "是否一致": _display_value(exact_match),
                    "说明": explanation,
                }
            )
        elif has_memory and not has_drain:
            base.update(
                {
                    "模板记忆模板": _display_value(memory_item.get("template")),
                    "Top-1 检索分数": _display_value(memory_item.get("score")),
                    "说明": explanation,
                }
            )
        elif has_drain and not has_memory:
            base.update(
                {
                    "Drain 模板": _display_value(drain_item.get("template")),
                    "说明": explanation,
                }
            )
        else:
            base.update(
                {
                    "Drain 模板": _display_value(drain_item.get("template")),
                    "模板记忆模板": _display_value(memory_item.get("template")),
                    "是否一致": _display_value(exact_match),
                    "说明": explanation,
                }
            )

        rows.append({key: _display_value(value) for key, value in base.items()})

    return rows


def format_final_recommendation_zh(
    final_answer: str,
    plan: dict | None = None,
    intermediate_outputs: dict | None = None,
    dataset_type: str = "unknown",
    memory_type: str = "unknown",
    dataset_memory_warning: str | None = None,
) -> str:
    """Render a Chinese Markdown recommendation body for the Agent-first UI."""
    plan = plan or {}
    outputs = intermediate_outputs or {}
    tool_sequence = plan.get("tool_sequence") or []
    tool_names = [translate_tool_name(tool) for tool in tool_sequence]
    drain_results = (outputs.get("drain_parse_logs") or {}).get("results", [])
    memory_results = (outputs.get("template_memory_parse_logs") or {}).get("results", [])
    comparison = outputs.get("compare_batch_parse_outputs") or {}
    has_comparison = bool(comparison)

    log_count = plan.get("log_count")
    if log_count is None:
        log_count = max(len(drain_results), len(memory_results), len(comparison.get("items", [])))

    lines = [
        (
            f"本次 Agent 共分析 **{log_count}** 条日志，调用了 "
            f"**{len(tool_sequence)}** 个工具：{', '.join(tool_names) or '未记录工具'}。"
        ),
        "",
        f"- Drain 解析模板数：{len(drain_results)}",
        f"- 模板记忆解析模板数：{len(memory_results)}",
    ]

    if has_comparison:
        exact_count = comparison.get("exact_match_count", 0)
        mismatch_count = comparison.get("mismatch_count", 0)
        match_rate = comparison.get("match_rate", 0.0)
        lines.extend(
            [
                f"- 一致数量：{exact_count}",
                f"- 不一致数量：{mismatch_count}",
                f"- 匹配率：{match_rate}",
                "",
            ]
        )

        if float(match_rate) == 0.0 and comparison.get("total", 0):
            lines.extend(
                [
                    "两个解析工具输出的模板全部不一致，通常说明当前模板记忆结果不适合直接作为最终模板使用。",
                    "",
                    "常见原因包括模板记忆库与日志数据集不匹配、日志格式差异较大，或某个解析工具产生了过度泛化模板。",
                    "",
                ]
            )
        else:
            lines.extend(
                [
                    f"两个解析工具的匹配率为 **{match_rate}**。匹配率越高，说明 Drain 与模板记忆解析越一致。",
                    "",
                ]
            )
    else:
        lines.extend(
            [
                "- 一致数量：未执行对比",
                "- 不一致数量：未执行对比",
                "- 匹配率：未执行对比",
                "",
                "本次任务没有执行多解析器对比，因此结论主要基于当前执行的工具结果。",
                "",
            ]
        )

    if dataset_memory_warning:
        lines.extend(
            [
                (
                    f"从当前输入看，上传日志更接近 **{dataset_type}**，但模板记忆库使用的是 "
                    f"**{memory_type}**，因此模板记忆解析可能返回不匹配数据集的模板。"
                ),
                "",
            ]
        )

    lines.append("**建议：**")
    if dataset_memory_warning:
        lines.append(
            f"优先切换到 {dataset_type} 模板记忆库后重新运行；在切换前，不建议接受模板记忆解析结果。"
        )
    elif has_comparison and comparison.get("mismatch_count", 0):
        lines.append("先查看不一致日志及其模板差异，再决定是否接受当前批量解析结果。")
    elif has_comparison:
        lines.append("两个解析工具结果整体一致，可以接受当前模板结果，并保留抽样复核。")
    elif memory_results:
        lines.append("当前建议主要基于模板记忆解析结果；如需更高置信度，建议运行复核解析结果任务。")
    elif drain_results:
        lines.append("当前结果适合作为快速基线；如需正式结论，建议继续执行模板记忆解析和对比。")
    else:
        lines.append(final_answer or "本次运行没有生成可用解析结果。")

    return "\n".join(lines)


def get_default_agent_demo_log() -> str:
    return (
        "Receiving block blk_38865049064139660 src: /10.250.19.102:54106 "
        "dest: /10.250.19.102:50010"
    )


def get_default_candidate_template() -> str:
    return "Receiving block <*> src: <*> dest: <*>"


def get_default_user_query() -> str:
    return "Please review this log parse, retrieve similar templates, and give a final recommendation."


def format_tool_outputs_for_display(tool_outputs: dict) -> dict:
    """Return a compact JSON-friendly view of tool outputs."""
    formatted = {}
    for key, value in tool_outputs.items():
        if isinstance(value, dict):
            formatted[key] = {
                "tool_name": value.get("tool_name", key),
                "has_error": bool(value.get("error")),
                "error": value.get("error"),
                "keys": list(value.keys()),
            }
            if "result" in value:
                formatted[key]["result"] = value["result"]
            if "results" in value:
                formatted[key]["results"] = value["results"]
            if "summary" in value:
                formatted[key]["summary"] = value["summary"]
            if "top_k" in value:
                formatted[key]["top_k"] = value["top_k"]
            if "exact_match" in value:
                formatted[key]["exact_match"] = value["exact_match"]
                formatted[key]["token_overlap"] = value.get("token_overlap")
        else:
            formatted[key] = value
    return formatted


def load_eval_summary_if_exists(
    summary_path: str = "results/agent_eval/agent_eval_summary.json",
) -> dict | None:
    path = Path(summary_path)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def summarize_agent_result_for_ui(result, trace, saved_paths: dict) -> dict:
    """Build a compact dictionary for Streamlit display."""
    return {
        "selected_action": getattr(result.plan, "action", ""),
        "plan_reason": getattr(result.plan, "reason", ""),
        "status": getattr(trace, "status", ""),
        "success": getattr(trace, "success", False),
        "tool_error_count": getattr(trace, "tool_error_count", 0),
        "final_answer": getattr(result, "final_answer", ""),
        "saved_json_path": saved_paths.get("json"),
        "saved_markdown_path": saved_paths.get("markdown"),
    }


def normalize_selected_logs_for_agent(raw_logs, max_logs: int = 10) -> list[str]:
    """Normalize selected logs from app data structures into text lines."""
    if raw_logs is None:
        return []

    values = []
    candidate_fields = ["Content", "content", "log", "message", "raw_log"]

    if hasattr(raw_logs, "columns") and hasattr(raw_logs, "__getitem__"):
        for field in candidate_fields:
            if field in raw_logs.columns:
                values = raw_logs[field].tolist()
                break
        if not values:
            values = raw_logs.iloc[:, 0].tolist() if len(raw_logs.columns) else []
    elif isinstance(raw_logs, list):
        if all(isinstance(item, str) for item in raw_logs):
            values = raw_logs
        elif all(isinstance(item, dict) for item in raw_logs):
            for item in raw_logs:
                for field in candidate_fields:
                    value = item.get(field)
                    if value:
                        values.append(value)
                        break
    else:
        values = [raw_logs]

    normalized = []
    for value in values:
        text = str(value).strip()
        if text:
            normalized.append(text)
        if len(normalized) >= max_logs:
            break
    return normalized


def build_agent_log_preview(logs: list[str], max_preview: int = 5) -> str:
    """Build a numbered multi-line preview for display."""
    return "\n".join(
        f"{index}. {log}"
        for index, log in enumerate(logs[:max_preview], start=1)
    )


def build_batch_candidate_templates_from_logs(logs: list[str]) -> list[str]:
    """Placeholder for future batch candidate template generation."""
    return []
