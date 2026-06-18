"""Lightweight Python-controlled log review agent."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable

from logpilot.ai_framework.agent_trace import (
    AgentRunTrace,
    ToolCallTrace,
    build_agent_run_trace,
    now_iso,
    save_agent_report_markdown,
    save_agent_trace_json,
    summarize_for_trace,
)
from logpilot.ai_framework.log_tools import (
    build_batch_parse_summary,
    build_review_summary,
    compare_batch_parse_outputs_tool,
    compare_template_strings,
    drain_parse_logs_tool,
    parse_with_template_memory_tool,
    retrieve_template_memory_tool,
    template_memory_parse_logs_tool,
)
from logpilot.ai_framework.models import build_chat_model
from logpilot.ai_framework.structured_log_parser import extract_json_object


ALLOWED_ACTIONS = {
    "retrieve_only",
    "parse_with_memory",
    "compare_with_candidate",
    "full_review",
}


AGENT_TASK_PLANS = {
    "High-confidence parse review": {
        "action": "high_confidence_parse_review",
        "reason": (
            "The task asks for reliable parsing and review, so the agent will use "
            "both Drain and Template Memory tools and compare their outputs."
        ),
        "tool_sequence": [
            "drain_parse_logs",
            "template_memory_parse_logs",
            "compare_batch_parse_outputs",
            "build_batch_parse_summary",
        ],
    },
    "Fast baseline parse": {
        "action": "fast_baseline_parse",
        "reason": "The task prioritizes speed, so the agent will use Drain and summarize the result.",
        "tool_sequence": ["drain_parse_logs", "build_batch_parse_summary"],
    },
    "Compare parser tools": {
        "action": "compare_parser_tools",
        "reason": "The task asks for parser comparison, so the agent will run Drain, Template Memory, and compare outputs.",
        "tool_sequence": [
            "drain_parse_logs",
            "template_memory_parse_logs",
            "compare_batch_parse_outputs",
            "build_batch_parse_summary",
        ],
    },
    "Find risky templates": {
        "action": "find_risky_templates",
        "reason": "The task asks for risky templates, so mismatches between parser tools should be highlighted.",
        "tool_sequence": [
            "drain_parse_logs",
            "template_memory_parse_logs",
            "compare_batch_parse_outputs",
            "build_batch_parse_summary",
        ],
    },
    "Explain parsing results": {
        "action": "explain_parsing_results",
        "reason": "The task asks for explanation, so the agent will parse with Template Memory and summarize results.",
        "tool_sequence": ["template_memory_parse_logs", "build_batch_parse_summary"],
    },
}


AGENT_TASK_ALIASES = {
    "快速解析当前日志": "Fast baseline parse",
    "复核解析结果": "High-confidence parse review",
    "查找风险日志": "Find risky templates",
}


def plan_tools_for_agent_task(
    agent_task: str,
    logs: list[str],
    offline: bool = True,
) -> dict:
    """Return a deterministic tool plan for an agent task."""
    resolved_task = AGENT_TASK_ALIASES.get(agent_task, agent_task)
    plan = AGENT_TASK_PLANS.get(resolved_task) or AGENT_TASK_PLANS[
        "High-confidence parse review"
    ]
    return {
        "action": plan["action"],
        "reason": plan["reason"],
        "tool_sequence": list(plan["tool_sequence"]),
        "offline": offline,
        "log_count": len(logs),
    }


def _build_tool_registry_context() -> str:
    try:
        from logpilot.ai_framework.tool_registry import build_tool_context_for_agent

        return build_tool_context_for_agent()
    except Exception:
        return ""


@dataclass
class AgentPlan:
    action: str
    reason: str


@dataclass
class LogReviewAgentResult:
    user_query: str
    log: str
    plan: AgentPlan
    tool_outputs: dict
    final_answer: str


def build_agent_planning_prompt(
    user_query: str,
    log: str,
    candidate_template: str | None = None,
) -> str:
    """Build a JSON-only action planning prompt for the review agent."""
    candidate_text = candidate_template or "未提供"
    tool_context = _build_tool_registry_context()
    return (
        "你是 LogPilot 日志复核 Agent 的规划器。\n"
        "你只负责选择下一步 action，不执行工具。\n"
        "允许的 action 只有：retrieve_only, parse_with_memory, "
        "compare_with_candidate, full_review。\n\n"
        "action 含义：\n"
        "- retrieve_only：只检索相似历史模板。\n"
        "- parse_with_memory：用 Template Memory RAG 解析日志。\n"
        "- compare_with_candidate：先解析日志，再和用户给定 candidate_template 比较。\n"
        "- full_review：检索相似模板 + RAG 解析 + 如有 candidate_template 则比较 + 生成复核报告。\n\n"
        f"{tool_context}\n\n"
        "请只输出 JSON object，不要输出 Markdown，不要输出额外解释文本。\n"
        "JSON object 格式必须是：\n"
        "{\n"
        '  "action": "full_review",\n'
        '  "reason": "The user asks for a complete log parsing review."\n'
        "}\n\n"
        f"用户请求：{user_query}\n"
        f"日志：{log}\n"
        f"candidate_template：{candidate_text}"
    )


def parse_agent_plan(text: str) -> AgentPlan:
    """Parse and validate an LLM action plan, falling back safely."""
    fallback = AgentPlan(
        action="full_review",
        reason="Invalid plan, fallback to full_review.",
    )
    try:
        data = extract_json_object(text)
    except Exception:
        return fallback

    action = str(data.get("action", "")).strip()
    reason = str(data.get("reason", "")).strip()

    if action not in ALLOWED_ACTIONS or not reason:
        return fallback

    return AgentPlan(action=action, reason=reason)


def plan_log_review_action(
    user_query: str,
    log: str,
    candidate_template: str | None = None,
) -> AgentPlan:
    """Ask the LLM for a JSON action plan, with deterministic fallback."""
    try:
        llm = build_chat_model()
        response = llm.invoke(
            build_agent_planning_prompt(user_query, log, candidate_template)
        )
        content = getattr(response, "content", response)
        if not isinstance(content, str):
            return AgentPlan(
                action="full_review",
                reason="LLM planning returned non-text content, fallback to full_review.",
            )
        return parse_agent_plan(content)
    except Exception:
        return AgentPlan(
            action="full_review",
            reason="LLM planning failed, fallback to full_review.",
        )


def _parsed_template(parse_output: dict) -> str | None:
    result = parse_output.get("result")
    if isinstance(result, dict):
        template = result.get("template")
        if isinstance(template, str) and template:
            return template
    return None


def _compare_if_possible(parse_output: dict, candidate_template: str | None) -> dict:
    if not candidate_template:
        return {
            "tool_name": "compare_template_strings",
            "error": "candidate_template is not provided.",
        }

    parsed_template = _parsed_template(parse_output)
    if not parsed_template:
        return {
            "tool_name": "compare_template_strings",
            "error": "parsed template is not available.",
        }

    return compare_template_strings(parsed_template, candidate_template)


def _duration_ms(started_at: str, ended_at: str) -> float:
    start = datetime.fromisoformat(started_at)
    end = datetime.fromisoformat(ended_at)
    return round((end - start).total_seconds() * 1000, 3)


def _run_traced_tool(
    tool_name: str,
    input_summary: dict,
    tool_func: Callable[..., dict],
    *args: Any,
    **kwargs: Any,
) -> tuple[dict, ToolCallTrace]:
    started_at = now_iso()
    output = None
    error = None
    try:
        output = tool_func(*args, **kwargs)
        if isinstance(output, dict) and output.get("error"):
            error = str(output.get("error"))
    except Exception as exc:
        error = str(exc)
        output = {"tool_name": tool_name, "error": error}
    ended_at = now_iso()

    trace = ToolCallTrace(
        tool_name=tool_name,
        input_summary=summarize_for_trace(input_summary),
        output_summary=summarize_for_trace(output),
        error=summarize_for_trace(error) if error else None,
        started_at=started_at,
        ended_at=ended_at,
        duration_ms=_duration_ms(started_at, ended_at),
    )
    return output, trace


def _tool_error_count(tool_traces: list[ToolCallTrace]) -> int:
    return sum(1 for tool_trace in tool_traces if tool_trace.error)


def _status_from_tool_traces(tool_traces: list[ToolCallTrace]) -> str:
    return "partial_success" if _tool_error_count(tool_traces) > 0 else "success"


def run_log_review_agent(
    user_query: str,
    log: str,
    template_csv_path: str,
    candidate_template: str | None = None,
    k: int = 3,
) -> LogReviewAgentResult:
    """Run the lightweight log review agent with Python-controlled tools."""
    plan = plan_log_review_action(user_query, log, candidate_template)
    tool_outputs: dict = {}

    if plan.action == "retrieve_only":
        retrieve_output = retrieve_template_memory_tool(log, template_csv_path, k=k)
        tool_outputs["retrieve_template_memory"] = retrieve_output
        final_answer = build_review_summary(
            log,
            retrieved_templates=retrieve_output.get("top_k", []),
        )

    elif plan.action == "parse_with_memory":
        parse_output = parse_with_template_memory_tool(log, template_csv_path, k=k)
        tool_outputs["parse_with_template_memory"] = parse_output
        final_answer = build_review_summary(
            log,
            parsed_result=parse_output,
        )

    elif plan.action == "compare_with_candidate":
        parse_output = parse_with_template_memory_tool(log, template_csv_path, k=k)
        compare_output = _compare_if_possible(parse_output, candidate_template)
        tool_outputs["parse_with_template_memory"] = parse_output
        tool_outputs["compare_template_strings"] = compare_output
        final_answer = build_review_summary(
            log,
            parsed_result=parse_output,
            comparison_result=compare_output,
        )

    else:
        retrieve_output = retrieve_template_memory_tool(log, template_csv_path, k=k)
        parse_output = parse_with_template_memory_tool(log, template_csv_path, k=k)
        tool_outputs["retrieve_template_memory"] = retrieve_output
        tool_outputs["parse_with_template_memory"] = parse_output

        compare_output = None
        if candidate_template:
            compare_output = _compare_if_possible(parse_output, candidate_template)
            tool_outputs["compare_template_strings"] = compare_output

        final_answer = build_review_summary(
            log,
            parsed_result=parse_output,
            retrieved_templates=retrieve_output.get("top_k", []),
            comparison_result=compare_output,
        )

    return LogReviewAgentResult(
        user_query=user_query,
        log=log,
        plan=plan,
        tool_outputs=tool_outputs,
        final_answer=final_answer,
    )


def _save_trace_outputs(
    trace: AgentRunTrace,
    save_outputs: bool,
) -> dict:
    if not save_outputs:
        return {}
    return {
        "json": save_agent_trace_json(trace),
        "markdown": save_agent_report_markdown(trace),
    }


def run_log_review_agent_with_trace(
    user_query: str,
    log: str,
    template_csv_path: str,
    candidate_template: str | None = None,
    k: int = 3,
    save_outputs: bool = True,
) -> tuple[LogReviewAgentResult, AgentRunTrace, dict]:
    """Run the log review agent and export a standard trace/report."""
    started_at = now_iso()
    tool_traces: list[ToolCallTrace] = []

    try:
        plan = plan_log_review_action(user_query, log, candidate_template)
        tool_outputs: dict = {}

        if plan.action == "retrieve_only":
            retrieve_output, retrieve_trace = _run_traced_tool(
                "retrieve_template_memory",
                {"log": log, "template_csv_path": template_csv_path, "k": k},
                retrieve_template_memory_tool,
                log,
                template_csv_path,
                k=k,
            )
            tool_traces.append(retrieve_trace)
            tool_outputs["retrieve_template_memory"] = retrieve_output
            final_answer = build_review_summary(
                log,
                retrieved_templates=retrieve_output.get("top_k", []),
            )

        elif plan.action == "parse_with_memory":
            parse_output, parse_trace = _run_traced_tool(
                "parse_with_template_memory",
                {"log": log, "template_csv_path": template_csv_path, "k": k},
                parse_with_template_memory_tool,
                log,
                template_csv_path,
                k=k,
            )
            tool_traces.append(parse_trace)
            tool_outputs["parse_with_template_memory"] = parse_output
            final_answer = build_review_summary(log, parsed_result=parse_output)

        elif plan.action == "compare_with_candidate":
            parse_output, parse_trace = _run_traced_tool(
                "parse_with_template_memory",
                {"log": log, "template_csv_path": template_csv_path, "k": k},
                parse_with_template_memory_tool,
                log,
                template_csv_path,
                k=k,
            )
            tool_traces.append(parse_trace)
            tool_outputs["parse_with_template_memory"] = parse_output

            compare_output, compare_trace = _run_traced_tool(
                "compare_template_strings",
                {
                    "parsed_template": _parsed_template(parse_output),
                    "candidate_template": candidate_template,
                },
                _compare_if_possible,
                parse_output,
                candidate_template,
            )
            tool_traces.append(compare_trace)
            tool_outputs["compare_template_strings"] = compare_output
            final_answer = build_review_summary(
                log,
                parsed_result=parse_output,
                comparison_result=compare_output,
            )

        else:
            retrieve_output, retrieve_trace = _run_traced_tool(
                "retrieve_template_memory",
                {"log": log, "template_csv_path": template_csv_path, "k": k},
                retrieve_template_memory_tool,
                log,
                template_csv_path,
                k=k,
            )
            tool_traces.append(retrieve_trace)
            tool_outputs["retrieve_template_memory"] = retrieve_output

            parse_output, parse_trace = _run_traced_tool(
                "parse_with_template_memory",
                {"log": log, "template_csv_path": template_csv_path, "k": k},
                parse_with_template_memory_tool,
                log,
                template_csv_path,
                k=k,
            )
            tool_traces.append(parse_trace)
            tool_outputs["parse_with_template_memory"] = parse_output

            compare_output = None
            if candidate_template:
                compare_output, compare_trace = _run_traced_tool(
                    "compare_template_strings",
                    {
                        "parsed_template": _parsed_template(parse_output),
                        "candidate_template": candidate_template,
                    },
                    _compare_if_possible,
                    parse_output,
                    candidate_template,
                )
                tool_traces.append(compare_trace)
                tool_outputs["compare_template_strings"] = compare_output

            final_answer = build_review_summary(
                log,
                parsed_result=parse_output,
                retrieved_templates=retrieve_output.get("top_k", []),
                comparison_result=compare_output,
            )

        result = LogReviewAgentResult(
            user_query=user_query,
            log=log,
            plan=plan,
            tool_outputs=tool_outputs,
            final_answer=final_answer,
        )
        ended_at = now_iso()
        trace = build_agent_run_trace(
            user_query=user_query,
            log=log,
            selected_action=plan.action,
            plan_reason=plan.reason,
            tool_calls=tool_traces,
            final_answer=final_answer,
            success=True,
            status=_status_from_tool_traces(tool_traces),
            tool_error_count=_tool_error_count(tool_traces),
            started_at=started_at,
            ended_at=ended_at,
        )
        saved_paths = _save_trace_outputs(trace, save_outputs)
        return result, trace, saved_paths

    except Exception as exc:
        ended_at = now_iso()
        plan = AgentPlan(
            action="full_review",
            reason="Unexpected agent failure, trace generated for debugging.",
        )
        final_answer = f"Log review agent failed: {exc}"
        result = LogReviewAgentResult(
            user_query=user_query,
            log=log,
            plan=plan,
            tool_outputs={"error": str(exc)},
            final_answer=final_answer,
        )
        trace = build_agent_run_trace(
            user_query=user_query,
            log=log,
            selected_action=plan.action,
            plan_reason=plan.reason,
            tool_calls=tool_traces,
            final_answer=final_answer,
            success=False,
            status="failed",
            tool_error_count=_tool_error_count(tool_traces),
            started_at=started_at,
            ended_at=ended_at,
        )
        saved_paths = _save_trace_outputs(trace, save_outputs)
        return result, trace, saved_paths


def _build_offline_parse_trace(
    log: str,
    retrieved_templates: list[dict],
) -> tuple[dict, ToolCallTrace]:
    started_at = now_iso()
    top_template = (
        retrieved_templates[0].get("template", "")
        if retrieved_templates
        else "Receiving block <*> src: <*> dest: <*>"
    )
    output = {
        "tool_name": "offline_parse_with_template_memory",
        "log": log,
        "result": {
            "template": top_template,
            "variables": [
                "blk_38865049064139660",
                "/10.250.19.102:54106",
                "/10.250.19.102:50010",
            ],
            "confidence": 0.99,
            "reason": (
                "Offline demo uses the top retrieved historical template as a "
                "deterministic parsing suggestion. No real LLM call was made."
            ),
        },
    }
    ended_at = now_iso()
    trace = ToolCallTrace(
        tool_name="offline_parse_with_template_memory",
        input_summary=summarize_for_trace(
            {
                "log": log,
                "top_retrieved_template": top_template,
                "mode": "offline_demo",
            }
        ),
        output_summary=summarize_for_trace(output),
        error=None,
        started_at=started_at,
        ended_at=ended_at,
        duration_ms=_duration_ms(started_at, ended_at),
    )
    return output, trace


def run_log_review_agent_offline_demo_with_trace(
    user_query: str,
    log: str,
    template_csv_path: str,
    candidate_template: str | None = None,
    k: int = 3,
    save_outputs: bool = True,
) -> tuple[LogReviewAgentResult, AgentRunTrace, dict]:
    """Run an offline happy-path demo with deterministic tool outputs."""
    started_at = now_iso()
    tool_traces: list[ToolCallTrace] = []
    plan = AgentPlan(
        action="full_review",
        reason="Offline demo mode uses a deterministic full_review plan.",
    )

    try:
        retrieve_output, retrieve_trace = _run_traced_tool(
            "retrieve_template_memory",
            {"log": log, "template_csv_path": template_csv_path, "k": k},
            retrieve_template_memory_tool,
            log,
            template_csv_path,
            k=k,
        )
        tool_traces.append(retrieve_trace)

        parse_output, parse_trace = _build_offline_parse_trace(
            log,
            retrieve_output.get("top_k", []),
        )
        tool_traces.append(parse_trace)

        tool_outputs = {
            "retrieve_template_memory": retrieve_output,
            "offline_parse_with_template_memory": parse_output,
        }

        compare_output = None
        if candidate_template:
            compare_output, compare_trace = _run_traced_tool(
                "compare_template_strings",
                {
                    "parsed_template": _parsed_template(parse_output),
                    "candidate_template": candidate_template,
                },
                _compare_if_possible,
                parse_output,
                candidate_template,
            )
            tool_traces.append(compare_trace)
            tool_outputs["compare_template_strings"] = compare_output

        final_answer = (
            "Offline demo mode: parsing is based on deterministic retrieved-template "
            "fallback rather than a real LLM call.\n"
            + build_review_summary(
                log,
                parsed_result=parse_output,
                retrieved_templates=retrieve_output.get("top_k", []),
                comparison_result=compare_output,
            )
        )

        result = LogReviewAgentResult(
            user_query=user_query,
            log=log,
            plan=plan,
            tool_outputs=tool_outputs,
            final_answer=final_answer,
        )
        ended_at = now_iso()
        trace = build_agent_run_trace(
            user_query=user_query,
            log=log,
            selected_action=plan.action,
            plan_reason=plan.reason,
            tool_calls=tool_traces,
            final_answer=final_answer,
            success=True,
            status=_status_from_tool_traces(tool_traces),
            tool_error_count=_tool_error_count(tool_traces),
            started_at=started_at,
            ended_at=ended_at,
        )
        saved_paths = _save_trace_outputs(trace, save_outputs)
        return result, trace, saved_paths

    except Exception as exc:
        ended_at = now_iso()
        final_answer = f"Offline log review demo failed: {exc}"
        result = LogReviewAgentResult(
            user_query=user_query,
            log=log,
            plan=plan,
            tool_outputs={"error": str(exc)},
            final_answer=final_answer,
        )
        trace = build_agent_run_trace(
            user_query=user_query,
            log=log,
            selected_action=plan.action,
            plan_reason=plan.reason,
            tool_calls=tool_traces,
            final_answer=final_answer,
            success=False,
            status="failed",
            tool_error_count=_tool_error_count(tool_traces),
            started_at=started_at,
            ended_at=ended_at,
        )
        saved_paths = _save_trace_outputs(trace, save_outputs)
        return result, trace, saved_paths


def run_current_logs_agent_offline_with_trace(
    user_query: str,
    logs: list[str],
    template_csv_path: str,
    drain_config: dict | None = None,
    max_logs: int = 10,
    save_outputs: bool = True,
):
    """Run a deterministic offline agent over the currently selected logs."""
    started_at = now_iso()
    selected_logs = [log for log in logs if str(log).strip()][:max_logs]
    tool_traces: list[ToolCallTrace] = []
    plan = AgentPlan(
        action="parse_current_logs",
        reason=(
            "Offline current-logs agent uses parser tools on the logs selected "
            "in the Streamlit sidebar."
        ),
    )

    try:
        drain_outputs, drain_trace = _run_traced_tool(
            "drain_parse_logs",
            {"log_count": len(selected_logs), "drain_config": drain_config or {}},
            drain_parse_logs_tool,
            selected_logs,
            drain_config=drain_config,
        )
        tool_traces.append(drain_trace)

        rag_outputs, rag_trace = _run_traced_tool(
            "template_memory_parse_logs",
            {
                "log_count": len(selected_logs),
                "template_csv_path": template_csv_path,
                "offline": True,
            },
            template_memory_parse_logs_tool,
            selected_logs,
            template_csv_path,
            k=3,
            offline=True,
        )
        tool_traces.append(rag_trace)

        comparison_outputs, comparison_trace = _run_traced_tool(
            "compare_batch_parse_outputs",
            {
                "drain_result_count": len(drain_outputs.get("results", [])),
                "rag_result_count": len(rag_outputs.get("results", [])),
            },
            compare_batch_parse_outputs_tool,
            drain_outputs,
            rag_outputs,
        )
        tool_traces.append(comparison_trace)

        summary_outputs, summary_trace = _run_traced_tool(
            "build_batch_parse_summary",
            {"log_count": len(selected_logs)},
            lambda *args, **kwargs: {
                "tool_name": "build_batch_parse_summary",
                "summary": build_batch_parse_summary(*args, **kwargs),
                "error": None,
            },
            selected_logs,
            drain_outputs=drain_outputs,
            rag_outputs=rag_outputs,
            comparison_outputs=comparison_outputs,
        )
        tool_traces.append(summary_trace)

        tool_outputs = {
            "drain_parse_logs": drain_outputs,
            "template_memory_parse_logs": rag_outputs,
            "compare_batch_parse_outputs": comparison_outputs,
            "build_batch_parse_summary": summary_outputs,
        }
        final_answer = summary_outputs.get("summary", "")
        result = LogReviewAgentResult(
            user_query=user_query,
            log=f"{len(selected_logs)} current selected logs",
            plan=plan,
            tool_outputs=tool_outputs,
            final_answer=final_answer,
        )
        ended_at = now_iso()
        trace = build_agent_run_trace(
            user_query=user_query,
            log=f"Current selected logs count: {len(selected_logs)}",
            selected_action=plan.action,
            plan_reason=plan.reason,
            tool_calls=tool_traces,
            final_answer=final_answer,
            success=True,
            status=_status_from_tool_traces(tool_traces),
            tool_error_count=_tool_error_count(tool_traces),
            started_at=started_at,
            ended_at=ended_at,
        )
        saved_paths = _save_trace_outputs(trace, save_outputs)
        return result, trace, saved_paths

    except Exception as exc:
        ended_at = now_iso()
        final_answer = f"Current logs offline agent failed: {exc}"
        result = LogReviewAgentResult(
            user_query=user_query,
            log=f"{len(selected_logs)} current selected logs",
            plan=plan,
            tool_outputs={"error": str(exc)},
            final_answer=final_answer,
        )
        trace = build_agent_run_trace(
            user_query=user_query,
            log=f"Current selected logs count: {len(selected_logs)}",
            selected_action=plan.action,
            plan_reason=plan.reason,
            tool_calls=tool_traces,
            final_answer=final_answer,
            success=False,
            status="failed",
            tool_error_count=_tool_error_count(tool_traces),
            started_at=started_at,
            ended_at=ended_at,
        )
        saved_paths = _save_trace_outputs(trace, save_outputs)
        return result, trace, saved_paths


def run_agent_task_on_logs_offline_with_trace(
    agent_task: str,
    user_query: str,
    logs: list[str],
    template_csv_path: str | None = None,
    drain_config: dict | None = None,
    max_logs: int = 10,
    save_outputs: bool = True,
    dataset_name: str | None = None,
):
    """Run a task-driven offline agent over logs with strategy-based tools."""
    started_at = now_iso()
    selected_logs = [log for log in logs if str(log).strip()][:max_logs]
    plan = plan_tools_for_agent_task(agent_task, selected_logs, offline=True)
    agent_plan = AgentPlan(action=plan["action"], reason=plan["reason"])
    tool_traces: list[ToolCallTrace] = []
    intermediate_outputs: dict = {}

    try:
        for tool_name in plan["tool_sequence"]:
            if tool_name == "drain_parse_logs":
                output, trace = _run_traced_tool(
                    tool_name,
                    {
                        "log_count": len(selected_logs),
                        "drain_config": drain_config or {},
                        "dataset_name": dataset_name,
                    },
                    drain_parse_logs_tool,
                    selected_logs,
                    drain_config=drain_config,
                    dataset_name=dataset_name,
                )
            elif tool_name == "template_memory_parse_logs":
                if not template_csv_path:
                    raise ValueError(
                        "template_csv_path is required when template_memory_parse_logs is in the tool sequence."
                    )
                output, trace = _run_traced_tool(
                    tool_name,
                    {
                        "log_count": len(selected_logs),
                        "template_csv_path": template_csv_path,
                        "offline": True,
                        "dataset_name": dataset_name,
                    },
                    template_memory_parse_logs_tool,
                    selected_logs,
                    template_csv_path,
                    k=3,
                    offline=True,
                    dataset_name=dataset_name,
                )
            elif tool_name == "compare_batch_parse_outputs":
                output, trace = _run_traced_tool(
                    tool_name,
                    {
                        "drain_result_count": len(
                            intermediate_outputs.get("drain_parse_logs", {}).get(
                                "results", []
                            )
                        ),
                        "rag_result_count": len(
                            intermediate_outputs.get(
                                "template_memory_parse_logs", {}
                            ).get("results", [])
                        ),
                    },
                    compare_batch_parse_outputs_tool,
                    intermediate_outputs.get("drain_parse_logs"),
                    intermediate_outputs.get("template_memory_parse_logs"),
                )
            elif tool_name == "build_batch_parse_summary":
                output, trace = _run_traced_tool(
                    tool_name,
                    {"task": agent_task, "log_count": len(selected_logs)},
                    lambda *args, **kwargs: {
                        "tool_name": "build_batch_parse_summary",
                        "summary": build_batch_parse_summary(*args, **kwargs),
                        "error": None,
                    },
                    selected_logs,
                    drain_outputs=intermediate_outputs.get("drain_parse_logs"),
                    rag_outputs=intermediate_outputs.get("template_memory_parse_logs"),
                    comparison_outputs=intermediate_outputs.get(
                        "compare_batch_parse_outputs"
                    ),
                    task=agent_task,
                    tools_used=plan["tool_sequence"],
                )
            else:
                output = {"tool_name": tool_name, "error": f"Unknown tool: {tool_name}"}
                now = now_iso()
                trace = ToolCallTrace(
                    tool_name=tool_name,
                    input_summary={},
                    output_summary=output,
                    error=output["error"],
                    started_at=now,
                    ended_at=now,
                    duration_ms=0.0,
                )

            intermediate_outputs[tool_name] = output
            tool_traces.append(trace)

        summary_output = intermediate_outputs.get("build_batch_parse_summary", {})
        final_answer = summary_output.get("summary") or "No final summary was generated."
        result = LogReviewAgentResult(
            user_query=user_query,
            log=f"{len(selected_logs)} selected logs",
            plan=agent_plan,
            tool_outputs=intermediate_outputs,
            final_answer=final_answer,
        )
        ended_at = now_iso()
        trace = build_agent_run_trace(
            user_query=user_query,
            log=f"Agent task: {agent_task}; selected logs count: {len(selected_logs)}",
            selected_action=agent_plan.action,
            plan_reason=agent_plan.reason,
            tool_calls=tool_traces,
            final_answer=final_answer,
            success=True,
            status=_status_from_tool_traces(tool_traces),
            tool_error_count=_tool_error_count(tool_traces),
            started_at=started_at,
            ended_at=ended_at,
        )
        saved_paths = _save_trace_outputs(trace, save_outputs)
        return result, trace, saved_paths, plan, intermediate_outputs

    except Exception as exc:
        ended_at = now_iso()
        final_answer = f"Task-driven offline agent failed: {exc}"
        result = LogReviewAgentResult(
            user_query=user_query,
            log=f"{len(selected_logs)} selected logs",
            plan=agent_plan,
            tool_outputs=intermediate_outputs,
            final_answer=final_answer,
        )
        trace = build_agent_run_trace(
            user_query=user_query,
            log=f"Agent task: {agent_task}; selected logs count: {len(selected_logs)}",
            selected_action=agent_plan.action,
            plan_reason=agent_plan.reason,
            tool_calls=tool_traces,
            final_answer=final_answer,
            success=False,
            status="failed",
            tool_error_count=_tool_error_count(tool_traces),
            started_at=started_at,
            ended_at=ended_at,
        )
        saved_paths = _save_trace_outputs(trace, save_outputs)
        return result, trace, saved_paths, plan, intermediate_outputs
