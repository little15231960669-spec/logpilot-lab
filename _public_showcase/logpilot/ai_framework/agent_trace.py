"""Agent run tracing and report export helpers."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4


VALID_RUN_STATUSES = {"success", "partial_success", "failed"}


@dataclass
class ToolCallTrace:
    tool_name: str
    input_summary: dict
    output_summary: dict | None
    error: str | None
    started_at: str
    ended_at: str
    duration_ms: float


@dataclass
class AgentRunTrace:
    run_id: str
    user_query: str
    log: str
    selected_action: str
    plan_reason: str
    tool_calls: list[ToolCallTrace]
    final_answer: str
    success: bool
    status: str
    tool_error_count: int
    started_at: str
    ended_at: str
    duration_ms: float


def now_iso() -> str:
    """Return the current UTC timestamp in ISO-8601 format."""
    return datetime.now(timezone.utc).isoformat()


def generate_run_id() -> str:
    """Generate a unique run id for trace files."""
    return f"agent_run_{uuid4().hex[:12]}"


def summarize_for_trace(value, max_length: int = 500):
    """Recursively summarize trace values and truncate long strings."""
    if isinstance(value, str):
        if len(value) <= max_length:
            return value
        return value[:max_length] + "...[truncated]"

    if isinstance(value, dict):
        return {
            str(key): summarize_for_trace(item, max_length=max_length)
            for key, item in value.items()
        }

    if isinstance(value, list):
        return [summarize_for_trace(item, max_length=max_length) for item in value]

    if isinstance(value, tuple):
        return [summarize_for_trace(item, max_length=max_length) for item in value]

    return value


def _duration_ms(started_at: str, ended_at: str) -> float:
    start = datetime.fromisoformat(started_at)
    end = datetime.fromisoformat(ended_at)
    return round((end - start).total_seconds() * 1000, 3)


def build_agent_run_trace(
    user_query: str,
    log: str,
    selected_action: str,
    plan_reason: str,
    tool_calls: list[ToolCallTrace],
    final_answer: str,
    success: bool,
    started_at: str,
    ended_at: str,
    status: str | None = None,
    tool_error_count: int | None = None,
) -> AgentRunTrace:
    """Build a complete agent run trace."""
    resolved_tool_error_count = (
        sum(1 for tool_call in tool_calls if tool_call.error)
        if tool_error_count is None
        else tool_error_count
    )
    if status is None:
        if not success:
            status = "failed"
        elif resolved_tool_error_count > 0:
            status = "partial_success"
        else:
            status = "success"
    if status not in VALID_RUN_STATUSES:
        raise ValueError(f"Invalid agent run status: {status}")

    return AgentRunTrace(
        run_id=generate_run_id(),
        user_query=summarize_for_trace(user_query),
        log=summarize_for_trace(log),
        selected_action=selected_action,
        plan_reason=summarize_for_trace(plan_reason),
        tool_calls=tool_calls,
        final_answer=summarize_for_trace(final_answer),
        success=success,
        status=status,
        tool_error_count=resolved_tool_error_count,
        started_at=started_at,
        ended_at=ended_at,
        duration_ms=_duration_ms(started_at, ended_at),
    )


def agent_trace_to_dict(trace: AgentRunTrace) -> dict:
    """Convert an agent trace dataclass into a JSON-ready dictionary."""
    return summarize_for_trace(asdict(trace))


def save_agent_trace_json(
    trace: AgentRunTrace,
    output_dir: str = "results/agent_runs",
) -> str:
    """Save an agent trace as JSON and return the file path."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    file_path = output_path / f"{trace.run_id}.json"
    file_path.write_text(
        json.dumps(agent_trace_to_dict(trace), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return str(file_path)


def render_agent_report_markdown(trace: AgentRunTrace) -> str:
    """Render a human-readable Markdown report for an agent trace."""
    lines = [
        "# Log Review Agent Run Report",
        "",
        f"- run_id: `{trace.run_id}`",
        f"- selected action: `{trace.selected_action}`",
        f"- plan reason: {trace.plan_reason}",
        f"- status: `{trace.status}`",
        f"- success: {trace.success}",
        f"- tool_error_count: {trace.tool_error_count}",
        f"- duration_ms: {trace.duration_ms}",
        "",
        "## Input Log",
        "",
        "```text",
        trace.log,
        "```",
        "",
        "## Tool Call Timeline",
        "",
    ]

    if not trace.tool_calls:
        lines.append("No tool calls recorded.")
    else:
        for index, call in enumerate(trace.tool_calls, start=1):
            lines.extend(
                [
                    f"### {index}. {call.tool_name}",
                    "",
                    f"- started_at: {call.started_at}",
                    f"- ended_at: {call.ended_at}",
                    f"- duration_ms: {call.duration_ms}",
                    f"- error: {call.error or 'None'}",
                    "",
                    "Input summary:",
                    "",
                    "```json",
                    json.dumps(call.input_summary, ensure_ascii=False, indent=2),
                    "```",
                    "",
                    "Output summary:",
                    "",
                    "```json",
                    json.dumps(call.output_summary, ensure_ascii=False, indent=2),
                    "```",
                    "",
                ]
            )

    lines.extend(
        [
            "## Final Answer",
            "",
            "```text",
            trace.final_answer,
            "```",
        ]
    )

    return "\n".join(lines)


def save_agent_report_markdown(
    trace: AgentRunTrace,
    output_dir: str = "results/agent_runs",
) -> str:
    """Save a Markdown agent report and return the file path."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    file_path = output_path / f"{trace.run_id}.md"
    file_path.write_text(render_agent_report_markdown(trace), encoding="utf-8")
    return str(file_path)
