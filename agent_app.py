from __future__ import annotations

from datetime import datetime
from html import escape
from pathlib import Path

import streamlit as st

from logpilot.ai_framework.log_review_agent import (
    plan_tools_for_agent_task,
    run_agent_task_on_logs_offline_with_trace,
)
from logpilot.ai_framework.ui_helpers import (
    build_agent_log_preview,
    build_memory_dataset_warning,
    infer_dataset_type_from_name_or_logs,
    infer_template_memory_type,
    normalize_selected_logs_for_agent,
    task_requires_template_memory,
    validate_template_memory_csv_text,
)


DEMO_LOGS = [
    "Receiving block blk_38865049064139660 src: /10.250.19.102:54106 dest: /10.250.19.102:50010",
    "Served block blk_777 to /10.250.19.103:50010",
    "Deleting block blk_888 file /data/current/subdir0/blk_888",
    "PacketResponder 1 for block blk_999 terminating",
    "Verification succeeded for blk_111",
]

TASK_LABEL_TO_KEY = {
    "快速解析当前日志": "Fast baseline parse",
    "复核解析结果": "High-confidence parse review",
    "查找风险日志": "Find risky templates",
}

TASK_DEFAULT_QUERIES = {
    "快速解析当前日志": "请快速解析当前选中的日志，生成基础模板结果。",
    "复核解析结果": "请复核当前日志，选择合适的解析工具，对比解析结果并给出建议。",
    "查找风险日志": "请找出当前日志中不同解析工具输出不一致的样本，并给出复核建议。",
}

TASK_PLAN_REASONS = {
    "快速解析当前日志": "使用 Drain 快速生成基线模板，适合快速查看当前日志的基础解析结果。",
    "复核解析结果": "该任务需要更可靠的解析结果，因此 Agent 将同时调用 Drain 和模板记忆解析，并比较两者差异。",
    "查找风险日志": "Agent 会找出不同解析工具输出不一致的日志，帮助定位需要人工复核的风险样本。",
}

ACTION_LABELS = {
    "high_confidence_parse_review": "复核解析结果",
    "fast_baseline_parse": "快速解析当前日志",
    "find_risky_templates": "查找风险日志",
    "compare_parser_tools": "复核解析结果",
    "explain_parsing_results": "复核解析结果",
}

TOOL_LABELS = {
    "drain_parse_logs": "Drain 基线解析",
    "template_memory_parse_logs": "模板记忆解析",
    "compare_batch_parse_outputs": "解析结果对比",
    "build_batch_parse_summary": "生成解释与建议",
}

TOOL_DESCRIPTIONS = {
    "drain_parse_logs": "生成基线模板",
    "template_memory_parse_logs": "检索模板记忆并解析",
    "compare_batch_parse_outputs": "比较模板是否一致",
    "build_batch_parse_summary": "汇总结果并生成建议",
}

DATA_SOURCE_LABELS = {
    "HDFS 示例": "hdfs_sample",
    "上传文件": "upload",
    "手动输入": "manual",
}

BGL_TEMPLATE_PATH = "data/template_memory/bgl_templates.csv"
HDFS_TEMPLATE_PATH = "data/template_memory/hdfs_templates.csv"


def inject_css() -> None:
    st.markdown(
        """
        <style>
        .main-header {
            font-size: 30px;
            font-weight: 700;
            color: #1F2937;
            margin-bottom: 6px;
            line-height: 1.25;
        }
        .subtitle {
            font-size: 14px;
            color: #6B7280;
            margin-bottom: 24px;
            line-height: 1.6;
        }
        .section-title {
            font-size: 22px;
            font-weight: 700;
            color: #1F2937;
            margin-top: 24px;
            margin-bottom: 14px;
            line-height: 1.35;
        }
        .subsection-title {
            font-size: 18px;
            font-weight: 650;
            color: #1F2937;
            margin-top: 18px;
            margin-bottom: 10px;
        }
        .metric-card {
            background: #FFFFFF;
            border: 1px solid #E5E7EB;
            border-radius: 10px;
            padding: 15px 17px;
            min-height: 88px;
        }
        .metric-label {
            font-size: 13px;
            color: #6B7280;
            margin-bottom: 8px;
            line-height: 1.3;
        }
        .metric-value {
            font-size: 23px;
            font-weight: 700;
            color: #1F2937;
            line-height: 1.25;
            word-break: break-word;
        }
        .metric-hint {
            font-size: 12px;
            color: #9CA3AF;
            margin-top: 6px;
            line-height: 1.4;
        }
        .recommendation-card {
            background: #FFFFFF;
            border: 1px solid #E5E7EB;
            border-left: 4px solid #2563EB;
            border-radius: 10px;
            padding: 18px 20px;
            font-size: 15px;
            color: #1F2937;
            line-height: 1.8;
        }
        .plan-card {
            background: #FFFFFF;
            border: 1px solid #E5E7EB;
            border-radius: 10px;
            padding: 18px 20px;
            font-size: 15px;
            color: #1F2937;
            line-height: 1.7;
        }
        .status-pill {
            display: inline-block;
            padding: 4px 10px;
            border-radius: 999px;
            font-size: 13px;
            font-weight: 600;
        }
        .status-success {
            color: #166534;
            background: #DCFCE7;
        }
        .status-partial {
            color: #92400E;
            background: #FEF3C7;
        }
        .status-failed {
            color: #991B1B;
            background: #FEE2E2;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_metric_card(label: str, value: object, hint: str = "", allow_html: bool = False) -> None:
    safe_label = escape(str(label))
    safe_value = str(value) if allow_html else escape(str(value))
    safe_hint = escape(str(hint))
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{safe_label}</div>
            <div class="metric-value">{safe_value}</div>
            <div class="metric-hint">{safe_hint}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_section_title(title: str) -> None:
    st.markdown(f'<div class="section-title">{escape(title)}</div>', unsafe_allow_html=True)


def render_subsection_title(title: str) -> None:
    st.markdown(f'<div class="subsection-title">{escape(title)}</div>', unsafe_allow_html=True)


def status_pill(status: str) -> str:
    if status == "success":
        return '<span class="status-pill status-success">成功</span>'
    if status == "partial_success":
        return '<span class="status-pill status-partial">部分成功</span>'
    return '<span class="status-pill status-failed">失败</span>'


def load_hdfs_sample_logs() -> list[str]:
    path = Path("examples/hdfs_sample.log")
    if path.exists():
        return [
            line.strip()
            for line in path.read_text(encoding="utf-8", errors="ignore").splitlines()
            if line.strip()
        ]
    return DEMO_LOGS


def read_uploaded_logs(uploaded_file) -> list[str]:
    if uploaded_file is None:
        return []
    text = uploaded_file.getvalue().decode("utf-8", errors="ignore")
    return [line.strip() for line in text.splitlines() if line.strip()]


def save_uploaded_memory_csv(csv_text: str) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = Path("results/template_memory") / f"uploaded_memory_{timestamp}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(csv_text, encoding="utf-8")
    return str(path)


def tool_label(tool_name: str) -> str:
    return TOOL_LABELS.get(tool_name, tool_name or "-")


def action_label(action: str) -> str:
    return ACTION_LABELS.get(action, action or "-")


def format_rate(value) -> str:
    if value is None:
        return "未执行对比"
    if isinstance(value, (int, float)):
        return f"{value * 100:.1f}%"
    return str(value)


def short_log(log: str, limit: int = 120) -> str:
    text = str(log or "").strip()
    return text if len(text) <= limit else text[:limit] + "..."


def display_value(value) -> str:
    if value is None or value == "":
        return "-"
    if isinstance(value, bool):
        return "一致" if value else "不一致"
    return str(value)


def build_tool_trace_rows(trace) -> list[dict]:
    rows = []
    for index, call in enumerate(getattr(trace, "tool_calls", []) or [], start=1):
        rows.append(
            {
                "步骤": index,
                "工具": tool_label(call.tool_name),
                "作用": TOOL_DESCRIPTIONS.get(call.tool_name, "执行工具"),
                "状态": "失败" if call.error else "成功",
                "耗时(ms)": getattr(call, "duration_ms", "-"),
                "错误": call.error or "-",
            }
        )
    return rows


def build_parsed_rows(intermediate_outputs: dict) -> list[dict]:
    drain_rows = {
        item.get("index"): item
        for item in intermediate_outputs.get("drain_parse_logs", {}).get("results", [])
    }
    memory_rows = {
        item.get("index"): item
        for item in intermediate_outputs.get("template_memory_parse_logs", {}).get("results", [])
    }
    comparison_rows = {
        item.get("index"): item
        for item in intermediate_outputs.get("compare_batch_parse_outputs", {}).get("items", [])
    }

    rows = []
    for index in sorted(set(drain_rows) | set(memory_rows) | set(comparison_rows)):
        drain_item = drain_rows.get(index, {})
        memory_item = memory_rows.get(index, {})
        comparison_item = comparison_rows.get(index, {})
        exact_match = comparison_item.get("exact_match")
        log = (
            drain_item.get("content")
            or memory_item.get("content")
            or drain_item.get("log")
            or memory_item.get("log")
            or "-"
        )
        if comparison_rows:
            match_text = display_value(exact_match)
            note = "两个工具给出相同模板" if exact_match else "两个工具输出不同，建议复核"
        else:
            match_text = "未对比"
            note = "当前任务未执行对比"

        row = {
            "序号": index,
            "日志正文": short_log(log),
        }
        if drain_rows:
            row["Drain 模板"] = display_value(
                comparison_item.get("drain_template") or drain_item.get("template")
            )
        if memory_rows:
            row["模板记忆模板"] = display_value(
                comparison_item.get("rag_template") or memory_item.get("template")
            )
        row["是否一致"] = match_text
        row["说明"] = note
        rows.append(row)
    return rows


def build_risk_rows(intermediate_outputs: dict) -> list[dict]:
    comparison_items = intermediate_outputs.get("compare_batch_parse_outputs", {}).get("items", [])
    drain_rows = {
        item.get("index"): item
        for item in intermediate_outputs.get("drain_parse_logs", {}).get("results", [])
    }
    memory_rows = {
        item.get("index"): item
        for item in intermediate_outputs.get("template_memory_parse_logs", {}).get("results", [])
    }

    rows = []
    for item in comparison_items:
        if item.get("exact_match") is not False:
            continue
        index = item.get("index")
        drain_item = drain_rows.get(index, {})
        memory_item = memory_rows.get(index, {})
        log = (
            drain_item.get("content")
            or memory_item.get("content")
            or item.get("log")
            or "-"
        )
        rows.append(
            {
                "序号": index,
                "日志正文": short_log(log),
                "Drain 模板": display_value(item.get("drain_template") or drain_item.get("template")),
                "模板记忆模板": display_value(item.get("rag_template") or memory_item.get("template")),
                "风险原因": "不同解析工具输出不一致，建议人工复核。",
            }
        )
    return rows


def recommendation_content(intermediate_outputs: dict, dataset_warning: str | None) -> tuple[list[tuple[str, object]], str]:
    drain_results = intermediate_outputs.get("drain_parse_logs", {}).get("results", [])
    memory_results = intermediate_outputs.get("template_memory_parse_logs", {}).get("results", [])
    comparison = intermediate_outputs.get("compare_batch_parse_outputs", {}) or {}
    mismatch_count = comparison.get("mismatch_count", "未执行对比")
    exact_count = comparison.get("exact_match_count", "未执行对比")
    match_rate = comparison.get("match_rate", None) if comparison else None
    used_tools = [
        tool_label(name)
        for name in [
            "drain_parse_logs",
            "template_memory_parse_logs",
            "compare_batch_parse_outputs",
            "build_batch_parse_summary",
        ]
        if name in intermediate_outputs
    ]

    items = [
        ("本次分析日志数", max(len(drain_results), len(memory_results), len(comparison.get("items", [])))),
        ("调用工具", "、".join(used_tools) or "-"),
        ("Drain 解析模板数", len(drain_results)),
        ("模板记忆解析模板数", len(memory_results)),
        ("一致数量", exact_count),
        ("不一致数量", mismatch_count),
        ("匹配率", format_rate(match_rate)),
    ]

    if match_rate == 0 and isinstance(mismatch_count, int) and mismatch_count > 0:
        advice = (
            "当前两个解析工具输出差异较大，不建议直接接受模板记忆解析结果；"
            "建议优先查看不一致日志，并确认模板记忆库是否与当前日志数据集匹配。"
        )
    elif dataset_warning:
        advice = "当前日志与模板记忆库可能不匹配，建议切换到匹配的数据集模板库后重新运行。"
    elif comparison and isinstance(mismatch_count, int) and mismatch_count > 0:
        advice = "存在不一致模板，建议优先查看风险样本并进行人工复核。"
    elif comparison:
        advice = "Drain 和模板记忆解析结果整体一致，可以接受当前模板结果，并保留少量抽样复核。"
    elif memory_results:
        advice = "本次结果主要来自模板记忆解析；如需更高置信度，建议运行复核解析结果任务。"
    else:
        advice = "本次为快速解析，适合快速查看模板分布；正式复核建议运行带模板记忆对比的任务。"
    return items, advice


def display_download(path_value: str | None, button_label: str, mime: str) -> None:
    if not path_value:
        st.caption("本次运行未生成文件。")
        return
    path = Path(path_value)
    if path.exists():
        data = path.read_bytes() if mime == "application/json" else path.read_text(encoding="utf-8")
        st.download_button(button_label, data=data, file_name=path.name, mime=mime)
    else:
        st.warning("文件不存在，无法下载。")


st.set_page_config(page_title="LogPilot Agent Lab", layout="wide")
inject_css()

st.markdown('<div class="main-header">LogPilot Agent Lab</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">面向日志解析的 Agent 工具编排实验台：自动调用解析工具、检索历史模板、对比结果并生成复核建议</div>',
    unsafe_allow_html=True,
)

render_section_title("Select an AI Skill / task goal")
task_label = st.selectbox("Select an AI Skill / task goal", list(TASK_LABEL_TO_KEY.keys()), index=1)
task_key = TASK_LABEL_TO_KEY[task_label]
requires_memory = task_requires_template_memory(task_label)
user_query = st.text_area(
    "任务说明",
    value=TASK_DEFAULT_QUERIES[task_label],
    height=88,
    key=f"user_query_{task_label}",
)

with st.sidebar:
    render_subsection_title("数据配置")
    data_source_label = st.radio("数据源", list(DATA_SOURCE_LABELS.keys()), index=0)
    max_agent_logs = st.slider("本次分析日志数", 1, 500, 5)

    data_source = DATA_SOURCE_LABELS[data_source_label]
    source_name = data_source_label
    if data_source == "hdfs_sample":
        selected_logs = load_hdfs_sample_logs()
        source_name = "hdfs_sample.log"
    elif data_source == "upload":
        uploaded = st.file_uploader("上传 .log / .txt", type=["log", "txt"])
        source_name = uploaded.name if uploaded is not None else "uploaded_file"
        selected_logs = read_uploaded_logs(uploaded)
    else:
        manual_text = st.text_area("手动输入日志", value="\n".join(DEMO_LOGS[:3]), height=180)
        selected_logs = [line.strip() for line in manual_text.splitlines() if line.strip()]
        source_name = "manual_input"

dataset_type = infer_dataset_type_from_name_or_logs(source_name, selected_logs)
logs_to_agent = normalize_selected_logs_for_agent(selected_logs, max_logs=max_agent_logs)

render_section_title("数据上下文")
ctx_col_1, ctx_col_2, ctx_col_3 = st.columns(3)
with ctx_col_1:
    render_metric_card("数据源", data_source_label)
with ctx_col_2:
    render_metric_card("已加载日志数", len(selected_logs))
with ctx_col_3:
    render_metric_card("本次分析日志数", len(logs_to_agent))

with st.expander("查看当前分析范围内的日志预览", expanded=False):
    if logs_to_agent:
        st.code(build_agent_log_preview(logs_to_agent, max_preview=len(logs_to_agent)), language="text")
    else:
        st.warning("当前没有可分析的日志。")

template_csv_path: str | None = None
memory_type = "unknown"
dataset_memory_warning = None

if requires_memory:
    with st.sidebar:
        render_subsection_title("模板记忆库")
        memory_source = st.radio(
            "模板记忆库来源",
            ["自动匹配", "HDFS 模板库", "BGL 模板库", "上传模板记忆 CSV"],
        )
        if memory_source == "自动匹配":
            template_csv_path = (
                BGL_TEMPLATE_PATH
                if dataset_type == "BGL" and Path(BGL_TEMPLATE_PATH).exists()
                else HDFS_TEMPLATE_PATH
            )
            st.caption(f"自动选择：{Path(template_csv_path).name}")
        elif memory_source == "HDFS 模板库":
            template_csv_path = HDFS_TEMPLATE_PATH
        elif memory_source == "BGL 模板库":
            template_csv_path = BGL_TEMPLATE_PATH
        else:
            uploaded_memory = st.file_uploader("上传模板记忆 CSV", type=["csv"])
            if uploaded_memory is not None:
                text = uploaded_memory.getvalue().decode("utf-8", errors="ignore")
                is_valid, message, normalized_csv = validate_template_memory_csv_text(text)
                if is_valid:
                    template_csv_path = save_uploaded_memory_csv(normalized_csv)
                    st.success("模板记忆库已上传。")
                else:
                    st.warning(message)

    render_section_title("模板记忆库")
    memory_type = infer_template_memory_type(template_csv_path or "")
    mem_col_1, mem_col_2 = st.columns(2)
    with mem_col_1:
        render_metric_card("记忆库来源", memory_source)
    with mem_col_2:
        render_metric_card("记忆库类型", memory_type)
    dataset_memory_warning = build_memory_dataset_warning(dataset_type, memory_type) if template_csv_path else None
    if dataset_memory_warning:
        st.warning(dataset_memory_warning)

planned = plan_tools_for_agent_task(task_key, logs_to_agent, offline=True)

render_section_title("预计执行策略")
tools_html = "".join(f"<li>{escape(tool_label(tool))}</li>" for tool in planned.get("tool_sequence", []))
st.markdown(
    f"""
    <div class="plan-card">
        <div><strong>任务目标：</strong>{escape(task_label)}</div>
        <div><strong>Agent 选择原因：</strong>{escape(TASK_PLAN_REASONS[task_label])}</div>
        <div><strong>预计工具链：</strong></div>
        <ol>{tools_html}</ol>
    </div>
    """,
    unsafe_allow_html=True,
)

run_clicked = st.button("运行 Agent", type="primary")
if run_clicked:
    if not logs_to_agent:
        st.warning("当前没有可分析的日志。")
    elif requires_memory and not template_csv_path:
        st.warning("请先选择或上传模板记忆库。")
    else:
        st.session_state["agent_run_result"] = run_agent_task_on_logs_offline_with_trace(
            agent_task=task_key,
            user_query=user_query,
            logs=logs_to_agent,
            template_csv_path=template_csv_path,
            max_logs=len(logs_to_agent),
            save_outputs=True,
            dataset_name=dataset_type,
        )
        st.session_state["agent_run_context"] = {
            "task_label": task_label,
            "dataset_memory_warning": dataset_memory_warning,
        }

run_payload = st.session_state.get("agent_run_result")
if run_payload:
    result, trace, saved_paths, plan, intermediate_outputs = run_payload
    run_context = st.session_state.get("agent_run_context", {})
    comparison = intermediate_outputs.get("compare_batch_parse_outputs", {}) or {}
    mismatch_count = comparison.get("mismatch_count", "未执行对比")

    render_section_title("本次运行总览")
    overview_cols = st.columns(3)
    with overview_cols[0]:
        render_metric_card("运行状态", status_pill(trace.status), allow_html=True)
    with overview_cols[1]:
        render_metric_card("任务目标", run_context.get("task_label", task_label))
    with overview_cols[2]:
        render_metric_card("分析日志数", plan.get("log_count", len(logs_to_agent)))
    overview_cols_2 = st.columns(3)
    with overview_cols_2[0]:
        render_metric_card("匹配率", format_rate(comparison.get("match_rate") if comparison else None))
    with overview_cols_2[1]:
        hint = "需要复核" if isinstance(mismatch_count, int) and mismatch_count > 0 else ""
        render_metric_card("不一致数量", mismatch_count, hint=hint)
    with overview_cols_2[2]:
        render_metric_card("工具错误数", trace.tool_error_count)

    with st.expander("查看工具执行过程", expanded=False):
        st.caption("以下为 Agent 本次实际调用的工具顺序与执行状态。")
        st.dataframe(build_tool_trace_rows(trace), use_container_width=True, hide_index=True)
        for index, call in enumerate(trace.tool_calls, start=1):
            with st.expander(f"第 {index} 步：{tool_label(call.tool_name)}", expanded=False):
                st.json(call.output_summary)

    render_section_title("解析结果")
    st.caption("下表展示 Agent 对日志正文 Content 的解析结果；时间戳、日志级别、logger 等头部字段不会参与模板比较。")
    parsed_rows = build_parsed_rows(intermediate_outputs)
    if parsed_rows:
        st.dataframe(parsed_rows, use_container_width=True, hide_index=True)
    else:
        st.info("本次运行没有生成可展示的解析结果。")

    render_section_title("风险样本")
    risk_rows = build_risk_rows(intermediate_outputs)
    if risk_rows:
        st.dataframe(risk_rows, use_container_width=True, hide_index=True)
    elif comparison:
        st.info("未发现明显风险样本。")
    else:
        st.info("当前任务未执行多工具对比。")

    render_section_title("Agent 解释与建议")
    summary_items, advice = recommendation_content(
        intermediate_outputs,
        run_context.get("dataset_memory_warning"),
    )
    bullets = "".join(
        f"<li><strong>{escape(label)}：</strong>{escape(str(value))}</li>"
        for label, value in summary_items
    )
    st.markdown(
        f"""
        <div class="recommendation-card">
            <ul>{bullets}</ul>
            <div>{escape(advice)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    render_section_title("报告下载")
    col_a, col_b = st.columns(2)
    with col_a:
        display_download(saved_paths.get("json"), "下载 JSON Trace", "application/json")
    with col_b:
        display_download(saved_paths.get("markdown"), "下载 Markdown 报告", "text/markdown")
    with st.expander("查看报告保存路径", expanded=False):
        st.caption(f"JSON Trace：{saved_paths.get('json') or '-'}")
        st.caption(f"Markdown 报告：{saved_paths.get('markdown') or '-'}")
