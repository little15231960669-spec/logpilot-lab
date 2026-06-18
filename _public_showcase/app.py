from html import escape
import os
from textwrap import dedent

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from logpilot.data.loader import (
    SAMPLE_DATASETS,
    load_log_file,
    load_log_lines,
    records_to_dataframe,
)
from logpilot.parsers.drain_parser import DrainLogParser
from logpilot.parsers.llm_parser import LLMLogParser
from logpilot.parsers.hybrid_parser import HybridLogParser
from logpilot.llm.factory import BACKEND_OPTIONS, create_llm_backend
from logpilot.evaluation.metrics import summarize_parsing_result
from logpilot.evaluation.comparison import (
    summary_to_row,
    compare_line_level_templates,
    compute_template_agreement,
)
from logpilot.evolution.diff import analyze_template_evolution


load_dotenv()


st.set_page_config(
    page_title="LogPilot",
    page_icon="🧭",
    layout="wide",
)


# ---------- Helpers ----------
def inject_style():
    """
    Inject a clean product-style visual system.
    Keep the UI simple and avoid over-styling Streamlit internals.
    """
    st.markdown(
        dedent(
            """
            <style>
            :root {
                --lp-blue: #1E88E5;
                --lp-green: #4CAF50;
                --lp-bg: #F8F9FA;
                --lp-border: #E5E7EB;
                --lp-text: #263238;
                --lp-muted: #6B7280;
            }

            html, body, .stApp {
                font-family: "PingFang SC", "Inter", "Microsoft YaHei", sans-serif;
                color: var(--lp-text);
            }

            .stApp {
                background: #FFFFFF;
            }

            .block-container {
                padding-top: 1.4rem;
                padding-bottom: 3rem;
                max-width: 1500px;
            }

            h1 {
                font-size: 2.1rem !important;
                margin-bottom: 0.15rem !important;
                letter-spacing: -0.02em;
            }

            h2, h3 {
                color: var(--lp-text);
                letter-spacing: -0.01em;
            }

            /* Buttons */
            div[data-testid="stButton"] > button,
            div[data-testid="stDownloadButton"] > button {
                border-radius: 8px !important;
                border: 1px solid var(--lp-border) !important;
                box-shadow: 0 2px 8px rgba(38, 50, 56, 0.08) !important;
                transition: all 120ms ease-in-out !important;
            }

            div[data-testid="stButton"] > button:hover,
            div[data-testid="stDownloadButton"] > button:hover {
                transform: translateY(-1px);
                border-color: var(--lp-blue) !important;
                box-shadow: 0 6px 14px rgba(30, 136, 229, 0.16) !important;
            }

            /* Compact ribbon */
            .ribbon-card {
                background: #FFFFFF;
                border: 1px solid var(--lp-border);
                border-radius: 10px;
                padding: 10px 12px;
                min-height: 62px;
                box-shadow: 0 1px 4px rgba(38, 50, 56, 0.06);
            }

            .ribbon-label {
                font-size: 0.78rem;
                color: var(--lp-muted);
                margin-bottom: 5px;
                white-space: nowrap;
            }

            .ribbon-value {
                font-size: 1.0rem;
                font-weight: 650;
                color: var(--lp-text);
                white-space: nowrap;
                overflow: hidden;
                text-overflow: ellipsis;
            }

            .subtle-text {
                color: var(--lp-muted);
                font-size: 0.88rem;
                margin-top: 0.35rem;
                margin-bottom: 0.8rem;
            }

            /* Evolution badges */
            .evolution-badge {
                border-radius: 10px;
                padding: 10px 12px;
                border: 1px solid var(--lp-border);
                background: var(--lp-bg);
                min-height: 62px;
                box-shadow: 0 1px 4px rgba(38, 50, 56, 0.05);
            }

            .badge-label {
                font-size: 0.78rem;
                color: var(--lp-muted);
                margin-bottom: 5px;
            }

            .badge-count {
                font-size: 1.1rem;
                font-weight: 700;
            }

            .badge-stable {
                background: #E8F5E9;
                border-color: #C8E6C9;
            }

            .badge-stable .badge-count {
                color: #2E7D32;
            }

            .badge-mutated {
                background: #E3F2FD;
                border-color: #BBDEFB;
            }

            .badge-mutated .badge-count {
                color: #1565C0;
            }

            .badge-emerging {
                background: #F1F8E9;
                border-color: #DCEDC8;
            }

            .badge-emerging .badge-count {
                color: #558B2F;
            }

            .badge-vanishing {
                background: #ECEFF1;
                border-color: #CFD8DC;
            }

            .badge-vanishing .badge-count {
                color: #455A64;
            }
                        /* Compact result cards */
            .mini-metric-grid {
                display: grid;
                grid-template-columns: repeat(2, minmax(0, 1fr));
                gap: 12px;
                margin: 0.8rem 0 1.2rem 0;
            }

            .mini-metric-card {
                background: #FFFFFF;
                border: 1px solid var(--lp-border);
                border-radius: 10px;
                padding: 12px 14px;
                box-shadow: 0 1px 4px rgba(38, 50, 56, 0.06);
            }

            .mini-metric-label {
                font-size: 0.82rem;
                color: var(--lp-muted);
                margin-bottom: 6px;
            }

            .mini-metric-value {
                font-size: 1.12rem;
                font-weight: 650;
                color: var(--lp-text);
                line-height: 1.1;
            }
            </style>
            """
        ),
        unsafe_allow_html=True,
    )

def run_drain_parser(
    records,
    text_field: str,
    similarity_threshold: float = 0.4,
    depth: int = 4,
    max_clusters: int = 1000,
):
    parser = DrainLogParser(
        similarity_threshold=similarity_threshold,
        depth=depth,
        max_clusters=max_clusters,
    )
    return parser.parse(records, text_field=text_field)


def run_llm_parser(
    records,
    text_field: str,
    backend_name: str = "MockBackend",
    batch_size: int = 8,
):
    backend = create_llm_backend(backend_name)
    parser = LLMLogParser(
        backend=backend,
        batch_size=batch_size,
    )
    return parser.parse(records, text_field=text_field)


def run_hybrid_parser(
    records,
    backend_name: str = "MockBackend",
    drain_similarity_threshold: float = 0.4,
    drain_depth: int = 4,
    drain_max_clusters: int = 1000,
    llm_batch_size: int = 8,
    review_top_k: int = 10,
    risk_threshold: float = 0.5,
    small_cluster_threshold: int = 1,
    high_wildcard_threshold: float = 0.5,
):
    backend = create_llm_backend(backend_name)
    parser = HybridLogParser(
        llm_backend=backend,
        drain_similarity_threshold=drain_similarity_threshold,
        drain_depth=drain_depth,
        drain_max_clusters=drain_max_clusters,
        llm_batch_size=llm_batch_size,
        review_top_k=review_top_k,
        risk_threshold=risk_threshold,
        small_cluster_threshold=small_cluster_threshold,
        high_wildcard_threshold=high_wildcard_threshold,
    )
    return parser.parse(
        records,
        drain_text_field="masked_content",
        llm_text_field="content",
    )


def is_openai_configured() -> bool:
    # Check whether the OpenAI-compatible backend has local credentials.
    return bool(os.getenv("OPENAI_API_KEY")) and bool(os.getenv("OPENAI_BASE_URL"))


def get_llm_model_options() -> list[str]:
    # Present product-facing model names instead of implementation backend classes.
    return [
        "Mock Demo",
        "qwen3.6-plus",
        "qwen3.5-plus",
        "qwen-plus",
        "qwen-turbo",
        "自定义模型",
    ]


def resolve_backend_and_model(selected_model: str, custom_model: str = "") -> tuple[str, str]:
    # Map UI model selection to the backend identifier expected by the parser factory.
    if selected_model == "Mock Demo":
        return "MockBackend", "Mock Demo"

    model_name = custom_model.strip() if selected_model == "自定义模型" else selected_model
    return "OpenAICompatibleBackend", model_name


def get_model_display_name(model_name: str) -> str:
    # Normalize the model name used in user-facing UI.
    return model_name or "未选择模型"


def render_basic_metrics(summary: dict, show_ai_metrics: bool = False):
    metric_cols = st.columns(6)
    metric_cols[0].metric("日志数", summary["total_logs"])
    metric_cols[1].metric("模板数", summary["template_count"])
    metric_cols[2].metric("Cluster 数", summary["cluster_count"])
    metric_cols[3].metric("平均延迟 / 条", f'{summary["avg_latency_ms"]} ms')
    metric_cols[4].metric("总延迟", f'{summary["total_latency_ms"]} ms')

    valid_json_rate = summary.get("valid_json_rate")
    if valid_json_rate is None:
        metric_cols[5].metric("Valid JSON Rate", "-")
    else:
        metric_cols[5].metric("Valid JSON Rate", f"{valid_json_rate * 100:.1f}%")

    if show_ai_metrics:
        ai_cols = st.columns(5)
        ai_cols[0].metric("Input Tokens Est.", summary.get("input_tokens_est", 0))
        ai_cols[1].metric("Output Tokens Est.", summary.get("output_tokens_est", 0))
        ai_cols[2].metric("Total Tokens Est.", summary.get("total_tokens_est", 0))

        avg_conf = summary.get("avg_confidence")
        ai_cols[3].metric("Avg Confidence", avg_conf if avg_conf is not None else "-")

        avg_risk = summary.get("avg_risk_score")
        ai_cols[4].metric("Avg Risk Score", avg_risk if avg_risk is not None else "-")


def render_llm_direct_metrics(summary: dict):
    """
    Render compact LLM Direct result metrics.
    """
    render_compact_metric_row(
        [
            ("日志数", summary.get("total_logs", 0)),
            ("模板数", summary.get("template_count", 0)),
        ],
        columns=2,
    )

def render_compact_metric_row(items: list[tuple[str, str]], columns: int = 4):
    """
    Render small product-style metric cards.

    This implementation avoids nested multi-line HTML wrappers, so it will not
    leak stray closing tags such as </div> in Streamlit Markdown.
    """
    if not items:
        return

    column_count = max(1, min(columns, len(items)))
    cols = st.columns(column_count)

    for idx, (label, value) in enumerate(items):
        with cols[idx % column_count]:
            card_html = (
                "<div class='mini-metric-card'>"
                f"<div class='mini-metric-label'>{escape(str(label))}</div>"
                f"<div class='mini-metric-value'>{escape(str(value))}</div>"
                "</div>"
            )
            st.markdown(card_html, unsafe_allow_html=True)

def render_template_summary(parsed_df: pd.DataFrame):
    template_summary = (
        parsed_df.groupby("template")
        .agg(count=("line_id", "count"))
        .reset_index()
        .sort_values("count", ascending=False)
    )
    st.dataframe(template_summary, use_container_width=True, height=300)


def safe_dataframe(df: pd.DataFrame, columns: list[str], height: int = 360):
    existing_columns = [col for col in columns if col in df.columns]
    st.dataframe(df[existing_columns], use_container_width=True, height=height)


def remember_recent_run(mode_name: str, summary: dict | None = None):
    # Keep lightweight run status for the top ribbon without changing parser behavior.
    st.session_state["recent_run"] = {
        "mode": mode_name,
        "summary": summary or {},
    }


def render_metric_ribbon(
    data_source_label: str,
    log_count: int,
    start_line: int,
    llm_model_display: str,
):
    """
    Render a compact top metric ribbon.

    All values are from current input or session_state.
    No fake performance numbers are used.
    """
    end_line = start_line + max(log_count, 1) - 1
    recent_run = st.session_state.get("recent_run", {})
    current_mode = recent_run.get("mode") or "待运行"

    items = [
        ("🪵 数据源", f"{data_source_label} · {log_count} lines"),
        ("📍 行范围", f"{start_line} - {end_line}" if log_count else "-"),
        ("🤖 LLM 模型", llm_model_display or "未选择"),
        ("⚙️ 最近任务", current_mode),
    ]

    cols = st.columns(4)

    for col, (label, value) in zip(cols, items):
        with col:
            st.markdown(
                dedent(
                    f"""
                    <div class="ribbon-card">
                        <div class="ribbon-label">{escape(str(label))}</div>
                        <div class="ribbon-value" title="{escape(str(value))}">
                            {escape(str(value))}
                        </div>
                    </div>
                    """
                ),
                unsafe_allow_html=True,
            )

def render_hybrid_summary_cards(parsed_df: pd.DataFrame):
    # Render compact Hybrid routing conclusions without chart noise.
    total_logs = len(parsed_df)
    reviewed_count = (
        int(parsed_df["used_llm_review"].sum())
        if "used_llm_review" in parsed_df.columns
        else 0
    )
    non_reviewed_count = max(0, total_logs - reviewed_count)
    fallback_count = (
        int(parsed_df["fallback_to_drain"].sum())
        if "fallback_to_drain" in parsed_df.columns
        else 0
    )
    llm_suggested_count = 0

    if "used_llm_review" in parsed_df.columns and "fallback_to_drain" in parsed_df.columns:
        llm_suggested_count = int(
            (
                (parsed_df["used_llm_review"] == True)
                & (parsed_df["fallback_to_drain"] == False)
            ).sum()
        )

    render_compact_metric_row(
        [
            ("已复核日志", f"{reviewed_count} / {total_logs}"),
            ("未复核日志", f"{non_reviewed_count} / {total_logs}"),
            ("LLM 建议", llm_suggested_count),
            ("回退 Drain", fallback_count),
        ],
        columns=4,
    )


def attach_llm_backend_error(line_compare_df: pd.DataFrame, llm_df: pd.DataFrame) -> pd.DataFrame:
    # Preserve backend_error in comparison output when LLM backend returns failures.
    if "backend_error" not in llm_df.columns or "backend_error" in line_compare_df.columns:
        return line_compare_df

    error_part = llm_df[["line_id", "backend_error"]].copy()
    return line_compare_df.merge(error_part, on="line_id", how="left")


def _evolution_counts(evolution_summary_df: pd.DataFrame | None) -> dict[str, int | str]:
    if evolution_summary_df is None:
        return {
            "stable": "Run analysis to update",
            "rewritten_candidate": "Run analysis to update",
            "new": "Run analysis to update",
            "disappeared": "Run analysis to update",
        }

    if evolution_summary_df.empty:
        return {
            "stable": 0,
            "rewritten_candidate": 0,
            "new": 0,
            "disappeared": 0,
        }

    count_map = dict(
        zip(
            evolution_summary_df["evolution_type"],
            evolution_summary_df["count"],
        )
    )
    return {
        "stable": int(count_map.get("stable", 0)),
        "rewritten_candidate": int(count_map.get("rewritten_candidate", 0)),
        "new": int(count_map.get("new", 0)),
        "disappeared": int(count_map.get("disappeared", 0)),
    }


def render_status_badges(evolution_summary_df: pd.DataFrame | None = None):
    """
    Render evolution lifecycle counts as compact badges.
    """
    counts = _evolution_counts(evolution_summary_df)

    badge_items = [
        ("Stable", counts["stable"], "badge-stable"),
        ("Mutated", counts["rewritten_candidate"], "badge-mutated"),
        ("Emerging", counts["new"], "badge-emerging"),
        ("Vanishing", counts["disappeared"], "badge-vanishing"),
    ]

    cols = st.columns(4)

    for col, (label, count, css_class) in zip(cols, badge_items):
        with col:
            st.markdown(
                dedent(
                    f"""
                    <div class="evolution-badge {css_class}">
                        <div class="badge-label">{escape(str(label))}</div>
                        <div class="badge-count">{escape(str(count))}</div>
                    </div>
                    """
                ),
                unsafe_allow_html=True,
            )
# ---------- Page ----------

inject_style()
# ---------- UI State ----------
# Initialize this flag before any section uses it.
# The actual toggle is rendered later in the sidebar settings area.
if "show_design_notes" not in st.session_state:
    st.session_state["show_design_notes"] = False

show_design_notes = bool(st.session_state["show_design_notes"])
st.title("🧭 LogPilot")
st.caption("大模型日志解析与模板演化分析平台 | LLM-powered Log Parsing & Template Evolution Lab")

if show_design_notes:
    with st.expander("查看产品定位说明", expanded=False):
        st.markdown(
            """
LogPilot 用于展示 Drain、LLM Direct、Hybrid 在日志解析场景中的适用边界，
并通过延迟、Token、Valid JSON Rate、Fallback Rate、模板演化等指标，
呈现一个更接近真实 AI 产品的闭环。
"""
        )

st.divider()

# ---------- Sidebar: Data Source ----------

st.sidebar.header("① 数据来源")

data_source = st.sidebar.radio(
    "选择日志来源",
    options=[
        "内置样例",
        "上传文件",
        "本地路径",
    ],
    index=0,
)

dataset_hint = st.sidebar.selectbox(
    "日志类型提示",
    options=[
        "HDFS sample",
        "BGL sample",
        "Generic log",
    ],
    index=0,
)

start_line = st.sidebar.number_input(
    "起始行号",
    min_value=1,
    value=1,
    step=1,
    help="用于从大文件中读取指定片段。例如从第 100000 行开始读取。",
)

max_lines = st.sidebar.slider(
    "读取行数",
    min_value=5,
    max_value=5000,
    value=50,
    step=5,
    help="建议真实 LLM 调用时先控制在 5-50 行。",
)

records = []
source_desc = ""
source_label = ""

try:
    if data_source == "内置样例":
        sample_name = st.sidebar.selectbox(
            "选择内置样例",
            options=list(SAMPLE_DATASETS.keys()),
        )
        selected_path = SAMPLE_DATASETS[sample_name]
        records = load_log_file(
            selected_path,
            max_lines=max_lines,
            dataset_name=sample_name,
            start_line=start_line,
        )
        source_desc = f"{sample_name} | {selected_path}"
        source_label = sample_name

    elif data_source == "上传文件":
        uploaded_file = st.sidebar.file_uploader(
            "上传 .log / .txt 文件",
            type=["log", "txt", "csv"],
        )

        if uploaded_file is not None:
            text = uploaded_file.getvalue().decode("utf-8", errors="ignore")
            lines = text.splitlines()
            records = load_log_lines(
                lines,
                max_lines=max_lines,
                dataset_name=dataset_hint,
                start_line=start_line,
            )
            source_desc = f"Uploaded file | {uploaded_file.name}"
            source_label = uploaded_file.name
        else:
            source_desc = "尚未上传文件"
            source_label = "Uploaded file"

    elif data_source == "本地路径":
        local_path = st.sidebar.text_input(
            "本地日志路径",
            placeholder=r"D:\path\to\HDFS_full.log",
        )

        if local_path.strip():
            records = load_log_file(
                local_path.strip(),
                max_lines=max_lines,
                dataset_name=dataset_hint,
                start_line=start_line,
            )
            source_desc = f"Local path | {local_path.strip()}"
            source_label = local_path.strip().split("\\")[-1].split("/")[-1] or "Local path"
        else:
            source_desc = "尚未输入本地路径"
            source_label = "Local path"

except Exception as exc:
    st.error(f"日志加载失败：{exc}")
    st.stop()


if not records:
    st.info("请先在左侧选择内置样例、上传日志文件，或输入本地日志路径。")
    st.stop()

raw_df = records_to_dataframe(records)


# ---------- Sidebar: Parser Config ----------

st.sidebar.header("② 解析配置")

similarity_threshold = st.sidebar.slider(
    "Drain 相似度阈值",
    min_value=0.1,
    max_value=1.0,
    value=0.4,
    step=0.05,
)

depth = st.sidebar.slider(
    "Drain 树深度",
    min_value=2,
    max_value=8,
    value=4,
    step=1,
)

max_clusters = st.sidebar.number_input(
    "Drain 最大 Cluster 数",
    min_value=100,
    max_value=10000,
    value=1000,
    step=100,
)


st.sidebar.header("③ LLM 配置")

selected_llm_model = st.sidebar.selectbox(
    "LLM 模型",
    options=get_llm_model_options(),
    index=0,
)

custom_llm_model = ""
if selected_llm_model == "自定义模型":
    custom_llm_model = st.sidebar.text_input(
        "自定义模型名称",
        placeholder="例如：qwen3.6-plus / qwen-plus / 你的模型名",
    )

llm_backend_name, llm_model_name = resolve_backend_and_model(
    selected_model=selected_llm_model,
    custom_model=custom_llm_model,
)

llm_model_display = get_model_display_name(llm_model_name)

batch_size = st.sidebar.slider(
    "LLM Batch Size",
    min_value=1,
    max_value=20,
    value=8,
    step=1,
)

if llm_backend_name == "MockBackend":
    st.sidebar.info("当前为 Mock Demo，不调用真实 API，适合开源演示。")
else:
    if is_openai_configured():
        st.sidebar.success("已检测到本地 API 配置。")
    else:
        st.sidebar.warning("需要在本地 .env 中配置 OPENAI_API_KEY 和 OPENAI_BASE_URL。")
with st.sidebar.expander("⚙️ 显示设置", expanded=False):
    st.toggle(
        "显示设计说明",
        key="show_design_notes",
        help="打开后显示模块设计说明，适合答辩或面试讲解。",
    )

# Refresh local variable after the widget is rendered.
show_design_notes = bool(st.session_state["show_design_notes"])
# ---------- Top summary ----------

ribbon_slot = st.empty()
with ribbon_slot.container():
    render_metric_ribbon(
        data_source_label=source_label or data_source,
        log_count=len(records),
        start_line=start_line,
        llm_model_display=llm_model_display,
    )

st.markdown(
    f'<div class="subtle-text">当前数据：{source_desc}</div>',
    unsafe_allow_html=True,
)
# ---------- Tabs ----------

tab_preview, tab_single, tab_compare, tab_hybrid, tab_evolution = st.tabs(
    [
        "📄 数据预览",
        "🔧 单方法解析",
        "📊 方法对比",
        "🤖 Hybrid Review",
        "🧬 模板演化",
    ]
)


with tab_preview:
    st.subheader("数据预览")

    if show_design_notes:
        with st.expander("查看字段说明", expanded=False):
            st.markdown(
                """
这里展示从原始日志中提取出的三个层次：

- `raw_log`：原始日志；
- `content`：去掉时间、日志级别、组件名后的日志主体；
- `masked_content`：对明显变量进行正则遮蔽后的文本，适合给 Drain 使用。
"""
            )

    safe_dataframe(
        raw_df,
        ["line_id", "raw_log", "content", "masked_content"],
        height=420,
    )


with tab_single:
    st.subheader("单方法解析")

    method = st.radio(
        "选择解析方法",
        options=[
            "Drain baseline",
            "LLM Direct",
        ],
        horizontal=True,
    )

    if method == "Drain baseline":
        parse_target = st.radio(
            "Drain 解析对象",
            options=[
                "masked_content",
                "content",
                "raw_log",
            ],
            index=0,
            horizontal=True,
        )

        if st.button("运行 Drain baseline", type="primary"):
            with st.spinner("正在运行 Drain baseline..."):
                parsed_df = run_drain_parser(
                    records=records,
                    text_field=parse_target,
                    similarity_threshold=similarity_threshold,
                    depth=depth,
                    max_clusters=max_clusters,
                )
                summary = summarize_parsing_result(parsed_df)

            remember_recent_run("Drain baseline", summary)
            with ribbon_slot.container():
                render_metric_ribbon(
                    data_source_label=source_label or data_source,
                    log_count=len(records),
                    start_line=start_line,
                    llm_model_display=llm_model_display,
                )

            st.success("Drain 解析完成。")

            render_basic_metrics(summary)

            st.markdown("#### 解析结果")
            safe_dataframe(
                parsed_df,
                [
                    "line_id",
                    "content",
                    "masked_content",
                    "template",
                    "cluster_id",
                    "final_cluster_size",
                    "change_type",
                    "latency_ms",
                ],
                height=420,
            )

            st.markdown("#### 模板统计")
            render_template_summary(parsed_df)

            csv = parsed_df.to_csv(index=False).encode("utf-8-sig")
            st.download_button(
                "下载 Drain 结果 CSV",
                data=csv,
                file_name="drain_results.csv",
                mime="text/csv",
            )

    else:
        parse_target = st.radio(
            "LLM 解析对象",
            options=[
                "content",
                "masked_content",
                "raw_log",
            ],
            index=0,
            horizontal=True,
        )

        if st.button("运行 LLM Direct", type="primary"):
            with st.spinner(f"正在运行 LLM Direct：{llm_model_display}..."):
                if llm_backend_name == "OpenAICompatibleBackend" and llm_model_name:
                    os.environ["OPENAI_MODEL"] = llm_model_name
                parsed_df = run_llm_parser(
                    records=records,
                    text_field=parse_target,
                    backend_name=llm_backend_name,
                    batch_size=batch_size,
                )
                summary = summarize_parsing_result(parsed_df)

            remember_recent_run(f"LLM Direct · {llm_model_display}", summary)
            with ribbon_slot.container():
                render_metric_ribbon(
                    data_source_label=source_label or data_source,
                    log_count=len(records),
                    start_line=start_line,
                    llm_model_display=llm_model_display,
                )

            st.toast("LLM Direct 解析完成。", icon="✅")

            render_llm_direct_metrics(summary)

            st.markdown("#### 解析结果")
            safe_dataframe(
                parsed_df,
                [
                    "line_id",
                    "content",
                    "template",
                    "variables",
                ],
                height=420,
            )

            failed_count = 0
            if "valid_json" in parsed_df.columns:
                failed_count = int((parsed_df["valid_json"] == False).sum())
            
            if failed_count > 0:
                st.warning(
                    f"有 {failed_count} 条日志未获得合法 LLM 输出。可以降低 Batch Size，或在下方调试信息中查看 API 错误。"
                )
            
            with st.expander("API 错误 / Raw Response 调试信息", expanded=False):
                safe_dataframe(
                    parsed_df,
                    [
                        "line_id",
                        "valid_json",
                        "backend_error",
                        "raw_response",
                        "cleaned_response",
                    ],
                    height=320,
                )
            
            csv = parsed_df.to_csv(index=False).encode("utf-8-sig")
            st.download_button(
                "下载 LLM 结果 CSV",
                data=csv,
                file_name="llm_results.csv",
                mime="text/csv",
            )


with tab_compare:
    st.subheader("方法对比：Drain vs LLM Direct")

    if show_design_notes:
        with st.expander("查看方法对比设计说明", expanded=False):
            st.markdown(
            """
    该模块不是为了证明 LLM 一定优于 Drain，而是对比两类策略在模板数量、延迟、Token、
    Valid JSON Rate、输出一致性上的差异。真实产品中，二者的差异样本通常应进入 Review 或 Hybrid 流程。
    """
            )

    if st.button("运行 Drain + LLM 对比", type="primary"):
        with st.spinner("正在运行 Drain 和 LLM Direct..."):
            drain_df = run_drain_parser(
                records=records,
                text_field="masked_content",
                similarity_threshold=similarity_threshold,
                depth=depth,
                max_clusters=max_clusters,
            )

            if llm_backend_name == "OpenAICompatibleBackend" and llm_model_name:
                os.environ["OPENAI_MODEL"] = llm_model_name

            llm_df = run_llm_parser(
                records=records,
                text_field="content",
                backend_name=llm_backend_name,
                batch_size=batch_size,
            )

            drain_summary = summarize_parsing_result(drain_df)
            llm_summary = summarize_parsing_result(llm_df)

            comparison_summary = pd.DataFrame(
                [
                    summary_to_row("Drain baseline", drain_summary),
                    summary_to_row(f"LLM Direct - {llm_model_display}", llm_summary),
                ]
            )

            line_compare_df = compare_line_level_templates(drain_df, llm_df)
            line_compare_df = attach_llm_backend_error(line_compare_df, llm_df)
            agreement_rate = compute_template_agreement(line_compare_df)

        remember_recent_run("方法对比", llm_summary)
        with ribbon_slot.container():
            render_metric_ribbon(
                data_source_label=source_label or data_source,
                log_count=len(records),
                start_line=start_line,
                llm_model_display=llm_model_display,
            )

        st.success("方法对比完成。")

        with st.expander("方法详细指标", expanded=False):
            safe_dataframe(
                comparison_summary,
                [
                    "method",
                    "total_logs",
                    "template_count",
                    "avg_latency_ms",
                    "total_tokens_est",
                ],
                height=150,
            )

        render_compact_metric_row(
            [
                ("模板一致率", f"{agreement_rate * 100:.1f}%"),
                ("Drain 模板数", drain_summary.get("template_count", 0)),
                ("LLM 模板数", llm_summary.get("template_count", 0)),
            ],
            columns=3,
        )

        st.markdown("#### 行级模板对比")
        safe_dataframe(
            line_compare_df,
            [
                "content",
                "drain_template",
                "llm_template",
                "same_template",
            ],
            height=460,
        )

        if show_design_notes:
            with st.expander("查看方法对比结果解读", expanded=False):
                st.info(
                    "如果 Drain 更快且模板更稳定，它适合作为主路径；如果 LLM 对变量、语义或自然语言日志更友好，"
                    "它适合作为 Review / 修正 / 解释层。二者不一致的样本，是 Hybrid 需要重点处理的对象。"
                )

        csv = line_compare_df.to_csv(index=False).encode("utf-8-sig")
        st.download_button(
            "下载对比结果 CSV",
            data=csv,
            file_name="method_comparison.csv",
            mime="text/csv",
        )


with tab_hybrid:
    st.subheader("Hybrid Review：Drain 全量解析 + LLM 风险样本复核")

    if show_design_notes:
        with st.expander("查看 Hybrid 设计说明", expanded=False):
            st.markdown(
            """
    Hybrid 模式体现真实 AI 产品思路：Drain 先解析全部日志，系统计算 risk_score，
    只把高风险样本交给 LLM Review。LLM 输出作为建议，不直接无条件替换；LLM 失败时 fallback 到 Drain。
    """
            )

    h_col_1, h_col_2, h_col_3 = st.columns(3)

    with h_col_1:
        review_top_k = st.slider(
            "Review Top-K 风险日志",
            min_value=1,
            max_value=max(1, min(100, len(records))),
            value=min(10, len(records)),
            step=1,
        )

    with h_col_2:
        risk_threshold = st.slider(
            "Risk Threshold",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.05,
        )

    with h_col_3:
        small_cluster_threshold = st.slider(
            "Small Cluster Threshold",
            min_value=1,
            max_value=10,
            value=1,
            step=1,
        )

    high_wildcard_threshold = st.slider(
        "High Wildcard Ratio Threshold",
        min_value=0.1,
        max_value=1.0,
        value=0.5,
        step=0.05,
    )

    if st.button("运行 Hybrid Review", type="primary"):
        with st.spinner("正在运行 Hybrid Review..."):
            if llm_backend_name == "OpenAICompatibleBackend" and llm_model_name:
                os.environ["OPENAI_MODEL"] = llm_model_name

            parsed_df = run_hybrid_parser(
                records=records,
                backend_name=llm_backend_name,
                drain_similarity_threshold=similarity_threshold,
                drain_depth=depth,
                drain_max_clusters=max_clusters,
                llm_batch_size=batch_size,
                review_top_k=review_top_k,
                risk_threshold=risk_threshold,
                small_cluster_threshold=small_cluster_threshold,
                high_wildcard_threshold=high_wildcard_threshold,
            )
            summary = summarize_parsing_result(parsed_df)

        remember_recent_run("Hybrid Review", summary)
        with ribbon_slot.container():
            render_metric_ribbon(
                data_source_label=source_label or data_source,
                log_count=len(records),
                start_line=start_line,
                llm_model_display=llm_model_display,
            )

        st.success("Hybrid Review 完成。")

        review_rate = summary.get("review_rate")
        fallback_rate = summary.get("fallback_rate")
        render_compact_metric_row(
            [
                ("日志数", summary.get("total_logs", 0)),
                ("模板数", summary.get("template_count", 0)),
                ("Review Rate", "-" if review_rate is None else f"{review_rate * 100:.1f}%"),
                ("Fallback Rate", "-" if fallback_rate is None else f"{fallback_rate * 100:.1f}%"),
            ],
            columns=4,
        )

        render_hybrid_summary_cards(parsed_df)

        st.markdown("#### Hybrid 结果（LLM recommendation / suggestion）")
        safe_dataframe(
            parsed_df,
            [
                "content",
                "drain_template",
                "llm_template",
                "template",
                "risk_reason",
                "hybrid_action",
            ],
            height=460,
        )

        with st.expander("详细调试信息", expanded=False):
            safe_dataframe(
                parsed_df,
                [
                    "line_id",
                    "content",
                    "used_llm_review",
                    "fallback_to_drain",
                    "risk_score",
                    "risk_reason",
                    "drain_template",
                    "llm_template",
                    "llm_variables",
                    "llm_confidence",
                    "latency_ms",
                ],
                height=360,
            )

        csv = parsed_df.to_csv(index=False).encode("utf-8-sig")
        st.download_button(
            "下载 Hybrid 结果 CSV",
            data=csv,
            file_name="hybrid_results.csv",
            mime="text/csv",
        )


with tab_evolution:
    st.subheader("模板演化分析")

    if show_design_notes:
        with st.expander("查看模板演化设计说明", expanded=False):
            st.markdown(
                """
该模块用于分析两个日志窗口之间的模板生命周期变化：
stable 表示两个窗口都出现，new 表示只在后一个窗口出现，
disappeared 表示只在前一个窗口出现，rewritten_candidate 表示表面不同但相似度较高，可能是模板改写。
"""
            )

    if len(records) < 4:
        st.warning("模板演化分析至少建议读取 4 行日志。")
        st.stop()

    line_ids = [record["line_id"] for record in records]
    min_line_id = min(line_ids)
    max_line_id = max(line_ids)

    e_col_1, e_col_2, e_col_3 = st.columns(3)

    with e_col_1:
        evolution_parser = st.selectbox(
            "演化分析解析器",
            options=[
                "Drain baseline",
                "LLM Direct",
            ],
            index=0,
        )

    with e_col_2:
        window_mode = st.selectbox(
            "窗口选择方式",
            options=[
                "自动前后窗口",
                "自定义窗口",
            ],
            index=1,
        )

    with e_col_3:
        rewrite_similarity_threshold = st.slider(
            "改写候选相似度阈值",
            min_value=0.1,
            max_value=1.0,
            value=0.5,
            step=0.05,
        )

    if window_mode == "自动前后窗口":
        window_size = st.slider(
            "窗口大小",
            min_value=2,
            max_value=max(2, len(records) // 2),
            value=max(2, len(records) // 2),
            step=1,
        )

        old_records = records[:window_size]
        new_records = records[-window_size:]

        window_desc = (
            f"Window A：当前读取片段前 {len(old_records)} 行；"
            f"Window B：当前读取片段后 {len(new_records)} 行。"
        )

    else:
        st.markdown("#### 自定义窗口选择")

        custom_col_1, custom_col_2, custom_col_3 = st.columns(3)

        with custom_col_1:
            old_start_line = st.number_input(
                "Window A 起始 line_id",
                min_value=int(min_line_id),
                max_value=int(max_line_id),
                value=int(min_line_id),
                step=1,
            )

        with custom_col_2:
            default_new_start = min(
                int(max_line_id),
                int(min_line_id) + max(1, len(records) // 2),
            )

            new_start_line = st.number_input(
                "Window B 起始 line_id",
                min_value=int(min_line_id),
                max_value=int(max_line_id),
                value=default_new_start,
                step=1,
            )

        with custom_col_3:
            window_size = st.number_input(
                "窗口大小",
                min_value=2,
                max_value=max(2, len(records)),
                value=min(100, max(2, len(records) // 2)),
                step=1,
            )

        def select_window_by_line_id(all_records, start_line_id: int, size: int):
            end_line_id = start_line_id + size - 1
            return [
                record for record in all_records
                if start_line_id <= record["line_id"] <= end_line_id
            ]

        old_records = select_window_by_line_id(
            records,
            start_line_id=int(old_start_line),
            size=int(window_size),
        )

        new_records = select_window_by_line_id(
            records,
            start_line_id=int(new_start_line),
            size=int(window_size),
        )

        window_desc = (
            f"Window A：line_id {old_start_line} - {old_start_line + window_size - 1}；"
            f"Window B：line_id {new_start_line} - {new_start_line + window_size - 1}。"
        )

    st.markdown(
        f'<div class="subtle-text">当前窗口：{window_desc}</div>',
        unsafe_allow_html=True,
    )

    if not old_records or not new_records:
        st.warning("当前窗口没有读取到日志。请调整起始 line_id 或窗口大小。")
        st.stop()

    st.markdown("#### 演化状态快照栏")
    evolution_badge_slot = st.empty()
    with evolution_badge_slot.container():
        render_status_badges()

    preview_col_1, preview_col_2 = st.columns(2)

    with preview_col_1:
        st.markdown("#### Window A")
        safe_dataframe(
            records_to_dataframe(old_records),
            ["line_id", "content", "masked_content"],
            height=240,
        )

    with preview_col_2:
        st.markdown("#### Window B")
        safe_dataframe(
            records_to_dataframe(new_records),
            ["line_id", "content", "masked_content"],
            height=240,
        )

    if st.button("运行模板演化分析", type="primary"):
        with st.spinner("正在解析两个窗口并分析模板演化..."):
            if evolution_parser == "Drain baseline":
                old_df = run_drain_parser(
                    records=old_records,
                    text_field="masked_content",
                    similarity_threshold=similarity_threshold,
                    depth=depth,
                    max_clusters=max_clusters,
                )

                new_df = run_drain_parser(
                    records=new_records,
                    text_field="masked_content",
                    similarity_threshold=similarity_threshold,
                    depth=depth,
                    max_clusters=max_clusters,
                )

            else:
                if llm_backend_name == "OpenAICompatibleBackend" and llm_model_name:
                    os.environ["OPENAI_MODEL"] = llm_model_name

                old_df = run_llm_parser(
                    records=old_records,
                    text_field="content",
                    backend_name=llm_backend_name,
                    batch_size=batch_size,
                )

                new_df = run_llm_parser(
                    records=new_records,
                    text_field="content",
                    backend_name=llm_backend_name,
                    batch_size=batch_size,
                )

            old_summary = summarize_parsing_result(old_df)
            new_summary = summarize_parsing_result(new_df)

            evolution_summary_df, evolution_detail_df = analyze_template_evolution(
                old_df=old_df,
                new_df=new_df,
                old_window_name="Window A",
                new_window_name="Window B",
                rewrite_similarity_threshold=rewrite_similarity_threshold,
            )

        remember_recent_run("模板演化分析", new_summary)
        with ribbon_slot.container():
            render_metric_ribbon(
                data_source_label=source_label or data_source,
                log_count=len(records),
                start_line=start_line,
                llm_model_display=llm_model_display,
            )
        with evolution_badge_slot.container():
            render_status_badges(evolution_summary_df)

        st.success("模板演化分析完成。")

        st.markdown("#### 演化类型统计")
        st.dataframe(evolution_summary_df, use_container_width=True, height=180)

        st.markdown("#### 演化细节")

        if evolution_detail_df.empty:
            st.info("未检测到明显模板演化。")
        else:
            st.dataframe(evolution_detail_df, use_container_width=True, height=460)

        csv = evolution_detail_df.to_csv(index=False).encode("utf-8-sig")
        st.download_button(
            "下载模板演化结果 CSV",
            data=csv,
            file_name="template_evolution.csv",
            mime="text/csv",
        )
