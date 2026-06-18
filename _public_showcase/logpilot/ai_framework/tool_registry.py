"""Tool metadata registry for LogPilot agent workflows."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


VALID_RISK_LEVELS = {"low", "medium", "high"}
AGENT_TASK_ACTIONS = [
    "high_confidence_parse_review",
    "fast_baseline_parse",
    "compare_parser_tools",
    "find_risky_templates",
    "explain_parsing_results",
]


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    input_fields: dict[str, str]
    output_fields: dict[str, str]
    risk_level: str
    requires_llm: bool
    deterministic: bool
    suitable_actions: list[str] = field(default_factory=list)
    failure_modes: list[str] = field(default_factory=list)
    example_input: dict[str, Any] | None = None
    example_output: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if self.risk_level not in VALID_RISK_LEVELS:
            raise ValueError(f"Invalid risk_level for {self.name}: {self.risk_level}")


_TOOL_SPECS = [
    ToolSpec(
        name="retrieve_template_memory",
        description="Retrieve similar historical log templates using lightweight lexical similarity.",
        input_fields={
            "log": "Raw log line to search against template memory.",
            "template_csv_path": "CSV file containing historical templates.",
            "k": "Maximum number of templates to return.",
        },
        output_fields={
            "tool_name": "Tool identifier.",
            "log": "Input log line.",
            "top_k": "Ranked list of retrieved templates with description and score.",
        },
        risk_level="low",
        requires_llm=False,
        deterministic=True,
        suitable_actions=["retrieve_only", "full_review"],
        failure_modes=["template CSV missing or empty"],
        example_input={
            "log": "Receiving block blk_123 src: /a dest: /b",
            "template_csv_path": "data/template_memory/hdfs_templates.csv",
            "k": 3,
        },
        example_output={
            "top_k": [
                {
                    "template": "Receiving block <*> src: <*> dest: <*>",
                    "score": 0.622,
                }
            ]
        },
    ),
    ToolSpec(
        name="parse_with_template_memory",
        description="Parse a log line with Template Memory RAG using LangChain structured output and JSON fallback.",
        input_fields={
            "log": "Raw log line to parse.",
            "template_csv_path": "CSV file containing historical templates.",
            "k": "Number of retrieved templates to use as context.",
        },
        output_fields={
            "tool_name": "Tool identifier.",
            "log": "Input log line.",
            "result": "Structured parse result with template, variables, confidence, and reason.",
            "error": "Error summary when parsing fails.",
        },
        risk_level="medium",
        requires_llm=True,
        deterministic=False,
        suitable_actions=["parse_with_memory", "compare_with_candidate", "full_review"],
        failure_modes=[
            "LLM connection error",
            "structured output failure",
            "JSON validation failure",
        ],
        example_input={
            "log": "Receiving block blk_123 src: /a dest: /b",
            "template_csv_path": "data/template_memory/hdfs_templates.csv",
            "k": 3,
        },
        example_output={
            "result": {
                "template": "Receiving block <*> src: <*> dest: <*>",
                "variables": ["blk_123", "/a", "/b"],
                "confidence": 0.98,
            }
        },
    ),
    ToolSpec(
        name="compare_template_strings",
        description="Compare two template strings with exact match and token overlap metrics.",
        input_fields={
            "template_a": "First template string.",
            "template_b": "Second template string.",
        },
        output_fields={
            "tool_name": "Tool identifier.",
            "exact_match": "Whether the templates are exactly identical.",
            "token_overlap": "Normalized token overlap score.",
            "only_in_a": "Tokens only present in template_a.",
            "only_in_b": "Tokens only present in template_b.",
            "explanation": "Deterministic comparison explanation.",
        },
        risk_level="low",
        requires_llm=False,
        deterministic=True,
        suitable_actions=["compare_with_candidate", "full_review"],
        failure_modes=[],
        example_input={
            "template_a": "Receiving block <*> src: <*> dest: <*>",
            "template_b": "Receiving block <*> src: <*> dest: <*>",
        },
        example_output={"exact_match": True, "token_overlap": 1.0},
    ),
    ToolSpec(
        name="build_review_summary",
        description="Build a deterministic review summary from parsed result, retrieved templates, and comparison output.",
        input_fields={
            "log": "Raw input log.",
            "parsed_result": "Optional parse tool output.",
            "retrieved_templates": "Optional retrieved template list.",
            "comparison_result": "Optional template comparison output.",
        },
        output_fields={
            "summary": "Deterministic text review report.",
        },
        risk_level="low",
        requires_llm=False,
        deterministic=True,
        suitable_actions=[
            "retrieve_only",
            "parse_with_memory",
            "compare_with_candidate",
            "full_review",
        ],
        failure_modes=[],
        example_input={
            "log": "Receiving block blk_123 src: /a dest: /b",
            "parsed_result": {"template": "Receiving block <*> src: <*> dest: <*>"},
        },
        example_output={"summary": "Log review report: ..."},
    ),
    ToolSpec(
        name="offline_parse_with_template_memory",
        description=(
            "Offline demo / portfolio display parser that uses the top retrieved "
            "template as a deterministic parsing suggestion. It does not represent "
            "real LLM parsing."
        ),
        input_fields={
            "log": "Raw log line to parse in offline demo mode.",
            "top_retrieved_template": "Top template returned by retrieval.",
            "mode": "Expected to be offline_demo.",
        },
        output_fields={
            "tool_name": "Tool identifier.",
            "log": "Input log line.",
            "result": "Mock parse result for demonstration only.",
        },
        risk_level="low",
        requires_llm=False,
        deterministic=True,
        suitable_actions=["full_review"],
        failure_modes=["missing retrieved template"],
        example_input={
            "log": "Receiving block blk_123 src: /a dest: /b",
            "top_retrieved_template": "Receiving block <*> src: <*> dest: <*>",
            "mode": "offline_demo",
        },
        example_output={
            "result": {
                "template": "Receiving block <*> src: <*> dest: <*>",
                "confidence": 0.99,
            }
        },
    ),
    ToolSpec(
        name="drain_parse_logs",
        description="Parse the current selected logs with the existing Drain parser tool.",
        input_fields={
            "logs": "Batch of selected log lines.",
            "drain_config": "Optional Drain parser configuration.",
        },
        output_fields={
            "tool_name": "Tool identifier.",
            "drain_tool_mode": "existing or fallback.",
            "results": "Per-log Drain parse results.",
            "error": "Error summary when fallback mode is used.",
        },
        risk_level="low",
        requires_llm=False,
        deterministic=True,
        suitable_actions=[
            "parse_current_logs",
            "high_confidence_parse_review",
            "fast_baseline_parse",
            "compare_parser_tools",
            "find_risky_templates",
        ],
        failure_modes=["Drain parser failure triggers fallback parser"],
    ),
    ToolSpec(
        name="template_memory_parse_logs",
        description=(
            "Parse current selected logs with Template Memory. Offline mode is "
            "deterministic and does not require an LLM; online mode may call an LLM."
        ),
        input_fields={
            "logs": "Batch of selected log lines.",
            "template_csv_path": "CSV file containing historical templates.",
            "k": "Number of templates to retrieve.",
            "offline": "Whether to avoid LLM calls.",
        },
        output_fields={
            "tool_name": "Tool identifier.",
            "mode": "offline or online.",
            "results": "Per-log Template Memory parse results.",
            "error": "Tool-level error summary.",
        },
        risk_level="medium",
        requires_llm=False,
        deterministic=True,
        suitable_actions=[
            "parse_current_logs",
            "high_confidence_parse_review",
            "compare_parser_tools",
            "find_risky_templates",
            "explain_parsing_results",
        ],
        failure_modes=["template memory file missing", "online LLM call failure"],
    ),
    ToolSpec(
        name="build_template_memory_from_logs",
        description=(
            "Build a temporary template memory CSV from current logs using "
            "lightweight deterministic template extraction."
        ),
        input_fields={
            "logs": "Raw logs used to build template memory.",
            "method": "Selection method such as first_30_percent or random.",
            "max_memory_logs": "Maximum number of logs used for memory construction.",
            "output_path": "Optional output CSV path.",
        },
        output_fields={
            "template": "Generated template column in the CSV.",
            "description": "Generated template description.",
            "dataset": "Dataset/source marker.",
            "source": "Memory construction method.",
            "count": "Number of logs mapped to the template.",
        },
        risk_level="medium",
        requires_llm=False,
        deterministic=True,
        suitable_actions=[
            "high_confidence_parse_review",
            "compare_parser_tools",
            "find_risky_templates",
            "explain_parsing_results",
        ],
        failure_modes=["input logs are empty", "output path is not writable"],
    ),
    ToolSpec(
        name="compare_batch_parse_outputs",
        description="Compare Drain and Template Memory/RAG batch parse outputs.",
        input_fields={
            "drain_outputs": "Drain batch parse tool output.",
            "rag_outputs": "Template Memory/RAG batch parse tool output.",
        },
        output_fields={
            "total": "Compared item count.",
            "exact_match_count": "Number of matching templates.",
            "mismatch_count": "Number of mismatching templates.",
            "match_rate": "Exact match rate.",
            "items": "Per-log comparison rows.",
        },
        risk_level="low",
        requires_llm=False,
        deterministic=True,
        suitable_actions=[
            "parse_current_logs",
            "high_confidence_parse_review",
            "compare_parser_tools",
            "find_risky_templates",
        ],
        failure_modes=[],
    ),
    ToolSpec(
        name="build_batch_parse_summary",
        description="Build a deterministic batch parsing review summary.",
        input_fields={
            "logs": "Batch of selected log lines.",
            "drain_outputs": "Drain batch parse tool output.",
            "rag_outputs": "Template Memory/RAG batch parse tool output.",
            "comparison_outputs": "Batch comparison tool output.",
        },
        output_fields={
            "summary": "Deterministic batch review summary.",
        },
        risk_level="low",
        requires_llm=False,
        deterministic=True,
        suitable_actions=["parse_current_logs", *AGENT_TASK_ACTIONS],
        failure_modes=[],
    ),
]


def get_tool_specs() -> list[ToolSpec]:
    """Return all registered tool specs."""
    return list(_TOOL_SPECS)


def list_tool_names() -> list[str]:
    """Return registered tool names."""
    return [spec.name for spec in _TOOL_SPECS]


def get_tool_spec(name: str) -> ToolSpec:
    """Return a registered tool spec by name."""
    for spec in _TOOL_SPECS:
        if spec.name == name:
            return spec
    raise KeyError(f"Unknown tool: {name}")


def tool_spec_to_dict(spec: ToolSpec) -> dict:
    """Convert a tool spec into a JSON-ready dictionary."""
    return asdict(spec)


def get_tool_registry_dict() -> dict[str, dict]:
    """Return the full registry keyed by tool name."""
    return {spec.name: tool_spec_to_dict(spec) for spec in _TOOL_SPECS}


def render_tool_registry_markdown() -> str:
    """Render the tool registry as Markdown documentation."""
    lines = [
        "# Tool Registry Overview",
        "",
        "| Tool | Risk | Requires LLM | Deterministic | Suitable Actions |",
        "| --- | --- | --- | --- | --- |",
    ]
    for spec in _TOOL_SPECS:
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{spec.name}`",
                    spec.risk_level,
                    str(spec.requires_llm),
                    str(spec.deterministic),
                    ", ".join(spec.suitable_actions),
                ]
            )
            + " |"
        )

    for spec in _TOOL_SPECS:
        lines.extend(
            [
                "",
                f"## {spec.name}",
                "",
                spec.description,
                "",
                f"- risk_level: `{spec.risk_level}`",
                f"- requires_llm: `{spec.requires_llm}`",
                f"- deterministic: `{spec.deterministic}`",
                f"- suitable_actions: {', '.join(spec.suitable_actions) or 'none'}",
                f"- failure_modes: {', '.join(spec.failure_modes) or 'none'}",
            ]
        )

    return "\n".join(lines)


def build_tool_context_for_agent() -> str:
    """Build compact tool context suitable for the agent planning prompt."""
    lines = ["Available tools:"]
    for spec in _TOOL_SPECS:
        lines.append(
            "- "
            f"{spec.name}: {spec.description} "
            f"(risk={spec.risk_level}, requires_llm={spec.requires_llm}, "
            f"deterministic={spec.deterministic}, "
            f"actions={','.join(spec.suitable_actions)})"
        )
    return "\n".join(lines)
