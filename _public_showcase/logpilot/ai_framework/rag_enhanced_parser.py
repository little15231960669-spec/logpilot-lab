"""RAG-enhanced structured log parsing with template memory."""

from __future__ import annotations

from typing import Any

from logpilot.ai_framework.config import get_openai_api_key
from logpilot.ai_framework.errors import StructuredParseError
from logpilot.ai_framework.models import build_chat_model
from logpilot.ai_framework.schemas import LogParseResult
from logpilot.ai_framework.structured_log_parser import extract_json_object
from logpilot.ai_framework.template_memory import (
    build_template_memory_context,
    retrieve_similar_templates,
)


def build_rag_log_parse_prompt(log: str, memory_context: str) -> str:
    """Build a Chinese JSON prompt with retrieved template memory context."""
    return (
        "你是日志解析专家。\n"
        "请参考相似历史模板来解析输入日志，但不要强行匹配不相关模板。\n"
        "如果历史模板有帮助，可以借鉴其结构；如果不相关，请根据原日志独立判断。\n\n"
        f"{memory_context}\n\n"
        "输出要求：\n"
        "- 输出必须是 JSON object。\n"
        "- 字段必须为 template、variables、confidence、reason。\n"
        "- template 中所有变量统一使用 <*>。\n"
        "- variables 按原日志出现顺序输出。\n"
        "- confidence 是 0 到 1 的数字。\n"
        "- reason 简要说明判断依据，包括是否参考了历史模板。\n"
        "- 不要输出 Markdown。\n"
        "- 不要输出额外解释文本。\n\n"
        "JSON object 示例：\n"
        "{\n"
        '  "template": "...",\n'
        '  "variables": ["..."],\n'
        '  "confidence": 0.95,\n'
        '  "reason": "..."\n'
        "}\n\n"
        f"日志：{log}"
    )


def _validate_log_parse_result(data: Any) -> LogParseResult:
    if hasattr(LogParseResult, "model_validate"):
        return LogParseResult.model_validate(data)
    return LogParseResult.parse_obj(data)


def _safe_error_summary(error: Exception) -> str:
    summary = str(error)
    api_key = get_openai_api_key()
    if api_key:
        summary = summary.replace(api_key, "***")
    return summary


def parse_log_with_template_memory(
    log: str,
    template_csv_path: str,
    k: int = 3,
) -> LogParseResult:
    """Parse a log line using retrieved historical templates as context."""
    retrieved = retrieve_similar_templates(log, template_csv_path, k=k)
    memory_context = build_template_memory_context(retrieved)
    prompt = build_rag_log_parse_prompt(log, memory_context)

    llm = build_chat_model()

    try:
        structured_llm = llm.with_structured_output(LogParseResult)
        structured_result = structured_llm.invoke(prompt)
        if isinstance(structured_result, LogParseResult):
            return structured_result
        return _validate_log_parse_result(structured_result)
    except Exception as structured_error:
        try:
            response = llm.invoke(prompt)
            content = getattr(response, "content", response)
            if not isinstance(content, str):
                raise StructuredParseError("LLM fallback response content is not text.")
            data = extract_json_object(content)
            return _validate_log_parse_result(data)
        except Exception as fallback_error:
            raise StructuredParseError(
                "RAG-enhanced structured log parsing failed. "
                f"structured_output_error={_safe_error_summary(structured_error)}; "
                f"fallback_error={_safe_error_summary(fallback_error)}"
            ) from fallback_error
