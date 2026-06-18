"""Structured log parsing with LangChain."""

from __future__ import annotations

import json
import re
from typing import Any

from logpilot.ai_framework.config import get_openai_api_key
from logpilot.ai_framework.errors import StructuredParseError
from logpilot.ai_framework.models import build_chat_model
from logpilot.ai_framework.schemas import LogParseResult


def build_log_parse_prompt(log: str) -> str:
    """Build the Chinese JSON prompt used by both parser paths."""
    return (
        "你是日志解析专家。\n"
        "请将输入日志解析为日志模板和变量。\n"
        "要求：\n"
        "- template 中所有变量统一使用 <*>。\n"
        "- variables 按原日志出现顺序输出。\n"
        "- confidence 是 0 到 1 的数字。\n"
        "- reason 简要说明判断依据。\n"
        "- 输出必须是 JSON object。\n"
        "- 不要输出 Markdown。\n"
        "- 不要输出额外解释文本。\n\n"
        "字段必须是：\n"
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


def _parse_json_dict(text: str) -> dict:
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON object: {exc}") from exc

    if isinstance(data, list):
        raise ValueError("JSON array is not supported; expected a JSON object.")
    if not isinstance(data, dict):
        raise ValueError("Expected a JSON object.")
    return data


def _find_first_json_object(text: str) -> str:
    start = text.find("{")
    if start == -1:
        raise ValueError("No JSON object found in model output.")

    depth = 0
    in_string = False
    escape = False

    for index in range(start, len(text)):
        char = text[index]

        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]

    raise ValueError("Unclosed JSON object in model output.")


def extract_json_object(text: str) -> dict:
    """Extract and parse the first JSON object from model output."""
    stripped = text.strip()
    if not stripped:
        raise ValueError("Model output is empty; expected a JSON object.")

    if stripped.startswith("["):
        raise ValueError("JSON array is not supported; expected a JSON object.")

    try:
        return _parse_json_dict(stripped)
    except ValueError:
        pass

    fenced_match = re.search(
        r"```(?:json|JSON)?\s*(.*?)\s*```",
        stripped,
        flags=re.DOTALL,
    )
    if fenced_match:
        return _parse_json_dict(fenced_match.group(1).strip())

    json_object_text = _find_first_json_object(stripped)
    return _parse_json_dict(json_object_text)


def parse_log_with_structured_output(log: str) -> LogParseResult:
    """Parse a log line using provider-native structured output."""
    llm = build_chat_model()
    structured_llm = llm.with_structured_output(LogParseResult)
    result = structured_llm.invoke(build_log_parse_prompt(log))

    if isinstance(result, LogParseResult):
        return result
    return _validate_log_parse_result(result)


def parse_log_with_json_fallback(log: str) -> LogParseResult:
    """Parse a log line using JSON prompting plus Pydantic validation."""
    llm = build_chat_model()
    response = llm.invoke(build_log_parse_prompt(log))
    content = getattr(response, "content", response)
    if not isinstance(content, str):
        raise StructuredParseError("LLM fallback response content is not text.")

    data = extract_json_object(content)
    return _validate_log_parse_result(data)


def _safe_error_summary(error: Exception) -> str:
    summary = str(error)
    api_key = get_openai_api_key()
    if api_key:
        summary = summary.replace(api_key, "***")
    return summary


def parse_log_with_langchain(log: str) -> LogParseResult:
    """Parse a log line, falling back to JSON prompting when needed."""
    try:
        return parse_log_with_structured_output(log)
    except Exception as structured_error:
        try:
            return parse_log_with_json_fallback(log)
        except Exception as fallback_error:
            structured_summary = _safe_error_summary(structured_error)
            fallback_summary = _safe_error_summary(fallback_error)
            raise StructuredParseError(
                "LangChain structured log parsing failed. "
                f"structured_output_error={structured_summary}; "
                f"fallback_error={fallback_summary}"
            ) from fallback_error
