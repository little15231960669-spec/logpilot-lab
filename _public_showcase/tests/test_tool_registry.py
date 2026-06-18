from __future__ import annotations

import pytest

from logpilot.ai_framework.tool_registry import (
    ToolSpec,
    build_tool_context_for_agent,
    get_tool_spec,
    get_tool_specs,
    list_tool_names,
    render_tool_registry_markdown,
    tool_spec_to_dict,
)


def test_get_tool_specs_returns_non_empty_list() -> None:
    assert get_tool_specs()


def test_list_tool_names_contains_core_tools() -> None:
    names = list_tool_names()

    assert "retrieve_template_memory" in names
    assert "parse_with_template_memory" in names
    assert "compare_template_strings" in names


def test_risk_levels_are_valid() -> None:
    for spec in get_tool_specs():
        assert spec.risk_level in {"low", "medium", "high"}


def test_get_tool_spec_returns_known_tool_spec() -> None:
    spec = get_tool_spec("retrieve_template_memory")

    assert isinstance(spec, ToolSpec)
    assert spec.name == "retrieve_template_memory"


def test_get_tool_spec_raises_for_unknown_tool() -> None:
    with pytest.raises(KeyError, match="Unknown tool"):
        get_tool_spec("missing_tool")


def test_tool_spec_to_dict_returns_dict() -> None:
    data = tool_spec_to_dict(get_tool_spec("compare_template_strings"))

    assert isinstance(data, dict)
    assert data["name"] == "compare_template_strings"


def test_render_tool_registry_markdown_contains_tool_names() -> None:
    markdown = render_tool_registry_markdown()

    assert "retrieve_template_memory" in markdown
    assert "parse_with_template_memory" in markdown


def test_build_tool_context_for_agent_contains_names_and_descriptions() -> None:
    context = build_tool_context_for_agent()

    assert "retrieve_template_memory" in context
    assert "Retrieve similar historical log templates" in context

