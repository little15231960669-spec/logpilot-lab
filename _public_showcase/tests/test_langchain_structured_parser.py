from __future__ import annotations

import pytest

from logpilot.ai_framework.structured_log_parser import extract_json_object


def test_extract_json_object_from_plain_json() -> None:
    data = extract_json_object(
        '{"template": "abc <*>", "variables": ["x"], '
        '"confidence": 0.9, "reason": "test"}'
    )

    assert data == {
        "template": "abc <*>",
        "variables": ["x"],
        "confidence": 0.9,
        "reason": "test",
    }


def test_extract_json_object_from_markdown_fence() -> None:
    data = extract_json_object(
        '```json\n{"template": "abc <*>", "variables": ["x"], '
        '"confidence": 0.9, "reason": "test"}\n```'
    )

    assert data["template"] == "abc <*>"
    assert data["variables"] == ["x"]


def test_extract_json_object_from_text_with_prefix_and_suffix() -> None:
    data = extract_json_object(
        'Result:\n{"template": "abc <*>", "variables": ["x"], '
        '"confidence": 0.9, "reason": "test"}\nPlease confirm.'
    )

    assert data["confidence"] == 0.9
    assert data["reason"] == "test"


def test_extract_json_object_raises_for_invalid_json() -> None:
    with pytest.raises(ValueError, match="JSON object"):
        extract_json_object("not valid json")


def test_extract_json_object_does_not_support_json_array() -> None:
    with pytest.raises(ValueError, match="JSON array is not supported"):
        extract_json_object('[{"template": "abc <*>"}]')
