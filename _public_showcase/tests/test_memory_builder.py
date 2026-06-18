from __future__ import annotations

import csv

from logpilot.ai_framework.memory_builder import (
    build_template_memory_from_logs,
    simple_template_from_log,
)


def test_simple_template_from_log_masks_block_id() -> None:
    template = simple_template_from_log("Receiving block blk_123 src: /data/a")

    assert "blk_123" not in template
    assert "<*>" in template


def test_simple_template_from_log_masks_ip_port() -> None:
    template = simple_template_from_log("connect from 10.250.19.102:54106 ok")

    assert "10.250.19.102:54106" not in template
    assert "<*>" in template


def test_build_template_memory_from_logs_writes_csv(tmp_path) -> None:
    output_path = tmp_path / "memory.csv"
    path = build_template_memory_from_logs(
        [
            "Receiving block blk_123 src: /10.250.19.102:54106",
            "Receiving block blk_456 src: /10.250.19.103:54106",
        ],
        output_path=str(output_path),
    )

    assert path == str(output_path)
    assert output_path.exists()


def test_build_template_memory_from_logs_outputs_required_columns(tmp_path) -> None:
    output_path = tmp_path / "memory.csv"
    build_template_memory_from_logs(["log 123"], output_path=str(output_path))

    with output_path.open(encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        assert "template" in (reader.fieldnames or [])
        assert "description" in (reader.fieldnames or [])
