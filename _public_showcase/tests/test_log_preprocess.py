from __future__ import annotations

from pathlib import Path

from logpilot.ai_framework.log_preprocess import extract_log_content, mask_variables
from logpilot.ai_framework.log_tools import (
    drain_parse_logs_tool,
    template_memory_parse_logs_tool,
)


HDFS_DATA_XCEIVER = (
    "081109 203518 143 INFO dfs.DataNode$DataXceiver: "
    "Receiving block blk_38865049064139660 src: /10.250.19.102:54106 "
    "dest: /10.250.19.102:50010"
)


def test_extract_log_content_hdfs_data_xceiver() -> None:
    assert extract_log_content(HDFS_DATA_XCEIVER, "HDFS") == (
        "Receiving block blk_38865049064139660 src: /10.250.19.102:54106 "
        "dest: /10.250.19.102:50010"
    )


def test_extract_log_content_hdfs_fs_namesystem() -> None:
    raw = (
        "081109 203518 35 INFO dfs.FSNamesystem: "
        "BLOCK* NameSystem.allocateBlock: /user/root/rand/_temporary/_task_200811092030_0001_m_000000_0/part-00000. blk_123"
    )

    assert extract_log_content(raw, "HDFS").startswith("BLOCK* NameSystem.allocateBlock:")


def test_extract_log_content_hdfs_packet_responder() -> None:
    raw = (
        "081109 203518 143 INFO dfs.DataNode$PacketResponder: "
        "PacketResponder 1 for block blk_38865049064139660 terminating"
    )

    assert extract_log_content(raw, "HDFS") == (
        "PacketResponder 1 for block blk_38865049064139660 terminating"
    )


def test_mask_variables_receiving_block() -> None:
    content = (
        "Receiving block blk_38865049064139660 src: /10.250.19.102:54106 "
        "dest: /10.250.19.102:50010"
    )

    assert mask_variables(content) == "Receiving block <*> src: <*> dest: <*>"


def test_extract_log_content_bgl_message() -> None:
    raw = (
        "2005-06-03-15.42.50.363779 R01-M0-N0-C:J12-U11 RAS KERNEL INFO "
        "instruction cache parity error corrected"
    )

    assert extract_log_content(raw, "BGL") == "instruction cache parity error corrected"


def test_drain_parse_logs_tool_uses_hdfs_content() -> None:
    output = drain_parse_logs_tool([HDFS_DATA_XCEIVER], dataset_name="HDFS")
    result = output["results"][0]

    assert result["content"].startswith("Receiving block")
    assert "081109" not in result["template"]
    assert "INFO" not in result["template"]
    assert "dfs.DataNode$DataXceiver" not in result["template"]


def test_template_memory_parse_logs_tool_uses_hdfs_content() -> None:
    output = template_memory_parse_logs_tool(
        [HDFS_DATA_XCEIVER],
        str(Path("data/template_memory/hdfs_templates.csv")),
        offline=True,
        dataset_name="HDFS",
    )
    result = output["results"][0]

    assert result["content"].startswith("Receiving block")
    assert result["template"] == "Receiving block <*> src: <*> dest: <*>"
