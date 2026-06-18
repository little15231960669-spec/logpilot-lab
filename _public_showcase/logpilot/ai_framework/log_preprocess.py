"""Reusable log content extraction and variable masking helpers."""

from __future__ import annotations

import re
from typing import Optional


LOG_LEVEL_PATTERN = r"(INFO|WARN|WARNING|ERROR|FATAL|DEBUG|TRACE)"


def extract_log_content(raw_log: str, dataset_name: Optional[str] = None) -> str:
    """Extract the message content from a raw log line."""
    text = str(raw_log or "").strip()
    dataset = (dataset_name or "").upper()

    if dataset == "HDFS":
        match = re.match(
            rf"^\d{{6}}\s+\d{{6}}\s+\d+\s+{LOG_LEVEL_PATTERN}\s+[^:]+:\s*(?P<content>.*)$",
            text,
        )
        if match:
            return match.group("content").strip()

    if dataset == "BGL":
        match = re.match(
            rf"^.*?\bRAS\s+\S+\s+{LOG_LEVEL_PATTERN}\s+(?P<content>.*)$",
            text,
        )
        if match:
            return match.group("content").strip()

    generic_hdfs = re.match(
        rf"^\d{{6}}\s+\d{{6}}\s+\d+\s+{LOG_LEVEL_PATTERN}\s+[^:]+:\s*(?P<content>.*)$",
        text,
    )
    if generic_hdfs:
        return generic_hdfs.group("content").strip()

    generic_bgl = re.match(
        rf"^.*?\bRAS\s+\S+\s+{LOG_LEVEL_PATTERN}\s+(?P<content>.*)$",
        text,
    )
    if generic_bgl:
        return generic_bgl.group("content").strip()

    return text


def mask_variables(content: str) -> str:
    """Mask common variable tokens in log content with <*>."""
    template = str(content or "").strip()
    template = re.sub(r"\bblk_[A-Za-z0-9_]+\b", "<*>", template)
    template = re.sub(r"/?\b\d{1,3}(?:\.\d{1,3}){3}:\d+\b", "<*>", template)
    template = re.sub(r"\b\d{4}-\d{2}-\d{2}[T\s]\d{2}:\d{2}:\d{2}(?:\.\d+)?\b", "<*>", template)
    template = re.sub(r"\b\d{2}:\d{2}:\d{2}(?:\.\d+)?\b", "<*>", template)
    template = re.sub(r"/[^\s]+", "<*>", template)
    template = re.sub(r"\b\d+\b", "<*>", template)
    template = re.sub(r"\s+", " ", template).strip()
    return template
