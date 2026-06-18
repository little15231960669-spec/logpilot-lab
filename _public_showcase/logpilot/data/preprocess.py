import re
from typing import Optional


LOG_LEVEL_PATTERN = r"(INFO|WARN|WARNING|ERROR|FATAL|DEBUG|TRACE)"


def extract_log_content(raw_log: str, dataset_name: Optional[str] = None) -> str:
    """
    Extract the message content part from a raw log line.

    For log parsing, it is usually better to parse the log message body
    instead of the full raw line with timestamp, log level, thread id, etc.
    """
    if raw_log is None:
        return ""

    raw_log = raw_log.strip()
    dataset = (dataset_name or "").lower()

    # HDFS example:
    # 081109 203518 143 INFO dfs.DataNode$DataXceiver: Receiving block ...
    if "hdfs" in dataset:
        match = re.match(
            r"^\S+\s+\S+\s+\S+\s+\S+\s+[^:]+:\s*(?P<content>.*)$",
            raw_log,
        )
        if match:
            return match.group("content").strip()

    # BGL example:
    # ... RAS KERNEL INFO instruction cache parity error corrected
    if "bgl" in dataset:
        match = re.search(
            rf"\b{LOG_LEVEL_PATTERN}\b\s+(?P<content>.*)$",
            raw_log,
        )
        if match:
            return match.group("content").strip()

    # Generic case 1:
    # timestamp INFO component: message
    match = re.search(
        rf"\b{LOG_LEVEL_PATTERN}\b\s+[^:]+:\s*(?P<content>.*)$",
        raw_log,
    )
    if match:
        return match.group("content").strip()

    # Generic case 2:
    # timestamp INFO message
    match = re.search(
        rf"\b{LOG_LEVEL_PATTERN}\b\s+(?P<content>.*)$",
        raw_log,
    )
    if match:
        return match.group("content").strip()

    # Fallback: split after the first ": "
    if ": " in raw_log:
        return raw_log.split(": ", 1)[1].strip()

    return raw_log


def mask_variables(content: str) -> str:
    """
    Apply lightweight regex masks before Drain.

    This is a common practice in log parsing:
    known variable-like tokens such as block IDs, IP addresses, paths,
    and numbers are normalized before template mining.
    """
    if content is None:
        return ""

    text = content.strip()

    patterns = [
        # HDFS block ids
        (r"\bblk_-?\d+\b", "<*>"),

        # IP address with optional leading slash and optional port
        (r"/?(?:\d{1,3}\.){3}\d{1,3}(?::\d+)?", "<*>"),

        # File paths, e.g. /user/root/rand/part-00000
        (r"(?<!\w)/(?:[\w.\-]+/)*[\w.\-]+", "<*>"),

        # Standalone numbers
        (r"\b\d+\b", "<*>"),
    ]

    for pattern, replacement in patterns:
        text = re.sub(pattern, replacement, text)

    # Clean repeated spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text