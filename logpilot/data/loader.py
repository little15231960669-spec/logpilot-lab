from pathlib import Path
from typing import List, Dict, Optional, Iterable

import pandas as pd

from logpilot.data.preprocess import extract_log_content, mask_variables


SAMPLE_DATASETS = {
    "HDFS sample": Path("examples/hdfs_sample.log"),
    "BGL sample": Path("examples/bgl_sample.log"),
}


def _build_record(
    line_id: int,
    raw_log: str,
    dataset_name: Optional[str] = None,
) -> Dict:
    content = extract_log_content(raw_log, dataset_name=dataset_name)
    masked_content = mask_variables(content)

    return {
        "line_id": line_id,
        "raw_log": raw_log,
        "content": content,
        "masked_content": masked_content,
    }


def load_log_file(
    file_path: str | Path,
    max_lines: int = 200,
    dataset_name: Optional[str] = None,
    start_line: int = 1,
) -> List[Dict]:
    """
    Load part of a local .log file.

    This function streams the file line by line, so it can read a selected
    segment from a large local log file without loading the whole file.
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"Log file not found: {file_path}")

    start_line = max(1, int(start_line))
    max_lines = max(1, int(max_lines))

    records = []
    loaded = 0

    with file_path.open("r", encoding="utf-8", errors="ignore") as f:
        for physical_line_id, line in enumerate(f, start=1):
            if physical_line_id < start_line:
                continue

            if loaded >= max_lines:
                break

            raw_log = line.rstrip("\n")
            if not raw_log.strip():
                continue

            records.append(
                _build_record(
                    line_id=physical_line_id,
                    raw_log=raw_log,
                    dataset_name=dataset_name,
                )
            )
            loaded += 1

    return records


def load_log_lines(
    lines: Iterable[str],
    max_lines: int = 200,
    dataset_name: Optional[str] = None,
    start_line: int = 1,
) -> List[Dict]:
    """
    Load logs from an iterable of text lines, such as Streamlit uploaded file.
    """
    start_line = max(1, int(start_line))
    max_lines = max(1, int(max_lines))

    records = []
    loaded = 0

    for physical_line_id, line in enumerate(lines, start=1):
        if physical_line_id < start_line:
            continue

        if loaded >= max_lines:
            break

        raw_log = str(line).rstrip("\n")
        if not raw_log.strip():
            continue

        records.append(
            _build_record(
                line_id=physical_line_id,
                raw_log=raw_log,
                dataset_name=dataset_name,
            )
        )
        loaded += 1

    return records


def records_to_dataframe(records: List[Dict]) -> pd.DataFrame:
    return pd.DataFrame(records)