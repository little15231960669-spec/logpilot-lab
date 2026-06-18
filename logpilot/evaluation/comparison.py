from typing import Dict, Any

import pandas as pd


def summary_to_row(method_name: str, summary: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert a metric summary dict into a row for method comparison.
    """
    return {
        "method": method_name,
        "total_logs": summary.get("total_logs", 0),
        "template_count": summary.get("template_count", 0),
        "cluster_count": summary.get("cluster_count", 0),
        "avg_latency_ms": summary.get("avg_latency_ms", 0.0),
        "total_latency_ms": summary.get("total_latency_ms", 0.0),
        "valid_json_rate": summary.get("valid_json_rate"),
        "avg_confidence": summary.get("avg_confidence"),
        "input_tokens_est": summary.get("input_tokens_est", 0.0),
        "output_tokens_est": summary.get("output_tokens_est", 0.0),
        "total_tokens_est": summary.get("total_tokens_est", 0.0),
    }


def compare_line_level_templates(
    drain_df: pd.DataFrame,
    llm_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compare Drain and LLM templates at line level.
    """
    drain_part = drain_df[
        [
            "line_id",
            "content",
            "template",
            "latency_ms",
        ]
    ].rename(
        columns={
            "template": "drain_template",
            "latency_ms": "drain_latency_ms",
        }
    )

    llm_part = llm_df[
        [
            "line_id",
            "template",
            "variables",
            "confidence",
            "valid_json",
            "latency_ms",
        ]
    ].rename(
        columns={
            "template": "llm_template",
            "latency_ms": "llm_latency_ms",
        }
    )

    merged = drain_part.merge(llm_part, on="line_id", how="outer")

    merged["same_template"] = (
        merged["drain_template"].fillna("")
        == merged["llm_template"].fillna("")
    )

    return merged


def compute_template_agreement(compare_df: pd.DataFrame) -> float:
    """
    Compute simple exact template agreement between two methods.
    """
    if compare_df.empty or "same_template" not in compare_df.columns:
        return 0.0

    return round(float(compare_df["same_template"].mean()), 4)