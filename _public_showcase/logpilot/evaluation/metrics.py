from typing import Dict, Any

import pandas as pd


def summarize_parsing_result(df: pd.DataFrame) -> Dict[str, Any]:
    if df.empty:
        return {
            "total_logs": 0,
            "template_count": 0,
            "cluster_count": 0,
            "avg_latency_ms": 0.0,
            "total_latency_ms": 0.0,
            "valid_json_rate": None,
            "avg_confidence": None,
            "input_tokens_est": 0.0,
            "output_tokens_est": 0.0,
            "total_tokens_est": 0.0,
            "review_rate": None,
            "fallback_rate": None,
            "avg_risk_score": None,
        }

    total_logs = len(df)
    template_count = df["template"].nunique() if "template" in df.columns else 0
    cluster_count = df["cluster_id"].nunique() if "cluster_id" in df.columns else 0

    avg_latency_ms = (
        float(df["latency_ms"].mean())
        if "latency_ms" in df.columns
        else 0.0
    )

    total_latency_ms = (
        float(df["latency_ms"].sum())
        if "latency_ms" in df.columns
        else 0.0
    )

    valid_json_rate = None
    if "valid_json" in df.columns:
        valid_json_rate = round(float(df["valid_json"].mean()), 4)
    elif "llm_valid_json" in df.columns:
        reviewed_df = df[df.get("used_llm_review", False) == True]
        if not reviewed_df.empty:
            valid_json_rate = round(float(reviewed_df["llm_valid_json"].mean()), 4)

    avg_confidence = None
    if "confidence" in df.columns:
        avg_confidence = round(float(df["confidence"].mean()), 4)
    elif "llm_confidence" in df.columns:
        reviewed_df = df[df.get("used_llm_review", False) == True]
        if not reviewed_df.empty:
            avg_confidence = round(float(reviewed_df["llm_confidence"].mean()), 4)

    input_tokens_est = (
        float(df["input_tokens_est"].sum())
        if "input_tokens_est" in df.columns
        else 0.0
    )

    output_tokens_est = (
        float(df["output_tokens_est"].sum())
        if "output_tokens_est" in df.columns
        else 0.0
    )

    total_tokens_est = input_tokens_est + output_tokens_est

    review_rate = None
    if "used_llm_review" in df.columns:
        review_rate = round(float(df["used_llm_review"].mean()), 4)

    fallback_rate = None
    if "fallback_to_drain" in df.columns:
        fallback_rate = round(float(df["fallback_to_drain"].mean()), 4)

    avg_risk_score = None
    if "risk_score" in df.columns:
        avg_risk_score = round(float(df["risk_score"].mean()), 4)

    return {
        "total_logs": total_logs,
        "template_count": template_count,
        "cluster_count": cluster_count,
        "avg_latency_ms": round(avg_latency_ms, 4),
        "total_latency_ms": round(total_latency_ms, 4),
        "valid_json_rate": valid_json_rate,
        "avg_confidence": avg_confidence,
        "input_tokens_est": round(input_tokens_est, 2),
        "output_tokens_est": round(output_tokens_est, 2),
        "total_tokens_est": round(total_tokens_est, 2),
        "review_rate": review_rate,
        "fallback_rate": fallback_rate,
        "avg_risk_score": avg_risk_score,
    }