import json
from typing import List, Dict, Any, Optional

import pandas as pd

from logpilot.parsers.drain_parser import DrainLogParser
from logpilot.parsers.llm_parser import LLMLogParser
from logpilot.llm.base import BaseLLMBackend


class HybridLogParser:
    """
    Hybrid parser: Drain first, LLM review second.

    Product logic:
    - Drain handles all logs efficiently.
    - Only risky or uncertain logs are sent to LLM review.
    - LLM output is treated as a recommendation, not an unconditional replacement.
    """

    def __init__(
        self,
        llm_backend: BaseLLMBackend,
        drain_similarity_threshold: float = 0.4,
        drain_depth: int = 4,
        drain_max_clusters: int = 1000,
        llm_batch_size: int = 8,
        review_top_k: int = 10,
        risk_threshold: float = 0.5,
        small_cluster_threshold: int = 1,
        high_wildcard_threshold: float = 0.5,
    ):
        self.llm_backend = llm_backend
        self.drain_similarity_threshold = drain_similarity_threshold
        self.drain_depth = drain_depth
        self.drain_max_clusters = drain_max_clusters
        self.llm_batch_size = llm_batch_size
        self.review_top_k = review_top_k
        self.risk_threshold = risk_threshold
        self.small_cluster_threshold = small_cluster_threshold
        self.high_wildcard_threshold = high_wildcard_threshold

    def _wildcard_ratio(self, template: str) -> float:
        if not template:
            return 0.0

        tokens = str(template).split()
        if not tokens:
            return 0.0

        wildcard_count = sum(1 for token in tokens if "<*>" in token)
        return wildcard_count / len(tokens)

    def _compute_risk(self, row: pd.Series) -> tuple[float, str]:
        """
        Compute a product-level risk score for deciding whether a log should
        enter LLM review.

        This is intentionally explainable and rule-based for MVP.
        """
        score = 0.0
        reasons = []

        cluster_size = row.get("final_cluster_size", 0)
        change_type = str(row.get("change_type", "")).lower()
        template = str(row.get("template", ""))
        parsed_text = str(row.get("parsed_text", ""))

        wildcard_ratio = self._wildcard_ratio(template)

        if pd.isna(cluster_size):
            cluster_size = 0

        if int(cluster_size) <= self.small_cluster_threshold:
            score += 0.4
            reasons.append("small_cluster")

        if "created" in change_type or "new" in change_type:
            score += 0.3
            reasons.append("new_cluster")

        if wildcard_ratio >= self.high_wildcard_threshold:
            score += 0.2
            reasons.append("high_wildcard_ratio")

        if template.strip() == parsed_text.strip():
            score += 0.3
            reasons.append("not_generalized")

        if not reasons:
            reasons.append("low_risk")

        return round(min(score, 1.0), 4), ",".join(reasons)

    def parse(
        self,
        records: List[Dict[str, Any]],
        drain_text_field: str = "masked_content",
        llm_text_field: str = "content",
    ) -> pd.DataFrame:
        # Step 1: Drain parses all logs.
        drain_parser = DrainLogParser(
            similarity_threshold=self.drain_similarity_threshold,
            depth=self.drain_depth,
            max_clusters=self.drain_max_clusters,
        )

        drain_df = drain_parser.parse(records, text_field=drain_text_field)

        if drain_df.empty:
            return drain_df

        # Step 2: Compute review risk.
        risk_results = drain_df.apply(self._compute_risk, axis=1)
        drain_df["risk_score"] = [item[0] for item in risk_results]
        drain_df["risk_reason"] = [item[1] for item in risk_results]

        # Step 3: Select risky logs for LLM review.
        review_candidates = (
            drain_df[drain_df["risk_score"] >= self.risk_threshold]
            .sort_values("risk_score", ascending=False)
            .head(self.review_top_k)
        )

        review_line_ids = set(review_candidates["line_id"].tolist())

        selected_records = [
            record for record in records
            if record["line_id"] in review_line_ids
        ]

        # Step 4: Run LLM review only on selected records.
        llm_review_df = pd.DataFrame()

        if selected_records:
            llm_parser = LLMLogParser(
                backend=self.llm_backend,
                batch_size=self.llm_batch_size,
            )
            llm_review_df = llm_parser.parse(
                selected_records,
                text_field=llm_text_field,
            )

        llm_map = {}

        if not llm_review_df.empty:
            for _, row in llm_review_df.iterrows():
                llm_map[row["line_id"]] = row.to_dict()

        # Step 5: Merge Drain result and LLM recommendation.
        rows = []

        for _, row in drain_df.iterrows():
            line_id = row["line_id"]
            llm_item: Optional[Dict[str, Any]] = llm_map.get(line_id)

            drain_template = row["template"]
            llm_template = ""
            llm_variables = "[]"
            llm_confidence = 0.0
            llm_valid_json = False
            llm_latency_ms = 0.0
            llm_input_tokens_est = 0.0
            llm_output_tokens_est = 0.0

            used_llm_review = llm_item is not None

            if llm_item:
                llm_template = llm_item.get("template", "")
                llm_variables = llm_item.get("variables", "[]")
                llm_confidence = llm_item.get("confidence", 0.0)
                llm_valid_json = bool(llm_item.get("valid_json", False))
                llm_latency_ms = float(llm_item.get("latency_ms", 0.0))
                llm_input_tokens_est = float(llm_item.get("input_tokens_est", 0.0))
                llm_output_tokens_est = float(llm_item.get("output_tokens_est", 0.0))

            fallback_to_drain = not (
                used_llm_review
                and llm_valid_json
                and str(llm_template).strip()
            )

            if fallback_to_drain:
                recommended_template = drain_template
                hybrid_action = "keep_drain"
            else:
                recommended_template = llm_template
                hybrid_action = "llm_review_suggested"

            rows.append(
                {
                    "line_id": line_id,
                    "raw_log": row.get("raw_log", ""),
                    "content": row.get("content", ""),
                    "masked_content": row.get("masked_content", ""),
                    "parsed_text": row.get("parsed_text", ""),
                    "parser": f"Hybrid Drain + {self.llm_backend.name}",
                    "template": recommended_template,
                    "drain_template": drain_template,
                    "llm_template": llm_template,
                    "llm_variables": llm_variables,
                    "llm_confidence": llm_confidence,
                    "llm_valid_json": llm_valid_json,
                    "cluster_id": row.get("cluster_id"),
                    "final_cluster_size": row.get("final_cluster_size"),
                    "change_type": row.get("change_type"),
                    "risk_score": row.get("risk_score"),
                    "risk_reason": row.get("risk_reason"),
                    "used_llm_review": used_llm_review,
                    "fallback_to_drain": fallback_to_drain,
                    "hybrid_action": hybrid_action,
                    "drain_latency_ms": row.get("latency_ms", 0.0),
                    "llm_latency_ms": round(llm_latency_ms, 4),
                    "latency_ms": round(float(row.get("latency_ms", 0.0)) + llm_latency_ms, 4),
                    "input_tokens_est": llm_input_tokens_est,
                    "output_tokens_est": llm_output_tokens_est,
                }
            )

        return pd.DataFrame(rows)