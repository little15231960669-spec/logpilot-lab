import time
from typing import List, Dict, Any

import pandas as pd
from drain3 import TemplateMiner
from drain3.template_miner_config import TemplateMinerConfig


class DrainLogParser:
    """
    Drain baseline parser.

    This parser supports parsing different fields:
    - raw_log: full log line
    - content: extracted message body
    - masked_content: message body after regex variable masking

    The final displayed template is the final cluster template after all logs
    have been processed, instead of the intermediate online template.
    """

    def __init__(
        self,
        similarity_threshold: float = 0.4,
        depth: int = 4,
        max_children: int = 100,
        max_clusters: int = 1000,
    ):
        config = TemplateMinerConfig()
        config.profiling_enabled = False

        config.drain_sim_th = similarity_threshold
        config.drain_depth = depth
        config.drain_max_children = max_children
        config.drain_max_clusters = max_clusters

        self.template_miner = TemplateMiner(config=config)

    def parse(
        self,
        records: List[Dict[str, Any]],
        text_field: str = "masked_content",
    ) -> pd.DataFrame:
        rows = []

        # Track the latest template for each cluster.
        # Drain is an online parser, so the template of a cluster can change
        # when more similar log messages arrive.
        cluster_template_map = {}
        cluster_size_map = {}

        for record in records:
            line_id = record["line_id"]
            raw_log = record.get("raw_log", "")
            content = record.get("content", "")
            masked_content = record.get("masked_content", "")

            log_text = record.get(text_field) or content or raw_log

            start_time = time.perf_counter()
            result = self.template_miner.add_log_message(log_text)
            latency_ms = (time.perf_counter() - start_time) * 1000

            cluster_id = result.get("cluster_id")
            online_template = (
                result.get("template_mined")
                or result.get("template")
                or log_text
            )

            if cluster_id is not None:
                cluster_template_map[cluster_id] = online_template
                cluster_size_map[cluster_id] = result.get("cluster_size")

            rows.append(
                {
                    "line_id": line_id,
                    "raw_log": raw_log,
                    "content": content,
                    "masked_content": masked_content,
                    "parsed_text": log_text,
                    "parser": "Drain",
                    "online_template": online_template,
                    "cluster_id": cluster_id,
                    "online_cluster_size": result.get("cluster_size"),
                    "change_type": result.get("change_type"),
                    "latency_ms": round(latency_ms, 4),
                }
            )

        df = pd.DataFrame(rows)

        if df.empty:
            return df

        # Replace intermediate templates with final cluster templates.
        df["template"] = df.apply(
            lambda row: cluster_template_map.get(
                row["cluster_id"],
                row["online_template"],
            ),
            axis=1,
        )

        df["final_cluster_size"] = df["cluster_id"].map(cluster_size_map)

        # Reorder columns for readability.
        columns = [
            "line_id",
            "raw_log",
            "content",
            "masked_content",
            "parsed_text",
            "parser",
            "template",
            "cluster_id",
            "final_cluster_size",
            "online_template",
            "online_cluster_size",
            "change_type",
            "latency_ms",
        ]

        return df[columns]