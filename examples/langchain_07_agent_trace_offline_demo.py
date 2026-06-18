from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from logpilot.ai_framework.log_review_agent import (
    run_log_review_agent_offline_demo_with_trace,
)


TEMPLATE_CSV_PATH = "data/template_memory/hdfs_templates.csv"
SAMPLE_LOG = (
    "Receiving block blk_38865049064139660 src: /10.250.19.102:54106 "
    "dest: /10.250.19.102:50010"
)
USER_QUERY = "请复核这条日志的解析结果，检索相似历史模板，并给出最终建议。"
CANDIDATE_TEMPLATE = "Receiving block <*> src: <*> dest: <*>"


def main() -> None:
    try:
        _result, trace, saved_paths = run_log_review_agent_offline_demo_with_trace(
            user_query=USER_QUERY,
            log=SAMPLE_LOG,
            template_csv_path=TEMPLATE_CSV_PATH,
            candidate_template=CANDIDATE_TEMPLATE,
            k=3,
            save_outputs=True,
        )
    except Exception as exc:
        print(f"Offline agent trace demo failed: {exc}")
        return

    final_answer_summary = trace.final_answer.replace("\n", " ")[:240]
    print(f"run_id: {trace.run_id}")
    print(f"status: {trace.status}")
    print(f"success: {trace.success}")
    print(f"tool_error_count: {trace.tool_error_count}")
    print(f"selected action: {trace.selected_action}")
    print(f"saved JSON path: {saved_paths.get('json', '')}")
    print(f"saved Markdown path: {saved_paths.get('markdown', '')}")
    print(f"final answer summary: {final_answer_summary}")


if __name__ == "__main__":
    main()
