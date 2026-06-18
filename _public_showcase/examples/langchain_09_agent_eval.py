from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from logpilot.ai_framework.agent_eval import (
    compute_agent_eval_summary,
    evaluate_agent_cases_offline,
    save_agent_eval_results_csv,
    save_agent_eval_summary_json,
)


EVAL_CSV_PATH = PROJECT_ROOT / "data" / "eval" / "agent_eval_cases.csv"
TEMPLATE_CSV_PATH = PROJECT_ROOT / "data" / "template_memory" / "hdfs_templates.csv"


def main() -> None:
    results = evaluate_agent_cases_offline(
        str(EVAL_CSV_PATH),
        str(TEMPLATE_CSV_PATH),
        save_outputs=True,
    )
    summary = compute_agent_eval_summary(results)
    result_csv_path = save_agent_eval_results_csv(results)
    summary_json_path = save_agent_eval_summary_json(summary)

    print(f"total_cases: {summary['total_cases']}")
    print(f"action_accuracy: {summary['action_accuracy']}")
    print(f"template_exact_match_rate: {summary['template_exact_match_rate']}")
    print(f"retrieval_top1_hit_rate: {summary['retrieval_top1_hit_rate']}")
    print(f"final_decision_accuracy: {summary['final_decision_accuracy']}")
    print(f"success_status_rate: {summary['success_status_rate']}")
    print(f"output result csv path: {result_csv_path}")
    print(f"output summary json path: {summary_json_path}")


if __name__ == "__main__":
    main()

