from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from logpilot.ai_framework.log_review_agent import plan_tools_for_agent_task


TASKS = [
    "High-confidence parse review",
    "Fast baseline parse",
    "Compare parser tools",
    "Find risky templates",
    "Explain parsing results",
]

LOGS = [
    "Receiving block blk_38865049064139660 src: /10.250.19.102:54106 dest: /10.250.19.102:50010",
    "Served block blk_777 to /10.250.19.103:50010",
]


def main() -> None:
    for task in TASKS:
        plan = plan_tools_for_agent_task(task, LOGS, offline=True)
        print(f"task: {task}")
        print(f"action: {plan['action']}")
        print(f"reason: {plan['reason']}")
        print(f"tool_sequence: {plan['tool_sequence']}")
        print()


if __name__ == "__main__":
    main()
