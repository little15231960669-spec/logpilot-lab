from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from logpilot.ai_framework.tool_registry import (
    build_tool_context_for_agent,
    get_tool_specs,
    render_tool_registry_markdown,
)


SAMPLE_OUTPUT_PATH = PROJECT_ROOT / "docs" / "samples" / "tool_registry_overview.md"


def main() -> None:
    specs = get_tool_specs()

    print("registered tools:")
    for spec in specs:
        print(f"- {spec.name}")

    print("\ntool details:")
    for spec in specs:
        print(f"\n{spec.name}")
        print(f"  description: {spec.description}")
        print(f"  risk_level: {spec.risk_level}")
        print(f"  requires_llm: {spec.requires_llm}")
        print(f"  deterministic: {spec.deterministic}")
        print(f"  suitable_actions: {spec.suitable_actions}")

    context = build_tool_context_for_agent()
    print("\nagent tool context summary:")
    print(context[:1000])

    SAMPLE_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    SAMPLE_OUTPUT_PATH.write_text(render_tool_registry_markdown(), encoding="utf-8")
    print(f"\nsaved markdown: {SAMPLE_OUTPUT_PATH.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()

