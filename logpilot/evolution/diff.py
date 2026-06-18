import re
from typing import Tuple, List, Dict, Set

import pandas as pd


def _normalize_template(template: str) -> str:
    if template is None:
        return ""
    text = str(template).strip()
    text = re.sub(r"\s+", " ", text)
    return text


def _tokenize_template(template: str) -> Set[str]:
    """
    Tokenize template for rough similarity computation.
    The wildcard <*> is kept because wildcard positions are useful.
    """
    template = _normalize_template(template)
    if not template:
        return set()

    return set(template.split())


def template_jaccard_similarity(template_a: str, template_b: str) -> float:
    tokens_a = _tokenize_template(template_a)
    tokens_b = _tokenize_template(template_b)

    if not tokens_a and not tokens_b:
        return 1.0
    if not tokens_a or not tokens_b:
        return 0.0

    return len(tokens_a & tokens_b) / len(tokens_a | tokens_b)


def build_template_table(parsed_df: pd.DataFrame, window_name: str) -> pd.DataFrame:
    """
    Build template-level table from parsed logs.
    """
    if parsed_df.empty:
        return pd.DataFrame(
            columns=[
                "window",
                "template",
                "count",
                "example_content",
            ]
        )

    df = parsed_df.copy()
    df["template"] = df["template"].map(_normalize_template)

    table = (
        df.groupby("template")
        .agg(
            count=("line_id", "count"),
            example_content=("content", "first"),
        )
        .reset_index()
        .sort_values("count", ascending=False)
    )

    table.insert(0, "window", window_name)
    return table


def analyze_template_evolution(
    old_df: pd.DataFrame,
    new_df: pd.DataFrame,
    old_window_name: str = "Window A",
    new_window_name: str = "Window B",
    rewrite_similarity_threshold: float = 0.5,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Analyze template evolution between two parsed windows.

    Types:
    - stable: exact same template appears in both windows
    - new: template appears only in new window
    - disappeared: template appears only in old window
    - rewritten_candidate: old/new templates are not exact same but have high token similarity
    """
    old_table = build_template_table(old_df, old_window_name)
    new_table = build_template_table(new_df, new_window_name)

    old_templates = set(old_table["template"])
    new_templates = set(new_table["template"])

    stable_templates = old_templates & new_templates
    old_only = old_templates - new_templates
    new_only = new_templates - old_templates

    detail_rows: List[Dict] = []

    old_count_map = dict(zip(old_table["template"], old_table["count"]))
    new_count_map = dict(zip(new_table["template"], new_table["count"]))
    old_example_map = dict(zip(old_table["template"], old_table["example_content"]))
    new_example_map = dict(zip(new_table["template"], new_table["example_content"]))

    for template in sorted(stable_templates):
        detail_rows.append(
            {
                "evolution_type": "stable",
                "old_template": template,
                "new_template": template,
                "similarity": 1.0,
                "old_count": old_count_map.get(template, 0),
                "new_count": new_count_map.get(template, 0),
                "old_example": old_example_map.get(template, ""),
                "new_example": new_example_map.get(template, ""),
                "explanation": "Template appears in both windows.",
            }
        )

    # Find rewritten candidates between old-only and new-only templates.
    matched_old = set()
    matched_new = set()

    candidates = []

    for old_template in old_only:
        for new_template in new_only:
            sim = template_jaccard_similarity(old_template, new_template)
            if sim >= rewrite_similarity_threshold:
                candidates.append((sim, old_template, new_template))

    candidates.sort(reverse=True, key=lambda x: x[0])

    for sim, old_template, new_template in candidates:
        if old_template in matched_old or new_template in matched_new:
            continue

        matched_old.add(old_template)
        matched_new.add(new_template)

        detail_rows.append(
            {
                "evolution_type": "rewritten_candidate",
                "old_template": old_template,
                "new_template": new_template,
                "similarity": round(sim, 4),
                "old_count": old_count_map.get(old_template, 0),
                "new_count": new_count_map.get(new_template, 0),
                "old_example": old_example_map.get(old_template, ""),
                "new_example": new_example_map.get(new_template, ""),
                "explanation": "Templates are not exactly same, but have high token-level similarity.",
            }
        )

    for template in sorted(old_only - matched_old):
        detail_rows.append(
            {
                "evolution_type": "disappeared",
                "old_template": template,
                "new_template": "",
                "similarity": 0.0,
                "old_count": old_count_map.get(template, 0),
                "new_count": 0,
                "old_example": old_example_map.get(template, ""),
                "new_example": "",
                "explanation": "Template appears only in the old window.",
            }
        )

    for template in sorted(new_only - matched_new):
        detail_rows.append(
            {
                "evolution_type": "new",
                "old_template": "",
                "new_template": template,
                "similarity": 0.0,
                "old_count": 0,
                "new_count": new_count_map.get(template, 0),
                "old_example": "",
                "new_example": new_example_map.get(template, ""),
                "explanation": "Template appears only in the new window.",
            }
        )

    detail_df = pd.DataFrame(detail_rows)

    if detail_df.empty:
        summary_df = pd.DataFrame(
            columns=[
                "evolution_type",
                "count",
            ]
        )
    else:
        summary_df = (
            detail_df.groupby("evolution_type")
            .size()
            .reset_index(name="count")
            .sort_values("count", ascending=False)
        )

    return summary_df, detail_df