from __future__ import annotations

from pathlib import Path

from logpilot.ai_framework.template_memory import (
    build_template_memory_context,
    lexical_similarity,
    load_template_records,
    normalize_for_retrieval,
    retrieve_similar_templates,
)


def _write_template_csv(path: Path) -> None:
    path.write_text(
        "template,description\n"
        "Receiving block <*> src: <*> dest: <*>,DataNode receives a block.\n"
        "Deleting block <*> file <*>,DataNode deletes a block file.\n"
        "Verification succeeded for <*>,Block verification succeeds.\n",
        encoding="utf-8",
    )


def test_normalize_for_retrieval_returns_tokens() -> None:
    tokens = normalize_for_retrieval(
        "Receiving block blk_123 src: /10.0.0.1:50010 dest: <*>"
    )

    assert tokens
    assert "receiving" in tokens
    assert "block" in tokens
    assert "blk_123" in tokens
    assert "wildcard" in tokens


def test_lexical_similarity_scores_similar_text_higher() -> None:
    query = "Receiving block blk_123 src: /10.0.0.1:50010 dest: /10.0.0.2:50010"
    similar = "Receiving block <*> src: <*> dest: <*>"
    unrelated = "Verification succeeded for <*>"

    assert lexical_similarity(query, similar) > lexical_similarity(query, unrelated)


def test_load_template_records_reads_temp_csv(tmp_path: Path) -> None:
    csv_path = tmp_path / "templates.csv"
    _write_template_csv(csv_path)

    records = load_template_records(str(csv_path))

    assert len(records) == 3
    assert records[0].template == "Receiving block <*> src: <*> dest: <*>"
    assert records[0].description == "DataNode receives a block."


def test_retrieve_similar_templates_returns_top_k(tmp_path: Path) -> None:
    csv_path = tmp_path / "templates.csv"
    _write_template_csv(csv_path)

    retrieved = retrieve_similar_templates(
        "Receiving block blk_123 src: /10.0.0.1:50010 dest: /10.0.0.2:50010",
        str(csv_path),
        k=2,
    )

    assert len(retrieved) == 2
    assert retrieved[0].template == "Receiving block <*> src: <*> dest: <*>"
    assert retrieved[0].score >= retrieved[1].score


def test_build_template_memory_context_contains_expected_fields(tmp_path: Path) -> None:
    csv_path = tmp_path / "templates.csv"
    _write_template_csv(csv_path)
    retrieved = retrieve_similar_templates(
        "Receiving block blk_123 src: /10.0.0.1:50010 dest: /10.0.0.2:50010",
        str(csv_path),
        k=1,
    )

    context = build_template_memory_context(retrieved)

    assert "template:" in context
    assert "description:" in context
    assert "score:" in context
