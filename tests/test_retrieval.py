"""Tests for hybrid retrieval helpers."""
from typing import Any

import pytest

from src.llm import retrieval


def test_escape_lucene() -> None:
    assert retrieval.escape_lucene('A+B && "C"') == r'A\+B \&\& \"C\"'


def test_rrf_merge() -> None:
    assert retrieval.rrf_merge([["a", "b"], ["b", "c"]], 10) == ["b", "a", "c"]


def test_hybrid_search_embed_failure_uses_full_text(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str, dict[str, Any]]] = []

    class FakeClient:
        def run_read(self, query: str, **params: Any) -> list[dict]:
            calls.append((query, params))
            if "fulltext.queryNodes" in query:
                return [{"arxiv_id": "2"}, {"arxiv_id": "1"}]
            if "UNWIND $ids" in query:
                assert params["ids"] == ["2", "1"]
                return [
                    {
                        "arxiv_id": "1",
                        "title": "One",
                        "url": "https://arxiv.org/abs/1",
                        "published_date": "2024",
                        "authors": ["A", "B"],
                        "citation_count": 3,
                        "num_references": 4,
                        "num_citing_papers": 5,
                    },
                    {
                        "arxiv_id": "2",
                        "title": "Two",
                        "url": "https://arxiv.org/abs/2",
                        "published_date": "2025",
                        "authors": ["C"],
                        "citation_count": 7,
                        "num_references": 8,
                        "num_citing_papers": 9,
                    },
                ]
            raise AssertionError("unexpected query")

    def fail_embed(text: str) -> list[float]:
        raise RuntimeError("embedding unavailable")

    monkeypatch.setattr(retrieval, "embed", fail_embed)

    results = retrieval.hybrid_search(FakeClient(), "graph RAG", "graph retrieval", k=2)

    assert [paper["arxiv_id"] for paper in results] == ["2", "1"]
    assert results[0]["authors"] == ["C"]
    assert len(calls) == 2
