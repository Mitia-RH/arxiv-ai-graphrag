"""Basic tests for ID cleanup, Cypher cleanup, and read-only query safety.
Run with: python -m pytest
"""
from typing import Any

import pytest

from src.database.neo4j_client import Neo4jClient
from src.database.schema import CONSTRAINTS, INDEXES, create_schema
from src.ingestion.arxiv_fetcher import clean_arxiv_id
from src.ingestion.citation_enricher import arxiv_id_from_doi
from src.ingestion import pipeline
from src.database.repository import add_citations, upsert_paper
from src.llm.graph_rag import CYPHER_SYSTEM_PROMPT, clean_cypher, is_read_only
from src.llm import llm_client


def test_create_schema_runs_constraints_and_indexes():
    calls = []

    class FakeClient:
        def run_write(self, query: str, **params: Any) -> list[dict]:
            calls.append(query)
            return []

    create_schema(FakeClient())

    expected_statements = CONSTRAINTS + INDEXES
    assert calls == expected_statements
    assert all("IF NOT EXISTS" in statement for statement in calls)


def test_clean_arxiv_id():
    assert clean_arxiv_id("2401.12345v2") == "2401.12345"
    assert clean_arxiv_id("http://arxiv.org/abs/2401.12345v1") == "2401.12345"
    assert clean_arxiv_id("2401.12345") == "2401.12345"


def test_arxiv_id_from_doi():
    assert arxiv_id_from_doi("https://doi.org/10.48550/arXiv.1706.03762") == "1706.03762"
    assert arxiv_id_from_doi("https://doi.org/10.1000/xyz") is None
    assert arxiv_id_from_doi(None) is None


def test_clean_cypher_removes_markdown():
    raw = "```cypher\nMATCH (p:Paper) RETURN p.title LIMIT 5;\n```"
    assert clean_cypher(raw) == "MATCH (p:Paper) RETURN p.title LIMIT 5"


def test_clean_cypher_removes_think_block():
    raw = "<think>reasoning</think>MATCH (p:Paper) RETURN p.title LIMIT 5"
    assert clean_cypher(raw) == "MATCH (p:Paper) RETURN p.title LIMIT 5"


def test_clean_cypher_removes_unbalanced_think_close():
    raw = "reasoning that lacks an opening tag</think>MATCH (p:Paper) RETURN p.title LIMIT 5"
    assert clean_cypher(raw) == "MATCH (p:Paper) RETURN p.title LIMIT 5"


def test_clean_cypher_extracts_query_after_preface():
    raw = "Here is the query:\nMATCH (p:Paper) RETURN p.title LIMIT 5;"
    assert clean_cypher(raw) == "MATCH (p:Paper) RETURN p.title LIMIT 5"


def test_chat_disables_thinking_and_strips_response(monkeypatch):
    captured = {}

    class FakeResponse:
        def raise_for_status(self):
            pass

        def json(self):
            return {"message": {"content": "reasoning</think>MATCH (p) RETURN p"}}

    def fake_post(url, *, json, timeout):
        captured["payload"] = json
        return FakeResponse()

    monkeypatch.setattr(llm_client.requests, "post", fake_post)

    assert llm_client.chat("system", "user") == "MATCH (p) RETURN p"
    assert captured["payload"]["think"] is False
    assert captured["payload"]["messages"][0]["content"] == "system\n/no_think"


def test_read_only_accepts_reads():
    assert is_read_only("MATCH (p:Paper) RETURN p.title LIMIT 5")
    # 'set' in a string literal must not trigger a false positive.
    assert is_read_only("MATCH (p:Paper) WHERE p.title CONTAINS 'data set' RETURN p.title")


def test_read_only_rejects_writes():
    assert not is_read_only("MATCH (p:Paper) DETACH DELETE p")
    assert not is_read_only("MATCH (p:Paper) SET p.title = 'x' RETURN p")
    assert not is_read_only("CREATE (p:Paper {arxiv_id: '1'})")
    assert not is_read_only("MATCH (p:Paper) RETURN p; MATCH (a) DELETE a")
    assert not is_read_only("Here is the query: MATCH (p) RETURN p")


def test_cypher_prompt_examples_are_read_only():
    queries = [
        line for line in CYPHER_SYSTEM_PROMPT.splitlines() if line.startswith("MATCH")
    ]

    assert queries
    assert all(is_read_only(query) for query in queries)


@pytest.mark.parametrize(
    ("force", "expected_enriched"),
    [(False, ["stub"]), (True, ["existing", "stub"])],
)
def test_pipeline_skips_existing_papers_but_not_stubs_unless_forced(
    monkeypatch, force, expected_enriched
):
    papers = [
        {"arxiv_id": "existing", "title": "Already ingested", "references": []},
        {"arxiv_id": "stub", "title": "Only a stub", "references": []},
    ]
    enriched_ids = []
    read_calls = []

    class FakeClient:
        def __enter__(self):
            return self

        def __exit__(self, *_):
            pass

        def run_read(self, query, **params):
            read_calls.append(True)
            assert params["ids"] == ["existing", "stub"]
            return [{"arxiv_id": "existing"}]

    monkeypatch.setattr(pipeline, "fetch_papers", lambda limit, offset=0: papers)
    monkeypatch.setattr(pipeline, "Neo4jClient", FakeClient)
    monkeypatch.setattr(pipeline, "create_schema", lambda client: None)
    monkeypatch.setattr(
        pipeline,
        "enrich_paper",
        lambda paper: enriched_ids.append(paper["arxiv_id"]) or paper,
    )
    monkeypatch.setattr(pipeline, "upsert_paper", lambda client, paper: None)
    monkeypatch.setattr(pipeline, "add_citations", lambda client, source, targets: None)

    assert pipeline.run(limit=2, force=force) == len(expected_enriched)
    assert enriched_ids == expected_enriched
    assert len(read_calls) == (0 if force else 1)


def test_add_citations_batches_targets_and_skips_empty_list():
    calls = []

    class FakeClient(Neo4jClient):
        def __init__(self) -> None:
            pass

        def run_write(self, query: str, **params: Any) -> list[dict]:
            calls.append((query, params))
            return []

    client = FakeClient()
    targets: list[dict] = [
        {"arxiv_id": "target-1", "title": "First"},
        {"arxiv_id": "target-2", "title": "Second"},
    ]

    add_citations(client, "source", targets)

    assert len(calls) == 1
    query, params = calls[0]
    assert "UNWIND $targets AS t" in query
    assert params == {"source_arxiv_id": "source", "targets": targets}

    empty_targets: list[dict] = []
    add_citations(client, "source", empty_targets)
    assert len(calls) == 1


def test_upsert_paper_uses_one_write_query():
    calls = []

    class FakeClient(Neo4jClient):
        def __init__(self) -> None:
            pass

        def run_write(self, query: str, **params: Any) -> list[dict]:
            calls.append((query, params))
            return []

    paper = {
        "arxiv_id": "paper",
        "title": "Title",
        "abstract": "Abstract",
        "published_date": "2024-01-01",
        "url": "https://arxiv.org/abs/paper",
        "authors": ["Author"],
        "categories": ["cs.AI"],
        "citation_count": 3,
    }

    upsert_paper(FakeClient(), paper)

    assert len(calls) == 1
    query, params = calls[0]
    assert query.count("CALL {") == 2
    assert "p.stub = false" in query
    assert params["authors"] == paper["authors"]
    assert params["categories"] == paper["categories"]
    assert params["citation_count"] == 3
