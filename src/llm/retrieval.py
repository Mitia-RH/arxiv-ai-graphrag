"""Hybrid full-text and vector retrieval for papers."""
from typing import Any

from src.llm.llm_client import embed


def escape_lucene(text: str) -> str:
    """Escape Lucene special characters for a plain keyword query."""
    special = set('+ - && || ! ( ) { } [ ] ^ " ~ * ? : \\ /'.replace(" ", ""))
    return "".join("\\" + char if char in special else char for char in text)


def rrf_merge(rankings: list[list[str]], k: int) -> list[str]:
    """Merge ranked lists using reciprocal rank fusion."""
    if k <= 0:
        return []

    scores: dict[str, float] = {}
    first_seen: dict[str, int] = {}
    position = 0
    for ranking in rankings:
        for rank, item in enumerate(ranking, start=1):
            scores[item] = scores.get(item, 0.0) + 1.0 / (60 + rank)
            if item not in first_seen:
                first_seen[item] = position
                position += 1

    return sorted(scores, key=lambda item: (-scores[item], first_seen[item]))[:k]


def hybrid_search(
    client: Any, keywords: str, question_en: str, k: int = 10
) -> list[dict]:
    """Search papers by full text and optional vector similarity."""
    full_text_query = """
        CALL db.index.fulltext.queryNodes('paper_text', $q)
        YIELD node
        WHERE node.stub = false
        RETURN node.arxiv_id AS arxiv_id
        LIMIT $k
    """
    try:
        full_text_rows = client.run_read(full_text_query, q=keywords, k=k)
    except Exception:
        full_text_rows = client.run_read(
            full_text_query, q=escape_lucene(keywords), k=k
        )

    full_text_ids = [
        row.get("arxiv_id") or row.get("node", {}).get("arxiv_id")
        for row in full_text_rows
        if row.get("arxiv_id") or row.get("node", {}).get("arxiv_id")
    ]

    vector_ids: list[str] = []
    try:
        vector = embed(question_en)
        vector_rows = client.run_read(
            """
            CALL db.index.vector.queryNodes('paper_embedding', $k, $v)
            YIELD node
            WHERE node.stub = false
            RETURN node.arxiv_id AS arxiv_id
            """,
            k=k,
            v=vector,
        )
        vector_ids = [
            row.get("arxiv_id") or row.get("node", {}).get("arxiv_id")
            for row in vector_rows
            if row.get("arxiv_id") or row.get("node", {}).get("arxiv_id")
        ]
    except Exception:
        vector_ids = []

    rankings = [full_text_ids]
    if vector_ids:
        rankings.append(vector_ids)
    merged_ids = rrf_merge(rankings, k)
    if not merged_ids:
        return []

    rows = client.run_read(
        """
        UNWIND $ids AS arxiv_id
        MATCH (p:Paper {arxiv_id: arxiv_id})
        WHERE p.stub = false
        OPTIONAL MATCH (a:Author)-[:WROTE]->(p)
        WITH arxiv_id, p, collect(DISTINCT a.name)[0..4] AS authors
        OPTIONAL MATCH (p)-[:CITES]->(ref:Paper)
        WHERE ref.stub = false
        WITH arxiv_id, p, authors, count(DISTINCT ref) AS num_references
        OPTIONAL MATCH (citing:Paper)-[:CITES]->(p)
        WHERE citing.stub = false
        RETURN p.arxiv_id AS arxiv_id,
               p.title AS title,
               p.url AS url,
               p.published_date AS published_date,
               authors,
               p.citation_count AS citation_count,
               num_references,
               count(DISTINCT citing) AS num_citing_papers
        """,
        ids=merged_ids,
    )

    rows_by_id = {
        row["arxiv_id"]: row
        for row in rows
        if row.get("arxiv_id")
    }
    results: list[dict] = []
    for arxiv_id in merged_ids:
        row = rows_by_id.get(arxiv_id)
        if not row:
            continue
        results.append({
            "arxiv_id": arxiv_id,
            "title": row.get("title"),
            "url": row.get("url"),
            "published_date": row.get("published_date", row.get("date")),
            "authors": list(row.get("authors") or [])[:4],
            "citation_count": row.get("citation_count"),
            "num_references": row.get(
                "num_references",
                row.get("reference_count", row.get("references_count", 0)),
            ),
            "num_citing_papers": row.get(
                "num_citing_papers",
                row.get("citing_count", row.get("cited_by_count", 0)),
            ),
        })
    return results
