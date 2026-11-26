"""Write operations (always using MERGE for idempotency) and statistics."""
from src.database.neo4j_client import Neo4jClient


def get_existing_ids(client: Neo4jClient, ids: list[str]) -> set[str]:
    """Return arXiv IDs that already exist as complete papers (not stubs)."""
    if not ids:
        return set()
    rows = client.run_read(
        """
        UNWIND $ids AS arxiv_id
        MATCH (p:Paper {arxiv_id: arxiv_id})
        WHERE p.stub = false
        RETURN p.arxiv_id AS arxiv_id
        """,
        ids=ids,
    )
    return {row["arxiv_id"] for row in rows}


def upsert_paper(client: Neo4jClient, paper: dict) -> None:
    """Create or update a paper, its authors, and its categories.

    `paper`: arxiv_id, title, abstract, published_date, url,
             authors (list[str]), categories (list[str]), citation_count (optional).
    """
    # stub=false means this paper is complete, even if it was previously
    # created as a cited reference by another paper.
    client.run_write(
        """
        MERGE (p:Paper {arxiv_id: $arxiv_id})
        SET p.title = $title, p.abstract = $abstract,
            p.published_date = $published_date, p.url = $url,
            p.citation_count = $citation_count, p.stub = false
        WITH p
        CALL {
            WITH p
            UNWIND $authors AS author_name
            MERGE (a:Author {name: author_name})
            MERGE (a)-[:WROTE]->(p)
            RETURN count(*) AS authors_processed
        }
        WITH p
        CALL {
            WITH p
            UNWIND $categories AS code
            MERGE (c:Category {code: code})
            MERGE (p)-[:BELONGS_TO]->(c)
            RETURN count(*) AS categories_processed
        }
        RETURN p
        """,
        arxiv_id=paper["arxiv_id"], title=paper["title"], abstract=paper["abstract"],
        published_date=paper["published_date"], url=paper["url"],
        citation_count=paper.get("citation_count"), authors=paper["authors"],
        categories=paper["categories"],
    )


def add_citations(
    client: Neo4jClient, source_arxiv_id: str, targets: list[dict]
) -> None:
    """Add citation edges, creating unknown targets as stubs."""
    if not targets:
        return
    client.run_write(
        """
        UNWIND $targets AS t
        MATCH (s:Paper {arxiv_id: $source_arxiv_id})
        MERGE (c:Paper {arxiv_id: t.arxiv_id})
          ON CREATE SET c.stub = true, c.title = t.title,
                        c.url = 'https://arxiv.org/abs/' + t.arxiv_id
        MERGE (s)-[:CITES]->(c)
        """,
        source_arxiv_id=source_arxiv_id, targets=targets,
    )


def get_graph_stats(client: Neo4jClient) -> dict:
    """Return counters displayed in the sidebar."""
    return client.run_read(
        """
        CALL { MATCH (p:Paper) WHERE p.stub = false RETURN count(p) AS papers }
        CALL { MATCH (p:Paper) WHERE p.stub = true RETURN count(p) AS stubs }
        CALL { MATCH (a:Author) RETURN count(a) AS authors }
        CALL { MATCH ()-[r:CITES]->() RETURN count(r) AS citations }
        RETURN papers, stubs, authors, citations
        """
    )[0]
