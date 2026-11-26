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

    `paper`: arxiv_id, title, abstract, published_date, url, authors, categories,
             optional citation_count and embedding.
    """
    # stub=false means this paper is complete, even if it was previously
    # created as a cited reference by another paper.
    embedding = paper.get("embedding")
    embedding_set = ", p.embedding = $embedding" if embedding is not None else ""
    query = """
        MERGE (p:Paper {arxiv_id: $arxiv_id})
        SET p.title = $title, p.abstract = $abstract,
            p.published_date = $published_date, p.url = $url,
            p.citation_count = $citation_count, p.stub = false""" + embedding_set + """
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
        """
    params = {
        "arxiv_id": paper["arxiv_id"],
        "title": paper["title"],
        "abstract": paper["abstract"],
        "published_date": paper["published_date"],
        "url": paper["url"],
        "citation_count": paper.get("citation_count"),
        "authors": paper["authors"],
        "categories": paper["categories"],
    }
    if embedding is not None:
        params["embedding"] = embedding
    client.run_write(query, **params)


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


def get_top_pagerank(client: Neo4jClient, n: int) -> list[dict]:
    """Return the `n` papers with the highest computed PageRank scores."""
    if n < 0:
        raise ValueError("n must be non-negative")
    return client.run_read(
        """
        MATCH (p:Paper)
        WHERE p.pagerank IS NOT NULL
        RETURN p.arxiv_id AS arxiv_id, p.title AS title, p.pagerank AS pagerank
        ORDER BY p.pagerank DESC, p.arxiv_id
        LIMIT $n
        """,
        n=n,
    )


def get_community_summary(client: Neo4jClient) -> list[dict]:
    """Return each computed community's size and three most-cited paper titles."""
    return client.run_read(
        """
        MATCH (p:Paper)
        WHERE p.community IS NOT NULL
        WITH p.community AS community_id, p
        ORDER BY coalesce(p.citation_count, 0) DESC, p.title ASC
        WITH community_id, count(p) AS size, collect(p.title)[0..3] AS top_cited_titles
        RETURN community_id, size, top_cited_titles
        ORDER BY size DESC, community_id
        """
    )
