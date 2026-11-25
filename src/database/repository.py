"""Fonctions d'écriture (toujours avec MERGE => idempotent) et statistiques."""
from src.database.neo4j_client import Neo4jClient


def upsert_paper(client: Neo4jClient, paper: dict) -> None:
    """Crée ou met à jour un papier, ses auteurs et ses catégories.

    `paper` : arxiv_id, title, abstract, published_date, url,
              authors (list[str]), categories (list[str]), citation_count (optionnel).
    """
    # stub=false : ce papier est "complet" (il a pu être créé auparavant
    # comme simple référence citée par un autre papier).
    client.run_write(
        """
        MERGE (p:Paper {arxiv_id: $arxiv_id})
        SET p.title = $title, p.abstract = $abstract,
            p.published_date = $published_date, p.url = $url,
            p.citation_count = $citation_count, p.stub = false
        """,
        arxiv_id=paper["arxiv_id"], title=paper["title"], abstract=paper["abstract"],
        published_date=paper["published_date"], url=paper["url"],
        citation_count=paper.get("citation_count"),
    )
    client.run_write(
        """
        MATCH (p:Paper {arxiv_id: $arxiv_id})
        UNWIND $authors AS author_name
        MERGE (a:Author {name: author_name})
        MERGE (a)-[:WROTE]->(p)
        """,
        arxiv_id=paper["arxiv_id"], authors=paper["authors"],
    )
    client.run_write(
        """
        MATCH (p:Paper {arxiv_id: $arxiv_id})
        UNWIND $categories AS code
        MERGE (c:Category {code: code})
        MERGE (p)-[:BELONGS_TO]->(c)
        """,
        arxiv_id=paper["arxiv_id"], categories=paper["categories"],
    )


def add_citation(client: Neo4jClient, source_arxiv_id: str, target: dict) -> None:
    """Ajoute (source)-[:CITES]->(cible). La cible est créée en "stub" si inconnue."""
    client.run_write(
        """
        MATCH (s:Paper {arxiv_id: $source_id})
        MERGE (t:Paper {arxiv_id: $target_id})
          ON CREATE SET t.stub = true, t.title = $target_title,
                        t.url = 'https://arxiv.org/abs/' + $target_id
        MERGE (s)-[:CITES]->(t)
        """,
        source_id=source_arxiv_id, target_id=target["arxiv_id"],
        target_title=target.get("title"),
    )


def get_graph_stats(client: Neo4jClient) -> dict:
    """Compteurs affichés dans la barre latérale."""
    return client.run_read(
        """
        CALL { MATCH (p:Paper) WHERE p.stub = false RETURN count(p) AS papers }
        CALL { MATCH (p:Paper) WHERE p.stub = true RETURN count(p) AS stubs }
        CALL { MATCH (a:Author) RETURN count(a) AS authors }
        CALL { MATCH ()-[r:CITES]->() RETURN count(r) AS citations }
        RETURN papers, stubs, authors, citations
        """
    )[0]
