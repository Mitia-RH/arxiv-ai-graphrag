"""Orchestrateur : Fetch -> Enrich -> Load.

Usage : python -m src.ingestion.pipeline --limit 50
"""
import argparse
import logging

import config
from src.database.neo4j_client import Neo4jClient
from src.database.repository import add_citation, upsert_paper
from src.database.schema import create_schema
from src.ingestion.arxiv_fetcher import fetch_papers
from src.ingestion.citation_enricher import enrich_paper

logger = logging.getLogger(__name__)


def run(limit: int = config.MAX_PAPERS) -> int:
    """Ingère `limit` papiers. Relançable sans doublons (MERGE). Retourne le nombre traité."""
    papers = fetch_papers(limit)
    with Neo4jClient() as client:
        create_schema(client)
        for i, paper in enumerate(papers, start=1):
            paper = enrich_paper(paper)
            upsert_paper(client, paper)
            for ref in paper["references"]:
                add_citation(client, paper["arxiv_id"], ref)
            logger.info("[%d/%d] %s | %d citations ajoutées | %s",
                        i, len(papers), paper["arxiv_id"], len(paper["references"]),
                        paper["title"][:60])
    return len(papers)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Ingestion arXiv cs.AI -> Neo4j")
    parser.add_argument("--limit", type=int, default=config.MAX_PAPERS)
    args = parser.parse_args()
    total = run(args.limit)
    logger.info("Terminé : %d papiers traités.", total)
