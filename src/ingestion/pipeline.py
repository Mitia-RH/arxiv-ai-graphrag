"""Orchestrator: fetch -> enrich -> load.

Usage : python -m src.ingestion.pipeline --limit 50 --offset 100
        python -m src.ingestion.pipeline --limit 50 --offset 100 --force
"""
import argparse
import logging

import requests

import config
from src.database.neo4j_client import Neo4jClient
from src.database.repository import add_citations, get_existing_ids, upsert_paper
from src.database.schema import create_schema
from src.ingestion.arxiv_fetcher import fetch_papers
from src.ingestion.citation_enricher import enrich_paper
from src.llm.llm_client import embed

logger = logging.getLogger(__name__)


def run(limit: int = config.MAX_PAPERS, offset: int = 0, force: bool = False) -> int:
    """Ingest papers from `offset`, skipping complete existing papers unless `force` is true."""
    papers = fetch_papers(limit, offset=offset)
    with Neo4jClient() as client:
        create_schema(client)
        skipped = 0
        if not force:
            existing_ids = get_existing_ids(client, [paper["arxiv_id"] for paper in papers])
            papers_to_process = [
                paper for paper in papers if paper["arxiv_id"] not in existing_ids
            ]
            skipped = len(papers) - len(papers_to_process)
        else:
            papers_to_process = papers
        logger.info("Skipped %d papers already in the database.", skipped)
        for i, paper in enumerate(papers_to_process, start=1):
            paper = enrich_paper(paper)
            try:
                paper["embedding"] = embed(f"{paper['title']}. {paper['abstract']}")
            except requests.RequestException as exc:
                logger.warning("Could not embed paper %s: %s", paper["arxiv_id"], exc)
            upsert_paper(client, paper)
            if paper["references"]:
                add_citations(client, paper["arxiv_id"], paper["references"])
            logger.info("[%d/%d] %s | added %d citations | %s",
                        i, len(papers_to_process), paper["arxiv_id"], len(paper["references"]),
                        paper["title"][:60])
    return len(papers_to_process)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Ingestion arXiv cs.AI -> Neo4j")
    parser.add_argument("--limit", type=int, default=config.MAX_PAPERS)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--force", action="store_true", help="Re-ingest papers already in the database")
    args = parser.parse_args()
    total = run(args.limit, offset=args.offset, force=args.force)
    logger.info("Done: processed %d papers.", total)
