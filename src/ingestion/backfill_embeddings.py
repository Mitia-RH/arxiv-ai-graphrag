"""Backfill missing embeddings for complete papers already stored in Neo4j."""
import argparse
import logging

from src.database.neo4j_client import Neo4jClient
from src.llm.llm_client import embed

logger = logging.getLogger(__name__)
BATCH_SIZE = 50


def backfill(limit: int | None = None) -> int:
    """Embed up to `limit` non-stub papers that do not yet have embeddings."""
    if limit is not None and limit < 0:
        raise ValueError("limit must be non-negative")
    if limit == 0:
        return 0

    processed = 0
    with Neo4jClient() as client:
        while limit is None or processed < limit:
            batch_size = BATCH_SIZE if limit is None else min(BATCH_SIZE, limit - processed)
            papers = client.run_read(
                """
                MATCH (p:Paper)
                WHERE p.stub = false AND p.embedding IS NULL
                RETURN p.arxiv_id AS arxiv_id, p.title AS title, p.abstract AS abstract
                ORDER BY p.arxiv_id
                LIMIT $batch_size
                """,
                batch_size=batch_size,
            )
            if not papers:
                break

            embeddings = [
                {
                    "arxiv_id": paper["arxiv_id"],
                    "embedding": embed(f"{paper['title']}. {paper['abstract']}"),
                }
                for paper in papers
            ]
            client.run_write(
                """
                UNWIND $papers AS item
                MATCH (p:Paper {arxiv_id: item.arxiv_id})
                SET p.embedding = item.embedding
                """,
                papers=embeddings,
            )
            processed += len(embeddings)
            logger.info("Backfilled %d paper embeddings.", processed)
    return processed


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Backfill missing paper embeddings in Neo4j")
    parser.add_argument("--limit", type=int, help="Maximum number of papers to embed")
    args = parser.parse_args()
    if args.limit is not None and args.limit < 0:
        parser.error("--limit must be non-negative")
    total = backfill(limit=args.limit)
    logger.info("Done: backfilled %d paper embeddings.", total)
