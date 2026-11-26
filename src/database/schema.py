"""Uniqueness constraints. `IF NOT EXISTS` makes this safe to run repeatedly."""
import logging

from src.database.neo4j_client import Neo4jClient

logger = logging.getLogger(__name__)

CONSTRAINTS = [
    "CREATE CONSTRAINT paper_arxiv_id IF NOT EXISTS FOR (p:Paper) REQUIRE p.arxiv_id IS UNIQUE",
    "CREATE CONSTRAINT author_name IF NOT EXISTS FOR (a:Author) REQUIRE a.name IS UNIQUE",
    "CREATE CONSTRAINT category_code IF NOT EXISTS FOR (c:Category) REQUIRE c.code IS UNIQUE",
    "CREATE CONSTRAINT concept_name IF NOT EXISTS FOR (k:Concept) REQUIRE k.name IS UNIQUE",
]


def create_schema(client: Neo4jClient) -> None:
    for statement in CONSTRAINTS:
        client.run_write(statement)
    logger.info("Verified %d Neo4j constraints.", len(CONSTRAINTS))
