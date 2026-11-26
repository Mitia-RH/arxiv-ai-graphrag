"""Neo4j connection wrapper that can be used with `with`."""
import logging
from typing import Any

from neo4j import GraphDatabase

import config

logger = logging.getLogger(__name__)


class Neo4jClient:
    def __init__(self) -> None:
        self._driver = GraphDatabase.driver(
            config.NEO4J_URI, auth=(config.NEO4J_USER, config.NEO4J_PASSWORD)
        )

    def __enter__(self) -> "Neo4jClient":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    def close(self) -> None:
        self._driver.close()

    def run_write(self, query: str, **params: Any) -> list[dict]:
        """Execute a query that can modify the database (ingestion)."""
        with self._driver.session() as session:
            return session.execute_write(lambda tx: [r.data() for r in tx.run(query, **params)])

    def run_read(self, query: str, **params: Any) -> list[dict]:
        """Execute a read-only query.

        Defense in depth: even if a dangerous query passed the GraphRAG filter,
        Neo4j would refuse to execute it in a read transaction.
        """
        with self._driver.session() as session:
            return session.execute_read(lambda tx: [r.data() for r in tx.run(query, **params)])
