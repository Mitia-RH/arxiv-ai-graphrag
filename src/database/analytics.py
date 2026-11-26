"""Compute PageRank and Leiden communities using Neo4j Graph Data Science."""
import argparse
import logging
import sys

from neo4j.exceptions import Neo4jError

from src.database.neo4j_client import Neo4jClient

logger = logging.getLogger(__name__)

_PROJECTIONS = ("cit", "cit_undirected")
_DROP_PROJECTION_QUERY = """
    CALL gds.graph.drop($graph_name, false)
    YIELD graphName
    RETURN graphName
"""


def compute_analytics(client: Neo4jClient) -> None:
    """Write PageRank and Leiden community properties for the citation graph."""
    try:
        try:
            client.run_write(
                """
                CALL gds.graph.project(
                    $graph_name,
                    'Paper',
                    {CITES: {orientation: 'NATURAL'}}
                )
                YIELD graphName
                RETURN graphName
                """,
                graph_name="cit",
            )
            client.run_write(
                """
                CALL gds.pageRank.write($graph_name, {writeProperty: 'pagerank'})
                YIELD nodePropertiesWritten
                RETURN nodePropertiesWritten
                """,
                graph_name="cit",
            )
            client.run_write(
                """
                CALL gds.graph.project(
                    $graph_name,
                    'Paper',
                    {CITES: {orientation: 'UNDIRECTED'}}
                )
                YIELD graphName
                RETURN graphName
                """,
                graph_name="cit_undirected",
            )
            client.run_write(
                """
                CALL gds.leiden.write($graph_name, {writeProperty: 'community'})
                YIELD communityCount
                RETURN communityCount
                """,
                graph_name="cit_undirected",
            )
        except Neo4jError as error:
            raise RuntimeError(
                "Neo4j Graph Data Science analytics failed. Ensure the "
                "'graph-data-science' plugin is installed and available."
            ) from error
    finally:
        active_error = sys.exc_info()[1]
        cleanup_errors: list[tuple[str, Neo4jError]] = []
        for graph_name in reversed(_PROJECTIONS):
            try:
                client.run_write(_DROP_PROJECTION_QUERY, graph_name=graph_name)
            except Neo4jError as error:
                cleanup_errors.append((graph_name, error))

        if cleanup_errors and active_error is None:
            graph_name, error = cleanup_errors[0]
            raise RuntimeError(
                f"Failed to drop the in-memory GDS projection '{graph_name}'."
            ) from error
        for graph_name, error in cleanup_errors:
            logger.warning(
                "Could not drop GDS projection %s while handling another error: %s",
                graph_name,
                error,
            )

    logger.info("Computed PageRank and Leiden communities.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Compute Neo4j graph analytics")
    parser.parse_args()
    with Neo4jClient() as client:
        compute_analytics(client)
