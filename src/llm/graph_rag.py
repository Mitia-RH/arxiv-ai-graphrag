"""GraphRAG: question -> Cypher -> Neo4j -> generated answer.

Each step is visible, and the generated Cypher is always returned for transparency.
"""
import json
import logging
import re

import config
from src.llm.llm_client import chat, strip_thinking

logger = logging.getLogger(__name__)

CYPHER_SYSTEM_PROMPT = """Generate read-only Cypher queries for Neo4j.
Respond ONLY with the query, without explanations or code fences.

Schema:
(:Author {name})-[:WROTE]->(:Paper {arxiv_id, title, abstract, published_date, url, citation_count, stub})
(:Paper)-[:BELONGS_TO]->(:Category {code})
(:Paper)-[:CITES]->(:Paper)

Rules:
- Titles and abstracts are in ENGLISH: search with toLower(...) CONTAINS 'lowercase english term'.
- stub=true means the paper is a cited reference only (no abstract). For full papers, use WHERE p.stub = false.
- Always end with LIMIT 10 (or fewer).
- Always return p.title and p.url when returning papers.

Examples:
Question: Which papers are cited the most?
MATCH (p:Paper)<-[:CITES]-(c:Paper) RETURN p.title AS title, p.url AS url, count(c) AS citation_count ORDER BY citation_count DESC LIMIT 10

Question: Which papers discuss reinforcement learning?
MATCH (p:Paper) WHERE p.stub = false AND (toLower(p.title) CONTAINS 'reinforcement learning' OR toLower(p.abstract) CONTAINS 'reinforcement learning') RETURN p.title AS title, p.url AS url, p.published_date AS date ORDER BY date DESC LIMIT 10

Question: Which authors have written the most papers?
MATCH (a:Author)-[:WROTE]->(p:Paper) RETURN a.name AS author, count(p) AS paper_count ORDER BY paper_count DESC LIMIT 10
"""

ANSWER_SYSTEM_PROMPT = """You are a scientific research assistant. Respond in English, in a few sentences,
using ONLY the provided results. Cite paper titles and arXiv links.
If there are no results, simply say that no papers were found."""

FORBIDDEN = re.compile(
    r"\b(CREATE|DELETE|DETACH|SET|MERGE|REMOVE|DROP|LOAD|FOREACH|CALL)\b", re.IGNORECASE
)


def clean_cypher(raw: str) -> str:
    """Extract a read-only query from the model's response."""
    text = strip_thinking(raw)
    text = re.sub(r"```(?:[a-zA-Z0-9_+-]+)?", "", text).strip()
    text = re.sub(r";\s*$", "", text).strip()
    query_start = re.search(
        r"\b(?:OPTIONAL\s+MATCH|MATCH|WITH)\b", text, flags=re.IGNORECASE
    )
    if query_start:
        text = text[query_start.start():]
    return text.strip()


def is_read_only(cypher: str) -> bool:
    """Return whether the query is a simple read (MATCH ... RETURN) with no write keywords."""
    # Strip string literals so a title containing 'set' is not incorrectly rejected.
    without_strings = re.sub(r"'[^']*'|\"[^\"]*\"", "''", cypher)
    starts_ok = re.match(r"\s*(MATCH|OPTIONAL\s+MATCH|WITH)\b", without_strings, re.IGNORECASE)
    return bool(starts_ok) and not FORBIDDEN.search(without_strings) and ";" not in without_strings


def _shrink(rows: list[dict]) -> str:
    """Shrink results to fit the model's small context window."""
    def cut(v):
        return v[:config.MAX_CHARS_PER_VALUE] if isinstance(v, str) else v
    small = [{k: cut(v) for k, v in row.items()} for row in rows[:config.MAX_ROWS_FOR_LLM]]
    return json.dumps(small, ensure_ascii=False, default=str)


def ask(client, question: str) -> dict:
    """Ask the graph a question. Return {answer, cypher, rows, error}."""
    result = {"answer": "", "cypher": "", "rows": [], "error": None}

    raw_output = chat(CYPHER_SYSTEM_PROMPT, f"Question: {question}")
    logger.debug("Raw LLM output: %s", raw_output)
    cypher = clean_cypher(raw_output)
    result["cypher"] = cypher
    if not is_read_only(cypher):
        result["error"] = "Query rejected: it is not read-only."
        return result

    try:
        rows = client.run_read(cypher)
    except Exception as error:  # Invalid Cypher syntax: one automatic correction attempt.
        logger.warning("Invalid Cypher (%s); retrying.", error)
        fix_prompt = (f"Question: {question}\nPrevious query: {cypher}\n"
                      f"Neo4j error: {error}\nFix the query.")
        raw_output = chat(CYPHER_SYSTEM_PROMPT, fix_prompt)
        logger.debug("Raw LLM output: %s", raw_output)
        cypher = clean_cypher(raw_output)
        result["cypher"] = cypher
        if not is_read_only(cypher):
            result["error"] = "Corrected query rejected: it is not read-only."
            return result
        try:
            rows = client.run_read(cypher)
        except Exception as error2:
            result["error"] = f"Invalid Cypher after correction: {error2}"
            return result

    result["rows"] = rows
    result["answer"] = chat(
        ANSWER_SYSTEM_PROMPT, f"Question: {question}\nResults: {_shrink(rows)}"
    )
    return result
