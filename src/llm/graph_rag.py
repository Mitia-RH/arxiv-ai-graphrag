"""GraphRAG: question -> Cypher -> Neo4j -> generated answer.

Each step is visible, and the generated Cypher is always returned for transparency.
"""
import json
import logging
import re
from typing import Any

import config
from src.llm.llm_client import chat, strip_thinking
from src.llm.retrieval import hybrid_search

logger = logging.getLogger(__name__)

CYPHER_SYSTEM_PROMPT = """Generate read-only Cypher queries for Neo4j.
Respond ONLY with the query, without explanations or code fences.

Schema:
(:Author {name})-[:WROTE]->(:Paper {arxiv_id, title, abstract, published_date, url, citation_count, stub})
(:Paper)-[:BELONGS_TO]->(:Category {code})
(:Paper)-[:CITES]->(:Paper)

Rules:
- Titles and abstracts are in ENGLISH: search with toLower(...) CONTAINS 'lowercase english term'.
- Use the singular form and the shortest useful term (for example, 'llm', not 'llms').
- When a concept has common synonyms or abbreviations, search each term in both the title and abstract and combine all matches with OR (for example, 'large language model' OR 'llm').
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

Question: Who are the co-authors of an author?
MATCH (a:Author)-[:WROTE]->(p:Paper)<-[:WROTE]-(coauthor:Author) WHERE toLower(a.name) CONTAINS 'author name' AND coauthor <> a RETURN DISTINCT coauthor.name AS coauthor ORDER BY coauthor LIMIT 10

Question: Which papers were written by an author?
MATCH (a:Author)-[:WROTE]->(p:Paper) WHERE toLower(a.name) CONTAINS 'author name' AND p.stub = false RETURN p.title AS title, p.url AS url, p.published_date AS date ORDER BY date DESC LIMIT 10

Question: How many papers were published each year?
MATCH (p:Paper) WHERE p.stub = false AND p.published_date IS NOT NULL RETURN substring(p.published_date, 0, 4) AS year, count(p) AS paper_count ORDER BY year DESC LIMIT 10
"""

ANSWER_SYSTEM_PROMPT = """You are a scientific research assistant. The user may write in any language.
Respond in the user's language, in a few sentences, using ONLY the provided results.
Cite paper titles and arXiv links.
If there are no results, simply say that no papers were found."""

CLASSIFY_SYSTEM_PROMPT = """Classify the question with exactly ONE word:
- structural: questions about graph structure or statistics, such as top authors, most cited papers,
  counts, or who cites whom
- topical: questions about papers on a subject or research topic
Reply only with structural or topical."""

SEARCH_TERMS_SYSTEM_PROMPT = """Prepare a search for scientific papers.
Return only a JSON object with string fields "keywords" and "question_en".
"keywords" must be an English Lucene query with useful synonyms joined by OR.
"question_en" must be the user's question translated into English."""

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


def classify_question(question: str) -> str:
    """Choose structural graph queries or topical paper retrieval."""
    answer = chat(CLASSIFY_SYSTEM_PROMPT, f"Question: {question}").strip().lower()
    return answer if answer in {"structural", "topical"} else "topical"


def extract_search_terms(question: str) -> dict[str, str]:
    """Extract English Lucene keywords and an English translation of the question."""
    raw_output = chat(SEARCH_TERMS_SYSTEM_PROMPT, f"Question: {question}")
    text = strip_thinking(raw_output)
    object_start = text.find("{")
    if object_start >= 0:
        try:
            parsed, _ = json.JSONDecoder().raw_decode(text[object_start:])
        except json.JSONDecodeError:
            pass
        else:
            if (
                isinstance(parsed, dict)
                and isinstance(parsed.get("keywords"), str)
                and parsed["keywords"].strip()
                and isinstance(parsed.get("question_en"), str)
                and parsed["question_en"].strip()
            ):
                return {
                    "keywords": parsed["keywords"],
                    "question_en": parsed["question_en"],
                }
    return {"keywords": question, "question_en": question}


def _answer_with_topical_search(
    client: Any, question: str, result: dict[str, Any]
) -> dict[str, Any]:
    search_terms = extract_search_terms(question)
    rows = hybrid_search(client, search_terms["keywords"], search_terms["question_en"])
    result["route"] = "topical"
    result["cypher"] = (
        "Hybrid: full-text + vector\n"
        f"Keywords: {search_terms['keywords']}\n"
        f"Question (English): {search_terms['question_en']}"
    )
    result["rows"] = rows
    result["answer"] = chat(
        ANSWER_SYSTEM_PROMPT, f"Question: {question}\nResults: {_shrink(rows)}"
    )
    return result


def ask(client: Any, question: str) -> dict[str, Any]:
    """Ask the graph a question and return {answer, cypher, rows, error, route}."""
    result: dict[str, Any] = {
        "answer": "",
        "cypher": "",
        "rows": [],
        "error": None,
        "route": classify_question(question),
    }

    if result["route"] == "topical":
        return _answer_with_topical_search(client, question, result)

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
    if not rows:
        return _answer_with_topical_search(client, question, result)
    result["answer"] = chat(
        ANSWER_SYSTEM_PROMPT, f"Question: {question}\nResults: {_shrink(rows)}"
    )
    return result
