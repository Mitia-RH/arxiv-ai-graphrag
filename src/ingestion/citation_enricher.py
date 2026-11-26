"""Enrich papers via OpenAlex with citation counts and references to other arXiv papers.

For simplicity, only keep references that have an arXiv ID. This ensures every
:Paper node has a unique `arxiv_id`, as required by the schema.
"""
import logging
import re
import time

import requests

import config
from src.ingestion.arxiv_fetcher import clean_arxiv_id

logger = logging.getLogger(__name__)

BASE_URL = "https://api.openalex.org"


def arxiv_id_from_doi(doi: str | None) -> str | None:
    """Convert 'https://doi.org/10.48550/arXiv.1706.03762' to '1706.03762' (None if not arXiv)."""
    if not doi:
        return None
    match = re.search(r"10\.48550/arxiv\.(.+)$", doi, flags=re.IGNORECASE)
    return clean_arxiv_id(match.group(1)) if match else None


def _get(path: str, params: dict | None = None) -> dict | None:
    """Send a GET request with a polite delay. Return None on 404; raise otherwise."""
    params = dict(params or {})
    if config.OPENALEX_EMAIL:
        params["mailto"] = config.OPENALEX_EMAIL
    response = requests.get(f"{BASE_URL}{path}", params=params, timeout=20)
    time.sleep(config.OPENALEX_DELAY_S)
    if response.status_code == 404:
        return None
    response.raise_for_status()
    return response.json()


def _resolve_arxiv_references(openalex_ids: list[str]) -> list[dict]:
    """Keep only arXiv papers from a list of OpenAlex IDs ('W123')."""
    refs = []
    for i in range(0, len(openalex_ids), 50):          # OpenAlex limits each filter to 50 values.
        chunk = openalex_ids[i:i + 50]
        data = _get("/works", {
            "filter": "openalex:" + "|".join(chunk),
            "select": "id,doi,title",
            "per-page": 50,
        })
        for work in (data or {}).get("results", []):
            arxiv_id = arxiv_id_from_doi(work.get("doi"))
            if arxiv_id:
                refs.append({"arxiv_id": arxiv_id, "title": work.get("title")})
    return refs


def enrich_paper(paper: dict) -> dict:
    """Add `citation_count` and `references` to a paper. Never fail."""
    paper["citation_count"] = None
    paper["references"] = []
    try:
        work = _get(f"/works/doi:10.48550/arXiv.{paper['arxiv_id']}")
        if work is None:
            logger.info("%s: not found in OpenAlex (perhaps too recent).", paper["arxiv_id"])
            return paper
        paper["citation_count"] = work.get("cited_by_count")
        ids = [url.rsplit("/", 1)[-1] for url in work.get("referenced_works", [])]
        paper["references"] = _resolve_arxiv_references(ids)
    except requests.RequestException as error:
        logger.warning("%s: OpenAlex request failed (%s); continuing.", paper["arxiv_id"], error)
    return paper
