"""Enrichissement via OpenAlex : nombre de citations + références vers d'autres papiers arXiv.

Choix de simplicité : on ne garde que les références qui ont elles-mêmes un identifiant
arXiv. Ainsi tous les nœuds :Paper ont un `arxiv_id` unique, comme le veut le schéma.
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
    """'https://doi.org/10.48550/arXiv.1706.03762' -> '1706.03762' (None si pas arXiv)."""
    if not doi:
        return None
    match = re.search(r"10\.48550/arxiv\.(.+)$", doi, flags=re.IGNORECASE)
    return clean_arxiv_id(match.group(1)) if match else None


def _get(path: str, params: dict | None = None) -> dict | None:
    """GET avec pause polie. Retourne None si 404, lève une exception sinon."""
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
    """Parmi des IDs OpenAlex ('W123'), garde ceux qui sont des papiers arXiv."""
    refs = []
    for i in range(0, len(openalex_ids), 50):          # OpenAlex limite à 50 valeurs par filtre
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
    """Ajoute `citation_count` et `references` au papier. Ne plante jamais."""
    paper["citation_count"] = None
    paper["references"] = []
    try:
        work = _get(f"/works/doi:10.48550/arXiv.{paper['arxiv_id']}")
        if work is None:
            logger.info("%s : inconnu d'OpenAlex (trop récent ?).", paper["arxiv_id"])
            return paper
        paper["citation_count"] = work.get("cited_by_count")
        ids = [url.rsplit("/", 1)[-1] for url in work.get("referenced_works", [])]
        paper["references"] = _resolve_arxiv_references(ids)
    except requests.RequestException as error:
        logger.warning("%s : OpenAlex a échoué (%s), on continue.", paper["arxiv_id"], error)
    return paper
