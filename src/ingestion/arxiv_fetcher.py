"""Récupération des articles arXiv de la catégorie cs.AI."""
import logging
import re

import config

logger = logging.getLogger(__name__)


def clean_arxiv_id(raw: str) -> str:
    """'http://arxiv.org/abs/2401.12345v2' -> '2401.12345' (sans URL ni version)."""
    raw = raw.strip().removeprefix("https://").removeprefix("http://")
    raw = raw.removeprefix("arxiv.org/abs/")
    return re.sub(r"v\d+$", "", raw)


def fetch_papers(limit: int = config.MAX_PAPERS) -> list[dict]:
    """Retourne les `limit` articles cs.AI les plus récents."""
    import arxiv  # import ici pour que les tests unitaires n'en dépendent pas

    # delay_seconds=3 : arXiv demande de ne pas dépasser 1 requête / 3 s.
    client = arxiv.Client(page_size=min(limit, 100), delay_seconds=3, num_retries=3)
    search = arxiv.Search(
        query=f"cat:{config.ARXIV_CATEGORY}",
        max_results=limit,
        sort_by=arxiv.SortCriterion.SubmittedDate,
    )
    papers = []
    for r in client.results(search):
        papers.append({
            "arxiv_id": clean_arxiv_id(r.get_short_id()),
            "title": " ".join(r.title.split()),
            "abstract": " ".join(r.summary.split()),   # supprime les retours à la ligne
            "published_date": r.published.date().isoformat(),
            "url": r.entry_id,
            "authors": [a.name for a in r.authors],
            "categories": list(r.categories),
        })
    logger.info("%d articles récupérés depuis arXiv.", len(papers))
    return papers
