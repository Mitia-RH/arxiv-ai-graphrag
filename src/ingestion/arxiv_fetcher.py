"""Fetch arXiv papers in the cs.AI category."""
import logging
import re

import config

logger = logging.getLogger(__name__)


def clean_arxiv_id(raw: str) -> str:
    """Convert 'http://arxiv.org/abs/2401.12345v2' to '2401.12345' (without URL or version)."""
    raw = raw.strip().removeprefix("https://").removeprefix("http://")
    raw = raw.removeprefix("arxiv.org/abs/")
    return re.sub(r"v\d+$", "", raw)


def fetch_papers(limit: int = config.MAX_PAPERS) -> list[dict]:
    """Return the `limit` most recent cs.AI papers."""
    import arxiv  # Import here so unit tests do not depend on this package.

    # arXiv requests no more than one request every three seconds.
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
            "abstract": " ".join(r.summary.split()),   # Remove line breaks.
            "published_date": r.published.date().isoformat(),
            "url": r.entry_id,
            "authors": [a.name for a in r.authors],
            "categories": list(r.categories),
        })
    logger.info("Fetched %d papers from arXiv.", len(papers))
    return papers
