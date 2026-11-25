"""Tests simples : nettoyage des IDs, nettoyage du Cypher, sécurité lecture seule.
Lancer : python -m pytest
"""
from src.ingestion.arxiv_fetcher import clean_arxiv_id
from src.ingestion.citation_enricher import arxiv_id_from_doi
from src.llm.graph_rag import clean_cypher, is_read_only


def test_clean_arxiv_id():
    assert clean_arxiv_id("2401.12345v2") == "2401.12345"
    assert clean_arxiv_id("http://arxiv.org/abs/2401.12345v1") == "2401.12345"
    assert clean_arxiv_id("2401.12345") == "2401.12345"


def test_arxiv_id_from_doi():
    assert arxiv_id_from_doi("https://doi.org/10.48550/arXiv.1706.03762") == "1706.03762"
    assert arxiv_id_from_doi("https://doi.org/10.1000/xyz") is None
    assert arxiv_id_from_doi(None) is None


def test_clean_cypher_removes_markdown():
    raw = "```cypher\nMATCH (p:Paper) RETURN p.title LIMIT 5;\n```"
    assert clean_cypher(raw) == "MATCH (p:Paper) RETURN p.title LIMIT 5"


def test_read_only_accepts_reads():
    assert is_read_only("MATCH (p:Paper) RETURN p.title LIMIT 5")
    # 'set' dans une chaîne de caractères ne doit pas déclencher de faux positif
    assert is_read_only("MATCH (p:Paper) WHERE p.title CONTAINS 'data set' RETURN p.title")


def test_read_only_rejects_writes():
    assert not is_read_only("MATCH (p:Paper) DETACH DELETE p")
    assert not is_read_only("MATCH (p:Paper) SET p.title = 'x' RETURN p")
    assert not is_read_only("CREATE (p:Paper {arxiv_id: '1'})")
    assert not is_read_only("MATCH (p:Paper) RETURN p; MATCH (a) DELETE a")
    assert not is_read_only("Voici la requête : MATCH (p) RETURN p")
