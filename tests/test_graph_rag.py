"""Tests for GraphRAG routing and search-term extraction."""
from typing import Any

import pytest

from src.llm import graph_rag


def test_classify_question_defaults_unexpected_response_to_topical(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(graph_rag, "chat", lambda system, user: "I'm not sure")

    assert graph_rag.classify_question("papers about diffusion") == "topical"


def test_extract_search_terms_parses_json_after_noise(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        graph_rag,
        "chat",
        lambda system, user: """Searching now...
{"keywords": "generative OR diffusion OR \\"text-to-image\\"", "question_en": "What is text-to-image generation?"}""",
    )

    assert graph_rag.extract_search_terms("¿Qué es la generación de imágenes?") == {
        "keywords": "generative OR diffusion OR \"text-to-image\"",
        "question_en": "What is text-to-image generation?",
    }


@pytest.mark.parametrize("response", ["not JSON", "{broken}", '{"keywords": 4}'])
def test_extract_search_terms_falls_back_on_garbage(
    monkeypatch: pytest.MonkeyPatch, response: str
) -> None:
    monkeypatch.setattr(graph_rag, "chat", lambda system, user: response)

    assert graph_rag.extract_search_terms("papers about diffusion") == {
        "keywords": "papers about diffusion",
        "question_en": "papers about diffusion",
    }


def test_ask_routes_topical_questions_to_hybrid_search(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_chat(system: str, user: str) -> str:
        if system == graph_rag.CLASSIFY_SYSTEM_PROMPT:
            return "topical"
        if system == graph_rag.SEARCH_TERMS_SYSTEM_PROMPT:
            return '{"keywords": "diffusion OR generative", "question_en": "Diffusion models"}'
        if system == graph_rag.ANSWER_SYSTEM_PROMPT:
            return "Found relevant papers."
        raise AssertionError(f"Unexpected prompt: {system}")

    search_calls: list[tuple[Any, str, str]] = []

    def fake_hybrid_search(client: Any, keywords: str, question_en: str) -> list[dict]:
        search_calls.append((client, keywords, question_en))
        return [{"title": "A paper", "url": "https://arxiv.org/abs/1"}]

    monkeypatch.setattr(graph_rag, "chat", fake_chat)
    monkeypatch.setattr(graph_rag, "hybrid_search", fake_hybrid_search)
    client = object()

    result = graph_rag.ask(client, "papers about diffusion")

    assert result["route"] == "topical"
    assert search_calls == [(client, "diffusion OR generative", "Diffusion models")]
    assert "hybrid: full-text + vector" in result["cypher"].lower()
    assert "Keywords: diffusion OR generative" in result["cypher"]
    assert result["answer"] == "Found relevant papers."
    assert result["rows"][0]["title"] == "A paper"


def test_ask_routes_structural_questions_to_cypher(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_chat(system: str, user: str) -> str:
        if system == graph_rag.CLASSIFY_SYSTEM_PROMPT:
            return "structural"
        if system == graph_rag.CYPHER_SYSTEM_PROMPT:
            return "MATCH (a:Author) RETURN a.name AS author LIMIT 10"
        if system == graph_rag.ANSWER_SYSTEM_PROMPT:
            return "The author is Ada."
        raise AssertionError(f"Unexpected prompt: {system}")

    monkeypatch.setattr(graph_rag, "chat", fake_chat)

    class FakeClient:
        def run_read(self, query: str) -> list[dict]:
            return [{"author": "Ada"}]

    result = graph_rag.ask(FakeClient(), "Who wrote the most papers?")

    assert result["route"] == "structural"
    assert result["cypher"] == "MATCH (a:Author) RETURN a.name AS author LIMIT 10"
    assert result["rows"] == [{"author": "Ada"}]
    assert result["answer"] == "The author is Ada."


def test_ask_falls_back_to_topical_when_structural_query_has_no_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_chat(system: str, user: str) -> str:
        if system == graph_rag.CLASSIFY_SYSTEM_PROMPT:
            return "structural"
        if system == graph_rag.CYPHER_SYSTEM_PROMPT:
            return "MATCH (p:Paper) RETURN p.title AS title LIMIT 10"
        if system == graph_rag.SEARCH_TERMS_SYSTEM_PROMPT:
            return '{"keywords": "author OR papers", "question_en": "Top authors"}'
        if system == graph_rag.ANSWER_SYSTEM_PROMPT:
            return "Here are relevant papers."
        raise AssertionError(f"Unexpected prompt: {system}")

    monkeypatch.setattr(graph_rag, "chat", fake_chat)
    monkeypatch.setattr(
        graph_rag,
        "hybrid_search",
        lambda client, keywords, question_en: [{"title": "Relevant paper"}],
    )

    class FakeClient:
        def run_read(self, query: str) -> list[dict]:
            return []

    result = graph_rag.ask(FakeClient(), "Who are the top authors?")

    assert result["route"] == "topical"
    assert result["rows"] == [{"title": "Relevant paper"}]
    assert "Keywords: author OR papers" in result["cypher"]
    assert result["answer"] == "Here are relevant papers."
