"""Tests for pure paper graph conversion helpers."""
import pytest

from src.ui.graph_view import build_neighborhood_graph


def test_build_neighborhood_graph_includes_typed_one_hop_edges():
    nodes, edges = build_neighborhood_graph(
        {"arxiv_id": "center", "title": "Center paper"},
        [{"name": "Author One"}],
        [{"arxiv_id": "cited", "title": "Cited paper"}],
        [{"arxiv_id": "citing", "title": "Citing paper"}],
    )

    assert nodes == [
        {"id": "paper:center", "label": "Center paper", "kind": "paper"},
        {"id": "author:Author One", "label": "Author One", "kind": "author"},
        {"id": "paper:cited", "label": "Cited paper", "kind": "paper"},
        {"id": "paper:citing", "label": "Citing paper", "kind": "paper"},
    ]
    assert edges == [
        {"from": "author:Author One", "to": "paper:center", "label": "WROTE"},
        {"from": "paper:center", "to": "paper:cited", "label": "CITES"},
        {"from": "paper:citing", "to": "paper:center", "label": "CITES"},
    ]


def test_build_neighborhood_graph_caps_nodes_and_keeps_only_valid_edges():
    nodes, edges = build_neighborhood_graph(
        {"arxiv_id": "center", "title": "Center paper"},
        [{"name": "Author One"}, {"name": "Author Two"}],
        [{"arxiv_id": "cited", "title": "Cited paper"}],
        [{"arxiv_id": "citing", "title": "Citing paper"}],
        max_nodes=3,
    )

    assert len(nodes) == 3
    assert len(edges) == 2
    assert {node["id"] for node in nodes} == {
        "paper:center",
        "author:Author One",
        "paper:cited",
    }


def test_build_neighborhood_graph_requires_room_for_center():
    with pytest.raises(ValueError, match="at least one"):
        build_neighborhood_graph({"arxiv_id": "center"}, [], [], [], max_nodes=0)
