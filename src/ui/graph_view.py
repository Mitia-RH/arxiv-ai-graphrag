"""Pure helpers for building paper-neighbourhood graph data."""
from collections.abc import Mapping, Sequence


def build_neighborhood_graph(
    paper: Mapping[str, str],
    authors: Sequence[Mapping[str, str]],
    cited: Sequence[Mapping[str, str]],
    citing: Sequence[Mapping[str, str]],
    max_nodes: int = 50,
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    """Build PyVis-compatible nodes and edges for a one-hop paper neighbourhood."""
    if max_nodes < 1:
        raise ValueError("max_nodes must be at least one")

    paper_id = f"paper:{paper['arxiv_id']}"
    nodes = [{
        "id": paper_id,
        "label": paper.get("title") or paper["arxiv_id"],
        "kind": "paper",
    }]
    selected_ids = {paper_id}
    relations = [
        ("author", authors),
        ("cited", cited),
        ("citing", citing),
    ]
    candidates: list[tuple[str, str, str, str]] = []
    for relation, group in relations:
        for item in group:
            if relation == "author":
                name = item.get("name", "").strip()
                if name:
                    candidates.append((f"author:{name}", name, "author", relation))
            else:
                arxiv_id = item.get("arxiv_id", "").strip()
                if arxiv_id:
                    candidates.append((
                        f"paper:{arxiv_id}",
                        item.get("title") or arxiv_id,
                        "paper",
                        relation,
                    ))

    grouped_candidates = [
        [candidate for candidate in candidates if candidate[3] == relation]
        for relation, _ in relations
    ]
    for index in range(max(map(len, grouped_candidates), default=0)):
        for group in grouped_candidates:
            if index >= len(group) or len(nodes) >= max_nodes:
                continue
            node_id, label, kind, _ = group[index]
            if node_id not in selected_ids:
                nodes.append({"id": node_id, "label": label, "kind": kind})
                selected_ids.add(node_id)
        if len(nodes) >= max_nodes:
            break

    edges: list[dict[str, str]] = []
    edge_keys: set[tuple[str, str, str]] = set()
    for node_id, _, _, relation in candidates:
        if node_id not in selected_ids:
            continue
        if relation == "author":
            source, target, label = node_id, paper_id, "WROTE"
        elif relation == "cited":
            source, target, label = paper_id, node_id, "CITES"
        else:
            source, target, label = node_id, paper_id, "CITES"
        edge_key = (source, target, label)
        if edge_key not in edge_keys:
            edges.append({"from": source, "to": target, "label": label})
            edge_keys.add(edge_key)

    return nodes, edges
