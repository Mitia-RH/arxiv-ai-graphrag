"""Streamlit interface: GraphRAG chat and statistics. Run with: streamlit run app.py"""
import logging

import streamlit as st
from pyvis.network import Network

import config
from src.database.analytics import compute_analytics
from src.database.neo4j_client import Neo4jClient
from src.database.repository import (
    get_community_summary,
    get_graph_stats,
    get_paper_neighborhood,
    get_papers_per_month,
    get_top_pagerank,
)
from src.database.schema import create_schema
from src.ingestion import pipeline
from src.llm import graph_rag
from src.llm.llm_client import is_available
from src.ui.graph_view import build_neighborhood_graph

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
st.set_page_config(page_title="ArXiv AI GraphRAG", page_icon="🕸️")


@st.cache_resource
def get_client() -> Neo4jClient:
    client = Neo4jClient()
    create_schema(client)
    return client


try:
    client = get_client()
except Exception as error:
    st.error(f"Neo4j is unavailable: {error}\n\nHave you run `docker compose up -d`?")
    st.stop()

# ---------- Sidebar ----------
with st.sidebar:
    st.header("Neo4j database")
    stats = get_graph_stats(client)
    st.metric("Papers (cs.AI)", stats["papers"])
    st.metric("Cited papers (references)", stats["stubs"])
    st.metric("Authors", stats["authors"])
    st.metric("Citations", stats["citations"])

    st.header("Local LLM")
    st.write(f"Model: `{config.OLLAMA_MODEL}`")
    st.write("✅ Ollama is running" if is_available() else "❌ Ollama is unreachable")

    st.header("Ingestion")
    count_col, offset_col = st.columns(2)
    nb = count_col.number_input("Number of papers", 5, 500, 20, step=5)
    offset = offset_col.number_input("Offset", min_value=0, value=0, step=100)
    if st.button("Ingest new papers"):
        with st.spinner("Fetching and enriching papers (this may take a few minutes)..."):
            pipeline.run(int(nb), offset=int(offset))
        st.rerun()

chat_tab, insights_tab, graph_tab = st.tabs(["Chat", "Insights", "Graph"])


def show_details(msg: dict) -> None:
    with st.expander("View the Cypher query and raw data"):
        st.code(msg["cypher"], language="cypher")
        st.json(msg["rows"])


def show_route(msg: dict) -> None:
    st.caption(f"Route: {msg.get('route', 'unknown')}")


with chat_tab:
    st.title("🕸️ ArXiv AI — GraphRAG")
    st.caption("Ask a question about AI research (answers are based on the Neo4j graph).")

    if "history" not in st.session_state:
        st.session_state.history = []

    for msg in st.session_state.history:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
            if msg["role"] == "assistant":
                show_route(msg)
                show_details(msg)

    if question := st.chat_input("e.g. Which papers discuss LLM agents?"):
        st.session_state.history.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.write(question)
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    res = graph_rag.ask(client, question)
                except Exception as error:  # For example, Ollama is unavailable.
                    res = {
                        "answer": "",
                        "cypher": "",
                        "rows": [],
                        "error": str(error),
                        "route": "error",
                    }
            text = res["answer"] or f"⚠️ {res['error']}"
            st.write(text)
            msg = {
                "role": "assistant",
                "content": text,
                "cypher": res["cypher"],
                "rows": res["rows"],
                "route": res["route"],
            }
            show_route(msg)
            show_details(msg)
        st.session_state.history.append(msg)

with insights_tab:
    st.header("Insights")
    top_papers = get_top_pagerank(client, 10)
    communities = get_community_summary(client)

    if not top_papers or not communities:
        st.info("Graph analytics have not been computed yet.")
        if st.button("Compute analytics", key="compute_analytics"):
            try:
                compute_analytics(client)
            except RuntimeError as error:
                st.error(f"Could not compute analytics: {error}")
            else:
                st.success("Analytics computed.")
                st.rerun()
    else:
        st.subheader("Top 10 papers by PageRank")
        st.dataframe(top_papers, hide_index=True, use_container_width=True)

        st.subheader("Community summary")
        st.dataframe(communities, hide_index=True, use_container_width=True)

    st.subheader("Papers per month")
    monthly_counts = get_papers_per_month(client)
    if monthly_counts:
        st.bar_chart(monthly_counts, x="month", y="paper_count")
    else:
        st.info("No publication-date data is available.")

with graph_tab:
    st.header("Paper neighbourhood")
    top_papers = get_top_pagerank(client, 10)
    titles = {paper["arxiv_id"]: paper["title"] for paper in top_papers}
    selected_id = st.selectbox(
        "Top papers by PageRank",
        options=[None, *titles],
        format_func=lambda arxiv_id: (
            "Select a paper" if arxiv_id is None else titles[arxiv_id]
        ),
    )
    manual_id = st.text_input("Or enter an arXiv ID")
    arxiv_id = manual_id.strip() or selected_id

    if arxiv_id is None:
        st.info("Select a top paper or enter an arXiv ID to explore its neighbourhood.")
    else:
        neighborhood = get_paper_neighborhood(client, arxiv_id)
        if neighborhood is None:
            st.warning(f"No paper found with arXiv ID `{arxiv_id}`.")
        else:
            nodes, edges = build_neighborhood_graph(
                neighborhood["paper"],
                neighborhood["authors"],
                neighborhood["cited"],
                neighborhood["citing"],
                max_nodes=50,
            )
            network = Network(
                height="650px",
                width="100%",
                directed=True,
                cdn_resources="in_line",
            )
            for node in nodes:
                is_author = node["kind"] == "author"
                network.add_node(
                    node["id"],
                    label=node["label"],
                    title=node["kind"].title(),
                    shape="dot",
                    color="#f0ad4e" if is_author else "#4c78a8",
                )
            for edge in edges:
                network.add_edge(
                    edge["from"], edge["to"], label=edge["label"], arrows="to"
                )
            st.components.v1.html(network.generate_html(), height=680, scrolling=True)
