"""Streamlit interface: GraphRAG chat and statistics. Run with: streamlit run app.py"""
import logging

import streamlit as st

import config
from src.database.neo4j_client import Neo4jClient
from src.database.repository import get_graph_stats
from src.database.schema import create_schema
from src.ingestion import pipeline
from src.llm import graph_rag
from src.llm.llm_client import is_available

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
    nb = st.number_input("Number of papers", 5, 500, 20, step=5)
    if st.button("Ingest new papers"):
        with st.spinner("Fetching and enriching papers (this may take a few minutes)..."):
            pipeline.run(int(nb))
        st.rerun()

# ---------- Chat ----------
st.title("🕸️ ArXiv AI — GraphRAG")
st.caption("Ask a question about AI research (answers are based on the Neo4j graph).")

if "history" not in st.session_state:
    st.session_state.history = []


def show_details(msg: dict) -> None:
    with st.expander("View the Cypher query and raw data"):
        st.code(msg["cypher"], language="cypher")
        st.json(msg["rows"])


for msg in st.session_state.history:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if msg["role"] == "assistant":
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
                res = {"answer": "", "cypher": "", "rows": [], "error": str(error)}
        text = res["answer"] or f"⚠️ {res['error']}"
        st.write(text)
        msg = {"role": "assistant", "content": text, "cypher": res["cypher"], "rows": res["rows"]}
        show_details(msg)
    st.session_state.history.append(msg)
