"""Interface Streamlit : chat GraphRAG + statistiques. Lancer : streamlit run app.py"""
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
    st.error(f"Neo4j inaccessible : {error}\n\nAs-tu lancé `docker compose up -d` ?")
    st.stop()

# ---------- Barre latérale ----------
with st.sidebar:
    st.header("Base Neo4j")
    stats = get_graph_stats(client)
    st.metric("Papiers (cs.AI)", stats["papers"])
    st.metric("Papiers cités (références)", stats["stubs"])
    st.metric("Auteurs", stats["authors"])
    st.metric("Citations", stats["citations"])

    st.header("LLM local")
    st.write(f"Modèle : `{config.OLLAMA_MODEL}`")
    st.write("✅ Ollama actif" if is_available() else "❌ Ollama injoignable")

    st.header("Ingestion")
    nb = st.number_input("Nombre de papiers", 5, 500, 20, step=5)
    if st.button("Ingérer de nouveaux papiers"):
        with st.spinner("Collecte et enrichissement en cours (quelques minutes)..."):
            pipeline.run(int(nb))
        st.rerun()

# ---------- Chat ----------
st.title("🕸️ ArXiv AI — GraphRAG")
st.caption("Pose une question sur l'état de l'art en IA (réponses issues du graphe Neo4j).")

if "history" not in st.session_state:
    st.session_state.history = []


def show_details(msg: dict) -> None:
    with st.expander("Voir la requête Cypher et les données brutes"):
        st.code(msg["cypher"], language="cypher")
        st.json(msg["rows"])


for msg in st.session_state.history:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if msg["role"] == "assistant":
            show_details(msg)

if question := st.chat_input("Ex : Quels papiers parlent d'agents LLM ?"):
    st.session_state.history.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.write(question)
    with st.chat_message("assistant"):
        with st.spinner("Je réfléchis..."):
            try:
                res = graph_rag.ask(client, question)
            except Exception as error:  # ex. Ollama éteint
                res = {"answer": "", "cypher": "", "rows": [], "error": str(error)}
        text = res["answer"] or f"⚠️ {res['error']}"
        st.write(text)
        msg = {"role": "assistant", "content": text, "cypher": res["cypher"], "rows": res["rows"]}
        show_details(msg)
    st.session_state.history.append(msg)
