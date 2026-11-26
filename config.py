"""Centralized configuration loaded from .env (never hardcode passwords)."""
import os

from dotenv import load_dotenv

load_dotenv()

# --- Neo4j ---
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "")

# --- Ollama (LLM local) ---
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen3:4b")
EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text")
EMBED_DIM = 768

# --- Ingestion ---
ARXIV_CATEGORY = "cs.AI"                         # Fixed project scope.
MAX_PAPERS = int(os.getenv("MAX_PAPERS", "100"))
OPENALEX_EMAIL = os.getenv("OPENALEX_EMAIL", "")
OPENALEX_DELAY_S = 0.2                           # Polite delay between OpenAlex requests.

# --- GraphRAG ---
MAX_ROWS_FOR_LLM = 15      # Result rows passed to the LLM (small context window).
MAX_CHARS_PER_VALUE = 300  # Truncate long text values (e.g. abstracts).
