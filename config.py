"""Configuration centralisée : tout vient du fichier .env (jamais de mot de passe en dur)."""
import os

from dotenv import load_dotenv

load_dotenv()

# --- Neo4j ---
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "")

# --- Ollama (LLM local) ---
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "mistral")

# --- Ingestion ---
ARXIV_CATEGORY = "cs.AI"                         # périmètre fixe du projet
MAX_PAPERS = int(os.getenv("MAX_PAPERS", "100"))
OPENALEX_EMAIL = os.getenv("OPENALEX_EMAIL", "")
OPENALEX_DELAY_S = 0.2                           # pause polie entre deux appels OpenAlex

# --- GraphRAG ---
MAX_ROWS_FOR_LLM = 15      # lignes de résultat transmises au LLM (petite fenêtre de contexte)
MAX_CHARS_PER_VALUE = 300  # troncature des textes longs (ex. résumés)
