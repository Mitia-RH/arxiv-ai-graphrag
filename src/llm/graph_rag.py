"""GraphRAG : Question -> Cypher -> Neo4j -> Réponse rédigée.

Chaque étape est visible et le Cypher généré est toujours renvoyé (transparence).
"""
import json
import logging
import re

import config
from src.llm.llm_client import chat

logger = logging.getLogger(__name__)

CYPHER_SYSTEM_PROMPT = """Tu génères des requêtes Cypher (Neo4j) en LECTURE SEULE.
Réponds UNIQUEMENT avec la requête, sans explication ni balises.

Schéma :
(:Author {name})-[:WROTE]->(:Paper {arxiv_id, title, abstract, published_date, url, citation_count, stub})
(:Paper)-[:BELONGS_TO]->(:Category {code})
(:Paper)-[:CITES]->(:Paper)

Règles :
- Titres et résumés sont en ANGLAIS : cherche avec toLower(...) CONTAINS 'mot anglais minuscule'.
- stub=true signifie papier seulement cité (pas de résumé). Pour les papiers complets : WHERE p.stub = false.
- Toujours terminer par LIMIT 10 (ou moins).
- Renvoie toujours p.title et p.url quand tu retournes des papiers.

Exemples :
Question: Quels sont les papiers les plus cités ?
MATCH (p:Paper)<-[:CITES]-(c:Paper) RETURN p.title AS title, p.url AS url, count(c) AS nb_citations ORDER BY nb_citations DESC LIMIT 10

Question: Quels papiers parlent de reinforcement learning ?
MATCH (p:Paper) WHERE p.stub = false AND (toLower(p.title) CONTAINS 'reinforcement learning' OR toLower(p.abstract) CONTAINS 'reinforcement learning') RETURN p.title AS title, p.url AS url, p.published_date AS date ORDER BY date DESC LIMIT 10

Question: Quels auteurs ont écrit le plus de papiers ?
MATCH (a:Author)-[:WROTE]->(p:Paper) RETURN a.name AS author, count(p) AS nb_papers ORDER BY nb_papers DESC LIMIT 10
"""

ANSWER_SYSTEM_PROMPT = """Tu es un assistant de veille scientifique. Réponds en français, en quelques phrases,
UNIQUEMENT à partir des résultats fournis. Cite les titres et les liens arXiv.
Si les résultats sont vides, dis simplement qu'aucun papier n'a été trouvé."""

FORBIDDEN = re.compile(
    r"\b(CREATE|DELETE|DETACH|SET|MERGE|REMOVE|DROP|LOAD|FOREACH|CALL)\b", re.IGNORECASE
)


def clean_cypher(raw: str) -> str:
    """Retire les balises Markdown ```cypher ... ``` et le ';' final ajoutés par le modèle."""
    text = re.sub(r"```(?:cypher)?", "", raw, flags=re.IGNORECASE).strip()
    return text.split(";")[0].strip()


def is_read_only(cypher: str) -> bool:
    """Vrai si la requête est une lecture simple (MATCH ... RETURN) sans mot-clé d'écriture."""
    # On vide les chaînes de caractères : un titre contenant 'set' ne doit pas être refusé.
    without_strings = re.sub(r"'[^']*'|\"[^\"]*\"", "''", cypher)
    starts_ok = re.match(r"\s*(MATCH|OPTIONAL\s+MATCH|WITH)\b", without_strings, re.IGNORECASE)
    return bool(starts_ok) and not FORBIDDEN.search(without_strings) and ";" not in without_strings


def _shrink(rows: list[dict]) -> str:
    """Réduit les résultats pour tenir dans la petite fenêtre de contexte du modèle."""
    def cut(v):
        return v[:config.MAX_CHARS_PER_VALUE] if isinstance(v, str) else v
    small = [{k: cut(v) for k, v in row.items()} for row in rows[:config.MAX_ROWS_FOR_LLM]]
    return json.dumps(small, ensure_ascii=False, default=str)


def ask(client, question: str) -> dict:
    """Pose une question au graphe. Retourne {answer, cypher, rows, error}."""
    result = {"answer": "", "cypher": "", "rows": [], "error": None}

    cypher = clean_cypher(chat(CYPHER_SYSTEM_PROMPT, f"Question: {question}"))
    result["cypher"] = cypher
    if not is_read_only(cypher):
        result["error"] = "Requête refusée : elle n'est pas en lecture seule."
        return result

    try:
        rows = client.run_read(cypher)
    except Exception as error:  # syntaxe Cypher invalide : 1 seule tentative d'auto-correction
        logger.warning("Cypher invalide (%s), nouvelle tentative.", error)
        fix_prompt = (f"Question: {question}\nRequête précédente: {cypher}\n"
                      f"Erreur Neo4j: {error}\nCorrige la requête.")
        cypher = clean_cypher(chat(CYPHER_SYSTEM_PROMPT, fix_prompt))
        result["cypher"] = cypher
        if not is_read_only(cypher):
            result["error"] = "Requête corrigée refusée : elle n'est pas en lecture seule."
            return result
        try:
            rows = client.run_read(cypher)
        except Exception as error2:
            result["error"] = f"Cypher invalide après correction : {error2}"
            return result

    result["rows"] = rows
    result["answer"] = chat(
        ANSWER_SYSTEM_PROMPT, f"Question: {question}\nRésultats: {_shrink(rows)}"
    )
    return result
