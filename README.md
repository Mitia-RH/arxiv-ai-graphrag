# ArXiv AI — Knowledge Graph & GraphRAG

Explore les articles **arXiv cs.AI** sous forme de graphe (**Neo4j**) et interroge-les en
français grâce à un **LLM local** (**Ollama + Mistral**). 100 % local, aucune API payante.

```
arXiv ──► OpenAlex (citations) ──► Neo4j ◄── Question ► LLM ► Cypher ► Réponse
         (ingestion)                          (GraphRAG, interface Streamlit)
```

## Prérequis
Python 3.11+, Docker, et [Ollama](https://ollama.com/download).

## 1. Lancer Neo4j
```bash
cp .env.example .env        # puis modifie NEO4J_PASSWORD (8 caractères min.)
docker compose up -d
```
Neo4j Browser : <http://localhost:7474> (identifiants du `.env`).

## 2. Installer Ollama et un modèle léger
```bash
ollama pull mistral         # ~4 Go de RAM
```
RAM limitée ? Essaie `ollama pull llama3.2:3b` (ou un modèle `ministral`) puis change
`OLLAMA_MODEL` dans `.env`.

## 3. Installer et ingérer
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m src.ingestion.pipeline --limit 50
```
Relançable à volonté : `MERGE` évite tout doublon.

## 4. Lancer l'interface
```bash
streamlit run app.py
```
Pose par exemple : *« Quels sont les papiers les plus cités ? »*. Sous chaque réponse,
le bloc « Voir la requête Cypher et les données brutes » permet de vérifier l'absence d'hallucination.

## Tests
```bash
python -m pytest
```

## Choix de conception (à connaître)
- **Citations** : on ne garde que les références qui ont elles-mêmes un ID arXiv. Celles qui ne
  sont pas dans le corpus deviennent des nœuds `:Paper {stub: true}` (titre + lien, sans résumé).
  C'est ce qui fait apparaître les papiers « fondateurs » très cités.
- **Auteurs** identifiés par leur nom : deux homonymes seront fusionnés (limite assumée, KISS).
- **Sécurité** : le Cypher généré est filtré (lecture seule) *et* exécuté dans une transaction
  de lecture Neo4j.
- Les `:Concept` (phase 2) ont leur contrainte mais ne sont pas encore alimentés.
- Les papiers très récents sont parfois absents d'OpenAlex : ils sont chargés sans citations.

## Structure
`config.py` · `src/ingestion/` (arXiv, OpenAlex, pipeline) · `src/database/` (client, schéma, repository)
· `src/llm/` (Ollama, GraphRAG) · `app.py` (Streamlit) · `tests/`
