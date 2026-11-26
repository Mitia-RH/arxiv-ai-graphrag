# ArXiv AI — Knowledge Graph & GraphRAG

Explore **arXiv cs.AI** papers as a **Neo4j** graph and query them in English using a
**local LLM** (**Ollama + Qwen3:4b**). 100% local, with no paid APIs.

```
arXiv ──► OpenAlex (citations) ──► Neo4j ◄── Question ► LLM ► Cypher ► Answer
         (ingestion)                          (GraphRAG, Streamlit interface)
```

## Requirements

[Conda](https://conda.io/projects/conda/en/latest/user-guide/install/index.html), Docker,
and [Ollama](https://ollama.com/download). The Conda environment installs Python 3.11.

## 1. Start Neo4j

```bash
cp .env.example .env        # Then update NEO4J_PASSWORD (at least 8 characters).
docker compose up -d
```

Neo4j Browser: <http://localhost:7474> (credentials are in `.env`).

## 2. Install Ollama and a lightweight model

```bash
ollama pull qwen3:4b
```

To use a different Ollama model, download it and update `OLLAMA_MODEL` in `.env`.

## 3. Ingérer

```bash
conda env create -f environment.yml
conda activate arxiv-ai-graphrag
python -m src.ingestion.pipeline --limit 50 --offset 0
```

`--limit` sets the number of papers to fetch, and `--offset` selects where to start in
arXiv's results. Increase the offset to ingest older batches (for example, use `--offset
50` for the next 50 results). Papers already ingested as full papers are skipped; use
`--force` to re-ingest them. Cited-paper stubs do not count as already ingested.
Activate the environment in each new terminal.

## 4. Launch the app

```bash
streamlit run app.py
```

Try asking: *"Which papers are cited the most?"* The **View the Cypher query and raw data**
section below each answer lets you inspect the results and verify the response.

## Tests

```bash
python -m pytest
```

## Design notes

- **Citations:** Only references with an arXiv ID are kept. References not in the corpus
  become `:Paper {stub: true}` nodes (title and URL, no abstract). This surfaces highly
  cited foundational papers.
- **Authors** are identified by name, so two authors with the same name are merged
  (an intentional KISS tradeoff).
- **Safety:** Generated Cypher is filtered to allow read-only queries and is also executed
  in a Neo4j read transaction.
- `:Concept` nodes have a constraint but are not populated yet (phase 2).
- Very recent papers may not yet be available in OpenAlex; they are ingested without citations.

## Project structure

`config.py` · `src/ingestion/` (arXiv, OpenAlex, pipeline) · `src/database/` (client, schema,
repository) · `src/llm/` (Ollama, GraphRAG) · `app.py` (Streamlit) · `tests/`
