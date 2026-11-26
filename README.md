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
ollama pull nomic-embed-text
```

To use a different chat model, download it and update `OLLAMA_MODEL` in `.env`. Set
`EMBED_MODEL` there to select a different embedding model.

## 3. Ingest

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

To add embeddings to existing papers that were ingested before semantic search was
enabled, run `python -m src.ingestion.backfill_embeddings`. Use `--limit` to cap the
number of papers processed, for example `--limit 500`.

## 4. Launch the app

```bash
streamlit run app.py
```

Try asking: *"Which papers are cited the most?"* The **View the Cypher query and raw data**
section below each answer lets you inspect the results and verify the response.
Use **Insights** for PageRank rankings, community summaries, and monthly publication counts;
the tab can compute analytics when they are missing. **Graph** renders the selected paper's
one-hop author and citation neighbourhood.

## Graph analytics

Neo4j starts with the Graph Data Science (GDS) plugin configured; the first container
start downloads it. After ingestion, compute PageRank scores and Leiden communities from
the **Insights** tab or with:

```bash
python -m src.database.analytics
```

The results are written to `Paper.pagerank` and `Paper.community`. The repository helpers
`get_top_pagerank(client, n)` and `get_community_summary(client)` retrieve the rankings and
community sizes with their three most-cited titles. These database procedures are invoked
only by application code through `client.run_write`, not by LLM-generated Cypher.

This project uses the GDS Community Edition with Neo4j Community Edition. GDS Community
Edition limits algorithm concurrency to four; the graph projections also need to fit in
available memory. Higher GDS concurrency requires an Enterprise Edition license. See the
[GDS system requirements](https://neo4j.com/docs/graph-data-science/current/installation/System-requirements/)
and [Docker installation guide](https://neo4j.com/docs/graph-data-science/current/installation/installation-docker/).

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
- **Indexes:** The full-text index on paper titles and abstracts supports text search.
  The published-date index speeds up date filtering and ordering, while the citation-count
  index supports efficient citation-based filtering and ranking.
- `:Concept` nodes have a constraint but are not populated yet (phase 2).
- Very recent papers may not yet be available in OpenAlex; they are ingested without citations.

## Project structure

`config.py` · `src/ingestion/` (arXiv, OpenAlex, pipeline) · `src/database/` (client, schema,
repository, analytics) · `src/llm/` (Ollama, GraphRAG) · `app.py` (Streamlit) · `tests/`
