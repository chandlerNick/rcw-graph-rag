# GraphRAG package architecture

This package contains the application's executable components:

- `data_ingest_crawler.py` crawls the RCW source and emits JSON/JSONL data.
- `graph_store.py` builds LanceDB and NetworkX stores, and exposes retrieval
  and the `rcw-graph-rag` command-line interface.
- `frontend.py` is the Streamlit chat interface. It imports `RCWGraphRAG`,
  displays retrieved provisions, and sends the question plus retrieved context
  to Google Gemini.
- `__init__.py` exports `RCWGraphRAG` and `main` for package consumers and the
  configured command-line entry point.

The package is under `src/rcw_graph_rag/`, so its Python module name uses
underscores. The frontend is launched by its file path from the repository
root, while the graph store is also available through the installed console
script declared in `pyproject.toml`.

## Data flow

### 1. Crawl

`data_ingest_crawler.py` reads pages from the wa-law.org RCW mirror (or from an
offline site checkout) and writes:

- `titles.json` and `chapters.json`: hierarchy metadata.
- `sections.jsonl`: one record per parsed RCW section, with text, metadata,
  amendment history, and a prepared `embedding_text`.
- `edges.jsonl`: hierarchy, ordering, cross-reference, and amendment edges.
- `crawl_report.json`: counts, warnings, and failures.

The default output directory is `rcw_data/`. The crawl uses an on-disk HTML
cache unless `--no-cache` is specified.

### 2. Build

Run `uv run rcw-graph-rag build <data_dir> <db_dir>`. The build:

1. Embeds each section's `embedding_text`.
2. Writes section text, metadata, and vectors to the LanceDB `sections` table.
3. Builds a NetworkX `MultiDiGraph` from hierarchy nodes and edges.
4. Writes `graph.pkl` and `meta.json` beside the LanceDB directory.

The default sentence-transformer model is
`CSI-lab/Washington-state-law-embedding-model-Base`. It is loaded locally by
Sentence Transformers; the GPU is used when supported by the installed
PyTorch/CUDA build. The optional `hash` embedder avoids loading a neural model
but has lower semantic retrieval quality. The selected embedder is recorded in
`meta.json` and loaded again for queries, including queries from the frontend.

### 3. Retrieve

`RCWGraphRAG.retrieve()` performs these steps:

1. Prefixes the query as required by the embedding model and finds the top-k
   cosine-similar LanceDB records.
2. Traverses the NetworkX graph around those matches, by default following
   `REFERENCES` and `NEXT_SECTION` edges.
3. Retrieves up to the requested number of linked section records from
   LanceDB.

Graph traversal is bounded by the selected number of hops. The CLI and frontend
expose retrieval controls; the frontend defaults to five vector matches, one
graph hop, and five extra linked sections.

### 4. Answer

The Streamlit frontend sends the conversation, current question, and retrieved
RCW text to the selected Gemini model. It requests a short summary first and a
more detailed answer second, with citations restricted to retrieved sections.
The UI also displays the sections retrieved and their source URLs.

Gemini is an external service: prompts and retrieved statutory text leave the
local machine for generation. The tool is an experimental research aid and
does not provide legal advice.

## Storage layout

For a database directory such as `rcw_db/`:

```text
rcw_db/
├── graph.pkl
├── meta.json
└── lancedb/
    └── sections.lance/
```

`graph.pkl` is a Python pickle containing the NetworkX graph. Load it only from
a database you built or otherwise trust. `meta.json` records the embedder and
node, edge, and section counts. LanceDB stores the searchable section records
and vectors.

## Paths and commands

Run project commands from the repository root unless an absolute path is used.
The default data and database directories are resolved relative to the current
working directory by the CLI; the frontend's default database path is resolved
relative to the repository root. Set `RCW_DB_DIR` to override the frontend
database path.

Examples:

```bash
uv run python src/rcw_graph_rag/data_ingest_crawler.py --help
uv run rcw-graph-rag build rcw_data rcw_db
uv run rcw-graph-rag query rcw_db "your question"
uv run streamlit run src/rcw_graph_rag/frontend.py
```

For setup requirements and a complete quick start, see the
[main README](../../README.md).
