# ContextLens

**Reducing LLM Context Overhead via Semantic Document Retrieval in Agentic Software Development Workflows**

MSDS 453 (Natural Language Processing) final project, Northwestern University.

## Overview

LLM coding agents routinely exhaust their context budget by concatenating full documents or file trees into every prompt, regardless of relevance. ContextLens is a three-layer retrieval architecture that selectively loads only relevant document chunks at each agent step and evicts stale context between steps — testing the claim that selective semantic retrieval cuts token consumption without a proportional loss in task performance.

## Architecture

1. **Topic segmentation** — `all-MiniLM-L6-v2` embeddings → UMAP → HDBSCAN → c-TF-IDF (code-aware tokenizer) to discover topic clusters over the corpus (125 topics from 14K+ chunks).
2. **Selective retrieval** — chunk embeddings indexed per-cluster in FAISS (`IndexFlatIP`), with soft routing: a query is matched to its nearest cluster centroids (default top-2) before searching, versus a flat-RAG baseline that searches everything. The index is built with `microsoft/codebert-base`; routing can use CodeBERT (default) or MiniLM centroids (`--minilm-routing`), and MiniLM routing performed better in evaluation (see Results).
3. **Context registry & episodic memory** — a `ContextRegistry` that deduplicates chunks (by id and content hash) and evicts stale context between steps, plus a knowledge-graph context store across agent sessions with configurable eviction policies (recency-based, PageRank-based) to bound cumulative token growth.

## Corpus

~14,030 chunks (avg. 422 tokens) built from:
- **GitHub** (docs, issues, docstrings, changelogs, READMEs): FastAPI, Supabase, Appwrite, Pydantic, Directus
- **Stack Overflow**: Q&A pairs (score-filtered) for the same tag set

## Results

Full tables, figures and discussion are in [`MSDS453_Semantic_RAG_Report.docx`](MSDS453_Semantic_RAG_Report.docx); raw outputs are in `data/results/`.

**Multi-step token use** (MiniLM routing, 18 five-step tasks, average total tokens per task):

| k | flat_accumulate | flat_reset | selective + evict_all | Reduction vs flat_accumulate |
|---|---|---|---|---|
| 5 | 5,729 | 2,442 | 370 | 93.5% |
| 10 | 11,040 | 4,793 | 370 | 96.6% |
| 20 | 20,992 | 9,646 | 370 | 98.2% |

`evict_all` clears the context after every step, so the agent keeps no memory between steps; whether final answers suffer from that was not measured. Keeping one step of history (`evict_older_than(1)`) used *more* tokens than the baseline (+15% to +87%).

**Single-step quality** (100 Stack Overflow queries): selective and flat retrieval score about the same on completion and precision (within ±0.03 at k=5 and k=20), but selective uses more tokens in a single step (+54% to +121%), because there is no earlier context to deduplicate against and the registry adds prompt overhead. The savings only appear across multiple steps.

**Routing model:** MiniLM routing gave far better answers than CodeBERT routing (completion 0.343 vs 0.033 at k=5) and was faster (93 ms vs 109 ms), even though CodeBERT scored higher on embedding similarity.

**Knowledge-graph eviction** (run after the report, `data/results/kg_eviction_summary_table.csv`): PageRank-based eviction cut cumulative tokens by 27% to 62% versus no eviction, depending on how many chunks it drops (55% at the `pagerank_p30` setting). In the three-session evaluation, episodic memory used about 7% more tokens in total than running without it, but its judge score rose from session to session (0.26 → 0.36 → 0.46, against 0.28 → 0.20 → 0.36 without).

## Repo layout

```
src/
├── scraper/        # GitHub + Stack Overflow data collection
├── preprocessing/  # cleaning, chunking, corpus pipeline
├── clustering/     # BERTopic topic segmentation
├── retrieval/      # CodeBERT embeddings + FAISS index + Retriever
├── graph/          # episodic knowledge graph, eviction/forget gate, session retrieval
├── agent/          # context registry / context manager
└── evaluation/     # single-step, multi-step, multi-session eval harnesses and ablations
data/results/       # evaluation outputs (CSV/JSON) and figures, timestamped per run
```

## Setup

```bash
pip install -r requirements.txt   # Python 3.10+ recommended
cp .env.example .env              # GITHUB_TOKEN / STACKEXCHANGE_KEY (optional), ANTHROPIC_API_KEY or OPENAI_API_KEY (for evaluation)
```

Data collection, preprocessing, and model artifacts (embeddings, FAISS indices, BERTopic model) are not checked into this repo. Regenerate them from the repo root in this order:

```bash
python src/scraper/run_scraper.py              # GitHub docs, issues, docstrings, changelogs
python src/scraper/stackoverflow_scraper.py    # Stack Overflow Q&A
python src/preprocessing/pipeline.py           # clean + chunk → data/processed/
python src/clustering/bertopic_cluster.py      # topic clusters → data/models/
python src/retrieval/build_index.py            # CodeBERT embeddings + per-cluster FAISS indices
python src/graph/build_corpus_graph.py         # corpus graph for episodic memory

python -m src.evaluation.harness --n 100 --top-k 5 --minilm-routing     # single-step
python -m src.evaluation.multistep --top-k 5 --evict-all                # multi-step
```

Each evaluation script lists its options with `--help`.

## Data and licensing

The code is released under the [MIT License](LICENSE).

The corpus is not part of this repo and is not covered by that license: Stack Overflow content is licensed under [CC BY-SA](https://stackoverflow.com/help/licensing), and GitHub content remains under each source repository's own license. Any text excerpts in `data/results/` stay under their original licenses.

## Author

Albin Anto Jose
