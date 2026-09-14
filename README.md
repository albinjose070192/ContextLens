# ContextLens

**Reducing LLM Context Overhead via Semantic Document Retrieval in Agentic Software Development Workflows**

MSDS 453 (Natural Language Processing) final project, Northwestern University.

## Overview

LLM coding agents routinely exhaust their context budget by concatenating full documents or file trees into every prompt, regardless of relevance. ContextLens is a three-layer retrieval architecture that selectively loads only relevant document chunks at each agent step and evicts stale context between steps — testing the claim that selective semantic retrieval cuts token consumption without a proportional loss in task performance.

## Architecture

1. **Topic segmentation** — `all-MiniLM-L6-v2` embeddings → UMAP → HDBSCAN → c-TF-IDF (code-aware tokenizer) to discover topic clusters over the corpus (125 topics from 14K+ chunks).
2. **Selective retrieval** — `microsoft/codebert-base` embeddings indexed per-cluster in FAISS (`IndexFlatIP`), with soft routing: a query is matched to its top-2 nearest cluster centroids before searching, versus a flat-RAG baseline that searches everything.
3. **Episodic memory** — a knowledge-graph context registry across agent sessions with configurable eviction policies (recency-based, PageRank-based) to bound cumulative token growth across multi-turn/multi-session workflows.

## Corpus

~14,030 chunks built from:
- **GitHub** (docs, issues, docstrings, changelogs, READMEs): FastAPI, Supabase, Appwrite, Pydantic, Directus
- **Stack Overflow**: Q&A pairs (score-filtered) for the same tag set

## Results (highlights)

- Selective (cluster-routed) retrieval vs. flat-RAG baseline, evaluated with an LLM-as-judge harness across single-step and multi-step agent tasks.
- Knowledge-graph eviction ablation: cumulative token usage reduced up to **55%** vs. a no-eviction baseline (PageRank-based eviction), with graceful degradation in retained context size.
- Multi-session evaluation across 3 simulated sessions comparing token cost and task win-rate with vs. without episodic KG memory.

Full experiment tables, figures, and discussion are in `MSDS453_Semantic_RAG_Report.docx`.

## Repo layout

```
src/
├── scraper/        # GitHub + Stack Overflow data collection
├── preprocessing/  # cleaning, chunking, corpus pipeline
├── clustering/      # BERTopic topic segmentation
├── retrieval/       # CodeBERT embeddings + FAISS index + Retriever
├── graph/           # episodic knowledge graph, eviction/forget gate, session retrieval
├── agent/           # context registry / context manager
└── evaluation/      # single-step, multi-step, multi-session eval harnesses
notebooks/
configs/
```

## Setup

```bash
pip install -r requirements.txt  # or see imports in src/
cp .env.example .env             # set GITHUB_TOKEN / STACKEXCHANGE_KEY (optional)
```

Data collection, preprocessing, and model artifacts (embeddings, FAISS indices, BERTopic model) are not checked into this repo — see `src/scraper/`, `src/preprocessing/`, `src/clustering/`, and `src/retrieval/build_index.py` to regenerate them from scratch.

## Author

Albin Anto Jose
