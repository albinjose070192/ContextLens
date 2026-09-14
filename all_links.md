# Project Resource Index
**MSDS 453 — Reducing LLM Context Overhead via Semantic Document Retrieval**

---

## Context Files
`/Users/albin/Desktop/context/Context - Semantic RAG Retrieval for Agentic Systems/`

| File | Covers |
|---|---|
| `MSDS453_SESSION_2026-05-16_DATA_COLLECTION_PREPROCESSING_LAYERS1_2_COMPLETE.md` | Data collection, preprocessing, BERTopic (Layer 1), FAISS indices (Layer 2) |
| `MSDS453_SESSION_2026-05-17_LAYER3_EVALUATION_COMPLETE.md` | Layer 3 (ContextRegistry), first evaluation runs |
| `MSDS453_SESSION_2026-05-17_ALL_EXPERIMENTS_COMPLETE.md` | All 10 original experiments (CodeBERT routing) |
| `MSDS453_SESSION_2026-05-17_MINILM_RERUNS_COMPLETE.md` | MiniLM re-runs, revised narrative, routing classifier ← **most current** |

---

## Source Code
`/Users/albin/semantic-rag/src/`

### Layer 1 — Segmentation
- `/Users/albin/semantic-rag/src/clustering/bertopic_cluster.py`

### Layer 2 — Retrieval
- `/Users/albin/semantic-rag/src/retrieval/retriever.py`
- `/Users/albin/semantic-rag/src/retrieval/build_index.py`

### Layer 3 — Context Management
- `/Users/albin/semantic-rag/src/agent/context_manager.py`

### Data Pipeline
- `/Users/albin/semantic-rag/src/scraper/stackoverflow_scraper.py`
- `/Users/albin/semantic-rag/src/scraper/github_scraper.py`
- `/Users/albin/semantic-rag/src/scraper/fetch_issues_only.py`
- `/Users/albin/semantic-rag/src/scraper/run_scraper.py`
- `/Users/albin/semantic-rag/src/preprocessing/chunk.py`
- `/Users/albin/semantic-rag/src/preprocessing/clean.py`
- `/Users/albin/semantic-rag/src/preprocessing/pipeline.py`

### Evaluation Scripts
| Path | Purpose |
|---|---|
| `/Users/albin/semantic-rag/src/evaluation/harness.py` | Single-step flat vs selective eval (`--minilm-routing`, `--openai`) |
| `/Users/albin/semantic-rag/src/evaluation/multistep.py` | Multi-step simulation (`--minilm-routing`, `--evict-all`) |
| `/Users/albin/semantic-rag/src/evaluation/ablation.py` | top_clusters ablation (`--minilm-routing`, `--openai`) |
| `/Users/albin/semantic-rag/src/evaluation/eviction_ablation.py` | Eviction strategy comparison (`--minilm-routing`, `--openai`) |
| `/Users/albin/semantic-rag/src/evaluation/embedding_ablation.py` | CodeBERT vs MiniLM routing + `MiniLMRoutingRetriever` class |
| `/Users/albin/semantic-rag/src/evaluation/per_tag.py` | Per-tag breakdown (loads eval CSVs) |
| `/Users/albin/semantic-rag/src/evaluation/noise_analysis.py` | Noise cluster analysis |
| `/Users/albin/semantic-rag/src/evaluation/threshold_ablation.py` | Score threshold ablation |
| `/Users/albin/semantic-rag/src/evaluation/dedup_analysis.py` | Dedup benefit quantification |
| `/Users/albin/semantic-rag/src/evaluation/qualitative_compare.py` | 3-query side-by-side response comparison |
| `/Users/albin/semantic-rag/src/evaluation/routing_confidence.py` | Gap/entropy vs outcome correlation |
| `/Users/albin/semantic-rag/src/evaluation/routing_classifier.py` | Logistic regression routing classifier |
| `/Users/albin/semantic-rag/src/evaluation/context_only_tokens.py` | Strips overhead, measures context-only tokens |

---

## Data

### Raw
- `/Users/albin/semantic-rag/data/raw/stackoverflow/stackoverflow_all_raw.json` — 826 Q&A pairs (fastapi/pydantic/supabase/appwrite)
- `/Users/albin/semantic-rag/data/raw/stackoverflow/fastapi_raw.json`
- `/Users/albin/semantic-rag/data/raw/stackoverflow/pydantic_raw.json`
- `/Users/albin/semantic-rag/data/raw/stackoverflow/supabase_raw.json`
- `/Users/albin/semantic-rag/data/raw/stackoverflow/appwrite_raw.json`
- `/Users/albin/semantic-rag/data/raw/github/tiangolo_fastapi_raw.json`
- `/Users/albin/semantic-rag/data/raw/github/pydantic_pydantic_raw.json`
- `/Users/albin/semantic-rag/data/raw/github/supabase_supabase_raw.json`
- `/Users/albin/semantic-rag/data/raw/github/appwrite_appwrite_raw.json`
- `/Users/albin/semantic-rag/data/raw/github/directus_directus_raw.json`

### Processed
- `/Users/albin/semantic-rag/data/processed/corpus_all.jsonl` — 14,030 chunks
- `/Users/albin/semantic-rag/data/processed/corpus_with_topics.jsonl` — chunks with BERTopic topic assignments
- `/Users/albin/semantic-rag/data/processed/corpus_stats.json`

### Models
- `/Users/albin/semantic-rag/data/models/bertopic_model` — saved BERTopic model
- `/Users/albin/semantic-rag/data/models/embeddings_minilm.npy` — 14,030 × 384 MiniLM embeddings
- `/Users/albin/semantic-rag/data/models/codebert_embeddings.npy` — 14,030 × 768 CodeBERT embeddings
- `/Users/albin/semantic-rag/data/models/cluster_centroids.json` — MiniLM centroids, 125 clusters ← **use for routing**
- `/Users/albin/semantic-rag/data/models/codebert_centroids.json` — CodeBERT centroids (misaligned, do not use for routing)
- `/Users/albin/semantic-rag/data/models/topic_info.json`
- `/Users/albin/semantic-rag/data/models/topic_chunk_map.json`
- `/Users/albin/semantic-rag/data/models/faiss_indices/topic_*.index` — 126 per-cluster FAISS indices (CodeBERT embeddings)

---

## Results
`/Users/albin/semantic-rag/data/results/`

### Single-Step Eval
| File | Config |
|---|---|
| `eval_20260517_030029.csv/.json` | k=5, CodeBERT routing |
| `eval_20260517_113748.csv/.json` | k=10, CodeBERT routing |
| `eval_20260517_110636.csv/.json` | k=20, CodeBERT routing |
| `eval_20260517_191449.csv/.json` | k=5, MiniLM routing, OpenAI judge ← **primary** |
| `eval_20260517_202227.csv/.json` | k=10, MiniLM routing, OpenAI judge |
| `eval_20260517_210244.csv/.json` | k=20, MiniLM routing, OpenAI judge |

### Multi-Step — CodeBERT routing, evict_older_than(1)
| File | Config |
|---|---|
| `multistep_20260517_124943_steps/tasks.csv/.json` | k=5 |
| `multistep_20260517_131126_steps/tasks.csv/.json` | k=10 |
| `multistep_20260517_131134_steps/tasks.csv/.json` | k=20 |

### Multi-Step — MiniLM routing, evict_older_than(1)
| File | Config |
|---|---|
| `multistep_20260517_200320_steps/tasks.csv/.json` | k=5 |
| `multistep_20260517_200356_steps/tasks.csv/.json` | k=10 |
| `multistep_20260517_200421_steps/tasks.csv/.json` | k=20 |

### Multi-Step — MiniLM routing, evict_all ← **primary token efficiency result**
| File | Config |
|---|---|
| `multistep_20260517_220427_steps/tasks.csv/.json` | k=5 — 93.5% reduction vs flat_accumulate |
| `multistep_20260517_220451_steps/tasks.csv/.json` | k=10 — 96.6% reduction vs flat_accumulate |
| `multistep_20260517_220516_steps/tasks.csv/.json` | k=20 — 98.2% reduction vs flat_accumulate |

### Ablations
| File | Experiment | Routing |
|---|---|---|
| `ablation_topclusters_20260517_131542.csv/.png` | top_clusters | CodeBERT |
| `ablation_topclusters_20260517_211545.csv/.png` | top_clusters | MiniLM ← current |
| `eviction_20260517_135058_steps/tasks.csv/.png` | Eviction strategies | CodeBERT |
| `eviction_20260517_211044_steps/tasks.csv/.png` | Eviction strategies | MiniLM ← current |
| `embedding_ablation_20260517_181019.csv/.png` | CodeBERT vs MiniLM routing | — |
| `dedup_20260517_180450.csv/.png` | Dedup quantification | CodeBERT |
| `noise_20260517_141243.csv/.png` | Noise cluster analysis | CodeBERT |
| `threshold_20260517_142748.csv/.png` | Score threshold ablation | CodeBERT |
| `per_tag_20260517_141216.csv/.png` | Per-tag breakdown | CodeBERT |
| `per_tag_20260517_210619.csv/.png` | Per-tag breakdown | MiniLM ← current |
| `routing_confidence.csv/.png` | Gap/entropy vs outcome correlation | MiniLM |
| `routing_classifier.png` | Logistic regression classifier | MiniLM |
| `routing_classifier_features.csv` | Classifier feature values per query | MiniLM |

---

## Other
- `/Users/albin/semantic-rag/proposal.md` — original project proposal
- `/Users/albin/semantic-rag/all_links.md` — this file
