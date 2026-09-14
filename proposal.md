# Reducing LLM Context Overhead via Semantic Document Retrieval in Agentic Software Development Workflows

**MSDS 453 — Natural Language Processing | Final Project Proposal**
**Albin Anto Jose**

---

## Topic

Large language model (LLM) agents used for software development tasks — such as automated code generation, bug triage, and documentation retrieval — routinely exhaust session token budgets within minutes. The dominant cause is indiscriminate context loading: at each agent step, full documents or entire file trees are concatenated into the prompt regardless of relevance. This inflates token consumption, increases inference cost, and degrades response quality as irrelevant context dilutes the signal.

This project proposes and evaluates a three-layer semantic retrieval architecture that selectively loads only contextually relevant document passages into the LLM prompt at each agent step, and evicts stale context between steps. The central claim is that selective context loading via semantic retrieval significantly reduces token consumption in agentic coding workflows without a proportional loss in task performance.

---

## Data Sources

**Primary — GitHub REST API**
Four open-source software repositories will be scraped: FastAPI, Supabase, Directus, and Appwrite. These represent a range of complexity and domain coverage (web frameworks, databases, headless CMS, authentication). For each repository the following artifact types are collected: README, English-language Markdown documentation, changelogs, Python module docstrings (extracted via AST parsing), and GitHub Issues with comments. The FastAPI corpus has been collected and validated (784 artifacts: 154 docs, 128 docstrings, 500 issues, 1 README, 1 changelog). The remaining three repositories will follow the same pipeline. Expected total: 3,000–7,500 artifacts across all four repositories.

**Secondary — Stack Overflow Public Data Dump**
The Stack Exchange data dump (archive.org, CC BY-SA 4.0) will be filtered to question-answer pairs that are tagged with the target repository technology stacks, have a score ≥ 5, and have an accepted answer. This provides community-validated natural language descriptions of real developer problems. Expected yield: 50,000–100,000 filtered QA pairs.

---

## Preprocessing

Raw artifacts will be processed through a four-stage pipeline prior to any modeling:

1. **Cleaning** — Markdown syntax, HTML tags, autodoc directives (`::: module.path`), and inline URLs are stripped, leaving plain prose and code. Autodoc stubs with no prose content are dropped.
2. **Chunking** — Text is split into overlapping windows of 256–512 tokens, measured using the tokenizer of the downstream embedding model (`AutoTokenizer` from Hugging Face) to ensure token counts are accurate to the model's vocabulary. A stride of 64 tokens is used to preserve sentence-boundary context.
3. **Deduplication** — Near-duplicate chunks are identified by content hash and removed. This addresses the 10 exact duplicates already observed in the FastAPI corpus and similar redundancy expected across repositories.
4. **Metadata tagging** — Each chunk is labeled with `source_type` (doc, docstring, issue, readme), `repo`, and `created_at` date to support downstream filtering and evaluation.

---

## Machine Learning Methods

The architecture consists of three sequential layers:

**Layer 1 — Semantic Segmentation (BERTopic)**
BERTopic (Grootendorst, 2022) is applied to the preprocessed corpus to group chunks into semantically coherent clusters. The pipeline is: sentence embeddings → UMAP dimensionality reduction → HDBSCAN density clustering → c-TF-IDF topic labeling per cluster. A custom CountVectorizer tokenizer is used to split camelCase and snake_case identifiers and suppress high-frequency code boilerplate (`def`, `return`, `self`, `import`) that would otherwise dominate topic labels. This layer provides coarse routing: a query is mapped to its most relevant cluster(s) before fine retrieval begins.

**Layer 2 — Dense Passage Retrieval (DPR Bi-Encoder)**
Within the relevant cluster, a bi-encoder (Karpukhin et al., 2020) encodes the agent query and all candidate chunks into a shared vector space and retrieves the top-k passages by cosine similarity. The embedding model is `sentence-transformers/all-MiniLM-L6-v2` as the baseline; CodeBERT and UniXcoder will be evaluated as code-aware alternatives, as general-domain models trained on Wikipedia and news text underperform on code identifiers and API terminology.

**Layer 3 — Selective Context Injection**
Only the top-k retrieved passages are loaded into the LLM context at each agent step. A context registry tracks content hashes of all currently loaded passages to prevent redundant reloading of content already in context. Between steps, passages that were not referenced in the previous agent response are evicted. This step-aware eviction is the core contribution: no prior RAG system manages context as a stateful resource across agent turns.

---

## Evaluation Metrics

| Metric | Description |
|---|---|
| **Token reduction %** | Tokens consumed per task (selective) vs. flat RAG baseline |
| **Retrieval Precision@k** | Fraction of top-k retrieved chunks judged relevant to the query |
| **Task completion rate** | Fraction of coding tasks completed correctly end-to-end |
| **Latency per agent step** | Wall-clock time including retrieval overhead |
| **Cost per task** | Estimated API cost at published token pricing |
| **CodeBLEU / Exact Match** | Output quality on code generation sub-tasks |

The comparison baseline is standard flat RAG (Lewis et al., 2020): retrieve top-k, concatenate all into a single context block, no clustering, no step-awareness.

---

## Related Work and Novelty

Lewis et al. (2020) established retrieval-augmented generation as the standard approach for knowledge-intensive NLP tasks. Their system retrieves top-k passages and concatenates them into a flat context block — effective for single-turn question answering but not designed for multi-step agentic workflows where context accumulates across turns.

Topic-guided retrieval has been explored using LDA (Blei et al., 2003), which assigns each document a probability distribution over latent topics and can pre-filter the retrieval space. However, LDA operates on bag-of-words representations and requires the number of topics K to be specified in advance. On mixed code-and-prose corpora, LDA topics are dominated by high-frequency programming tokens (`def`, `return`, `import`) that carry no semantic meaning. BERTopic avoids this by separating the semantic representation step (dense embeddings) from the labeling step (c-TF-IDF), producing coherent, interpretable topic clusters even over code-heavy text.

Karpukhin et al. (2020) introduced the DPR bi-encoder architecture for dense retrieval. This project uses DPR as the fine-retrieval layer within BERTopic clusters — a two-stage coarse-then-fine retrieval design not present in the original DPR work.

The primary novelty of this project is **step-aware selective context injection**: treating the LLM context window as a managed resource with explicit load and eviction operations across agent turns, tracked by a content registry. Standard RAG systems are stateless with respect to context — they retrieve and concatenate anew at each step. This project operationalizes context as a first-class concern in agentic workflows, which is the architectural change required to meaningfully reduce token overhead over multi-step sessions.
