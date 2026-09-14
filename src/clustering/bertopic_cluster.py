#!/usr/bin/env python3
"""
Layer 1 — BERTopic semantic segmentation.

Pipeline:
  1. Load corpus_all.jsonl
  2. Embed chunks with all-MiniLM-L6-v2
  3. Reduce dimensions with UMAP
  4. Cluster with HDBSCAN
  5. Label clusters with c-TF-IDF (custom code-aware tokenizer)
  6. Save model, embeddings, topic assignments, cluster centroids

Outputs:
  data/models/bertopic_model/          BERTopic serialised model
  data/models/embeddings_minilm.npy    chunk embeddings (14k × 384)
  data/models/cluster_centroids.json   mean embedding per cluster (for soft routing)
  data/models/topic_info.json          label + size per topic
  data/processed/corpus_with_topics.jsonl  corpus_all + topic_id field

Usage:
  python bertopic_cluster.py
"""

import re
import json
import logging
import numpy as np
from collections import defaultdict
from pathlib import Path

from bertopic import BERTopic
from hdbscan import HDBSCAN
from umap import UMAP
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

ROOT          = Path(__file__).parents[2]
CORPUS_IN     = ROOT / "data" / "processed" / "corpus_all.jsonl"
CORPUS_OUT    = ROOT / "data" / "processed" / "corpus_with_topics.jsonl"
MODELS_DIR    = ROOT / "data" / "models"
EMBED_PATH    = MODELS_DIR / "embeddings_minilm.npy"
MODEL_DIR     = MODELS_DIR / "bertopic_model"
CENTROIDS_OUT = MODELS_DIR / "cluster_centroids.json"
TOPIC_INFO    = MODELS_DIR / "topic_info.json"

# ---------------------------------------------------------------------------
# Custom c-TF-IDF tokenizer for code corpora
# ---------------------------------------------------------------------------

_CODE_STOPWORDS = {
    # Python keywords / builtins
    "def", "return", "self", "import", "class", "none", "true", "false",
    "pass", "raise", "try", "except", "finally", "yield", "lambda",
    "global", "nonlocal", "del", "assert", "with", "elif",
    "print", "len", "str", "int", "float", "bool", "list", "dict",
    "tuple", "set", "type", "object", "super",
    # Generic English stopwords relevant to code docs
    "the", "and", "for", "are", "this", "that", "with", "from",
    "not", "use", "used", "using", "can", "will", "also", "may",
    "get", "set", "add", "new", "one", "two", "any", "all",
    # Ultra-common doc filler
    "example", "note", "see", "via", "etc", "fig", "above", "below",
}


def _code_tokenizer(text: str) -> list[str]:
    """
    Tokenise text for c-TF-IDF cluster labelling.

    - Splits snake_case  →  individual words
    - Splits camelCase   →  individual words
    - Lowercases
    - Keeps tokens ≥ 3 chars, alpha-only, not in code stopwords
    """
    # snake_case → spaces
    text = text.replace("_", " ")
    # camelCase → spaces before each uppercase-following-lowercase
    text = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)
    text = text.lower()
    tokens = re.findall(r"\b[a-z][a-z]{2,}\b", text)   # ≥ 3 chars, pure alpha
    return [t for t in tokens if t not in _CODE_STOPWORDS]


# ---------------------------------------------------------------------------
# Load corpus
# ---------------------------------------------------------------------------

def load_corpus(path: Path) -> tuple[list[str], list[dict]]:
    """Return (texts, records) from the JSONL corpus."""
    texts, records = [], []
    with open(path, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            texts.append(rec["text"])
            records.append(rec)
    log.info(f"Loaded {len(texts):,} chunks from {path.name}")
    return texts, records


# ---------------------------------------------------------------------------
# Embed
# ---------------------------------------------------------------------------

def embed(texts: list[str], save_path: Path) -> np.ndarray:
    if save_path.exists():
        log.info(f"Loading cached embeddings from {save_path.name} ...")
        return np.load(save_path)

    log.info("Embedding chunks with all-MiniLM-L6-v2 ...")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(
        texts,
        batch_size=128,
        show_progress_bar=True,
        normalize_embeddings=True,   # unit vectors → cosine = dot product
    )
    np.save(save_path, embeddings)
    log.info(f"Embeddings saved → {save_path} (shape: {embeddings.shape})")
    return embeddings


# ---------------------------------------------------------------------------
# Build BERTopic model
# ---------------------------------------------------------------------------

def build_topic_model() -> BERTopic:
    umap_model = UMAP(
        n_components=10,
        n_neighbors=15,
        min_dist=0.0,
        metric="cosine",
        random_state=42,
    )

    hdbscan_model = HDBSCAN(
        min_cluster_size=15,
        min_samples=10,
        metric="euclidean",
        cluster_selection_method="eom",
        prediction_data=True,          # required for approximate_predict (soft routing)
    )

    vectorizer = CountVectorizer(
        tokenizer=_code_tokenizer,
        min_df=2,                      # term must appear in ≥ 2 chunks
        max_df=0.95,                   # ignore near-universal terms
    )

    return BERTopic(
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer,
        top_n_words=10,                # words per topic label
        nr_topics="auto",              # HDBSCAN decides K
        calculate_probabilities=False, # faster; we do soft routing via centroids
        verbose=True,
    )


# ---------------------------------------------------------------------------
# Cluster centroids  (used for soft routing at inference)
# ---------------------------------------------------------------------------

def compute_centroids(
    embeddings: np.ndarray,
    topics: list[int],
) -> dict[int, list[float]]:
    """Mean embedding per cluster (excluding noise cluster -1)."""
    buckets: dict[int, list[np.ndarray]] = defaultdict(list)
    for emb, tid in zip(embeddings, topics):
        if tid != -1:
            buckets[tid].append(emb)

    centroids = {
        tid: np.mean(vecs, axis=0).tolist()
        for tid, vecs in buckets.items()
    }
    log.info(f"Computed centroids for {len(centroids)} clusters.")
    return centroids


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load corpus
    texts, records = load_corpus(CORPUS_IN)

    # 2. Embed (cached after first run)
    embeddings = embed(texts, EMBED_PATH)

    # 3. Fit BERTopic
    log.info("Fitting BERTopic ...")
    topic_model = build_topic_model()
    topics, _ = topic_model.fit_transform(texts, embeddings=embeddings)
    log.info("BERTopic fitting complete.")

    # 4. Cluster summary
    topic_info_df = topic_model.get_topic_info()
    n_topics = len(topic_info_df[topic_info_df["Topic"] != -1])
    n_noise  = int((np.array(topics) == -1).sum())
    log.info(f"Topics found : {n_topics}")
    log.info(f"Noise chunks : {n_noise:,} ({100*n_noise/len(topics):.1f}%)")

    # 5. Save model
    log.info(f"Saving BERTopic model → {MODEL_DIR}")
    topic_model.save(str(MODEL_DIR), serialization="pickle", save_embedding_model=False)

    # 6. Save topic info JSON
    topic_info_out = {}
    for _, row in topic_info_df.iterrows():
        tid = int(row["Topic"])
        words = topic_model.get_topic(tid)
        topic_info_out[tid] = {
            "label":     " | ".join(w for w, _ in words[:5]) if words else "noise",
            "top_words": [w for w, _ in words] if words else [],
            "size":      int(row["Count"]),
        }
    with open(TOPIC_INFO, "w") as f:
        json.dump(topic_info_out, f, indent=2)
    log.info(f"Topic info saved → {TOPIC_INFO}")

    # 7. Compute and save cluster centroids
    centroids = compute_centroids(embeddings, topics)
    with open(CENTROIDS_OUT, "w") as f:
        json.dump(centroids, f)
    log.info(f"Centroids saved → {CENTROIDS_OUT}")

    # 8. Write corpus_with_topics.jsonl
    log.info(f"Writing corpus with topic assignments → {CORPUS_OUT}")
    with open(CORPUS_OUT, "w", encoding="utf-8") as f:
        for rec, tid in zip(records, topics):
            rec["topic_id"] = int(tid)
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # 9. Print topic overview
    log.info("\n=== Topic overview (top 20 by size) ===")
    top_topics = topic_info_df[topic_info_df["Topic"] != -1].head(20)
    for _, row in top_topics.iterrows():
        tid   = int(row["Topic"])
        size  = int(row["Count"])
        words = topic_model.get_topic(tid)
        label = ", ".join(w for w, _ in words[:5]) if words else "—"
        log.info(f"  Topic {tid:>3} ({size:>4} chunks)  {label}")

    log.info("\nDone.")


if __name__ == "__main__":
    main()
