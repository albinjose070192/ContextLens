#!/usr/bin/env python3
"""
Layer 2 — Build CodeBERT FAISS indices for DPR retrieval.

Steps:
  1. Load corpus_with_topics.jsonl (14k chunks + topic assignments)
  2. Embed all chunks with microsoft/codebert-base (mean pooling, L2-normalized)
  3. Build one FAISS IndexFlatIP per topic (including noise topic -1)
  4. Compute CodeBERT cluster centroids for soft routing
  5. Save everything to data/models/

Outputs:
  data/models/codebert_embeddings.npy       all chunk embeddings (14k × 768)
  data/models/faiss_indices/topic_{id}.index  per-cluster FAISS index
  data/models/topic_chunk_map.json           {topic_id: [corpus positions]}
  data/models/codebert_centroids.json        CodeBERT centroid per cluster

Usage:
  python build_index.py
"""

import json
import logging
import numpy as np
from collections import defaultdict
from pathlib import Path

import faiss
import torch
from transformers import AutoTokenizer, AutoModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

ROOT          = Path(__file__).parents[2]
CORPUS_IN     = ROOT / "data" / "processed" / "corpus_with_topics.jsonl"
MODELS_DIR    = ROOT / "data" / "models"
EMBED_PATH    = MODELS_DIR / "codebert_embeddings.npy"
INDICES_DIR   = MODELS_DIR / "faiss_indices"
MAP_PATH      = MODELS_DIR / "topic_chunk_map.json"
CENTROIDS_OUT = MODELS_DIR / "codebert_centroids.json"

CODEBERT_MODEL = "microsoft/codebert-base"
BATCH_SIZE     = 32    # conservative — CodeBERT is 768-dim, heavier than MiniLM
EMBED_DIM      = 768


# ---------------------------------------------------------------------------
# CodeBERT embedder
# ---------------------------------------------------------------------------

def load_codebert():
    log.info(f"Loading {CODEBERT_MODEL} ...")
    tokenizer = AutoTokenizer.from_pretrained(CODEBERT_MODEL)
    model     = AutoModel.from_pretrained(CODEBERT_MODEL)
    device    = "mps" if torch.backends.mps.is_available() else "cpu"
    model     = model.to(device).eval()
    log.info(f"Model loaded on {device}.")
    return tokenizer, model, device


def mean_pool(token_embeddings: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Mean-pool token embeddings, ignoring padding positions."""
    mask = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    summed = torch.sum(token_embeddings * mask, dim=1)
    counts = torch.clamp(mask.sum(dim=1), min=1e-9)
    return summed / counts


def embed_texts(
    texts: list[str],
    tokenizer,
    model,
    device: str,
    save_path: Path,
) -> np.ndarray:
    if save_path.exists():
        log.info(f"Loading cached CodeBERT embeddings from {save_path.name} ...")
        return np.load(save_path)

    log.info(f"Embedding {len(texts):,} chunks with CodeBERT (batch={BATCH_SIZE}) ...")
    all_embeddings = []

    for start in range(0, len(texts), BATCH_SIZE):
        batch = texts[start : start + BATCH_SIZE]
        inputs = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs)

        pooled = mean_pool(outputs.last_hidden_state, inputs["attention_mask"])
        # L2-normalize → cosine similarity = dot product
        pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
        all_embeddings.append(pooled.cpu().numpy())

        if (start // BATCH_SIZE) % 20 == 0:
            done = min(start + BATCH_SIZE, len(texts))
            log.info(f"  {done:,} / {len(texts):,} embedded ...")

    embeddings = np.concatenate(all_embeddings, axis=0).astype("float32")
    np.save(save_path, embeddings)
    log.info(f"CodeBERT embeddings saved → {save_path} (shape: {embeddings.shape})")
    return embeddings


# ---------------------------------------------------------------------------
# FAISS index builder
# ---------------------------------------------------------------------------

def build_faiss_indices(
    embeddings: np.ndarray,
    topic_ids: list[int],
) -> tuple[dict[int, faiss.Index], dict[int, list[int]]]:
    """
    Build one IndexFlatIP per topic (normalized vectors → cosine similarity).
    Returns (indices dict, topic_chunk_map dict).
    topic_chunk_map[topic_id] = [corpus positions for chunks in that topic]
    """
    # Group corpus positions by topic
    topic_positions: dict[int, list[int]] = defaultdict(list)
    for pos, tid in enumerate(topic_ids):
        topic_positions[tid].append(pos)

    indices: dict[int, faiss.Index] = {}
    for tid, positions in topic_positions.items():
        vecs = embeddings[positions].astype("float32")
        index = faiss.IndexFlatIP(EMBED_DIM)   # inner product on unit vecs = cosine
        index.add(vecs)
        indices[tid] = index

    log.info(
        f"Built {len(indices)} FAISS indices "
        f"({sum(len(p) for p in topic_positions.values()):,} total vectors)."
    )
    return indices, dict(topic_positions)


def save_faiss_indices(indices: dict[int, faiss.Index], out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    for tid, index in indices.items():
        path = out_dir / f"topic_{tid}.index"
        faiss.write_index(index, str(path))
    log.info(f"Saved {len(indices)} FAISS indices → {out_dir}")


# ---------------------------------------------------------------------------
# Cluster centroids (CodeBERT space, for soft routing)
# ---------------------------------------------------------------------------

def compute_codebert_centroids(
    embeddings: np.ndarray,
    topic_ids: list[int],
) -> dict[int, list[float]]:
    """Mean CodeBERT embedding per cluster (noise -1 excluded)."""
    buckets: dict[int, list[np.ndarray]] = defaultdict(list)
    for emb, tid in zip(embeddings, topic_ids):
        if tid != -1:
            buckets[tid].append(emb)

    centroids = {}
    for tid, vecs in buckets.items():
        mean_vec = np.mean(vecs, axis=0)
        # Re-normalize the centroid
        norm = np.linalg.norm(mean_vec)
        if norm > 0:
            mean_vec /= norm
        centroids[tid] = mean_vec.tolist()

    log.info(f"Computed CodeBERT centroids for {len(centroids)} clusters.")
    return centroids


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load corpus
    log.info(f"Loading corpus from {CORPUS_IN.name} ...")
    texts, topic_ids, records = [], [], []
    with open(CORPUS_IN, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            texts.append(rec["text"])
            topic_ids.append(rec["topic_id"])
            records.append(rec)
    log.info(f"Loaded {len(texts):,} chunks.")

    # 2. Embed with CodeBERT
    tokenizer, model, device = load_codebert()
    embeddings = embed_texts(texts, tokenizer, model, device, EMBED_PATH)

    # 3. Build FAISS indices
    indices, topic_chunk_map = build_faiss_indices(embeddings, topic_ids)

    # 4. Save FAISS indices
    save_faiss_indices(indices, INDICES_DIR)

    # 5. Save topic→chunk position map (needed at query time)
    with open(MAP_PATH, "w") as f:
        json.dump({str(k): v for k, v in topic_chunk_map.items()}, f)
    log.info(f"Topic-chunk map saved → {MAP_PATH}")

    # 6. Compute and save CodeBERT centroids (replaces MiniLM centroids for routing)
    centroids = compute_codebert_centroids(embeddings, topic_ids)
    with open(CENTROIDS_OUT, "w") as f:
        json.dump({str(k): v for k, v in centroids.items()}, f)
    log.info(f"CodeBERT centroids saved → {CENTROIDS_OUT}")

    # 7. Summary
    non_noise = {k: v for k, v in topic_chunk_map.items() if k != -1}
    log.info("\n=== Index build complete ===")
    log.info(f"  Clusters indexed  : {len(non_noise)}")
    log.info(f"  Noise index size  : {len(topic_chunk_map.get(-1, []))}")
    log.info(f"  Embedding dim     : {EMBED_DIM}")
    log.info(f"  Total vectors     : {len(texts):,}")
    sizes = sorted((len(v), k) for k, v in non_noise.items())
    log.info(f"  Largest cluster   : topic {sizes[-1][1]} ({sizes[-1][0]} chunks)")
    log.info(f"  Smallest cluster  : topic {sizes[0][1]} ({sizes[0][0]} chunks)")


if __name__ == "__main__":
    main()
