"""
Layer 2 — DPR Retriever.

Loads the CodeBERT FAISS indices and cluster centroids built by build_index.py.
Exposes a single Retriever class for use by the evaluation harness and agent.

Retrieval flow:
  1. Embed query with CodeBERT (mean pool, L2-normalize)
  2. Soft routing: cosine similarity to each cluster centroid → top-N clusters
  3. FAISS search within each selected cluster → top-k candidates
  4. Merge, deduplicate, re-rank by score, return top-k chunks
"""

import json
import logging
import numpy as np
from pathlib import Path

import faiss
import torch
from transformers import AutoTokenizer, AutoModel

log = logging.getLogger(__name__)

ROOT        = Path(__file__).parents[2]
MODELS_DIR  = ROOT / "data" / "models"
INDICES_DIR = MODELS_DIR / "faiss_indices"
MAP_PATH    = MODELS_DIR / "topic_chunk_map.json"
CENTROIDS   = MODELS_DIR / "codebert_centroids.json"
CORPUS_PATH = ROOT / "data" / "processed" / "corpus_with_topics.jsonl"

CODEBERT_MODEL = "microsoft/codebert-base"


class Retriever:
    """
    Two-stage retriever: cluster routing (BERTopic centroids) + FAISS search.

    Parameters
    ----------
    top_clusters : int
        Number of clusters to search per query (soft routing breadth).
    top_k : int
        Number of chunks to return.
    """

    def __init__(self, top_clusters: int = 2, top_k: int = 5):
        self.top_clusters = top_clusters
        self.top_k        = top_k
        self._load_model()
        self._load_corpus()
        self._load_centroids()
        self._load_indices()
        log.info(
            f"Retriever ready — {len(self._indices)} cluster indices, "
            f"top_clusters={top_clusters}, top_k={top_k}"
        )

    # -----------------------------------------------------------------------
    # Initialisation helpers
    # -----------------------------------------------------------------------

    def _load_model(self):
        log.info(f"Loading {CODEBERT_MODEL} ...")
        self._tokenizer = AutoTokenizer.from_pretrained(CODEBERT_MODEL)
        self._model     = AutoModel.from_pretrained(CODEBERT_MODEL)
        self._device    = "mps" if torch.backends.mps.is_available() else "cpu"
        self._model     = self._model.to(self._device).eval()
        log.info(f"CodeBERT on {self._device}.")

    def _load_corpus(self):
        """Load all chunks into a list indexed by corpus position."""
        log.info("Loading corpus ...")
        self._corpus: list[dict] = []
        with open(CORPUS_PATH, encoding="utf-8") as f:
            for line in f:
                self._corpus.append(json.loads(line))
        log.info(f"  {len(self._corpus):,} chunks in memory.")

    def _load_centroids(self):
        """Load CodeBERT cluster centroids as a {topic_id: np.ndarray} dict."""
        with open(CENTROIDS) as f:
            raw = json.load(f)
        self._centroids: dict[int, np.ndarray] = {
            int(k): np.array(v, dtype="float32") for k, v in raw.items()
        }
        # Stack into a matrix for fast batch cosine similarity
        self._centroid_ids  = sorted(self._centroids.keys())
        self._centroid_mat  = np.stack(
            [self._centroids[tid] for tid in self._centroid_ids], axis=0
        )  # shape: (n_clusters, 768)
        log.info(f"  {len(self._centroids)} cluster centroids loaded.")

    def _load_indices(self):
        """Load per-cluster FAISS indices and topic→corpus position map."""
        with open(MAP_PATH) as f:
            raw = json.load(f)
        self._topic_chunk_map: dict[int, list[int]] = {
            int(k): v for k, v in raw.items()
        }

        self._indices: dict[int, faiss.Index] = {}
        for path in INDICES_DIR.glob("topic_*.index"):
            tid = int(path.stem.split("_")[1])
            self._indices[tid] = faiss.read_index(str(path))

        log.info(f"  {len(self._indices)} FAISS indices loaded.")

    # -----------------------------------------------------------------------
    # Query embedding
    # -----------------------------------------------------------------------

    def _embed_query(self, query: str) -> np.ndarray:
        """Embed a single query string → L2-normalized float32 vector (768,)."""
        inputs = self._tokenizer(
            query,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True,
        ).to(self._device)

        with torch.no_grad():
            outputs = self._model(**inputs)

        mask   = inputs["attention_mask"].unsqueeze(-1).float()
        summed = torch.sum(outputs.last_hidden_state * mask, dim=1)
        pooled = summed / torch.clamp(mask.sum(dim=1), min=1e-9)
        pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
        return pooled.squeeze(0).cpu().numpy().astype("float32")

    # -----------------------------------------------------------------------
    # Soft routing
    # -----------------------------------------------------------------------

    def _route(self, query_emb: np.ndarray) -> list[int]:
        """Return top-N cluster IDs by cosine similarity to centroids."""
        scores = self._centroid_mat @ query_emb          # (n_clusters,)
        top_n  = min(self.top_clusters, len(self._centroid_ids))
        top_idx = np.argpartition(scores, -top_n)[-top_n:]
        top_idx = top_idx[np.argsort(scores[top_idx])[::-1]]
        return [self._centroid_ids[i] for i in top_idx]

    # -----------------------------------------------------------------------
    # Search
    # -----------------------------------------------------------------------

    def _search_cluster(
        self,
        topic_id: int,
        query_emb: np.ndarray,
        k: int,
    ) -> list[dict]:
        """Search one FAISS index, return up to k result dicts with scores."""
        index = self._indices.get(topic_id)
        if index is None or index.ntotal == 0:
            return []

        k_actual = min(k, index.ntotal)
        scores, local_positions = index.search(query_emb[np.newaxis, :], k_actual)
        scores         = scores[0]
        local_positions = local_positions[0]

        positions = self._topic_chunk_map.get(topic_id, [])
        results = []
        for score, local_pos in zip(scores, local_positions):
            if local_pos < 0 or local_pos >= len(positions):
                continue
            corpus_pos = positions[local_pos]
            chunk = dict(self._corpus[corpus_pos])
            chunk["score"]    = float(score)
            chunk["topic_id"] = topic_id
            results.append(chunk)

        return results

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------

    def retrieve(
        self,
        query: str,
        top_k: int = None,
        top_clusters: int = None,
    ) -> list[dict]:
        """
        Retrieve the top-k most relevant chunks for a query.

        Returns a list of chunk dicts (text, source_type, repo, url, score, …)
        sorted by descending cosine similarity score.
        """
        top_k       = top_k       or self.top_k
        top_clusters = top_clusters or self.top_clusters

        query_emb   = self._embed_query(query)
        cluster_ids = self._route(query_emb)

        candidates: list[dict] = []
        for tid in cluster_ids:
            hits = self._search_cluster(tid, query_emb, top_k * 2)
            candidates.extend(hits)

        # Deduplicate by chunk_id, keep highest score
        seen: dict[str, dict] = {}
        for hit in candidates:
            cid = hit["chunk_id"]
            if cid not in seen or hit["score"] > seen[cid]["score"]:
                seen[cid] = hit

        ranked = sorted(seen.values(), key=lambda x: x["score"], reverse=True)
        return ranked[:top_k]

    def retrieve_flat(self, query: str, top_k: int = None) -> list[dict]:
        """
        Flat RAG baseline: search ALL clusters (no routing), return top-k.
        Used by the evaluation harness to benchmark against selective retrieval.
        """
        top_k = top_k or self.top_k
        query_emb = self._embed_query(query)

        candidates: list[dict] = []
        for tid, index in self._indices.items():
            hits = self._search_cluster(tid, query_emb, top_k)
            candidates.extend(hits)

        seen: dict[str, dict] = {}
        for hit in candidates:
            cid = hit["chunk_id"]
            if cid not in seen or hit["score"] > seen[cid]["score"]:
                seen[cid] = hit

        ranked = sorted(seen.values(), key=lambda x: x["score"], reverse=True)
        return ranked[:top_k]
