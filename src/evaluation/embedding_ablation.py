"""
Embedding Model Ablation — CodeBERT vs MiniLM for Cluster Routing.

Tests whether the routing step (query → centroid similarity → top-N clusters)
benefits from code-aware embeddings (CodeBERT) vs general-purpose embeddings
(all-MiniLM-L6-v2).

Both conditions use CodeBERT FAISS indices for within-cluster retrieval.
Only the routing step differs.

Conditions:
  codebert_routing — current system: embed query with CodeBERT (768-dim),
                     compare to CodeBERT centroids, route to top-2 clusters.

  minilm_routing   — alternative: embed query with MiniLM (384-dim),
                     compare to MiniLM centroids, route to top-2 clusters,
                     then search CodeBERT FAISS within those clusters using
                     a CodeBERT query embedding.

Metrics:
  - Cluster overlap (do both models route to the same clusters?)
  - Mean retrieval score (cosine similarity of retrieved chunks)
  - Retrieval precision (LLM-as-judge, optional)
  - Latency

Usage:
    python -m src.evaluation.embedding_ablation --n 100 --top-k 10 --dry-run
    ANTHROPIC_API_KEY=... python -m src.evaluation.embedding_ablation --n 100 --top-k 10
"""

import argparse
import json
import logging
import os
import random
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import anthropic
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tiktoken
import torch
from transformers import AutoModel, AutoTokenizer

from src.retrieval.retriever import Retriever

log = logging.getLogger(__name__)

ROOT        = Path(__file__).parents[2]
SO_PATH     = ROOT / "data" / "raw" / "stackoverflow" / "stackoverflow_all_raw.json"
MODELS_DIR  = ROOT / "data" / "models"
RESULTS_DIR = ROOT / "data" / "results"

MINILM_MODEL    = "sentence-transformers/all-MiniLM-L6-v2"
MINILM_CENTROID = MODELS_DIR / "cluster_centroids.json"   # 384-dim, from BERTopic run

_ENC             = tiktoken.get_encoding("cl100k_base")
_MODEL_JUDGE     = "claude-haiku-4-5-20251001"
_INPUT_COST_PER_1M  = 0.80
_OUTPUT_COST_PER_1M = 4.00


# ---------------------------------------------------------------------------
# MiniLM embedder (routing only)
# ---------------------------------------------------------------------------

class MiniLMEmbedder:
    def __init__(self):
        log.info(f"Loading {MINILM_MODEL} ...")
        self._tokenizer = AutoTokenizer.from_pretrained(MINILM_MODEL)
        self._model     = AutoModel.from_pretrained(MINILM_MODEL)
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        self._model     = self._model.to(device).eval()
        self._device    = device
        log.info(f"MiniLM on {device}.")

        with open(MINILM_CENTROID) as f:
            raw = json.load(f)
        self._centroid_ids  = sorted(int(k) for k in raw.keys())
        self._centroid_mat  = np.stack(
            [np.array(raw[str(tid)], dtype="float32") for tid in self._centroid_ids],
            axis=0,
        )   # (125, 384)

    def embed(self, text: str) -> np.ndarray:
        inputs = self._tokenizer(
            text, return_tensors="pt", truncation=True,
            max_length=512, padding=True,
        ).to(self._device)
        with torch.no_grad():
            out = self._model(**inputs)
        mask   = inputs["attention_mask"].unsqueeze(-1).float()
        summed = torch.sum(out.last_hidden_state * mask, dim=1)
        pooled = summed / torch.clamp(mask.sum(dim=1), min=1e-9)
        pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
        return pooled.squeeze(0).cpu().numpy().astype("float32")

    def route(self, query: str, top_clusters: int) -> list[int]:
        emb    = self.embed(query)
        scores = self._centroid_mat @ emb
        top_n  = min(top_clusters, len(self._centroid_ids))
        top_idx = np.argpartition(scores, -top_n)[-top_n:]
        top_idx = top_idx[np.argsort(scores[top_idx])[::-1]]
        return [self._centroid_ids[i] for i in top_idx]


# ---------------------------------------------------------------------------
# LLM judge
# ---------------------------------------------------------------------------

class JudgeClient:
    def __init__(self, api_key: Optional[str] = None):
        key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise ValueError("ANTHROPIC_API_KEY not set.")
        self._client = anthropic.Anthropic(api_key=key)
        self.total_input_tokens  = 0
        self.total_output_tokens = 0

    def precision(self, query: str, ground_truth: str,
                  chunks: list[dict]) -> Optional[float]:
        if not chunks:
            return 0.0
        snippets = "\n---\n".join(c.get("text", "")[:300] for c in chunks)
        prompt = (
            "Evaluate retrieval quality for a RAG system.\n"
            'Output ONLY: {"precision_score": float 0.0-1.0}\n\n'
            f"Question: {query}\n\nGround truth: {ground_truth[:400]}\n\n"
            f"Passages:\n{snippets}"
        )
        for attempt in range(5):
            try:
                msg = self._client.messages.create(
                    model=_MODEL_JUDGE, max_tokens=64,
                    messages=[{"role": "user", "content": prompt}],
                )
                break
            except (anthropic.RateLimitError, anthropic.APIStatusError) as e:
                if isinstance(e, anthropic.APIStatusError) and e.status_code not in (429, 529):
                    raise
                time.sleep(2 ** attempt * 15)
        else:
            return None
        self.total_input_tokens  += msg.usage.input_tokens
        self.total_output_tokens += msg.usage.output_tokens
        text = msg.content[0].text.strip().strip("```json").strip("```").strip()
        try:
            return float(json.loads(text).get("precision_score"))
        except Exception:
            return None

    @property
    def total_cost_usd(self) -> float:
        return (
            self.total_input_tokens  / 1_000_000 * _INPUT_COST_PER_1M
            + self.total_output_tokens / 1_000_000 * _OUTPUT_COST_PER_1M
        )


# ---------------------------------------------------------------------------
# Retriever with MiniLM routing
# ---------------------------------------------------------------------------

class MiniLMRoutingRetriever:
    """
    Uses MiniLM centroids for routing, then delegates to the base Retriever's
    CodeBERT FAISS indices for within-cluster search.
    """

    def __init__(self, base_retriever: Retriever,
                 minilm: MiniLMEmbedder, top_clusters: int = 2):
        self._base         = base_retriever
        self._minilm       = minilm
        self._top_clusters = top_clusters

    def retrieve_flat(self, query: str, top_k: int = None) -> list[dict]:
        return self._base.retrieve_flat(query, top_k=top_k)

    def retrieve(self, query: str, top_k: int, top_clusters: int = None) -> list[dict]:
        tc = top_clusters or self._top_clusters

        # Route with MiniLM
        cluster_ids = self._minilm.route(query, tc)

        # Embed with CodeBERT for FAISS search
        query_emb = self._base._embed_query(query)

        candidates: list[dict] = []
        for tid in cluster_ids:
            hits = self._base._search_cluster(tid, query_emb, top_k * 2)
            candidates.extend(hits)

        seen: dict[str, dict] = {}
        for hit in candidates:
            cid = hit["chunk_id"]
            if cid not in seen or hit["score"] > seen[cid]["score"]:
                seen[cid] = hit

        return sorted(seen.values(), key=lambda x: x["score"], reverse=True)[:top_k]


# ---------------------------------------------------------------------------
# Ablation runner
# ---------------------------------------------------------------------------

def load_queries(n: int, seed: int) -> list[dict]:
    with open(SO_PATH, encoding="utf-8") as f:
        data = json.load(f)
    data = [d for d in data if d.get("title") and d.get("answer_body")]
    random.seed(seed)
    return random.sample(data, min(n, len(data)))


def run_embedding_ablation(
    codebert_retriever: Retriever,
    minilm_retriever: MiniLMRoutingRetriever,
    queries: list[dict],
    top_k: int,
    top_clusters: int,
    judge: Optional[JudgeClient],
) -> pd.DataFrame:
    rows = []

    for i, record in enumerate(queries):
        query = record["title"]
        gt    = record.get("answer_body", "")
        tag   = record.get("tag", "")

        # CodeBERT routing
        t0 = time.perf_counter()
        cb_chunks  = codebert_retriever.retrieve(query, top_k=top_k, top_clusters=top_clusters)
        cb_lat     = (time.perf_counter() - t0) * 1000
        cb_cluster_ids = set(c.get("topic_id") for c in cb_chunks)
        cb_mean_score  = float(np.mean([c["score"] for c in cb_chunks])) if cb_chunks else 0.0

        # MiniLM routing
        t0 = time.perf_counter()
        ml_chunks  = minilm_retriever.retrieve(query, top_k=top_k, top_clusters=top_clusters)
        ml_lat     = (time.perf_counter() - t0) * 1000
        ml_cluster_ids = set(c.get("topic_id") for c in ml_chunks)
        ml_mean_score  = float(np.mean([c["score"] for c in ml_chunks])) if ml_chunks else 0.0

        # Cluster overlap
        if cb_cluster_ids or ml_cluster_ids:
            overlap = len(cb_cluster_ids & ml_cluster_ids) / len(cb_cluster_ids | ml_cluster_ids)
        else:
            overlap = 1.0

        row = {
            "query":             query,
            "tag":               tag,
            "cb_mean_score":     round(cb_mean_score, 4),
            "ml_mean_score":     round(ml_mean_score, 4),
            "score_delta":       round(cb_mean_score - ml_mean_score, 4),
            "cluster_overlap":   round(overlap, 3),
            "cb_latency_ms":     round(cb_lat, 1),
            "ml_latency_ms":     round(ml_lat, 1),
            "cb_precision":      None,
            "ml_precision":      None,
        }

        if judge:
            row["cb_precision"] = judge.precision(query, gt, cb_chunks)
            row["ml_precision"] = judge.precision(query, gt, ml_chunks)

        rows.append(row)
        if (i + 1) % 20 == 0:
            log.info(f"  {i+1}/{len(queries)} done")

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Summary + plot
# ---------------------------------------------------------------------------

def summarize(df: pd.DataFrame, judge_used: bool) -> None:
    print("\n" + "=" * 65)
    print("EMBEDDING ABLATION — CodeBERT vs MiniLM Routing")
    print("=" * 65)
    print(f"\n{'Metric':<30} {'CodeBERT':>12} {'MiniLM':>12} {'Delta':>10}")
    print("-" * 65)
    print(f"{'Mean retrieval score':<30} {df['cb_mean_score'].mean():>12.4f} "
          f"{df['ml_mean_score'].mean():>12.4f} "
          f"{df['score_delta'].mean():>+10.4f}")
    print(f"{'Mean cluster overlap':<30} {'—':>12} {'—':>12} "
          f"{df['cluster_overlap'].mean():>10.3f}")
    print(f"{'Mean latency (ms)':<30} {df['cb_latency_ms'].mean():>12.1f} "
          f"{df['ml_latency_ms'].mean():>12.1f} "
          f"{(df['cb_latency_ms'] - df['ml_latency_ms']).mean():>+10.1f}")
    if judge_used and df["cb_precision"].notna().any():
        print(f"{'Mean retrieval precision':<30} {df['cb_precision'].mean():>12.4f} "
              f"{df['ml_precision'].mean():>12.4f} "
              f"{(df['cb_precision'] - df['ml_precision']).mean():>+10.4f}")
    print()
    print("Per-tag cluster overlap:")
    print(df.groupby("tag")["cluster_overlap"].mean().round(3).to_string())
    print("=" * 65)


def plot_embedding(df: pd.DataFrame, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    has_prec  = df["cb_precision"].notna().any()

    # --- Left: retrieval score comparison ---
    ax1 = axes[0]
    ax1.scatter(df["cb_mean_score"], df["ml_mean_score"],
                alpha=0.5, s=20, color="#2563eb")
    lims = [min(df["cb_mean_score"].min(), df["ml_mean_score"].min()) - 0.01,
            max(df["cb_mean_score"].max(), df["ml_mean_score"].max()) + 0.01]
    ax1.plot(lims, lims, "k--", linewidth=1, alpha=0.5, label="y=x (equal)")
    ax1.set_xlabel("CodeBERT Routing — Mean Retrieval Score", fontsize=11)
    ax1.set_ylabel("MiniLM Routing — Mean Retrieval Score", fontsize=11)
    ax1.set_title("Per-Query Retrieval Score", fontsize=12)
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.annotate(
        f"Mean Δ: {df['score_delta'].mean():+.4f}\n"
        f"Cluster overlap: {df['cluster_overlap'].mean():.1%}",
        xy=(0.05, 0.92), xycoords="axes fraction", fontsize=9,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8),
    )

    # --- Right: precision or cluster overlap histogram ---
    ax2 = axes[1]
    if has_prec:
        ax2.hist(df["cb_precision"].dropna(), bins=15, alpha=0.7,
                 color="#2563eb", label="CodeBERT routing", edgecolor="white")
        ax2.hist(df["ml_precision"].dropna(), bins=15, alpha=0.7,
                 color="#f97316", label="MiniLM routing", edgecolor="white")
        ax2.axvline(df["cb_precision"].mean(), color="#2563eb", linestyle="--",
                    label=f"CB mean: {df['cb_precision'].mean():.3f}")
        ax2.axvline(df["ml_precision"].mean(), color="#f97316", linestyle="--",
                    label=f"ML mean: {df['ml_precision'].mean():.3f}")
        ax2.set_xlabel("Retrieval Precision Score", fontsize=11)
        ax2.set_title("Precision Distribution by Routing Model", fontsize=12)
    else:
        ax2.hist(df["cluster_overlap"], bins=15, color="#6b7280",
                 edgecolor="white", alpha=0.8)
        ax2.axvline(df["cluster_overlap"].mean(), color="black", linestyle="--",
                    label=f"Mean overlap: {df['cluster_overlap'].mean():.1%}")
        ax2.set_xlabel("Jaccard Cluster Overlap (CodeBERT ∩ MiniLM / union)", fontsize=11)
        ax2.set_title("Routing Agreement Between Models", fontsize=12)
    ax2.set_ylabel("Number of Queries", fontsize=11)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3, axis="y")

    fig.suptitle("Embedding Ablation — CodeBERT vs MiniLM Cluster Routing",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    log.info(f"Plot saved → {out_path}")
    plt.close()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--n",            type=int,  default=100)
    p.add_argument("--top-k",        type=int,  default=10)
    p.add_argument("--top-clusters", type=int,  default=2)
    p.add_argument("--seed",         type=int,  default=42)
    p.add_argument("--dry-run",      action="store_true")
    p.add_argument("--api-key",      type=str,  default=None)
    p.add_argument("--log-level",    default="INFO")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO),
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    queries = load_queries(args.n, args.seed)

    log.info("Loading CodeBERT retriever ...")
    cb_retriever = Retriever(top_clusters=args.top_clusters, top_k=args.top_k)

    log.info("Loading MiniLM embedder ...")
    minilm = MiniLMEmbedder()
    ml_retriever = MiniLMRoutingRetriever(cb_retriever, minilm, args.top_clusters)

    judge = None if args.dry_run else JudgeClient(api_key=args.api_key)

    log.info(f"Running ablation on {len(queries)} queries ...")
    df = run_embedding_ablation(
        cb_retriever, ml_retriever, queries,
        args.top_k, args.top_clusters, judge,
    )

    summarize(df, judge is not None)
    if judge:
        print(f"Total judge cost: ${judge.total_cost_usd:.4f}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    df.to_csv(RESULTS_DIR / f"embedding_ablation_{ts}.csv", index=False)
    plot_embedding(df, RESULTS_DIR / f"embedding_ablation_{ts}.png")
