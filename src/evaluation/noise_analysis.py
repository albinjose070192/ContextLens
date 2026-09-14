"""
Noise Cluster Analysis (topic_id = -1).

Examines how often queries route chunks from the noise cluster (BERTopic topic -1)
and whether excluding those chunks improves retrieval precision.

For each query:
  1. Run retrieve() — tracks what fraction of returned chunks are from topic -1
  2. Run retrieve_flat() — same tracking
  3. Optionally run LLM precision judge on:
       (a) full retrieved set
       (b) set with noise chunks excluded

Outputs:
  - Noise routing frequency (% of retrieved chunks from topic -1)
  - Precision comparison: with vs without noise chunks
  - Plot: noise fraction histogram + precision delta

Usage:
    python -m src.evaluation.noise_analysis --n 100 --top-k 10 --dry-run
    ANTHROPIC_API_KEY=... python -m src.evaluation.noise_analysis --n 100 --top-k 10
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

from src.retrieval.retriever import Retriever

log = logging.getLogger(__name__)

ROOT        = Path(__file__).parents[2]
SO_PATH     = ROOT / "data" / "raw" / "stackoverflow" / "stackoverflow_all_raw.json"
RESULTS_DIR = ROOT / "data" / "results"

_ENC             = tiktoken.get_encoding("cl100k_base")
_MODEL_JUDGE     = "claude-haiku-4-5-20251001"
_INPUT_COST_PER_1M  = 0.80
_OUTPUT_COST_PER_1M = 4.00

NOISE_TOPIC_ID = -1


def _count_tokens(text: str) -> int:
    return len(_ENC.encode(text))


def _context_str(chunks: list[dict]) -> str:
    return "\n\n".join(
        f"[chunk_id={c.get('chunk_id','?')} | score={c.get('score',0):.4f}]\n"
        f"{c.get('text','')}"
        for c in chunks
    )


def _noise_stats(chunks: list[dict]) -> dict:
    if not chunks:
        return {"n_total": 0, "n_noise": 0, "noise_pct": 0.0,
                "mean_noise_score": None, "mean_signal_score": None}
    noise  = [c for c in chunks if c.get("topic_id") == NOISE_TOPIC_ID]
    signal = [c for c in chunks if c.get("topic_id") != NOISE_TOPIC_ID]
    return {
        "n_total":          len(chunks),
        "n_noise":          len(noise),
        "noise_pct":        round(len(noise) / len(chunks) * 100, 1),
        "mean_noise_score": round(np.mean([c["score"] for c in noise]), 4) if noise else None,
        "mean_signal_score":round(np.mean([c["score"] for c in signal]), 4) if signal else None,
    }


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
            f"Question: {query}\n\n"
            f"Ground truth (first 400 chars): {ground_truth[:400]}\n\n"
            f"Retrieved passages:\n{snippets}"
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
                wait = 2 ** attempt * 15
                log.warning(f"API error — waiting {wait}s")
                time.sleep(wait)
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
# Main analysis
# ---------------------------------------------------------------------------

def load_queries(n: int, seed: int) -> list[dict]:
    with open(SO_PATH, encoding="utf-8") as f:
        data = json.load(f)
    data = [d for d in data if d.get("title") and d.get("answer_body")]
    random.seed(seed)
    return random.sample(data, min(n, len(data)))


def run_noise_analysis(
    retriever: Retriever,
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

        # --- Selective retrieval ---
        sel_chunks = retriever.retrieve(query, top_k=top_k, top_clusters=top_clusters)
        sel_stats  = _noise_stats(sel_chunks)
        sel_no_noise = [c for c in sel_chunks if c.get("topic_id") != NOISE_TOPIC_ID]

        # --- Flat retrieval ---
        flat_chunks   = retriever.retrieve_flat(query, top_k=top_k)
        flat_stats    = _noise_stats(flat_chunks)
        flat_no_noise = [c for c in flat_chunks if c.get("topic_id") != NOISE_TOPIC_ID]

        row = {
            "query":   query,
            "tag":     tag,

            # Selective stats
            "sel_n_noise":          sel_stats["n_noise"],
            "sel_noise_pct":        sel_stats["noise_pct"],
            "sel_mean_noise_score": sel_stats["mean_noise_score"],
            "sel_mean_signal_score":sel_stats["mean_signal_score"],
            "sel_context_tokens":   _count_tokens(_context_str(sel_chunks)),
            "sel_no_noise_tokens":  _count_tokens(_context_str(sel_no_noise)),

            # Flat stats
            "flat_n_noise":          flat_stats["n_noise"],
            "flat_noise_pct":        flat_stats["noise_pct"],
            "flat_mean_noise_score": flat_stats["mean_noise_score"],
            "flat_mean_signal_score":flat_stats["mean_signal_score"],
            "flat_context_tokens":   _count_tokens(_context_str(flat_chunks)),
            "flat_no_noise_tokens":  _count_tokens(_context_str(flat_no_noise)),

            # LLM precision (filled below if judge available)
            "sel_precision_with_noise":    None,
            "sel_precision_without_noise": None,
            "flat_precision_with_noise":   None,
            "flat_precision_without_noise":None,
        }

        if judge:
            row["sel_precision_with_noise"]    = judge.precision(query, gt, sel_chunks)
            row["sel_precision_without_noise"] = judge.precision(query, gt, sel_no_noise)
            row["flat_precision_with_noise"]   = judge.precision(query, gt, flat_chunks)
            row["flat_precision_without_noise"]= judge.precision(query, gt, flat_no_noise)

        rows.append(row)
        if (i + 1) % 10 == 0:
            log.info(f"  {i+1}/{len(queries)} done")

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Summary + plotting
# ---------------------------------------------------------------------------

def summarize(df: pd.DataFrame) -> None:
    print("\n" + "=" * 70)
    print("NOISE CLUSTER ANALYSIS SUMMARY")
    print("=" * 70)

    print("\n--- Noise chunk frequency ---")
    print(f"  Selective:  {df['sel_noise_pct'].mean():.1f}% of retrieved chunks are noise "
          f"({df['sel_n_noise'].mean():.2f} / query)")
    print(f"  Flat RAG:   {df['flat_noise_pct'].mean():.1f}% of retrieved chunks are noise "
          f"({df['flat_n_noise'].mean():.2f} / query)")

    print(f"\n--- Mean cosine scores ---")
    print(f"  Selective  — noise chunks: {df['sel_mean_noise_score'].mean():.4f}  "
          f"signal chunks: {df['sel_mean_signal_score'].mean():.4f}")
    print(f"  Flat RAG   — noise chunks: {df['flat_mean_noise_score'].mean():.4f}  "
          f"signal chunks: {df['flat_mean_signal_score'].mean():.4f}")

    print(f"\n--- Token impact of removing noise chunks ---")
    print(f"  Selective:  {df['sel_context_tokens'].mean():.1f} → "
          f"{df['sel_no_noise_tokens'].mean():.1f} tokens "
          f"(Δ {df['sel_context_tokens'].mean() - df['sel_no_noise_tokens'].mean():.1f})")
    print(f"  Flat RAG:   {df['flat_context_tokens'].mean():.1f} → "
          f"{df['flat_no_noise_tokens'].mean():.1f} tokens "
          f"(Δ {df['flat_context_tokens'].mean() - df['flat_no_noise_tokens'].mean():.1f})")

    if df["sel_precision_with_noise"].notna().any():
        print(f"\n--- Precision impact of removing noise chunks ---")
        print(f"  Selective  with noise:    {df['sel_precision_with_noise'].mean():.4f}")
        print(f"  Selective  without noise: {df['sel_precision_without_noise'].mean():.4f}")
        print(f"  Flat RAG   with noise:    {df['flat_precision_with_noise'].mean():.4f}")
        print(f"  Flat RAG   without noise: {df['flat_precision_without_noise'].mean():.4f}")

    print(f"\n--- Per-tag noise frequency (selective) ---")
    print(df.groupby("tag")[["sel_noise_pct", "flat_noise_pct"]].mean().round(1).to_string())
    print("=" * 70)


def plot_noise(df: pd.DataFrame, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # --- Left: noise % histogram by condition ---
    ax1 = axes[0]
    ax1.hist(df["sel_noise_pct"],  bins=15, alpha=0.7,
             color="#2563eb", label="Selective RAG", edgecolor="white")
    ax1.hist(df["flat_noise_pct"], bins=15, alpha=0.7,
             color="#94a3b8", label="Flat RAG", edgecolor="white")
    ax1.axvline(df["sel_noise_pct"].mean(), color="#2563eb",
                linestyle="--", linewidth=1.5,
                label=f"Sel mean: {df['sel_noise_pct'].mean():.1f}%")
    ax1.axvline(df["flat_noise_pct"].mean(), color="#6b7280",
                linestyle="--", linewidth=1.5,
                label=f"Flat mean: {df['flat_noise_pct'].mean():.1f}%")
    ax1.set_xlabel("% Retrieved Chunks from Noise Cluster", fontsize=11)
    ax1.set_ylabel("Number of Queries", fontsize=11)
    ax1.set_title("Noise Chunk Frequency Distribution", fontsize=12)
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3, axis="y")

    # --- Right: precision delta (with vs without noise) or token delta ---
    ax2 = axes[1]
    has_precision = df["sel_precision_with_noise"].notna().any()

    if has_precision:
        sel_delta  = df["sel_precision_without_noise"] - df["sel_precision_with_noise"]
        flat_delta = df["flat_precision_without_noise"] - df["flat_precision_with_noise"]
        ax2.hist(sel_delta,  bins=15, alpha=0.7,
                 color="#2563eb", label="Selective RAG", edgecolor="white")
        ax2.hist(flat_delta, bins=15, alpha=0.7,
                 color="#94a3b8", label="Flat RAG", edgecolor="white")
        ax2.axvline(0, color="black", linewidth=1)
        ax2.axvline(sel_delta.mean(), color="#2563eb", linestyle="--",
                    label=f"Sel mean Δ: {sel_delta.mean():.3f}")
        ax2.axvline(flat_delta.mean(), color="#6b7280", linestyle="--",
                    label=f"Flat mean Δ: {flat_delta.mean():.3f}")
        ax2.set_xlabel("Precision Change (without noise − with noise)", fontsize=11)
        ax2.set_ylabel("Number of Queries", fontsize=11)
        ax2.set_title("Precision Δ When Noise Chunks Removed", fontsize=12)
    else:
        # Show token savings from removing noise chunks
        sel_tok_delta  = df["sel_context_tokens"]  - df["sel_no_noise_tokens"]
        flat_tok_delta = df["flat_context_tokens"] - df["flat_no_noise_tokens"]
        ax2.hist(sel_tok_delta,  bins=15, alpha=0.7,
                 color="#2563eb", label="Selective RAG", edgecolor="white")
        ax2.hist(flat_tok_delta, bins=15, alpha=0.7,
                 color="#94a3b8", label="Flat RAG", edgecolor="white")
        ax2.axvline(sel_tok_delta.mean(), color="#2563eb", linestyle="--",
                    label=f"Sel mean: {sel_tok_delta.mean():.0f} tok")
        ax2.axvline(flat_tok_delta.mean(), color="#6b7280", linestyle="--",
                    label=f"Flat mean: {flat_tok_delta.mean():.0f} tok")
        ax2.set_xlabel("Token Reduction from Removing Noise Chunks", fontsize=11)
        ax2.set_ylabel("Number of Queries", fontsize=11)
        ax2.set_title("Token Savings When Noise Chunks Removed", fontsize=12)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3, axis="y")

    fig.suptitle("Noise Cluster (topic_id = -1) Analysis",
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

    queries  = load_queries(args.n, args.seed)
    retriever = Retriever(top_clusters=args.top_clusters, top_k=args.top_k)
    judge = None if args.dry_run else JudgeClient(api_key=args.api_key)

    log.info(f"Running noise analysis on {len(queries)} queries ...")
    df = run_noise_analysis(retriever, queries, args.top_k, args.top_clusters, judge)

    summarize(df)

    if judge:
        print(f"\nTotal judge cost: ${judge.total_cost_usd:.4f}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    df.to_csv(RESULTS_DIR / f"noise_{ts}.csv", index=False)
    plot_noise(df, RESULTS_DIR / f"noise_{ts}.png")
