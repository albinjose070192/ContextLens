"""
Score Threshold Ablation.

Filters retrieved chunks by cosine similarity score before loading into the
registry.  Tests whether a minimum score threshold:
  (a) reduces context tokens (fewer chunks loaded)
  (b) improves retrieval precision (only high-confidence chunks kept)
  (c) flips the token reduction direction vs flat RAG (selective < flat tokens)

Thresholds tested: None (no filter), 0.50, 0.60, 0.70, 0.80, 0.90

Usage:
    # Token counts only (fast, no API key)
    python -m src.evaluation.threshold_ablation --n 50 --top-k 10 --dry-run

    # With LLM precision scoring
    ANTHROPIC_API_KEY=... python -m src.evaluation.threshold_ablation --n 50 --top-k 10
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

THRESHOLDS = [None, 0.50, 0.60, 0.70, 0.80, 0.90]


def _count_tokens(text: str) -> int:
    return len(_ENC.encode(text))


def _context_str(chunks: list[dict]) -> str:
    parts = []
    for c in chunks:
        parts.append(
            f"[chunk_id={c.get('chunk_id','?')} | score={c.get('score',0):.4f}]\n"
            f"{c.get('text','')}"
        )
    return "\n\n".join(parts)


def _apply_threshold(chunks: list[dict], threshold: Optional[float]) -> list[dict]:
    if threshold is None:
        return chunks
    return [c for c in chunks if c.get("score", 0) >= threshold]


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
            "Given the question and ground-truth answer, assess whether the "
            "retrieved passages contain enough information to answer it.\n"
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
# Runner
# ---------------------------------------------------------------------------

def load_queries(n: int, seed: int) -> list[dict]:
    with open(SO_PATH, encoding="utf-8") as f:
        data = json.load(f)
    data = [d for d in data if d.get("title") and d.get("answer_body")]
    random.seed(seed)
    return random.sample(data, min(n, len(data)))


def run_threshold_ablation(
    retriever: Retriever,
    queries: list[dict],
    top_k: int,
    top_clusters: int,
    judge: Optional[JudgeClient],
) -> pd.DataFrame:
    rows = []

    for threshold in THRESHOLDS:
        label = "none" if threshold is None else str(threshold)
        log.info(f"Threshold={label} ...")

        context_tokens_list  = []
        n_chunks_list        = []
        n_flat_chunks_list   = []
        flat_tokens_list     = []
        precision_sel_list   = []
        precision_flat_list  = []
        latencies_sel        = []
        latencies_flat       = []

        for record in queries:
            query = record["title"]
            gt    = record.get("answer_body", "")

            # --- Selective ---
            t0 = time.perf_counter()
            sel_chunks_raw = retriever.retrieve(
                query, top_k=top_k, top_clusters=top_clusters)
            sel_lat = (time.perf_counter() - t0) * 1000

            sel_chunks = _apply_threshold(sel_chunks_raw, threshold)
            sel_tokens = _count_tokens(_context_str(sel_chunks))
            context_tokens_list.append(sel_tokens)
            n_chunks_list.append(len(sel_chunks))
            latencies_sel.append(sel_lat)

            # --- Flat ---
            t0 = time.perf_counter()
            flat_chunks_raw = retriever.retrieve_flat(query, top_k=top_k)
            flat_lat = (time.perf_counter() - t0) * 1000

            flat_chunks = _apply_threshold(flat_chunks_raw, threshold)
            flat_tok = _count_tokens(_context_str(flat_chunks))
            flat_tokens_list.append(flat_tok)
            n_flat_chunks_list.append(len(flat_chunks))
            latencies_flat.append(flat_lat)

            if judge:
                precision_sel_list.append(
                    judge.precision(query, gt, sel_chunks))
                precision_flat_list.append(
                    judge.precision(query, gt, flat_chunks))

        rows.append({
            "threshold":              threshold if threshold is not None else "none",
            "mean_sel_context_tokens":  round(sum(context_tokens_list) / len(queries), 1),
            "mean_flat_context_tokens": round(sum(flat_tokens_list) / len(queries), 1),
            "mean_sel_chunks":          round(sum(n_chunks_list) / len(queries), 2),
            "mean_flat_chunks":         round(sum(n_flat_chunks_list) / len(queries), 2),
            "mean_sel_latency_ms":      round(sum(latencies_sel) / len(queries), 1),
            "mean_flat_latency_ms":     round(sum(latencies_flat) / len(queries), 1),
            "mean_sel_precision":
                round(sum(s for s in precision_sel_list if s is not None)
                      / max(len([s for s in precision_sel_list if s is not None]), 1), 4)
                if precision_sel_list else None,
            "mean_flat_precision":
                round(sum(s for s in precision_flat_list if s is not None)
                      / max(len([s for s in precision_flat_list if s is not None]), 1), 4)
                if precision_flat_list else None,
            "n_queries": len(queries),
        })
        log.info(
            f"  sel: {rows[-1]['mean_sel_context_tokens']:.0f} tok "
            f"({rows[-1]['mean_sel_chunks']:.1f} chunks)  "
            f"flat: {rows[-1]['mean_flat_context_tokens']:.0f} tok "
            f"({rows[-1]['mean_flat_chunks']:.1f} chunks)"
        )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_threshold(df: pd.DataFrame, top_k: int, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    has_precision = df["mean_sel_precision"].notna().any()

    x_labels = df["threshold"].astype(str).tolist()
    x        = range(len(x_labels))

    # --- Left: context tokens vs threshold ---
    ax1 = axes[0]
    ax1.plot(x, df["mean_sel_context_tokens"],
             marker="o", linewidth=2, color="#2563eb", label="Selective RAG")
    ax1.plot(x, df["mean_flat_context_tokens"],
             marker="s", linewidth=2, color="#94a3b8", linestyle="--", label="Flat RAG")
    ax1.set_xticks(x)
    ax1.set_xticklabels(x_labels, fontsize=10)
    ax1.set_xlabel("Min Score Threshold", fontsize=11)
    ax1.set_ylabel("Mean Context Tokens", fontsize=11)
    ax1.set_title(f"Context Tokens vs. Score Threshold (top_k={top_k})", fontsize=12)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # Mark where selective drops below flat
    for i, (s, f) in enumerate(
            zip(df["mean_sel_context_tokens"], df["mean_flat_context_tokens"])):
        if s < f:
            ax1.axvline(x=i, color="#16a34a", linestyle=":", alpha=0.6)
            ax1.annotate("sel < flat", (i, max(s, f)),
                         xytext=(4, 4), textcoords="offset points",
                         color="#16a34a", fontsize=8)

    # --- Right: precision vs threshold (if available), else chunk counts ---
    ax2 = axes[1]
    if has_precision:
        ax2.plot(x, df["mean_sel_precision"],
                 marker="o", linewidth=2, color="#2563eb", label="Selective")
        ax2.plot(x, df["mean_flat_precision"],
                 marker="s", linewidth=2, color="#94a3b8", linestyle="--", label="Flat")
        ax2.set_ylabel("Mean Retrieval Precision", fontsize=11)
        ax2.set_title("Precision vs. Score Threshold", fontsize=12)
    else:
        ax2.plot(x, df["mean_sel_chunks"],
                 marker="o", linewidth=2, color="#2563eb", label="Selective chunks")
        ax2.plot(x, df["mean_flat_chunks"],
                 marker="s", linewidth=2, color="#94a3b8", linestyle="--", label="Flat chunks")
        ax2.set_ylabel("Mean Chunks Loaded", fontsize=11)
        ax2.set_title("Chunks Loaded vs. Score Threshold", fontsize=12)
    ax2.set_xticks(x)
    ax2.set_xticklabels(x_labels, fontsize=10)
    ax2.set_xlabel("Min Score Threshold", fontsize=11)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    fig.suptitle("Score Threshold Ablation — Precision–Efficiency Tradeoff",
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
    p.add_argument("--n",            type=int,  default=50)
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

    df = run_threshold_ablation(retriever, queries, args.top_k, args.top_clusters, judge)

    print("\n" + "=" * 80)
    print("SCORE THRESHOLD ABLATION RESULTS")
    print("=" * 80)
    print(df.to_string(index=False))
    print("=" * 80)
    if judge:
        print(f"\nTotal judge cost: ${judge.total_cost_usd:.4f}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    df.to_csv(RESULTS_DIR / f"threshold_{ts}.csv", index=False)
    plot_threshold(df, args.top_k, RESULTS_DIR / f"threshold_{ts}.png")
