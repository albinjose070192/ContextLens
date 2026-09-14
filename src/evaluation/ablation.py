"""
top_clusters Ablation — Token Count vs. Retrieval Precision tradeoff curve.

For each value of top_clusters in [1, 2, 3, 5, 126]:
  - Retrieves top_k chunks per query using selective routing
  - Measures mean context tokens (no LLM needed)
  - Uses LLM-as-judge for retrieval precision (1 call per query)

Also plots the tradeoff curve: x=context_tokens, y=retrieval_precision,
one point per top_clusters value, saved to data/results/.

Usage:
    # Token counts only (no API key needed)
    python -m src.evaluation.ablation --n 50 --top-k 10 --dry-run

    # Full with precision scores
    ANTHROPIC_API_KEY=... python -m src.evaluation.ablation --n 50 --top-k 10
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
import openai as openai_lib
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
_OAI_MODEL_JUDGE        = "gpt-4o-mini"
_OAI_INPUT_COST_PER_1M  = 0.15
_OAI_OUTPUT_COST_PER_1M = 0.60

ALL_CLUSTERS = 126   # total FAISS indices built
TOP_CLUSTERS_VALUES = [1, 2, 3, 5, ALL_CLUSTERS]


def _count_tokens(text: str) -> int:
    return len(_ENC.encode(text))


def _context_str(chunks: list[dict]) -> str:
    parts = []
    for c in chunks:
        parts.append(
            f"[chunk_id={c.get('chunk_id','?')} | "
            f"repo={c.get('repo','?')} | "
            f"score={c.get('score',0):.4f}]\n{c.get('text','')}"
        )
    return "\n\n".join(parts)


def _precision_prompt(query: str, ground_truth: str, chunks: list[dict]) -> str:
    snippets = "\n---\n".join(c.get("text", "")[:300] for c in chunks)
    return (
        "You are evaluating retrieval quality for a RAG system.\n"
        "Given the question and the ground-truth answer, assess whether the "
        "retrieved passages contain enough information to answer the question.\n"
        'Output a JSON object with one key: "precision_score": float 0.0–1.0\n'
        "Output ONLY valid JSON, no explanation.\n\n"
        f"Question: {query}\n\n"
        f"Ground truth (first 400 chars): {ground_truth[:400]}\n\n"
        f"Retrieved passages:\n{snippets}"
    )


# ---------------------------------------------------------------------------
# LLM judge (minimal — precision only)
# ---------------------------------------------------------------------------

class JudgeClient:
    def __init__(self, api_key: Optional[str] = None, use_openai: bool = False):
        self._use_openai = use_openai
        if use_openai:
            key = api_key or os.environ.get("OPENAI_API_KEY")
            if not key:
                raise ValueError("OPENAI_API_KEY not set.")
            self._client = openai_lib.OpenAI(api_key=key)
            self._model   = _OAI_MODEL_JUDGE
            self._in_cost  = _OAI_INPUT_COST_PER_1M
            self._out_cost = _OAI_OUTPUT_COST_PER_1M
        else:
            key = api_key or os.environ.get("ANTHROPIC_API_KEY")
            if not key:
                raise ValueError("ANTHROPIC_API_KEY not set.")
            self._client = anthropic.Anthropic(api_key=key)
            self._model   = _MODEL_JUDGE
            self._in_cost  = _INPUT_COST_PER_1M
            self._out_cost = _OUTPUT_COST_PER_1M
        self.total_input_tokens  = 0
        self.total_output_tokens = 0

    def precision(self, query: str, ground_truth: str, chunks: list[dict]) -> Optional[float]:
        prompt = _precision_prompt(query, ground_truth, chunks)
        for attempt in range(5):
            try:
                if self._use_openai:
                    resp = self._client.chat.completions.create(
                        model=self._model, max_tokens=64,
                        messages=[{"role": "user", "content": prompt}],
                    )
                    in_tok  = resp.usage.prompt_tokens
                    out_tok = resp.usage.completion_tokens
                    text    = resp.choices[0].message.content
                else:
                    msg = self._client.messages.create(
                        model=self._model, max_tokens=64,
                        messages=[{"role": "user", "content": prompt}],
                    )
                    in_tok  = msg.usage.input_tokens
                    out_tok = msg.usage.output_tokens
                    text    = msg.content[0].text
                break
            except (anthropic.RateLimitError, anthropic.APIStatusError,
                    openai_lib.RateLimitError, openai_lib.APIStatusError) as e:
                status = getattr(e, "status_code", None)
                if status and status not in (429, 529):
                    raise
                wait = 2 ** attempt * 15
                log.warning(f"API error — waiting {wait}s (attempt {attempt+1}/5)")
                time.sleep(wait)
        else:
            return None

        self.total_input_tokens  += in_tok
        self.total_output_tokens += out_tok

        text = text.strip().strip("```json").strip("```").strip()
        try:
            return float(json.loads(text).get("precision_score"))
        except Exception:
            return None

    @property
    def total_cost_usd(self) -> float:
        return (
            self.total_input_tokens  / 1_000_000 * self._in_cost
            + self.total_output_tokens / 1_000_000 * self._out_cost
        )


# ---------------------------------------------------------------------------
# Ablation runner
# ---------------------------------------------------------------------------

def load_queries(n: int, seed: int, tags: Optional[list[str]] = None) -> list[dict]:
    with open(SO_PATH, encoding="utf-8") as f:
        data = json.load(f)
    data = [d for d in data if d.get("title") and d.get("answer_body")]
    if tags:
        data = [d for d in data if d.get("tag") in tags]
    random.seed(seed)
    return random.sample(data, min(n, len(data)))


def run_ablation(
    retriever: Retriever,
    queries: list[dict],
    top_k: int,
    top_clusters_values: list[int],
    judge: Optional[JudgeClient],
) -> pd.DataFrame:
    rows = []

    for tc in top_clusters_values:
        label = "all" if tc >= ALL_CLUSTERS else str(tc)
        log.info(f"top_clusters={label} ({len(queries)} queries) ...")

        token_counts  = []
        precision_scores = []
        latencies     = []

        for i, record in enumerate(queries):
            query = record["title"]
            gt    = record.get("answer_body", "")

            t0 = time.perf_counter()
            if tc >= ALL_CLUSTERS:
                chunks = retriever.retrieve_flat(query, top_k=top_k)
            else:
                chunks = retriever.retrieve(query, top_k=top_k, top_clusters=tc)
            latency_ms = (time.perf_counter() - t0) * 1000

            ctx_tokens = _count_tokens(_context_str(chunks))
            token_counts.append(ctx_tokens)
            latencies.append(latency_ms)

            if judge:
                score = judge.precision(query, gt, chunks)
                precision_scores.append(score)

            if (i + 1) % 10 == 0:
                log.info(f"  {i+1}/{len(queries)} done")

        rows.append({
            "top_clusters":          tc,
            "top_clusters_label":    label,
            "mean_context_tokens":   round(sum(token_counts) / len(token_counts), 1),
            "mean_latency_ms":       round(sum(latencies) / len(latencies), 1),
            "mean_precision_score":  round(
                sum(s for s in precision_scores if s is not None)
                / max(len([s for s in precision_scores if s is not None]), 1),
                4,
            ) if precision_scores else None,
            "n_queries": len(queries),
        })
        log.info(
            f"  → tokens={rows[-1]['mean_context_tokens']:.1f}  "
            f"latency={rows[-1]['mean_latency_ms']:.1f}ms  "
            f"precision={rows[-1]['mean_precision_score']}"
        )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_tradeoff(df: pd.DataFrame, top_k: int, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    labels = df["top_clusters_label"].tolist()
    tokens = df["mean_context_tokens"].tolist()
    latencies = df["mean_latency_ms"].tolist()

    has_precision = df["mean_precision_score"].notna().any()

    # --- Left: latency vs top_clusters ---
    ax1 = axes[0]
    ax1.plot(labels, latencies, marker="o", linewidth=2, color="#dc2626")
    ax1.set_xlabel("top_clusters", fontsize=12)
    ax1.set_ylabel("Mean Retrieve Latency (ms)", fontsize=12)
    ax1.set_title(f"Latency vs. Routing Breadth (top_k={top_k})", fontsize=13)
    ax1.grid(True, alpha=0.3)
    for x, y in zip(labels, latencies):
        ax1.annotate(f"{y:.0f}ms", (x, y), textcoords="offset points",
                     xytext=(0, 8), ha="center", fontsize=9)

    # --- Right: precision vs tokens tradeoff ---
    ax2 = axes[1]
    if has_precision:
        precision = df["mean_precision_score"].tolist()
        ax2.plot(tokens, precision, marker="o", linewidth=2, color="#16a34a")
        for x, y, lbl in zip(tokens, precision, labels):
            ax2.annotate(f"k={lbl}", (x, y), textcoords="offset points",
                         xytext=(6, 4), fontsize=9)
        ax2.set_xlabel("Mean Context Tokens", fontsize=12)
        ax2.set_ylabel("Mean Retrieval Precision", fontsize=12)
        ax2.set_title("Precision vs. Token Cost Tradeoff", fontsize=13)
    else:
        ax2.plot(labels, tokens, marker="s", linewidth=2, color="#2563eb")
        ax2.set_xlabel("top_clusters", fontsize=12)
        ax2.set_ylabel("Mean Context Tokens", fontsize=12)
        ax2.set_title(f"Context Size vs. Routing Breadth (top_k={top_k})", fontsize=13)
        for x, y in zip(labels, tokens):
            ax2.annotate(f"{y:.0f}", (x, y), textcoords="offset points",
                         xytext=(0, 8), ha="center", fontsize=9)

    ax2.grid(True, alpha=0.3)
    fig.suptitle(
        "top_clusters Ablation — Token Efficiency vs. Retrieval Quality",
        fontsize=14, fontweight="bold", y=1.02,
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    log.info(f"Plot saved → {out_path}")
    plt.close()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args():
    p = argparse.ArgumentParser(description="top_clusters ablation")
    p.add_argument("--n",         type=int,   default=50)
    p.add_argument("--top-k",     type=int,   default=10)
    p.add_argument("--seed",      type=int,   default=42)
    p.add_argument("--tags",      nargs="+",  default=None)
    p.add_argument("--dry-run",   action="store_true", help="Skip LLM judge")
    p.add_argument("--minilm-routing", action="store_true", help="Use MiniLM centroids for cluster routing")
    p.add_argument("--openai",    action="store_true", help="Use OpenAI (gpt-4o-mini) instead of Anthropic")
    p.add_argument("--api-key",   type=str,   default=None)
    p.add_argument("--log-level", default="INFO")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    queries = load_queries(args.n, args.seed, args.tags)
    log.info(f"Loaded {len(queries)} queries.")

    base_retriever = Retriever(top_clusters=ALL_CLUSTERS, top_k=args.top_k)
    if args.minilm_routing:
        from src.evaluation.embedding_ablation import MiniLMEmbedder, MiniLMRoutingRetriever
        log.info("Swapping to MiniLM cluster routing ...")
        retriever = MiniLMRoutingRetriever(base_retriever, MiniLMEmbedder(), ALL_CLUSTERS)
    else:
        retriever = base_retriever

    judge = None
    if not args.dry_run:
        judge = JudgeClient(api_key=args.api_key, use_openai=args.openai)

    df = run_ablation(retriever, queries, args.top_k, TOP_CLUSTERS_VALUES, judge)

    print("\n" + "=" * 60)
    print("TOP_CLUSTERS ABLATION RESULTS")
    print("=" * 60)
    print(df.to_string(index=False))
    print("=" * 60)

    if judge:
        print(f"\nTotal judge cost: ${judge.total_cost_usd:.4f}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = RESULTS_DIR / f"ablation_topclusters_{ts}.csv"
    df.to_csv(csv_path, index=False)
    log.info(f"Data saved → {csv_path}")

    plot_path = RESULTS_DIR / f"ablation_topclusters_{ts}.png"
    plot_tradeoff(df, args.top_k, plot_path)
