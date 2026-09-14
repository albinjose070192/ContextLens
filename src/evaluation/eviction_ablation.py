"""
Eviction Strategy Ablation — token accumulation and quality across strategies.

Compares four eviction strategies in a multi-step agent session:

  evict_all          — registry cleared after every step. Zero cross-step memory.
                       Minimum token cost, no continuity benefit.

  evict_older_than_1 — chunks older than 1 step evicted. Carries forward chunks
                       retrieved in the immediately preceding step.

  evict_older_than_2 — chunks older than 2 steps evicted. Broader carry-forward.
                       Context window can grow but is still bounded.

  no_eviction        — nothing ever evicted (accumulate mode, worst-case upper
                       bound). Equivalent to flat_accumulate in multistep.py.

Token counts require no LLM. Quality eval (--with-llm) calls the LLM on the
final step of each task only to keep API cost reasonable.

Usage:
    # Token counts + latency only (fast, no API key)
    python -m src.evaluation.eviction_ablation --n-tasks 20 --steps 5 --top-k 5

    # With quality eval on final step
    ANTHROPIC_API_KEY=... python -m src.evaluation.eviction_ablation \\
        --n-tasks 20 --steps 5 --top-k 5 --with-llm
"""

import argparse
import json
import logging
import os
import random
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import anthropic
import openai as openai_lib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import tiktoken

from src.agent.context_manager import ContextRegistry
from src.retrieval.retriever import Retriever
from src.evaluation.multistep import build_tasks

log = logging.getLogger(__name__)

ROOT        = Path(__file__).parents[2]
RESULTS_DIR = ROOT / "data" / "results"

_ENC             = tiktoken.get_encoding("cl100k_base")
_MODEL_JUDGE     = "claude-haiku-4-5-20251001"
_INPUT_COST_PER_1M  = 0.80
_OUTPUT_COST_PER_1M = 4.00
_OAI_MODEL_JUDGE        = "gpt-4o-mini"
_OAI_INPUT_COST_PER_1M  = 0.15
_OAI_OUTPUT_COST_PER_1M = 0.60


def _count_tokens(text: str) -> int:
    return len(_ENC.encode(text))


# ---------------------------------------------------------------------------
# Eviction strategies
# ---------------------------------------------------------------------------

@dataclass
class EvictionStrategy:
    name: str
    label: str          # short display name
    color: str          # plot color

STRATEGIES = [
    EvictionStrategy("no_eviction",        "No eviction",        "#dc2626"),
    EvictionStrategy("evict_older_than_2", "evict_older_than(2)","#f97316"),
    EvictionStrategy("evict_older_than_1", "evict_older_than(1)","#2563eb"),
    EvictionStrategy("evict_all",          "evict_all",          "#16a34a"),
]


def _apply_eviction(strategy: EvictionStrategy, registry: ContextRegistry) -> int:
    """Apply the given eviction strategy to the registry. Returns chunks evicted."""
    if strategy.name == "evict_all":
        n = registry.size
        registry.evict_all()
        return n
    elif strategy.name == "evict_older_than_1":
        return registry.evict_older_than(1)
    elif strategy.name == "evict_older_than_2":
        return registry.evict_older_than(2)
    elif strategy.name == "no_eviction":
        return 0
    return 0


# ---------------------------------------------------------------------------
# LLM judge (final-step quality only)
# ---------------------------------------------------------------------------

class JudgeClient:
    def __init__(self, api_key: Optional[str] = None, use_openai: bool = False):
        self._use_openai = use_openai
        if use_openai:
            key = api_key or os.environ.get("OPENAI_API_KEY")
            if not key:
                raise ValueError("OPENAI_API_KEY not set.")
            self._client = openai_lib.OpenAI(api_key=key)
            self._model = _OAI_MODEL_JUDGE
            self._in_cost  = _OAI_INPUT_COST_PER_1M
            self._out_cost = _OAI_OUTPUT_COST_PER_1M
        else:
            key = api_key or os.environ.get("ANTHROPIC_API_KEY")
            if not key:
                raise ValueError("ANTHROPIC_API_KEY not set.")
            self._client = anthropic.Anthropic(api_key=key)
            self._model = _MODEL_JUDGE
            self._in_cost  = _INPUT_COST_PER_1M
            self._out_cost = _OUTPUT_COST_PER_1M
        self.total_input_tokens  = 0
        self.total_output_tokens = 0

    def call(self, prompt: str, max_tokens: int = 256) -> tuple[str, int, int]:
        for attempt in range(5):
            try:
                if self._use_openai:
                    resp = self._client.chat.completions.create(
                        model=self._model, max_tokens=max_tokens,
                        messages=[{"role": "user", "content": prompt}],
                    )
                    in_tok  = resp.usage.prompt_tokens
                    out_tok = resp.usage.completion_tokens
                    text    = resp.choices[0].message.content
                else:
                    msg = self._client.messages.create(
                        model=self._model, max_tokens=max_tokens,
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
            raise RuntimeError("Rate limit retries exhausted.")
        self.total_input_tokens  += in_tok
        self.total_output_tokens += out_tok
        return text, in_tok, out_tok

    def completion_score(self, query: str, ground_truth: str, response: str) -> Optional[float]:
        prompt = (
            "You are an expert evaluator for a RAG system.\n"
            "Given the question, ground-truth answer, and model response, output:\n"
            '  {"completion_score": float 0.0-1.0}\n'
            "ONLY valid JSON, no explanation.\n\n"
            f"Question: {query}\n\nGround truth: {ground_truth[:400]}\n\n"
            f"Model response: {response}"
        )
        text, _, _ = self.call(prompt, max_tokens=64)
        try:
            clean = text.strip().strip("```json").strip("```").strip()
            return float(json.loads(clean).get("completion_score"))
        except Exception:
            return None

    @property
    def total_cost_usd(self) -> float:
        return (
            self.total_input_tokens  / 1_000_000 * self._in_cost
            + self.total_output_tokens / 1_000_000 * self._out_cost
        )


# ---------------------------------------------------------------------------
# Single-strategy runner for one task
# ---------------------------------------------------------------------------

def run_strategy_on_task(
    task: list[dict],
    strategy: EvictionStrategy,
    retriever: Retriever,
    top_k: int,
    top_clusters: int,
    judge: Optional[JudgeClient],
) -> dict:
    registry = ContextRegistry(token_budget=16384)
    registry.evict_all()

    step_rows = []
    cumulative_tokens = 0
    total_dedup_saves = 0

    n_steps = len(task)

    for step_idx, record in enumerate(task):
        query = record["title"]
        gt    = record.get("answer_body", "")
        registry.advance_step()

        t0 = time.perf_counter()
        chunks = retriever.retrieve(query, top_k=top_k, top_clusters=top_clusters)
        retrieve_ms = (time.perf_counter() - t0) * 1000

        load_stats = registry.load_chunks(chunks)
        total_dedup_saves += load_stats["skipped_duplicate"]

        # Apply eviction AFTER loading (so new chunks are never immediately evicted)
        evicted = _apply_eviction(strategy, registry)

        context_tokens = _count_tokens(registry.get_context_window())
        prompt_tokens  = _count_tokens(registry.build_prompt(query))
        cumulative_tokens += prompt_tokens

        # LLM quality eval on final step only
        completion_score = None
        if judge and step_idx == n_steps - 1:
            prompt = registry.build_prompt(query)
            response, _, _ = judge.call(prompt, max_tokens=512)
            completion_score = judge.completion_score(query, gt, response)

        step_rows.append({
            "step":              step_idx + 1,
            "query":             query,
            "n_chunks_in_window": registry.size,
            "n_newly_loaded":    load_stats["loaded"],
            "n_skipped_dup":     load_stats["skipped_duplicate"],
            "n_evicted":         evicted,
            "context_tokens":    context_tokens,
            "step_tokens":       prompt_tokens,
            "cumulative_tokens": cumulative_tokens,
            "retrieve_ms":       round(retrieve_ms, 1),
            "completion_score":  completion_score,
        })

    registry.evict_all()
    return {
        "strategy":          strategy.name,
        "steps":             step_rows,
        "total_tokens":      cumulative_tokens,
        "final_step_tokens": step_rows[-1]["step_tokens"],
        "final_n_chunks":    step_rows[-1]["n_chunks_in_window"],
        "total_dedup_saves": total_dedup_saves,
        "final_completion_score": step_rows[-1]["completion_score"],
    }


# ---------------------------------------------------------------------------
# Harness
# ---------------------------------------------------------------------------

class EvictionAblationHarness:
    def __init__(
        self,
        n_tasks: int = 20,
        steps_per_task: int = 5,
        top_k: int = 5,
        top_clusters: int = 2,
        seed: int = 42,
        tags: Optional[list[str]] = None,
        with_llm: bool = False,
        api_key: Optional[str] = None,
        minilm_routing: bool = False,
        use_openai: bool = False,
    ):
        self.n_tasks        = n_tasks
        self.steps_per_task = steps_per_task
        self.top_k          = top_k
        self.top_clusters   = top_clusters
        self.seed           = seed
        self.tags           = tags

        log.info("Loading retriever ...")
        base_retriever = Retriever(top_clusters=top_clusters, top_k=top_k)
        if minilm_routing:
            from src.evaluation.embedding_ablation import MiniLMEmbedder, MiniLMRoutingRetriever
            log.info("Swapping to MiniLM cluster routing ...")
            self.retriever = MiniLMRoutingRetriever(base_retriever, MiniLMEmbedder(), top_clusters)
        else:
            self.retriever = base_retriever

        self.judge: Optional[JudgeClient] = None
        if with_llm:
            self.judge = JudgeClient(api_key=api_key, use_openai=use_openai)

    def run(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        tasks = build_tasks(
            n_tasks=self.n_tasks,
            steps_per_task=self.steps_per_task,
            seed=self.seed,
            tags=self.tags,
        )

        step_rows = []
        task_rows = []

        for task_id, task in enumerate(tasks):
            tag = task[0].get("tag", "")
            log.info(f"Task {task_id+1}/{len(tasks)} [{tag}]")

            for strategy in STRATEGIES:
                result = run_strategy_on_task(
                    task, strategy, self.retriever,
                    self.top_k, self.top_clusters, self.judge,
                )
                for sr in result["steps"]:
                    step_rows.append({
                        "task_id":  task_id,
                        "tag":      tag,
                        "strategy": strategy.name,
                        "label":    strategy.label,
                        **sr,
                    })
                task_rows.append({
                    "task_id":             task_id,
                    "tag":                 tag,
                    "strategy":            strategy.name,
                    "label":               strategy.label,
                    "total_tokens":        result["total_tokens"],
                    "final_step_tokens":   result["final_step_tokens"],
                    "final_n_chunks":      result["final_n_chunks"],
                    "total_dedup_saves":   result["total_dedup_saves"],
                    "final_completion_score": result["final_completion_score"],
                })

        return pd.DataFrame(step_rows), pd.DataFrame(task_rows)

    # -----------------------------------------------------------------------

    def save(self, step_df: pd.DataFrame, task_df: pd.DataFrame) -> Path:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
        stem = RESULTS_DIR / f"eviction_{ts}"
        step_df.to_csv(f"{stem}_steps.csv", index=False)
        task_df.to_csv(f"{stem}_tasks.csv", index=False)
        log.info(f"Results saved → {stem}_steps.csv / _tasks.csv")
        return Path(f"{stem}_tasks.csv")

    # -----------------------------------------------------------------------

    @staticmethod
    def summarize(task_df: pd.DataFrame, step_df: pd.DataFrame) -> pd.DataFrame:
        rows = []
        baseline_tokens = None

        for strategy in STRATEGIES:
            t = task_df[task_df["strategy"] == strategy.name]
            s = step_df[step_df["strategy"] == strategy.name]

            mean_total   = round(t["total_tokens"].mean(), 1)
            mean_final   = round(t["final_step_tokens"].mean(), 1)
            mean_chunks  = round(t["final_n_chunks"].mean(), 1)
            mean_dedup   = round(t["total_dedup_saves"].mean(), 1)
            mean_completion = (
                round(t["final_completion_score"].dropna().mean(), 3)
                if t["final_completion_score"].notna().any() else None
            )

            if strategy.name == "no_eviction":
                baseline_tokens = mean_total

            reduction = (
                round((baseline_tokens - mean_total) / baseline_tokens * 100, 1)
                if baseline_tokens and strategy.name != "no_eviction"
                else 0.0
            )

            rows.append({
                "strategy":            strategy.label,
                "mean_total_tokens":   mean_total,
                "token_reduction_%":   reduction,
                "mean_final_tokens":   mean_final,
                "mean_final_chunks":   mean_chunks,
                "mean_dedup_saves":    mean_dedup,
                "mean_completion":     mean_completion,
            })

        return pd.DataFrame(rows)

    def print_summary(self, task_df: pd.DataFrame, step_df: pd.DataFrame) -> None:
        df = self.summarize(task_df, step_df)
        print("\n" + "=" * 75)
        print("EVICTION STRATEGY ABLATION SUMMARY")
        print(f"Tasks: {task_df['task_id'].nunique()}  |  "
              f"Steps/task: {step_df['step'].max()}  |  top_k: {self.top_k}")
        print("=" * 75)
        print(df.to_string(index=False))
        print("=" * 75 + "\n")
        if self.judge:
            print(f"Total LLM judge cost: ${self.judge.total_cost_usd:.4f}\n")


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_eviction(step_df: pd.DataFrame, task_df: pd.DataFrame,
                  top_k: int, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    strategy_map = {s.name: s for s in STRATEGIES}

    # --- Left: per-step mean token growth curves ---
    ax1 = axes[0]
    steps = sorted(step_df["step"].unique())
    for strategy in STRATEGIES:
        s_df = step_df[step_df["strategy"] == strategy.name]
        means = s_df.groupby("step")["step_tokens"].mean()
        ax1.plot(steps, [means[s] for s in steps],
                 marker="o", linewidth=2,
                 color=strategy.color, label=strategy.label)

    ax1.set_xlabel("Agent Step", fontsize=12)
    ax1.set_ylabel("Mean Step Tokens", fontsize=12)
    ax1.set_title(f"Token Growth per Step (top_k={top_k})", fontsize=13)
    ax1.legend(fontsize=9)
    ax1.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax1.grid(True, alpha=0.3)

    # --- Right: cumulative token distribution (box plot) ---
    ax2 = axes[1]
    data_by_strategy = [
        task_df[task_df["strategy"] == s.name]["total_tokens"].values
        for s in STRATEGIES
    ]
    labels = [s.label for s in STRATEGIES]
    colors = [s.color for s in STRATEGIES]

    bp = ax2.boxplot(data_by_strategy, patch_artist=True, notch=False,
                     medianprops=dict(color="black", linewidth=2))
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    ax2.set_xticks(range(1, len(STRATEGIES) + 1))
    ax2.set_xticklabels(labels, fontsize=9, rotation=12)
    ax2.set_ylabel("Cumulative Tokens (5-step task)", fontsize=12)
    ax2.set_title("Cumulative Token Distribution by Strategy", fontsize=13)
    ax2.grid(True, alpha=0.3, axis="y")

    # Annotate medians
    for i, vals in enumerate(data_by_strategy, 1):
        med = float(np.median(vals))
        ax2.annotate(f"{med:.0f}", (i, med),
                     textcoords="offset points", xytext=(8, 0),
                     fontsize=8, color="black")

    fig.suptitle(
        "Eviction Strategy Ablation — Token Efficiency vs. Context Continuity",
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
    p = argparse.ArgumentParser(description="Eviction strategy ablation")
    p.add_argument("--n-tasks",      type=int,  default=20)
    p.add_argument("--steps",        type=int,  default=5)
    p.add_argument("--top-k",        type=int,  default=5)
    p.add_argument("--top-clusters", type=int,  default=2)
    p.add_argument("--seed",         type=int,  default=42)
    p.add_argument("--tags",         nargs="+", default=None)
    p.add_argument("--with-llm",     action="store_true",
                   help="Run LLM quality eval on final step of each task")
    p.add_argument("--minilm-routing", action="store_true",
                   help="Use MiniLM centroids for cluster routing")
    p.add_argument("--openai",       action="store_true",
                   help="Use OpenAI (gpt-4o-mini) instead of Anthropic for judging")
    p.add_argument("--api-key",      type=str,  default=None)
    p.add_argument("--log-level",    default="INFO")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    harness = EvictionAblationHarness(
        n_tasks=args.n_tasks,
        steps_per_task=args.steps,
        top_k=args.top_k,
        top_clusters=args.top_clusters,
        seed=args.seed,
        tags=args.tags,
        with_llm=args.with_llm,
        api_key=args.api_key,
        minilm_routing=args.minilm_routing,
        use_openai=args.openai,
    )

    step_df, task_df = harness.run()
    harness.print_summary(task_df, step_df)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_path = RESULTS_DIR / f"eviction_{ts}.png"
    plot_eviction(step_df, task_df, args.top_k, plot_path)

    harness.save(step_df, task_df)
