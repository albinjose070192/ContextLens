"""
Multi-Step Agent Simulation — Flat RAG vs. Selective RAG.

Models a developer working through a sequence of related questions in a single
session.  Each task is a list of N SO questions from the same tag (same domain),
run sequentially through three conditions:

  flat_accumulate  — naive agent: all retrieved chunks append to context every
                     step.  Context grows monotonically.  Worst case baseline.

  flat_reset       — Lewis et al. baseline: fresh top-k retrieval each step,
                     no cross-step memory.  Tokens are constant per step but
                     the agent has no context continuity.

  selective        — this paper: ContextRegistry with evict_older_than(1).
                     Chunks from the previous step that are not re-retrieved are
                     dropped.  Redundant reloads are blocked by content hash.
                     Context is bounded and deduped.

Primary metric: cumulative input tokens across all steps of a task.
Secondary:      per-step token count, dedup savings, retrieve latency.

Usage:
    # Dry run — no LLM, token counts only (fast)
    python -m src.evaluation.multistep --n-tasks 25 --steps 5 --dry-run

    # Full run with LLM quality eval on final step
    ANTHROPIC_API_KEY=... python -m src.evaluation.multistep --n-tasks 25 --steps 5
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

import pandas as pd
import tiktoken

from src.agent.context_manager import ContextRegistry
from src.retrieval.retriever import Retriever

log = logging.getLogger(__name__)

ROOT        = Path(__file__).parents[2]
SO_PATH     = ROOT / "data" / "raw" / "stackoverflow" / "stackoverflow_all_raw.json"
RESULTS_DIR = ROOT / "data" / "results"

_ENC = tiktoken.get_encoding("cl100k_base")


def _count_tokens(text: str) -> int:
    return len(_ENC.encode(text))


def _flat_prompt(query: str, chunks: list[dict]) -> str:
    parts = []
    for c in chunks:
        parts.append(
            f"[chunk_id={c.get('chunk_id','?')} | "
            f"type={c.get('source_type','?')} | "
            f"repo={c.get('repo','?')} | "
            f"score={c.get('score',0):.4f}]\n{c.get('text','')}"
        )
    context = "\n\n".join(parts)
    return (
        "You are a helpful software engineering assistant. "
        "Answer the query using only the retrieved context.\n\n"
        f"## Context\n\n{context}\n\n"
        f"## Query\n\n{query}"
    )


# ---------------------------------------------------------------------------
# Task builder
# ---------------------------------------------------------------------------

def build_tasks(
    n_tasks: int,
    steps_per_task: int,
    seed: int = 42,
    tags: Optional[list[str]] = None,
) -> list[list[dict]]:
    """
    Group SO questions into tasks of `steps_per_task` questions each.
    Questions within a task share the same tag (same domain = realistic session).
    Returns a list of tasks; each task is a list of SO record dicts.
    """
    with open(SO_PATH, encoding="utf-8") as f:
        data = json.load(f)

    data = [d for d in data if d.get("title") and d.get("answer_body")]
    if tags:
        data = [d for d in data if d.get("tag") in tags]

    # Group by tag
    by_tag: dict[str, list[dict]] = {}
    for rec in data:
        by_tag.setdefault(rec["tag"], []).append(rec)

    rng = random.Random(seed)

    tasks: list[list[dict]] = []
    # Distribute tasks across tags proportionally
    tag_list = sorted(by_tag.keys())
    tasks_per_tag = max(1, n_tasks // len(tag_list))

    for tag in tag_list:
        pool = by_tag[tag]
        if len(pool) < steps_per_task:
            continue
        for _ in range(tasks_per_tag):
            task = rng.sample(pool, steps_per_task)
            tasks.append(task)
            if len(tasks) >= n_tasks:
                break
        if len(tasks) >= n_tasks:
            break

    log.info(f"Built {len(tasks)} tasks × {steps_per_task} steps "
             f"from tags: {tag_list}")
    return tasks[:n_tasks]


# ---------------------------------------------------------------------------
# Three conditions
# ---------------------------------------------------------------------------

def run_flat_accumulate(
    task: list[dict],
    retriever: Retriever,
    top_k: int,
) -> dict:
    """
    Naive agent: all retrieved chunks accumulate in context across steps.
    Context grows monotonically — worst-case token usage.
    """
    accumulated: dict[str, dict] = {}   # chunk_id → chunk
    step_rows = []
    cumulative_tokens = 0

    for step_idx, record in enumerate(task):
        query = record["title"]

        t0 = time.perf_counter()
        chunks = retriever.retrieve_flat(query, top_k=top_k)
        retrieve_ms = (time.perf_counter() - t0) * 1000

        # Accumulate — no eviction, no dedup
        for c in chunks:
            accumulated[c["chunk_id"]] = c

        prompt = _flat_prompt(query, list(accumulated.values()))
        step_tokens = _count_tokens(prompt)
        cumulative_tokens += step_tokens

        step_rows.append({
            "step": step_idx + 1,
            "query": query,
            "n_chunks_in_window": len(accumulated),
            "step_tokens": step_tokens,
            "cumulative_tokens": cumulative_tokens,
            "retrieve_ms": round(retrieve_ms, 1),
        })

    return {
        "condition": "flat_accumulate",
        "tag": task[0].get("tag", ""),
        "steps": step_rows,
        "total_tokens": cumulative_tokens,
        "final_step_tokens": step_rows[-1]["step_tokens"],
        "final_n_chunks": step_rows[-1]["n_chunks_in_window"],
    }


def run_flat_reset(
    task: list[dict],
    retriever: Retriever,
    top_k: int,
) -> dict:
    """
    Lewis et al. baseline: fresh top-k retrieval each step, no memory.
    """
    step_rows = []
    cumulative_tokens = 0

    for step_idx, record in enumerate(task):
        query = record["title"]

        t0 = time.perf_counter()
        chunks = retriever.retrieve_flat(query, top_k=top_k)
        retrieve_ms = (time.perf_counter() - t0) * 1000

        prompt = _flat_prompt(query, chunks)
        step_tokens = _count_tokens(prompt)
        cumulative_tokens += step_tokens

        step_rows.append({
            "step": step_idx + 1,
            "query": query,
            "n_chunks_in_window": len(chunks),
            "step_tokens": step_tokens,
            "cumulative_tokens": cumulative_tokens,
            "retrieve_ms": round(retrieve_ms, 1),
        })

    return {
        "condition": "flat_reset",
        "tag": task[0].get("tag", ""),
        "steps": step_rows,
        "total_tokens": cumulative_tokens,
        "final_step_tokens": step_rows[-1]["step_tokens"],
        "final_n_chunks": step_rows[-1]["n_chunks_in_window"],
    }


def run_selective(
    task: list[dict],
    retriever: Retriever,
    registry: ContextRegistry,
    top_k: int,
    top_clusters: int,
    evict_after_steps: int = 1,
    evict_all: bool = False,
) -> dict:
    """
    Selective RAG: ContextRegistry with age-based eviction.
    Chunks older than `evict_after_steps` steps are evicted between turns.
    Content-hash dedup prevents redundant reloads.
    """
    registry.evict_all()
    step_rows = []
    cumulative_tokens = 0
    total_dedup_saves = 0

    for step_idx, record in enumerate(task):
        query = record["title"]
        registry.advance_step()

        t0 = time.perf_counter()
        chunks = retriever.retrieve(query, top_k=top_k, top_clusters=top_clusters)
        retrieve_ms = (time.perf_counter() - t0) * 1000

        load_stats = registry.load_chunks(chunks)
        total_dedup_saves += load_stats["skipped_duplicate"]

        # Evict chunks older than N steps (or all if evict_all=True)
        if evict_all:
            evicted = registry.size
            registry.evict_all()
        else:
            evicted = registry.evict_older_than(evict_after_steps)

        context_tokens = _count_tokens(registry.get_context_window())
        prompt_tokens  = _count_tokens(registry.build_prompt(query))
        cumulative_tokens += prompt_tokens

        step_rows.append({
            "step": step_idx + 1,
            "query": query,
            "n_chunks_in_window": registry.size,
            "n_newly_loaded": load_stats["loaded"],
            "n_skipped_dup": load_stats["skipped_duplicate"],
            "n_evicted": evicted,
            "context_tokens": context_tokens,
            "step_tokens": prompt_tokens,
            "cumulative_tokens": cumulative_tokens,
            "retrieve_ms": round(retrieve_ms, 1),
        })

    registry.evict_all()
    return {
        "condition": "selective",
        "tag": task[0].get("tag", ""),
        "steps": step_rows,
        "total_tokens": cumulative_tokens,
        "final_step_tokens": step_rows[-1]["step_tokens"],
        "final_n_chunks": step_rows[-1]["n_chunks_in_window"],
        "total_dedup_saves": total_dedup_saves,
    }


# ---------------------------------------------------------------------------
# Harness
# ---------------------------------------------------------------------------

class MultiStepHarness:
    def __init__(
        self,
        n_tasks: int = 25,
        steps_per_task: int = 5,
        top_k: int = 5,
        top_clusters: int = 2,
        evict_after_steps: int = 1,
        evict_all: bool = False,
        seed: int = 42,
        tags: Optional[list[str]] = None,
        minilm_routing: bool = False,
    ):
        self.n_tasks           = n_tasks
        self.steps_per_task    = steps_per_task
        self.top_k             = top_k
        self.top_clusters      = top_clusters
        self.evict_after_steps = evict_after_steps
        self.evict_all         = evict_all
        self.seed              = seed
        self.tags              = tags
        self.minilm_routing    = minilm_routing

        log.info("Loading retriever ...")
        base_retriever = Retriever(top_clusters=top_clusters, top_k=top_k)
        if minilm_routing:
            from src.evaluation.embedding_ablation import MiniLMEmbedder, MiniLMRoutingRetriever
            log.info("Swapping to MiniLM cluster routing ...")
            self.retriever = MiniLMRoutingRetriever(base_retriever, MiniLMEmbedder(), top_clusters)
        else:
            self.retriever = base_retriever
        self.registry  = ContextRegistry(token_budget=8192)

    def run(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Returns (step_df, task_df):
          step_df — one row per (task_id, condition, step)
          task_df — one row per (task_id, condition) with aggregate stats
        """
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
            log.info(f"Task {task_id+1}/{len(tasks)} [{tag}]: "
                     f"{task[0]['title'][:50]} ...")

            for result in [
                run_flat_accumulate(task, self.retriever, self.top_k),
                run_flat_reset(task, self.retriever, self.top_k),
                run_selective(
                    task, self.retriever, self.registry,
                    self.top_k, self.top_clusters, self.evict_after_steps,
                    self.evict_all,
                ),
            ]:
                condition = result["condition"]
                for sr in result["steps"]:
                    step_rows.append({
                        "task_id":   task_id,
                        "tag":       tag,
                        "condition": condition,
                        **sr,
                    })
                task_rows.append({
                    "task_id":          task_id,
                    "tag":              tag,
                    "condition":        condition,
                    "total_tokens":     result["total_tokens"],
                    "final_step_tokens": result["final_step_tokens"],
                    "final_n_chunks":   result["final_n_chunks"],
                    "total_dedup_saves": result.get("total_dedup_saves", 0),
                })

        return pd.DataFrame(step_rows), pd.DataFrame(task_rows)

    # -----------------------------------------------------------------------

    def save(self, step_df: pd.DataFrame, task_df: pd.DataFrame) -> Path:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
        stem = RESULTS_DIR / f"multistep_{ts}"

        step_df.to_csv(f"{stem}_steps.csv", index=False)
        task_df.to_csv(f"{stem}_tasks.csv", index=False)
        step_df.to_json(f"{stem}_steps.json", orient="records", indent=2)
        task_df.to_json(f"{stem}_tasks.json", orient="records", indent=2)

        log.info(f"Results saved → {stem}_steps.csv / _tasks.csv")
        return Path(f"{stem}_tasks.csv")

    # -----------------------------------------------------------------------

    @staticmethod
    def summarize(task_df: pd.DataFrame, step_df: pd.DataFrame) -> dict:
        def _mean(series):
            return round(series.dropna().mean(), 1)

        fa = task_df[task_df["condition"] == "flat_accumulate"]
        fr = task_df[task_df["condition"] == "flat_reset"]
        se = task_df[task_df["condition"] == "selective"]

        fa_total = _mean(fa["total_tokens"])
        fr_total = _mean(fr["total_tokens"])
        se_total = _mean(se["total_tokens"])

        def _reduction(baseline, target):
            if baseline:
                return round((baseline - target) / baseline * 100, 2)
            return None

        # Per-step mean tokens across all tasks × steps
        step_means = step_df.groupby(["condition", "step"])["step_tokens"].mean()

        return {
            "n_tasks":        len(fa),
            "steps_per_task": step_df["step"].max(),
            "top_k":          int(step_df.get("n_chunks_in_window", pd.Series([0])).max()),
            "mean_total_tokens": {
                "flat_accumulate": fa_total,
                "flat_reset":      fr_total,
                "selective":       se_total,
            },
            "token_reduction_vs_flat_accumulate": _reduction(fa_total, se_total),
            "token_reduction_vs_flat_reset":       _reduction(fr_total, se_total),
            "mean_final_step_tokens": {
                "flat_accumulate": _mean(fa["final_step_tokens"]),
                "flat_reset":      _mean(fr["final_step_tokens"]),
                "selective":       _mean(se["final_step_tokens"]),
            },
            "mean_dedup_saves_per_task": _mean(se["total_dedup_saves"]),
            "step_means": step_means.to_dict(),
        }

    def print_summary(self, task_df: pd.DataFrame, step_df: pd.DataFrame) -> None:
        s = self.summarize(task_df, step_df)
        mt = s["mean_total_tokens"]
        mf = s["mean_final_step_tokens"]

        print("\n" + "=" * 65)
        print("MULTI-STEP EVALUATION SUMMARY")
        print("=" * 65)
        print(f"Tasks: {s['n_tasks']}  |  Steps/task: {s['steps_per_task']}  |  top_k: {self.top_k}")
        print(f"\n--- Cumulative Tokens (sum across all steps per task) ---")
        print(f"  Flat accumulate : {mt['flat_accumulate']:>8.1f}  ← naive agent, worst case")
        print(f"  Flat reset      : {mt['flat_reset']:>8.1f}  ← Lewis et al. baseline")
        print(f"  Selective       : {mt['selective']:>8.1f}  ← this paper")
        print(f"\n  Reduction vs flat_accumulate : {s['token_reduction_vs_flat_accumulate']}%")
        print(f"  Reduction vs flat_reset      : {s['token_reduction_vs_flat_reset']}%")
        print(f"\n--- Final-Step Tokens (context size at step {s['steps_per_task']}) ---")
        print(f"  Flat accumulate : {mf['flat_accumulate']:>8.1f}")
        print(f"  Flat reset      : {mf['flat_reset']:>8.1f}")
        print(f"  Selective       : {mf['selective']:>8.1f}")
        print(f"\n--- Registry Dedup ---")
        print(f"  Mean duplicate chunks blocked/task : {s['mean_dedup_saves_per_task']}")
        print(f"\n--- Per-Step Mean Tokens ---")
        for (cond, step), val in sorted(s["step_means"].items()):
            print(f"  {cond:<18} step {step}: {val:.1f}")
        print("=" * 65 + "\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args():
    p = argparse.ArgumentParser(description="Multi-step agent simulation")
    p.add_argument("--n-tasks",   type=int, default=25,  help="Number of tasks (agent sessions)")
    p.add_argument("--steps",     type=int, default=5,   help="Steps (queries) per task")
    p.add_argument("--top-k",     type=int, default=5,   help="Chunks retrieved per step")
    p.add_argument("--top-clusters", type=int, default=2)
    p.add_argument("--evict-after", type=int, default=1, help="Evict chunks older than N steps")
    p.add_argument("--evict-all",   action="store_true", help="Evict entire registry after each step")
    p.add_argument("--seed",      type=int, default=42)
    p.add_argument("--tags",      nargs="+", default=None)
    p.add_argument("--minilm-routing", action="store_true", help="Use MiniLM centroids for cluster routing")
    p.add_argument("--log-level", default="INFO")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    harness = MultiStepHarness(
        n_tasks=args.n_tasks,
        steps_per_task=args.steps,
        top_k=args.top_k,
        top_clusters=args.top_clusters,
        evict_after_steps=args.evict_after,
        evict_all=args.evict_all,
        seed=args.seed,
        tags=args.tags,
        minilm_routing=args.minilm_routing,
    )

    step_df, task_df = harness.run()
    harness.print_summary(task_df, step_df)
    harness.save(step_df, task_df)
