"""
kg_eviction_ablation.py — KG PageRank Eviction Ablation.

Compares four eviction strategies on the StackOverflow multi-step benchmark.
All conditions use the same routed retrieval (MiniLM cluster routing + CodeBERT
FAISS) so that only the eviction strategy varies.

Conditions
----------
  no_eviction        — all retrieved chunks accumulate each step (worst case)
  evict_all          — registry cleared before each step (no cross-step memory)
  evict_older_than_1 — only chunks loaded at the current step survive forward
  pagerank_p{τ}      — ForgetGate: PageRank × recency, evict bottom τ% per step

τ sweep over {10, 20, 30, 40} produces a token-reduction Pareto table.

Limitation
----------
SO queries have no cross-step continuity, so PageRank cannot demonstrate its
main multi-session advantage here.  This eval measures whether PageRank makes
smarter within-session eviction decisions than age-based baselines under the
same retrieval load.  See multisession_eval.py for the cross-session story.

Usage
-----
    # Token-only (no LLM judge):
    python -m src.evaluation.kg_eviction_ablation --n-tasks 25 --steps 5

    # Custom τ sweep:
    python -m src.evaluation.kg_eviction_ablation --n-tasks 25 --steps 5 \\
        --tau-sweep 10 20 30 40

    # Fewer tasks for a quick sanity check:
    python -m src.evaluation.kg_eviction_ablation --n-tasks 5 --steps 5
"""

import argparse
import logging
import pickle
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd
import tiktoken

from src.agent.context_manager import ContextRegistry
from src.evaluation.multistep import build_tasks
from src.graph.episodic_store import EpisodicStore
from src.graph.forget_gate import ForgetGate
from src.retrieval.retriever import Retriever

log = logging.getLogger(__name__)

ROOT        = Path(__file__).parents[2]
RESULTS_DIR = ROOT / "data" / "results"

_ENC = tiktoken.get_encoding("cl100k_base")

_TAU_DEFAULT: list[float] = [10.0, 20.0, 30.0, 40.0]


def _count_tokens(text: str) -> int:
    return len(_ENC.encode(text))


# ---------------------------------------------------------------------------
# Condition runners
# ---------------------------------------------------------------------------

def run_no_eviction(
    task: list[dict],
    retriever,
    top_k: int,
    top_clusters: int,
) -> dict:
    """
    Accumulate all retrieved chunks across steps — no eviction, no dedup.
    Worst-case token baseline (routed retrieval).
    """
    registry = ContextRegistry(token_budget=8192)
    step_rows = []
    cumulative_tokens = 0

    for step_idx, record in enumerate(task):
        query = record["title"]
        registry.advance_step()

        t0 = time.perf_counter()
        chunks = retriever.retrieve(query, top_k=top_k, top_clusters=top_clusters)
        retrieve_ms = (time.perf_counter() - t0) * 1000

        load_stats = registry.load_chunks(chunks)

        prompt_tokens = _count_tokens(registry.build_prompt(query))
        cumulative_tokens += prompt_tokens

        step_rows.append({
            "step":               step_idx + 1,
            "query":              query,
            "n_chunks_in_window": registry.size,
            "n_newly_loaded":     load_stats["loaded"],
            "n_skipped_dup":      load_stats["skipped_duplicate"],
            "n_evicted":          0,
            "step_tokens":        prompt_tokens,
            "cumulative_tokens":  cumulative_tokens,
            "retrieve_ms":        round(retrieve_ms, 1),
        })

    return {
        "condition":           "no_eviction",
        "tag":                 task[0].get("tag", ""),
        "steps":               step_rows,
        "total_tokens":        cumulative_tokens,
        "final_step_tokens":   step_rows[-1]["step_tokens"],
        "final_n_chunks":      step_rows[-1]["n_chunks_in_window"],
        "total_dedup_saves":   sum(r["n_skipped_dup"] for r in step_rows),
        "eviction_percentile": None,
    }


def run_evict_all(
    task: list[dict],
    retriever,
    top_k: int,
    top_clusters: int,
) -> dict:
    """
    Clear the registry before loading each step — no cross-step memory.
    Each step sees only freshly retrieved chunks.
    """
    registry = ContextRegistry(token_budget=8192)
    step_rows = []
    cumulative_tokens = 0

    for step_idx, record in enumerate(task):
        query = record["title"]
        registry.advance_step()
        registry.evict_all()

        t0 = time.perf_counter()
        chunks = retriever.retrieve(query, top_k=top_k, top_clusters=top_clusters)
        retrieve_ms = (time.perf_counter() - t0) * 1000

        load_stats = registry.load_chunks(chunks)

        prompt_tokens = _count_tokens(registry.build_prompt(query))
        cumulative_tokens += prompt_tokens

        step_rows.append({
            "step":               step_idx + 1,
            "query":              query,
            "n_chunks_in_window": registry.size,
            "n_newly_loaded":     load_stats["loaded"],
            "n_skipped_dup":      load_stats["skipped_duplicate"],
            "n_evicted":          0,  # whole registry replaced, not tracked per-chunk
            "step_tokens":        prompt_tokens,
            "cumulative_tokens":  cumulative_tokens,
            "retrieve_ms":        round(retrieve_ms, 1),
        })

    return {
        "condition":           "evict_all",
        "tag":                 task[0].get("tag", ""),
        "steps":               step_rows,
        "total_tokens":        cumulative_tokens,
        "final_step_tokens":   step_rows[-1]["step_tokens"],
        "final_n_chunks":      step_rows[-1]["n_chunks_in_window"],
        "total_dedup_saves":   sum(r["n_skipped_dup"] for r in step_rows),
        "eviction_percentile": None,
    }


def run_evict_older_than_1(
    task: list[dict],
    retriever,
    top_k: int,
    top_clusters: int,
) -> dict:
    """
    Age-based eviction: drop chunks loaded more than 1 step ago after each step.
    Content-hash dedup prevents redundant reloads.
    """
    registry = ContextRegistry(token_budget=8192)
    step_rows = []
    cumulative_tokens = 0

    for step_idx, record in enumerate(task):
        query = record["title"]
        registry.advance_step()

        t0 = time.perf_counter()
        chunks = retriever.retrieve(query, top_k=top_k, top_clusters=top_clusters)
        retrieve_ms = (time.perf_counter() - t0) * 1000

        load_stats = registry.load_chunks(chunks)
        n_evicted = registry.evict_older_than(1)

        prompt_tokens = _count_tokens(registry.build_prompt(query))
        cumulative_tokens += prompt_tokens

        step_rows.append({
            "step":               step_idx + 1,
            "query":              query,
            "n_chunks_in_window": registry.size,
            "n_newly_loaded":     load_stats["loaded"],
            "n_skipped_dup":      load_stats["skipped_duplicate"],
            "n_evicted":          n_evicted,
            "step_tokens":        prompt_tokens,
            "cumulative_tokens":  cumulative_tokens,
            "retrieve_ms":        round(retrieve_ms, 1),
        })

    registry.evict_all()
    return {
        "condition":           "evict_older_than_1",
        "tag":                 task[0].get("tag", ""),
        "steps":               step_rows,
        "total_tokens":        cumulative_tokens,
        "final_step_tokens":   step_rows[-1]["step_tokens"],
        "final_n_chunks":      step_rows[-1]["n_chunks_in_window"],
        "total_dedup_saves":   sum(r["n_skipped_dup"] for r in step_rows),
        "eviction_percentile": None,
    }


def run_pagerank_eviction(
    task: list[dict],
    retriever,
    corpus_artifact: dict,
    top_k: int,
    top_clusters: int,
    eviction_percentile: float,
    lambda_decay: float = 0.1,
    min_chunks: int = 5,
) -> dict:
    """
    ForgetGate eviction: PageRank centrality × recency decay.

    After each step, chunks whose combined score falls below the τ-th percentile
    are evicted from the registry.  A temporary EpisodicStore tracks query→chunk
    RETRIEVED edges so PageRank can score cross-query chunk importance.

    The EpisodicStore is ephemeral (no save) — only the token/eviction telemetry
    is retained in the returned dict.
    """
    registry  = ContextRegistry(token_budget=8192)
    tmp_path  = Path(tempfile.mktemp(suffix=".pkl"))
    store     = EpisodicStore(path=tmp_path, embed_queries=False)
    sid       = store.start_session(project_id="kg_ablation")

    gate = ForgetGate(
        corpus_artifact    = corpus_artifact,
        eviction_percentile= eviction_percentile,
        lambda_decay       = lambda_decay,
        min_chunks         = min_chunks,
    )

    step_rows = []
    cumulative_tokens = 0

    for step_idx, record in enumerate(task):
        query       = record["title"]
        step_number = step_idx + 1
        registry.advance_step()

        t0 = time.perf_counter()
        chunks = retriever.retrieve(query, top_k=top_k, top_clusters=top_clusters)
        retrieve_ms = (time.perf_counter() - t0) * 1000

        load_stats = registry.load_chunks(chunks)

        # Record retrieval in episodic store so PageRank has RETRIEVED edges
        qid = store.record_query(sid, text=query, step_number=step_number)
        for chunk in chunks:
            store.record_retrieved(
                qid,
                chunk["chunk_id"],
                score=chunk.get("score", 0.0),
                step=step_number,
            )

        # Run ForgetGate — evict bottom τ% by PageRank × recency
        evicted_ids = gate(store, registry, sid)
        for cid in evicted_ids:
            store.record_evicted(qid, cid, step=step_number, strategy="pagerank")

        prompt_tokens = _count_tokens(registry.build_prompt(query))
        cumulative_tokens += prompt_tokens

        step_rows.append({
            "step":               step_number,
            "query":              query,
            "n_chunks_in_window": registry.size,
            "n_newly_loaded":     load_stats["loaded"],
            "n_skipped_dup":      load_stats["skipped_duplicate"],
            "n_evicted":          len(evicted_ids),
            "step_tokens":        prompt_tokens,
            "cumulative_tokens":  cumulative_tokens,
            "retrieve_ms":        round(retrieve_ms, 1),
        })

    store.end_session(sid)
    if tmp_path.exists():
        tmp_path.unlink()

    return {
        "condition":           f"pagerank_p{int(eviction_percentile)}",
        "tag":                 task[0].get("tag", ""),
        "steps":               step_rows,
        "total_tokens":        cumulative_tokens,
        "final_step_tokens":   step_rows[-1]["step_tokens"],
        "final_n_chunks":      step_rows[-1]["n_chunks_in_window"],
        "total_dedup_saves":   sum(r["n_skipped_dup"] for r in step_rows),
        "eviction_percentile": eviction_percentile,
    }


# ---------------------------------------------------------------------------
# Harness
# ---------------------------------------------------------------------------

class KGEvictionAblation:
    """
    Runs the KG eviction ablation across n_tasks SO question sequences.

    Parameters
    ----------
    n_tasks          : number of SO tasks (agent sessions) to simulate
    steps_per_task   : queries per task
    top_k            : chunks retrieved per step
    top_clusters     : clusters searched per query (MiniLM routing breadth)
    tau_values       : eviction percentile values to sweep for PageRank condition
    lambda_decay     : ForgetGate recency decay rate λ
    min_chunks_gate  : ForgetGate min registry size before firing
    seed             : random seed for task sampling
    tags             : optional SO tag filter (list of strings)
    minilm_routing   : if True, use MiniLM centroids for cluster routing
    """

    def __init__(
        self,
        n_tasks:         int              = 25,
        steps_per_task:  int              = 5,
        top_k:           int              = 5,
        top_clusters:    int              = 2,
        tau_values:      list[float]      = None,
        lambda_decay:    float            = 0.1,
        min_chunks_gate: int              = 5,
        seed:            int              = 42,
        tags:            Optional[list[str]] = None,
        minilm_routing:  bool             = True,
    ):
        self.n_tasks         = n_tasks
        self.steps_per_task  = steps_per_task
        self.top_k           = top_k
        self.top_clusters    = top_clusters
        self.tau_values      = tau_values if tau_values is not None else list(_TAU_DEFAULT)
        self.lambda_decay    = lambda_decay
        self.min_chunks_gate = min_chunks_gate
        self.seed            = seed
        self.tags            = tags

        log.info("Loading retriever ...")
        base_retriever = Retriever(top_clusters=top_clusters, top_k=top_k)
        if minilm_routing:
            from src.evaluation.embedding_ablation import MiniLMEmbedder, MiniLMRoutingRetriever
            log.info("Using MiniLM cluster routing.")
            self.retriever = MiniLMRoutingRetriever(base_retriever, MiniLMEmbedder(), top_clusters)
        else:
            self.retriever = base_retriever

        corpus_path = ROOT / "data" / "models" / "corpus_graph.pkl"
        log.info("Loading corpus graph for ForgetGate (%s) ...", corpus_path)
        with open(corpus_path, "rb") as f:
            self.corpus_artifact = pickle.load(f)
        log.info(
            "  Corpus graph: %d nodes, %d edges",
            self.corpus_artifact["graph"].number_of_nodes(),
            self.corpus_artifact["graph"].number_of_edges(),
        )

    # -----------------------------------------------------------------------

    def run(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Returns
        -------
        step_df  — one row per (task_id, condition, step)
        task_df  — one row per (task_id, condition) with aggregate stats
        """
        tasks = build_tasks(
            n_tasks        = self.n_tasks,
            steps_per_task = self.steps_per_task,
            seed           = self.seed,
            tags           = self.tags,
        )

        step_rows: list[dict] = []
        task_rows: list[dict] = []

        for task_id, task in enumerate(tasks):
            tag = task[0].get("tag", "")
            log.info(
                "Task %d/%d [%s]: %s ...",
                task_id + 1, len(tasks), tag, task[0]["title"][:55],
            )

            conditions = [
                run_no_eviction(task, self.retriever, self.top_k, self.top_clusters),
                run_evict_all(task, self.retriever, self.top_k, self.top_clusters),
                run_evict_older_than_1(task, self.retriever, self.top_k, self.top_clusters),
            ]

            for tau in self.tau_values:
                conditions.append(
                    run_pagerank_eviction(
                        task                = task,
                        retriever           = self.retriever,
                        corpus_artifact     = self.corpus_artifact,
                        top_k               = self.top_k,
                        top_clusters        = self.top_clusters,
                        eviction_percentile = tau,
                        lambda_decay        = self.lambda_decay,
                        min_chunks          = self.min_chunks_gate,
                    )
                )

            for result in conditions:
                cond = result["condition"]
                for sr in result["steps"]:
                    step_rows.append({
                        "task_id":             task_id,
                        "tag":                 tag,
                        "condition":           cond,
                        "eviction_percentile": result.get("eviction_percentile"),
                        **sr,
                    })
                task_rows.append({
                    "task_id":             task_id,
                    "tag":                 tag,
                    "condition":           cond,
                    "eviction_percentile": result.get("eviction_percentile"),
                    "total_tokens":        result["total_tokens"],
                    "final_step_tokens":   result["final_step_tokens"],
                    "final_n_chunks":      result["final_n_chunks"],
                    "total_dedup_saves":   result.get("total_dedup_saves", 0),
                })

        return pd.DataFrame(step_rows), pd.DataFrame(task_rows)

    # -----------------------------------------------------------------------

    def save(self, step_df: pd.DataFrame, task_df: pd.DataFrame) -> Path:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
        stem = RESULTS_DIR / f"kg_eviction_ablation_{ts}"

        step_df.to_csv(f"{stem}_steps.csv", index=False)
        task_df.to_csv(f"{stem}_tasks.csv", index=False)
        step_df.to_json(f"{stem}_steps.json", orient="records", indent=2)
        task_df.to_json(f"{stem}_tasks.json", orient="records", indent=2)

        log.info("Results saved → %s_steps.csv / _tasks.csv", stem)
        return Path(f"{stem}_tasks.csv")

    # -----------------------------------------------------------------------

    def print_summary(self, task_df: pd.DataFrame, step_df: pd.DataFrame) -> None:
        means        = task_df.groupby("condition")["total_tokens"].mean()
        final_chunks = task_df.groupby("condition")["final_n_chunks"].mean()
        baseline     = means.get("no_eviction", None)

        def _pct(val):
            if baseline and baseline > 0:
                return f"{(baseline - val) / baseline * 100:+.1f}%"
            return "—"

        ordered = (
            ["no_eviction", "evict_all", "evict_older_than_1"]
            + [f"pagerank_p{int(t)}" for t in sorted(self.tau_values)]
        )

        print("\n" + "=" * 72)
        print("KG EVICTION ABLATION SUMMARY")
        print("=" * 72)
        print(
            f"Tasks: {self.n_tasks}  |  Steps/task: {self.steps_per_task}  "
            f"|  top_k: {self.top_k}  |  top_clusters: {self.top_clusters}"
        )
        print(f"\n{'Condition':<26} {'Mean Total Tokens':>18} {'vs no_eviction':>16} {'Final chunks':>13}")
        print("-" * 75)
        for cond in ordered:
            if cond not in means:
                continue
            val = means[cond]
            fc  = final_chunks.get(cond, float("nan"))
            print(f"  {cond:<24} {val:>18.1f} {_pct(val):>16} {fc:>13.1f}")

        print(f"\n--- PageRank τ Pareto ---")
        print(f"  {'τ':>4}  {'tokens':>10}  {'reduction':>10}  {'mean evicted/task':>18}")
        print("  " + "-" * 48)
        for tau in sorted(self.tau_values):
            cond = f"pagerank_p{int(tau)}"
            if cond not in means:
                continue
            val = means[cond]
            cond_steps   = step_df[step_df["condition"] == cond]
            mean_evicted = cond_steps.groupby("task_id")["n_evicted"].sum().mean()
            print(
                f"  {int(tau):>4}  {val:>10.1f}  {_pct(val):>10}  {mean_evicted:>18.1f}"
            )

        print(f"\n--- Per-Step Mean Tokens ---")
        step_means = step_df.groupby(["condition", "step"])["step_tokens"].mean()
        for cond in ordered:
            for step in sorted(step_df["step"].unique()):
                val = step_means.get((cond, step))
                if val is not None:
                    print(f"  {cond:<26} step {step}: {val:.1f}")

        print("=" * 72 + "\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args():
    p = argparse.ArgumentParser(description="KG eviction ablation — PageRank vs age-based baselines")
    p.add_argument("--n-tasks",          type=int,   default=25,
                   help="Number of SO tasks (agent sessions)")
    p.add_argument("--steps",            type=int,   default=5,
                   help="Queries per task")
    p.add_argument("--top-k",            type=int,   default=5,
                   help="Chunks retrieved per step")
    p.add_argument("--top-clusters",     type=int,   default=2,
                   help="Clusters searched per query")
    p.add_argument("--tau-sweep",        type=float, nargs="+",
                   default=[10.0, 20.0, 30.0, 40.0],
                   help="Eviction percentile values to sweep (e.g. --tau-sweep 10 20 30 40)")
    p.add_argument("--lambda-decay",     type=float, default=0.1,
                   help="ForgetGate recency decay rate λ")
    p.add_argument("--min-chunks-gate",  type=int,   default=5,
                   help="ForgetGate min registry size before firing")
    p.add_argument("--seed",             type=int,   default=42)
    p.add_argument("--tags",             nargs="+",  default=None,
                   help="Restrict to specific SO tags")
    p.add_argument("--no-minilm-routing", action="store_true",
                   help="Use CodeBERT routing instead of MiniLM (not recommended)")
    p.add_argument("--log-level",        default="INFO")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    logging.basicConfig(
        level  = getattr(logging, args.log_level.upper(), logging.INFO),
        format = "%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    ablation = KGEvictionAblation(
        n_tasks         = args.n_tasks,
        steps_per_task  = args.steps,
        top_k           = args.top_k,
        top_clusters    = args.top_clusters,
        tau_values      = args.tau_sweep,
        lambda_decay    = args.lambda_decay,
        min_chunks_gate = args.min_chunks_gate,
        seed            = args.seed,
        tags            = args.tags,
        minilm_routing  = not args.no_minilm_routing,
    )

    step_df, task_df = ablation.run()
    ablation.print_summary(task_df, step_df)
    ablation.save(step_df, task_df)
