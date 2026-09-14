"""
multisession_eval.py — Multi-session episodic memory evaluation.

Compares two conditions on a fixed 15-query sequence (3 sessions × 5 steps)
drawn from the FastAPI / Pydantic / Supabase domain:

  no_kg    — MiniLMRoutingRetriever + ContextRegistry(evict_older_than=1).
             Each session is independent: no cross-session memory.

  with_kg  — SessionRetriever + EpisodicStore + ForgetGate.
             The episodic store persists across all three sessions.
             Cross-session chunk history informs document scoring (β·context_frac,
             γ·history_score) and enables push-back detection for similar queries.

Both conditions use a fresh ContextRegistry per session.

Metrics
-------
  - step_tokens / cumulative_tokens (across all 3 sessions)
  - n_newly_loaded, n_skipped_dup, n_evicted per step
  - cross_session_reused  : chunks loaded this step that appeared in a prior session
  - push_back_detected    : a similar prior query was found in the episodic store
  - judge_score           : LLM relevance score 0.0–1.0 (optional, --judge flag)
  - ab_winner             : A/B comparison per step  (optional, --judge flag)

Expected finding
----------------
In Sessions 2 and 3, the with_kg condition loads fewer novel chunks and uses
fewer tokens while maintaining comparable retrieval precision — episodic memory
compounds in value over time.

Usage
-----
    # Token metrics only (no LLM judge):
    python -m src.evaluation.multisession_eval

    # With LLM judge (requires OPENAI_API_KEY):
    OPENAI_API_KEY=... python -m src.evaluation.multisession_eval --judge

    # Adjust push-back threshold (default 0.88 may be strict; try 0.75):
    python -m src.evaluation.multisession_eval --push-back-threshold 0.75
"""

import argparse
import json
import logging
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Optional
import pickle

import numpy as np
import pandas as pd
import tiktoken

from src.agent.context_manager import ContextRegistry
from src.graph.episodic_store import EpisodicStore
from src.graph.forget_gate import ForgetGate
from src.graph.session_retriever import SessionRetriever
from src.retrieval.retriever import Retriever

log = logging.getLogger(__name__)

ROOT        = Path(__file__).parents[2]
RESULTS_DIR = ROOT / "data" / "results"

_ENC = tiktoken.get_encoding("cl100k_base")

# ---------------------------------------------------------------------------
# Fixed 15-query evaluation sequence (3 sessions × 5 steps)
# ---------------------------------------------------------------------------

SESSIONS: list[dict] = [
    {
        "name":  "session_1_setup",
        "label": "Session 1 — Project setup",
        "queries": [
            "How do I structure a FastAPI app with routers?",
            "How do I define Pydantic request/response models?",
            "How do I connect FastAPI to Supabase?",
            "How do I add JWT authentication to FastAPI?",
            "How do I handle CORS in FastAPI?",
        ],
    },
    {
        "name":  "session_2_features",
        "label": "Session 2 — Feature extension",
        "queries": [
            "How do I add background tasks in FastAPI?",
            "How do I validate nested Pydantic models?",
            "How do I use Supabase Row Level Security?",
            "How do I refresh JWT tokens in FastAPI?",      # similar to S1 step 4
            "How do I write pytest tests for FastAPI endpoints?",
        ],
    },
    {
        "name":  "session_3_debug",
        "label": "Session 3 — Debug and optimise",
        "queries": [
            "FastAPI returning 422 errors — how do I debug Pydantic validation?",
            "How do I profile slow Pydantic models?",
            "Supabase queries timing out — what should I check?",
            "How do I add rate limiting middleware to FastAPI?",
            "How do I structure FastAPI dependency injection for auth?",  # similar to S1 step 4
        ],
    },
]

# ---------------------------------------------------------------------------
# Token counting
# ---------------------------------------------------------------------------

def _count_tokens(text: str) -> int:
    return len(_ENC.encode(text))


def _format_chunks_for_judge(chunks: list[dict], max_chars: int = 6000) -> str:
    parts = []
    total = 0
    for c in chunks:
        header = (
            f"[chunk_id={c.get('chunk_id','?')} | "
            f"type={c.get('source_type','?')} | "
            f"repo={c.get('repo','?')} | "
            f"score={c.get('score', 0.0):.3f}]"
        )
        body = c.get("text", "")
        block = f"{header}\n{body}"
        if total + len(block) > max_chars:
            break
        parts.append(block)
        total += len(block)
    return "\n\n".join(parts) if parts else "(no chunks retrieved)"


# ---------------------------------------------------------------------------
# Condition A: No KG
# ---------------------------------------------------------------------------

def _run_no_kg(
    retriever,
    top_k: int,
    top_clusters: int,
) -> list[dict]:
    """
    Run all 3 sessions under the no_kg condition.

    Each session is fully independent: fresh ContextRegistry, no episodic store.
    Eviction strategy: evict_older_than(1) — keeps only previous + current step chunks.
    """
    step_rows: list[dict] = []
    cumulative_tokens = 0
    session_chunk_sets: list[set[str]] = []   # chunk_ids retrieved per session

    for sess_idx, session in enumerate(SESSIONS):
        sess_num     = sess_idx + 1
        global_offset = sess_idx * 5

        registry         = ContextRegistry(token_budget=8192)
        session_retrieved = set()

        # Prior sessions' chunks (for cross-session reuse tracking)
        prior_chunks: set[str] = set()
        for s in session_chunk_sets:
            prior_chunks |= s

        for step_idx, query in enumerate(session["queries"]):
            step_num    = step_idx + 1
            global_step = global_offset + step_num
            registry.advance_step()

            t0      = time.perf_counter()
            chunks  = retriever.retrieve(query, top_k=top_k, top_clusters=top_clusters)
            retrieve_ms = (time.perf_counter() - t0) * 1000

            ids_before  = set(registry.loaded_ids)
            load_stats  = registry.load_chunks(chunks)
            n_evicted   = registry.evict_older_than(1)

            retrieved_ids = {c["chunk_id"] for c in chunks}
            session_retrieved |= retrieved_ids
            n_cross = len(retrieved_ids & prior_chunks)

            prompt_tokens      = _count_tokens(registry.build_prompt(query))
            cumulative_tokens += prompt_tokens

            step_rows.append({
                "session_num":           sess_num,
                "session_name":          session["name"],
                "step_num":              step_num,
                "global_step":           global_step,
                "query":                 query,
                "condition":             "no_kg",
                "chunks":                chunks,        # kept for LLM judge; stripped before CSV
                "n_chunks_in_window":    registry.size,
                "n_retrieved":           len(chunks),
                "n_newly_loaded":        load_stats["loaded"],
                "n_skipped_dup":         load_stats["skipped_duplicate"],
                "n_evicted":             n_evicted,
                "n_cross_session_reused": n_cross,
                "push_back_detected":    False,
                "step_tokens":           prompt_tokens,
                "cumulative_tokens":     cumulative_tokens,
                "retrieve_ms":           round(retrieve_ms, 1),
                "judge_score":           None,
                "ab_winner":             None,
            })

        session_chunk_sets.append(session_retrieved)

    return step_rows


# ---------------------------------------------------------------------------
# Condition B: With KG
# ---------------------------------------------------------------------------

def _run_with_kg(
    base_retriever,
    minilm,
    corpus_artifact: dict,
    top_clusters:       int,
    gate_percentile:    float,
    lambda_decay:       float,
    token_threshold:    int,
    push_back_threshold: float,
) -> list[dict]:
    """
    Run all 3 sessions under the with_kg condition.

    A single EpisodicStore persists across sessions (ephemeral, not saved to disk).
    Each session gets a fresh ContextRegistry but shares the episodic graph.
    A ForgetGate is wired into the SessionRetriever to bound within-session context.
    """
    step_rows: list[dict] = []
    cumulative_tokens = 0

    tmp_path = Path(tempfile.mktemp(suffix=".pkl"))
    store    = EpisodicStore(path=tmp_path, embed_queries=False)
    gate     = ForgetGate(
        corpus_artifact    = corpus_artifact,
        eviction_percentile= gate_percentile,
        lambda_decay       = lambda_decay,
    )

    prior_session_ids:    list[str]  = []
    prior_session_chunks: set[str]   = set()  # chunk_ids retrieved in any prior session

    for sess_idx, session in enumerate(SESSIONS):
        sess_num      = sess_idx + 1
        global_offset = sess_idx * 5

        registry = ContextRegistry(token_budget=8192)
        sid      = store.start_session(project_id="multisession_eval")

        sr = SessionRetriever(
            base_retriever       = base_retriever,
            minilm               = minilm,
            store                = store,
            corpus_artifact      = corpus_artifact,
            registry             = registry,
            session_id           = sid,
            top_clusters         = top_clusters,
            forget_gate_fn       = gate,
            token_threshold      = token_threshold,
            push_back_threshold  = push_back_threshold,
        )

        for step_idx, query in enumerate(session["queries"]):
            step_num    = step_idx + 1
            global_step = global_offset + step_num
            registry.advance_step()

            n_before    = registry.size
            t0          = time.perf_counter()
            chunks, qid = sr.retrieve(query, step_number=global_step)
            retrieve_ms = (time.perf_counter() - t0) * 1000
            n_after     = registry.size

            # Evictions = what the gate removed (registry may have shrunk)
            n_newly_loaded = len(chunks)
            n_evicted      = max(0, n_before + n_newly_loaded - n_after)

            # Cross-session reuse: newly loaded chunks that were seen in prior sessions
            newly_loaded_ids = {c["chunk_id"] for c in chunks}
            n_cross = len(newly_loaded_ids & prior_session_chunks)

            # Push-back: did the episodic store contain a similar prior query?
            push_back = False
            query_emb = np.array(
                store._G.nodes.get(qid, {}).get("embedding", []),
                dtype=np.float32,
            )
            if query_emb.size > 0:
                similar = store.find_similar_queries(
                    query_emb,
                    threshold=push_back_threshold,
                )
                similar = [h for h in similar if h["query_id"] != qid]
                push_back = len(similar) > 0

            prompt_tokens      = _count_tokens(registry.build_prompt(query))
            cumulative_tokens += prompt_tokens

            step_rows.append({
                "session_num":            sess_num,
                "session_name":           session["name"],
                "step_num":               step_num,
                "global_step":            global_step,
                "query":                  query,
                "condition":              "with_kg",
                "chunks":                 chunks,
                "n_chunks_in_window":     n_after,
                "n_retrieved":            len(chunks),
                "n_newly_loaded":         n_newly_loaded,
                "n_skipped_dup":          0,    # SR only returns newly loaded chunks
                "n_evicted":              n_evicted,
                "n_cross_session_reused": n_cross,
                "push_back_detected":     push_back,
                "step_tokens":            prompt_tokens,
                "cumulative_tokens":      cumulative_tokens,
                "retrieve_ms":            round(retrieve_ms, 1),
                "judge_score":            None,
                "ab_winner":              None,
            })

        store.end_session(sid)

        # Accumulate this session's retrieved chunks for future sessions
        prior_session_ids.append(sid)
        for q in store._session_query_ids(sid):
            for _, target, data in store._G.out_edges(q, data=True):
                if data.get("relation") == "RETRIEVED":
                    prior_session_chunks.add(target)

    if tmp_path.exists():
        tmp_path.unlink()

    return step_rows


# ---------------------------------------------------------------------------
# LLM judge
# ---------------------------------------------------------------------------

def _judge_steps(
    no_kg_rows:   list[dict],
    with_kg_rows: list[dict],
    client,
    model: str = "gpt-4o-mini",
) -> dict[int, dict]:
    """
    Run LLM judge on every step using the OpenAI client.

    Returns {global_step: {"no_kg_score": float, "with_kg_score": float, "ab_winner": str}}
    """
    no_kg_by_step   = {r["global_step"]: r for r in no_kg_rows}
    with_kg_by_step = {r["global_step"]: r for r in with_kg_rows}
    results: dict[int, dict] = {}

    for gstep in sorted(no_kg_by_step.keys()):
        row_a  = no_kg_by_step[gstep]
        row_b  = with_kg_by_step.get(gstep)
        query  = row_a["query"]

        chunks_a = _format_chunks_for_judge(row_a["chunks"])
        chunks_b = _format_chunks_for_judge(row_b["chunks"]) if row_b else "(unavailable)"

        # ── Direct relevance scoring ───────────────────────────────────────
        def _score(chunks_text: str) -> float:
            prompt = (
                "You are evaluating retrieval quality for a code assistance RAG system.\n\n"
                f"Query: {query}\n\n"
                "Retrieved chunks:\n"
                f"{chunks_text}\n\n"
                "Score how well these chunks answer the query.\n"
                'Return ONLY valid JSON: {"score": <float 0.0-1.0>, "reasoning": "<1 sentence>"}\n'
                "Where 1.0 = directly and completely answers the query, 0.0 = irrelevant."
            )
            try:
                resp = client.chat.completions.create(
                    model=model,
                    max_tokens=64,
                    response_format={"type": "json_object"},
                    messages=[{"role": "user", "content": prompt}],
                )
                raw = resp.choices[0].message.content.strip()
                return float(json.loads(raw)["score"])
            except Exception as e:
                log.warning("Judge scoring failed for step %d: %s", gstep, e)
                return float("nan")

        score_a = _score(chunks_a)
        score_b = _score(chunks_b) if row_b else float("nan")

        # ── A/B comparison ────────────────────────────────────────────────
        ab_winner = "N/A"
        if row_b:
            ab_prompt = (
                "You are comparing two sets of retrieved chunks for a code assistance query.\n\n"
                f"Query: {query}\n\n"
                "=== Set A (no_kg) ===\n"
                f"{chunks_a}\n\n"
                "=== Set B (with_kg) ===\n"
                f"{chunks_b}\n\n"
                "Which set better answers the query? Consider relevance, completeness, "
                "and non-redundancy.\n"
                'Return ONLY valid JSON: {"winner": "A" or "B" or "TIE", "reasoning": "<1 sentence>"}'
            )
            try:
                resp = client.chat.completions.create(
                    model=model,
                    max_tokens=64,
                    response_format={"type": "json_object"},
                    messages=[{"role": "user", "content": ab_prompt}],
                )
                raw = resp.choices[0].message.content.strip()
                ab_winner = json.loads(raw)["winner"].upper()
            except Exception as e:
                log.warning("A/B judge failed for step %d: %s", gstep, e)

        results[gstep] = {
            "no_kg_score":   score_a,
            "with_kg_score": score_b,
            "ab_winner":     ab_winner,
        }
        log.info(
            "Step %2d: no_kg=%.2f  with_kg=%.2f  winner=%s  [%s]",
            gstep, score_a, score_b, ab_winner, query[:60],
        )

    return results


# ---------------------------------------------------------------------------
# Harness
# ---------------------------------------------------------------------------

class MultisessionEval:
    """
    Runs the multi-session evaluation across two conditions.

    Parameters
    ----------
    top_k                : chunks retrieved per step (no_kg condition)
    top_clusters         : clusters searched per query
    gate_percentile      : ForgetGate eviction percentile (with_kg)
    lambda_decay         : ForgetGate recency decay rate
    token_threshold      : ForgetGate token threshold trigger
    push_back_threshold  : cosine similarity for push-back detection (0.88 default,
                           consider 0.75–0.80 based on real data)
    judge_model          : Anthropic model for LLM judge
    """

    def __init__(
        self,
        top_k:               int   = 5,
        top_clusters:        int   = 2,
        gate_percentile:     float = 20.0,
        lambda_decay:        float = 0.1,
        token_threshold:     int   = 3000,
        push_back_threshold: float = 0.88,
        judge_model:         str   = "gpt-4o-mini",
    ):
        self.top_k               = top_k
        self.top_clusters        = top_clusters
        self.gate_percentile     = gate_percentile
        self.lambda_decay        = lambda_decay
        self.token_threshold     = token_threshold
        self.push_back_threshold = push_back_threshold
        self.judge_model         = judge_model

        log.info("Loading retriever ...")
        self.base_retriever = Retriever(top_clusters=top_clusters, top_k=top_k)

        log.info("Loading MiniLM embedder ...")
        from src.evaluation.embedding_ablation import MiniLMEmbedder, MiniLMRoutingRetriever
        self.minilm   = MiniLMEmbedder()
        self.retriever = MiniLMRoutingRetriever(self.base_retriever, self.minilm, top_clusters)

        corpus_path = ROOT / "data" / "models" / "corpus_graph.pkl"
        log.info("Loading corpus graph (%s) ...", corpus_path)
        with open(corpus_path, "rb") as f:
            self.corpus_artifact = pickle.load(f)
        log.info(
            "  Corpus graph: %d nodes, %d edges",
            self.corpus_artifact["graph"].number_of_nodes(),
            self.corpus_artifact["graph"].number_of_edges(),
        )

    # -----------------------------------------------------------------------

    def run(self, judge: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Run both conditions and return (step_df, session_df).

        step_df    — one row per (condition, global_step)
        session_df — one row per (condition, session_num) with aggregate stats
        """
        log.info("=== Running no_kg condition ===")
        no_kg_rows = _run_no_kg(
            retriever    = self.retriever,
            top_k        = self.top_k,
            top_clusters = self.top_clusters,
        )

        log.info("=== Running with_kg condition ===")
        with_kg_rows = _run_with_kg(
            base_retriever       = self.base_retriever,
            minilm               = self.minilm,
            corpus_artifact      = self.corpus_artifact,
            top_clusters         = self.top_clusters,
            gate_percentile      = self.gate_percentile,
            lambda_decay         = self.lambda_decay,
            token_threshold      = self.token_threshold,
            push_back_threshold  = self.push_back_threshold,
        )

        if judge:
            log.info("=== Running LLM judge ===")
            import openai
            client        = openai.OpenAI()
            judge_results = _judge_steps(no_kg_rows, with_kg_rows, client, self.judge_model)
            for row in no_kg_rows:
                jr = judge_results.get(row["global_step"], {})
                row["judge_score"] = jr.get("no_kg_score")
                row["ab_winner"]   = jr.get("ab_winner")
            for row in with_kg_rows:
                jr = judge_results.get(row["global_step"], {})
                row["judge_score"] = jr.get("with_kg_score")
                row["ab_winner"]   = jr.get("ab_winner")

        # Build step_df — strip "chunks" column (not CSV-serialisable)
        all_rows = []
        for row in no_kg_rows + with_kg_rows:
            all_rows.append({k: v for k, v in row.items() if k != "chunks"})
        step_df = pd.DataFrame(all_rows).sort_values(["condition", "global_step"])

        # Build session_df — aggregate per (condition, session_num)
        session_rows = []
        for cond, cond_rows in [("no_kg", no_kg_rows), ("with_kg", with_kg_rows)]:
            by_session: dict[int, list[dict]] = {}
            for r in cond_rows:
                by_session.setdefault(r["session_num"], []).append(r)

            for sess_num, rows in sorted(by_session.items()):
                last_row = rows[-1]
                session_rows.append({
                    "condition":              cond,
                    "session_num":            sess_num,
                    "session_name":           rows[0]["session_name"],
                    "total_tokens":           sum(r["step_tokens"] for r in rows),
                    "cumulative_tokens_end":  last_row["cumulative_tokens"],
                    "mean_step_tokens":       round(sum(r["step_tokens"] for r in rows) / len(rows), 1),
                    "total_newly_loaded":     sum(r["n_newly_loaded"] for r in rows),
                    "total_skipped_dup":      sum(r["n_skipped_dup"] for r in rows),
                    "total_evicted":          sum(r["n_evicted"] for r in rows),
                    "total_cross_reused":     sum(r["n_cross_session_reused"] for r in rows),
                    "push_back_count":        sum(1 for r in rows if r["push_back_detected"]),
                    "mean_judge_score":       _safe_mean([r["judge_score"] for r in rows]),
                    "ab_wins":               sum(1 for r in rows if r.get("ab_winner") == ("A" if cond == "no_kg" else "B")),
                })
        session_df = pd.DataFrame(session_rows)

        return step_df, session_df

    # -----------------------------------------------------------------------

    def save(self, step_df: pd.DataFrame, session_df: pd.DataFrame) -> Path:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
        stem = RESULTS_DIR / f"multisession_eval_{ts}"

        step_df.to_csv(f"{stem}_steps.csv", index=False)
        session_df.to_csv(f"{stem}_sessions.csv", index=False)
        step_df.to_json(f"{stem}_steps.json", orient="records", indent=2)
        session_df.to_json(f"{stem}_sessions.json", orient="records", indent=2)

        log.info("Results saved → %s_steps.csv / _sessions.csv", stem)
        return Path(f"{stem}_sessions.csv")

    # -----------------------------------------------------------------------

    def print_summary(self, step_df: pd.DataFrame, session_df: pd.DataFrame) -> None:
        no_kg  = session_df[session_df["condition"] == "no_kg"]
        with_kg = session_df[session_df["condition"] == "with_kg"]

        print("\n" + "=" * 72)
        print("MULTI-SESSION EVALUATION SUMMARY")
        print("=" * 72)
        print(
            f"top_k={self.top_k}  |  top_clusters={self.top_clusters}  |  "
            f"push_back_threshold={self.push_back_threshold}  |  "
            f"gate_percentile={self.gate_percentile}"
        )

        print(f"\n{'':30} {'no_kg':>12} {'with_kg':>12} {'delta':>10}")
        print("-" * 68)

        def _row(label, col, fmt=".1f"):
            a = no_kg[col].sum() if col in no_kg else float("nan")
            b = with_kg[col].sum() if col in with_kg else float("nan")
            try:
                d = b - a
                d_str = f"{d:+.1f}"
            except Exception:
                d_str = "—"
            print(f"  {label:<28} {a:>12{fmt}} {b:>12{fmt}} {d_str:>10}")

        _row("Total tokens (all sessions)", "total_tokens")
        _row("Total newly loaded chunks",   "total_newly_loaded", ".0f")
        _row("Total skipped duplicates",    "total_skipped_dup", ".0f")
        _row("Total evicted chunks",        "total_evicted", ".0f")
        _row("Cross-session reused chunks", "total_cross_reused", ".0f")
        _row("Push-back detections",        "push_back_count", ".0f")

        print(f"\n--- Per-session token breakdown ---")
        print(f"  {'Session':<32} {'no_kg':>10} {'with_kg':>10} {'delta':>8}")
        print("  " + "-" * 62)
        for sess_num in sorted(no_kg["session_num"].unique()):
            sess_name = SESSIONS[sess_num - 1]["label"]
            a = no_kg[no_kg["session_num"] == sess_num]["total_tokens"].values[0]
            b_row = with_kg[with_kg["session_num"] == sess_num]
            b = b_row["total_tokens"].values[0] if len(b_row) else float("nan")
            pct = f"{(b - a)/a*100:+.1f}%" if a else "—"
            print(f"  {sess_name:<32} {a:>10.0f} {b:>10.0f} {pct:>8}")

        print(f"\n--- Per-step token breakdown ---")
        print(f"  {'Step':>6} {'Session':>10} {'Query':<45} {'no_kg':>7} {'with_kg':>8} {'pb':>4}")
        print("  " + "-" * 84)
        no_kg_steps  = step_df[step_df["condition"] == "no_kg"].sort_values("global_step")
        with_kg_steps = step_df[step_df["condition"] == "with_kg"].sort_values("global_step")
        for _, row_a in no_kg_steps.iterrows():
            gs = row_a["global_step"]
            row_b = with_kg_steps[with_kg_steps["global_step"] == gs]
            tok_b = row_b["step_tokens"].values[0] if len(row_b) else float("nan")
            pb    = "✓" if (len(row_b) and row_b["push_back_detected"].values[0]) else " "
            q     = row_a["query"][:43]
            print(
                f"  {gs:>6}  S{row_a['session_num']}:step{row_a['step_num']}  "
                f"{q:<45} {row_a['step_tokens']:>7.0f} {tok_b:>8.0f} {pb:>4}"
            )

        if session_df["mean_judge_score"].notna().any():
            print(f"\n--- LLM Judge Scores (mean per session) ---")
            print(f"  {'Session':<32} {'no_kg':>8} {'with_kg':>9} {'AB wins (kg)':>14}")
            print("  " + "-" * 66)
            for sess_num in sorted(no_kg["session_num"].unique()):
                sess_name = SESSIONS[sess_num - 1]["label"]
                a_score = no_kg[no_kg["session_num"] == sess_num]["mean_judge_score"].values[0]
                b_row   = with_kg[with_kg["session_num"] == sess_num]
                b_score = b_row["mean_judge_score"].values[0] if len(b_row) else float("nan")
                b_wins  = b_row["ab_wins"].values[0] if len(b_row) else 0
                a_s = f"{a_score:.2f}" if a_score == a_score else "—"
                b_s = f"{b_score:.2f}" if b_score == b_score else "—"
                print(f"  {sess_name:<32} {a_s:>8} {b_s:>9} {b_wins:>14}")

        print("=" * 72 + "\n")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_mean(values: list) -> Optional[float]:
    vals = [v for v in values if v is not None and v == v]  # drop None and NaN
    return round(sum(vals) / len(vals), 3) if vals else None


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args():
    p = argparse.ArgumentParser(description="Multi-session episodic memory evaluation")
    p.add_argument("--top-k",               type=int,   default=5)
    p.add_argument("--top-clusters",        type=int,   default=2)
    p.add_argument("--gate-percentile",     type=float, default=20.0,
                   help="ForgetGate eviction percentile for with_kg condition")
    p.add_argument("--lambda-decay",        type=float, default=0.1)
    p.add_argument("--token-threshold",     type=int,   default=3000,
                   help="ForgetGate token threshold trigger")
    p.add_argument("--push-back-threshold", type=float, default=0.88,
                   help="Cosine similarity for push-back detection (try 0.75 if no hits)")
    p.add_argument("--judge",               action="store_true",
                   help="Run LLM judge (requires OPENAI_API_KEY)")
    p.add_argument("--judge-model",         default="gpt-4o-mini")
    p.add_argument("--log-level",           default="INFO")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    logging.basicConfig(
        level  = getattr(logging, args.log_level.upper(), logging.INFO),
        format = "%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    eval_ = MultisessionEval(
        top_k               = args.top_k,
        top_clusters        = args.top_clusters,
        gate_percentile     = args.gate_percentile,
        lambda_decay        = args.lambda_decay,
        token_threshold     = args.token_threshold,
        push_back_threshold = args.push_back_threshold,
        judge_model         = args.judge_model,
    )

    step_df, session_df = eval_.run(judge=args.judge)
    eval_.print_summary(step_df, session_df)
    eval_.save(step_df, session_df)
