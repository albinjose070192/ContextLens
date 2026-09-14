"""
Evaluation Harness — Flat RAG vs. Selective RAG.

Runs the same query set through two retrieval conditions and records:
  - Input token count (primary metric)
  - Retrieval Precision@k  (LLM-as-judge, optional)
  - Task completion score  (LLM-as-judge, optional)
  - Latency per step
  - Estimated cost

Usage (dry run — no LLM, token counts only):
    python -m src.evaluation.harness --n 100 --dry-run

Usage (full eval with LLM judge):
    ANTHROPIC_API_KEY=... python -m src.evaluation.harness --n 100 --top-k 5

Results are saved to data/results/eval_<timestamp>.csv and .json.
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
import pandas as pd
import tiktoken

from src.agent.context_manager import ContextRegistry
from src.retrieval.retriever import Retriever

log = logging.getLogger(__name__)

ROOT         = Path(__file__).parents[2]
SO_PATH      = ROOT / "data" / "raw" / "stackoverflow" / "stackoverflow_all_raw.json"
RESULTS_DIR  = ROOT / "data" / "results"

# tiktoken encoder for token counting (same as context_manager.py)
_ENC = tiktoken.get_encoding("cl100k_base")

# Anthropic pricing (per 1M tokens, as of mid-2025)
# Using claude-haiku-4-5 for main agent (cheap), same model for judge
_MODEL_AGENT  = "claude-haiku-4-5-20251001"
_MODEL_JUDGE  = "claude-haiku-4-5-20251001"
_INPUT_COST_PER_1M  = 0.80   # USD (Anthropic Haiku)
_OUTPUT_COST_PER_1M = 4.00   # USD

_OAI_MODEL_AGENT = "gpt-4o-mini"
_OAI_MODEL_JUDGE = "gpt-4o-mini"
_OAI_INPUT_COST_PER_1M  = 0.15   # USD
_OAI_OUTPUT_COST_PER_1M = 0.60   # USD


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------

def _build_flat_prompt(query: str, chunks: list[dict]) -> str:
    """Flat RAG prompt: all retrieved chunks concatenated, no registry."""
    parts = []
    for c in chunks:
        header = (
            f"[chunk_id={c.get('chunk_id','?')} | "
            f"type={c.get('source_type','?')} | "
            f"repo={c.get('repo','?')} | "
            f"score={c.get('score',0):.4f}]"
        )
        parts.append(f"{header}\n{c.get('text','')}")

    context = "\n\n".join(parts)
    return (
        "You are a helpful software engineering assistant. "
        "Answer the query using only the retrieved context provided below. "
        "Be concise and precise.\n\n"
        f"## Retrieved Context\n\n{context}\n\n"
        f"## Query\n\n{query}"
    )


def _build_judge_prompt(query: str, ground_truth: str, response: str) -> str:
    return (
        "You are an expert evaluator for a retrieval-augmented generation system.\n"
        "Given the question, the ground-truth answer, and the model's response, "
        "output a JSON object with exactly two keys:\n"
        '  "completion_score": float 0.0–1.0  (did the response correctly answer the question?)\n'
        '  "precision_score":  float 0.0–1.0  (was the response grounded in relevant information, '
        "with no hallucination?)\n"
        "Output ONLY valid JSON. No explanation.\n\n"
        f"Question: {query}\n\n"
        f"Ground truth: {ground_truth}\n\n"
        f"Model response: {response}"
    )


def _build_retrieval_precision_prompt(query: str, ground_truth: str, chunks: list[dict]) -> str:
    snippets = "\n---\n".join(c.get("text", "")[:300] for c in chunks)
    return (
        "You are evaluating retrieval quality for a RAG system.\n"
        "Given the question and the ground-truth answer, assess whether the retrieved "
        "passages contain enough information to answer the question.\n"
        'Output a JSON object with one key: "precision_score": float 0.0–1.0\n'
        "Output ONLY valid JSON.\n\n"
        f"Question: {query}\n\n"
        f"Ground truth (first 400 chars): {ground_truth[:400]}\n\n"
        f"Retrieved passages:\n{snippets}"
    )


# ---------------------------------------------------------------------------
# LLM caller
# ---------------------------------------------------------------------------

class LLMClient:
    """Thin wrapper supporting Anthropic or OpenAI with usage tracking."""

    def __init__(self, api_key: Optional[str] = None, use_openai: bool = False):
        self._use_openai = use_openai
        if use_openai:
            key = api_key or os.environ.get("OPENAI_API_KEY")
            if not key:
                raise ValueError("OPENAI_API_KEY not set.")
            self._client = openai_lib.OpenAI(api_key=key)
            self._model_agent = _OAI_MODEL_AGENT
            self._model_judge = _OAI_MODEL_JUDGE
            self._in_cost  = _OAI_INPUT_COST_PER_1M
            self._out_cost = _OAI_OUTPUT_COST_PER_1M
        else:
            key = api_key or os.environ.get("ANTHROPIC_API_KEY")
            if not key:
                raise ValueError(
                    "ANTHROPIC_API_KEY not set. "
                    "Run with --dry-run to skip LLM calls, or use --openai."
                )
            self._client = anthropic.Anthropic(api_key=key)
            self._model_agent = _MODEL_AGENT
            self._model_judge = _MODEL_JUDGE
            self._in_cost  = _INPUT_COST_PER_1M
            self._out_cost = _OUTPUT_COST_PER_1M
        self.total_input_tokens  = 0
        self.total_output_tokens = 0

    def call(self, prompt: str, model: str = None, max_tokens: int = 512) -> tuple[str, int, int]:
        """Returns (response_text, input_tokens, output_tokens). Retries on rate limits."""
        model = model or self._model_agent
        for attempt in range(5):
            try:
                if self._use_openai:
                    resp = self._client.chat.completions.create(
                        model=model,
                        max_tokens=max_tokens,
                        messages=[{"role": "user", "content": prompt}],
                    )
                    in_tok  = resp.usage.prompt_tokens
                    out_tok = resp.usage.completion_tokens
                    text    = resp.choices[0].message.content
                else:
                    msg = self._client.messages.create(
                        model=model,
                        max_tokens=max_tokens,
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
                log.warning(f"API error {status} — waiting {wait}s (attempt {attempt+1}/5)")
                time.sleep(wait)
        else:
            raise RuntimeError("Rate limit retries exhausted after 5 attempts.")
        self.total_input_tokens  += in_tok
        self.total_output_tokens += out_tok
        return text, in_tok, out_tok

    def judge(self, prompt: str) -> dict:
        """Call the judge model, parse JSON response."""
        text, _, _ = self.call(prompt, model=self._model_judge, max_tokens=128)
        try:
            clean = text.strip().strip("```json").strip("```").strip()
            return json.loads(clean)
        except json.JSONDecodeError:
            log.warning(f"Judge returned non-JSON: {text!r}")
            return {"completion_score": None, "precision_score": None}

    @property
    def total_cost_usd(self) -> float:
        return (
            self.total_input_tokens  / 1_000_000 * self._in_cost
            + self.total_output_tokens / 1_000_000 * self._out_cost
        )


# ---------------------------------------------------------------------------
# Single-step evaluation
# ---------------------------------------------------------------------------

def _count_tokens(text: str) -> int:
    return len(_ENC.encode(text))


def run_flat(
    query: str,
    retriever: Retriever,
    top_k: int,
    llm: Optional[LLMClient],
    ground_truth: str,
) -> dict:
    t0 = time.perf_counter()
    chunks = retriever.retrieve_flat(query, top_k=top_k)
    retrieve_ms = (time.perf_counter() - t0) * 1000

    prompt = _build_flat_prompt(query, chunks)
    prompt_tokens = _count_tokens(prompt)

    result = {
        "condition": "flat",
        "n_chunks": len(chunks),
        "prompt_tokens": prompt_tokens,
        "retrieve_ms": round(retrieve_ms, 1),
        "llm_input_tokens": None,
        "llm_output_tokens": None,
        "llm_ms": None,
        "completion_score": None,
        "precision_score": None,
        "retrieval_precision_score": None,
        "cost_usd": None,
    }

    if llm:
        t1 = time.perf_counter()
        response, in_tok, out_tok = llm.call(prompt)
        result["llm_ms"] = round((time.perf_counter() - t1) * 1000, 1)
        result["llm_input_tokens"]  = in_tok
        result["llm_output_tokens"] = out_tok
        result["cost_usd"] = round(
            in_tok / 1_000_000 * _INPUT_COST_PER_1M
            + out_tok / 1_000_000 * _OUTPUT_COST_PER_1M,
            6,
        )

        judge_result = llm.judge(_build_judge_prompt(query, ground_truth, response))
        result["completion_score"] = judge_result.get("completion_score")
        result["precision_score"]  = judge_result.get("precision_score")

        ret_prec = llm.judge(_build_retrieval_precision_prompt(query, ground_truth, chunks))
        result["retrieval_precision_score"] = ret_prec.get("precision_score")

    return result


def run_selective(
    query: str,
    retriever: Retriever,
    registry: ContextRegistry,
    top_k: int,
    top_clusters: int,
    llm: Optional[LLMClient],
    ground_truth: str,
) -> dict:
    registry.advance_step()

    t0 = time.perf_counter()
    chunks = retriever.retrieve(query, top_k=top_k, top_clusters=top_clusters)
    retrieve_ms = (time.perf_counter() - t0) * 1000

    load_stats = registry.load_chunks(chunks)
    prompt = registry.build_prompt(query)
    prompt_tokens = _count_tokens(prompt)

    result = {
        "condition": "selective",
        "n_chunks": len(chunks),
        "n_loaded": load_stats["loaded"],
        "n_skipped_dup": load_stats["skipped_duplicate"],
        "prompt_tokens": prompt_tokens,
        "retrieve_ms": round(retrieve_ms, 1),
        "llm_input_tokens": None,
        "llm_output_tokens": None,
        "llm_ms": None,
        "completion_score": None,
        "precision_score": None,
        "retrieval_precision_score": None,
        "cost_usd": None,
    }

    if llm:
        t1 = time.perf_counter()
        response, in_tok, out_tok = llm.call(prompt)
        result["llm_ms"] = round((time.perf_counter() - t1) * 1000, 1)
        result["llm_input_tokens"]  = in_tok
        result["llm_output_tokens"] = out_tok
        result["cost_usd"] = round(
            in_tok / 1_000_000 * _INPUT_COST_PER_1M
            + out_tok / 1_000_000 * _OUTPUT_COST_PER_1M,
            6,
        )

        # Evict unreferenced chunks (use citations if present, else evict all)
        cited = ContextRegistry.parse_citations(response)
        if cited:
            registry.evict_unreferenced(cited)
        else:
            registry.evict_all()

        judge_result = llm.judge(_build_judge_prompt(query, ground_truth, response))
        result["completion_score"] = judge_result.get("completion_score")
        result["precision_score"]  = judge_result.get("precision_score")

        ret_prec = llm.judge(_build_retrieval_precision_prompt(query, ground_truth, chunks))
        result["retrieval_precision_score"] = ret_prec.get("precision_score")
    else:
        # Dry run: evict all so registry doesn't accumulate across queries
        registry.evict_all()

    return result


# ---------------------------------------------------------------------------
# Main harness
# ---------------------------------------------------------------------------

class EvaluationHarness:
    def __init__(
        self,
        n_queries: int = 50,
        top_k: int = 5,
        top_clusters: int = 2,
        seed: int = 42,
        dry_run: bool = False,
        api_key: Optional[str] = None,
        tags: Optional[list[str]] = None,
        minilm_routing: bool = False,
        use_openai: bool = False,
    ):
        self.n_queries    = n_queries
        self.top_k        = top_k
        self.top_clusters = top_clusters
        self.seed         = seed
        self.dry_run      = dry_run
        self.tags         = tags  # filter SO data by tag; None = all

        log.info("Loading retriever ...")
        base_retriever = Retriever(top_clusters=top_clusters, top_k=top_k)
        if minilm_routing:
            from src.evaluation.embedding_ablation import MiniLMEmbedder, MiniLMRoutingRetriever
            log.info("Swapping to MiniLM cluster routing ...")
            minilm = MiniLMEmbedder()
            self.retriever = MiniLMRoutingRetriever(base_retriever, minilm, top_clusters)
        else:
            self.retriever = base_retriever
        self.registry  = ContextRegistry(token_budget=8192)

        self.llm: Optional[LLMClient] = None
        if not dry_run:
            self.llm = LLMClient(api_key=api_key, use_openai=use_openai)

    # -----------------------------------------------------------------------

    def load_queries(self) -> list[dict]:
        with open(SO_PATH, encoding="utf-8") as f:
            data = json.load(f)

        if self.tags:
            data = [d for d in data if d.get("tag") in self.tags]

        # Keep only records with non-empty question and answer
        data = [
            d for d in data
            if d.get("title") and d.get("answer_body")
        ]

        random.seed(self.seed)
        n = min(self.n_queries, len(data))
        sample = random.sample(data, n)
        log.info(f"Sampled {n} queries from {len(data)} available SO records.")
        return sample

    # -----------------------------------------------------------------------

    def run(self) -> pd.DataFrame:
        queries = self.load_queries()
        rows = []

        for i, record in enumerate(queries):
            qid   = record.get("question_id", i)
            query = record["title"]
            gt    = record.get("answer_body", "")
            tag   = record.get("tag", "")

            log.info(f"[{i+1}/{len(queries)}] {tag} | {query[:60]}")

            base = {
                "question_id": qid,
                "tag": tag,
                "query": query,
            }

            # --- Flat RAG ---
            flat = run_flat(
                query=query,
                retriever=self.retriever,
                top_k=self.top_k,
                llm=self.llm,
                ground_truth=gt,
            )
            rows.append({**base, **flat})

            # --- Selective RAG ---
            self.registry.evict_all()  # fresh registry per query
            sel = run_selective(
                query=query,
                retriever=self.retriever,
                registry=self.registry,
                top_k=self.top_k,
                top_clusters=self.top_clusters,
                llm=self.llm,
                ground_truth=gt,
            )
            rows.append({**base, **sel})

        df = pd.DataFrame(rows)
        return df

    # -----------------------------------------------------------------------

    def save(self, df: pd.DataFrame) -> Path:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
        stem = RESULTS_DIR / f"eval_{ts}"

        df.to_csv(f"{stem}.csv", index=False)
        df.to_json(f"{stem}.json", orient="records", indent=2)

        log.info(f"Results saved → {stem}.csv / .json")
        return Path(f"{stem}.csv")

    # -----------------------------------------------------------------------

    @staticmethod
    def summarize(df: pd.DataFrame) -> dict:
        flat = df[df["condition"] == "flat"]
        sel  = df[df["condition"] == "selective"]

        def _mean(series):
            return round(series.dropna().mean(), 4) if not series.dropna().empty else None

        flat_tok = flat["prompt_tokens"].mean()
        sel_tok  = sel["prompt_tokens"].mean()
        reduction_pct = round((flat_tok - sel_tok) / flat_tok * 100, 2) if flat_tok else None

        summary = {
            "n_queries": len(flat),
            "top_k": int(df.get("n_chunks", pd.Series([None])).dropna().iloc[0]) if len(flat) else None,
            "flat_mean_prompt_tokens":      round(flat_tok, 1),
            "selective_mean_prompt_tokens": round(sel_tok, 1),
            "token_reduction_pct":          reduction_pct,
            "flat_mean_retrieve_ms":        _mean(flat["retrieve_ms"]),
            "selective_mean_retrieve_ms":   _mean(sel["retrieve_ms"]),
            "flat_mean_completion_score":   _mean(flat["completion_score"]),
            "selective_mean_completion_score": _mean(sel["completion_score"]),
            "flat_mean_precision_score":    _mean(flat["precision_score"]),
            "selective_mean_precision_score": _mean(sel["precision_score"]),
            "flat_mean_retrieval_precision":    _mean(flat["retrieval_precision_score"]),
            "selective_mean_retrieval_precision": _mean(sel["retrieval_precision_score"]),
            "flat_total_cost_usd":      round(flat["cost_usd"].dropna().sum(), 4) if "cost_usd" in flat else None,
            "selective_total_cost_usd": round(sel["cost_usd"].dropna().sum(), 4) if "cost_usd" in sel else None,
        }
        return summary

    def print_summary(self, df: pd.DataFrame) -> None:
        s = self.summarize(df)
        print("\n" + "=" * 60)
        print("EVALUATION SUMMARY")
        print("=" * 60)
        print(f"Queries evaluated     : {s['n_queries']}")
        print(f"\n--- Token Counts (mean prompt tokens) ---")
        print(f"  Flat RAG            : {s['flat_mean_prompt_tokens']:.1f}")
        print(f"  Selective RAG       : {s['selective_mean_prompt_tokens']:.1f}")
        print(f"  Reduction           : {s['token_reduction_pct']}%")
        print(f"\n--- Latency (mean retrieve ms) ---")
        print(f"  Flat RAG            : {s['flat_mean_retrieve_ms']}")
        print(f"  Selective RAG       : {s['selective_mean_retrieve_ms']}")
        if s["flat_mean_completion_score"] is not None:
            print(f"\n--- Quality (LLM-as-judge) ---")
            print(f"  Task completion     flat={s['flat_mean_completion_score']}  sel={s['selective_mean_completion_score']}")
            print(f"  Response precision  flat={s['flat_mean_precision_score']}  sel={s['selective_mean_precision_score']}")
            print(f"  Retrieval P@k       flat={s['flat_mean_retrieval_precision']}  sel={s['selective_mean_retrieval_precision']}")
            print(f"\n--- Cost ---")
            print(f"  Flat RAG            : ${s['flat_total_cost_usd']}")
            print(f"  Selective RAG       : ${s['selective_total_cost_usd']}")
        print("=" * 60 + "\n")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _parse_args():
    p = argparse.ArgumentParser(description="Flat RAG vs. Selective RAG evaluation harness")
    p.add_argument("--n",            type=int,   default=50,    help="Number of queries to evaluate")
    p.add_argument("--top-k",        type=int,   default=5,     help="Chunks to retrieve per query")
    p.add_argument("--top-clusters", type=int,   default=2,     help="Clusters to route to (selective only)")
    p.add_argument("--seed",         type=int,   default=42,    help="Random seed for query sampling")
    p.add_argument("--tags",         nargs="+",  default=None,  help="Filter by SO tag(s): fastapi pydantic supabase appwrite")
    p.add_argument("--dry-run",       action="store_true",      help="Skip LLM calls; measure tokens and latency only")
    p.add_argument("--minilm-routing",action="store_true",      help="Use MiniLM centroids for cluster routing")
    p.add_argument("--openai",        action="store_true",      help="Use OpenAI (gpt-4o-mini) instead of Anthropic")
    p.add_argument("--api-key",      type=str,   default=None,  help="Anthropic API key (overrides env var)")
    p.add_argument("--log-level",    default="INFO",            help="Logging level")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    harness = EvaluationHarness(
        n_queries=args.n,
        top_k=args.top_k,
        top_clusters=args.top_clusters,
        seed=args.seed,
        dry_run=args.dry_run,
        minilm_routing=args.minilm_routing,
        use_openai=args.openai,
        api_key=args.api_key,
        tags=args.tags,
    )

    df = harness.run()
    harness.print_summary(df)
    harness.save(df)
