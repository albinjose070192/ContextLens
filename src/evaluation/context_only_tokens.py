"""
Context-Only Token Measurement.

Strips system-prompt and query overhead from prompt_tokens in eval CSVs
to isolate the actual retrieved-context token count — the metric that
reflects retrieval cost without prompt scaffolding noise.

Overhead layout
---------------
Flat prompt (harness._build_flat_prompt):
  PREAMBLE + "## Retrieved Context\n\n" + CONTEXT + "\n\n## Query\n\n" + QUERY

Selective prompt (ContextRegistry.build_prompt):
  SYSTEM + "\n\n## Retrieved Context\n\n" + CONTEXT + "\n\n## Query\n\n"
  + QUERY + "\n\nWhen answering, cite ..."

Both have the same system/query overhead; selective adds a citation instruction.
We subtract both to isolate CONTEXT tokens.
"""

import tiktoken
import pandas as pd
from pathlib import Path

ROOT        = Path(__file__).parents[2]
RESULTS_DIR = ROOT / "data" / "results"

_ENC = tiktoken.get_encoding("cl100k_base")

# Fixed preamble shared by both conditions
_FLAT_SYSTEM = (
    "You are a helpful software engineering assistant. "
    "Answer the query using only the retrieved context provided below. "
    "Be concise and precise."
)
_FLAT_SCAFFOLD   = "\n\n## Retrieved Context\n\n\n\n## Query\n\n"
_SEL_EXTRA       = "\n\nWhen answering, cite the chunk_id(s) you relied on using the format: [used: chunk_id_1, chunk_id_2]"

_FLAT_FIXED_TOKENS = len(_ENC.encode(_FLAT_SYSTEM + _FLAT_SCAFFOLD))
_SEL_EXTRA_TOKENS  = len(_ENC.encode(_SEL_EXTRA))


def context_only_tokens(row: pd.Series) -> int:
    query_tokens = len(_ENC.encode(str(row["query"])))
    overhead = _FLAT_FIXED_TOKENS + query_tokens
    if row["condition"] == "selective":
        overhead += _SEL_EXTRA_TOKENS
    return max(0, int(row["prompt_tokens"]) - overhead)


def analyze(csv_path: Path, label: str) -> None:
    df = pd.read_csv(csv_path)
    df["context_only_tokens"] = df.apply(context_only_tokens, axis=1)

    flat = df[df.condition == "flat"]
    sel  = df[df.condition == "selective"]

    flat_mean  = flat["context_only_tokens"].mean()
    sel_mean   = sel["context_only_tokens"].mean()
    flat_ptok  = flat["prompt_tokens"].mean()
    sel_ptok   = sel["prompt_tokens"].mean()

    overhead_flat = flat_ptok - flat_mean
    overhead_sel  = sel_ptok  - sel_mean
    reduction_prompt  = (flat_ptok  - sel_ptok)  / flat_ptok  * 100
    reduction_context = (flat_mean  - sel_mean)  / flat_mean  * 100

    print(f"\n{'─'*65}")
    print(f"  {label}")
    print(f"{'─'*65}")
    print(f"  {'':25} {'Flat':>10} {'Selective':>12} {'Δ':>8}")
    print(f"  {'prompt_tokens (mean)':25} {flat_ptok:>10.1f} {sel_ptok:>12.1f} "
          f"{reduction_prompt:>+7.1f}%")
    print(f"  {'overhead_tokens (mean)':25} {overhead_flat:>10.1f} {overhead_sel:>12.1f}")
    print(f"  {'context_only (mean)':25} {flat_mean:>10.1f} {sel_mean:>12.1f} "
          f"{reduction_context:>+7.1f}%")


def main():
    print("=" * 65)
    print("CONTEXT-ONLY TOKEN MEASUREMENT (MiniLM routing)")
    print("Single-step eval — stripping system prompt + query overhead")
    print("=" * 65)

    analyze(RESULTS_DIR / "eval_20260517_191449.csv", "k=5  (eval_191449)")
    analyze(RESULTS_DIR / "eval_20260517_202227.csv", "k=10 (eval_202227)")
    analyze(RESULTS_DIR / "eval_20260517_210244.csv", "k=20 (eval_210244)")

    print(f"\n{'─'*65}")
    print(f"  Fixed overhead per query:")
    print(f"    Flat system+scaffold : {_FLAT_FIXED_TOKENS} tokens")
    print(f"    Selective extra (cite): {_SEL_EXTRA_TOKENS} tokens")
    print("=" * 65)


if __name__ == "__main__":
    main()
