"""
Dedup Benefit Quantification.

Analyses the content-hash deduplication benefit of the ContextRegistry across
multi-step sessions.  Loads existing multistep step CSVs (no new runs required)
and computes:

  - Total duplicate chunks blocked per session (by k value)
  - Estimated token savings from dedup (n_skipped × mean_tokens_per_chunk)
  - How dedup benefit scales with top_k and session step
  - Fraction of retrieval attempts that were duplicates

Usage:
    python -m src.evaluation.dedup_analysis
"""

import logging
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

ROOT        = Path(__file__).parents[2]
RESULTS_DIR = ROOT / "data" / "results"

# Multistep step CSVs keyed by approximate top_k
# (identified by inspecting max n_chunks_in_window)
STEP_FILES = {
    5:  "multistep_20260517_124943_steps.csv",
    10: "multistep_20260517_131126_steps.csv",
    20: "multistep_20260517_131134_steps.csv",
}


def load_selective_steps(files: dict[int, str]) -> pd.DataFrame:
    frames = []
    for top_k, fname in files.items():
        path = RESULTS_DIR / fname
        if not path.exists():
            log.warning(f"Missing: {path}")
            continue
        df = pd.read_csv(path)
        sel = df[df["condition"] == "selective"].copy()
        sel["top_k"] = top_k
        frames.append(sel)
    return pd.concat(frames, ignore_index=True)


def estimate_tokens_per_chunk(df: pd.DataFrame) -> pd.Series:
    """
    Estimate mean tokens per chunk from context_tokens / n_chunks_in_window.
    Only use rows where window is non-empty.
    """
    valid = df[df["n_chunks_in_window"] > 0].copy()
    valid["tokens_per_chunk"] = valid["context_tokens"] / valid["n_chunks_in_window"]
    return valid.groupby("top_k")["tokens_per_chunk"].mean()


def compute_dedup_stats(df: pd.DataFrame) -> pd.DataFrame:
    tpc = estimate_tokens_per_chunk(df)

    rows = []
    for top_k, group in df.groupby("top_k"):
        n_tasks       = group["task_id"].nunique()
        n_steps       = group["step"].max()
        total_attempts = (group["n_newly_loaded"] + group["n_skipped_dup"]).sum()
        total_dupes   = group["n_skipped_dup"].sum()
        dup_rate      = total_dupes / max(total_attempts, 1)
        mean_tpc      = tpc.get(top_k, 0)
        tokens_saved  = total_dupes * mean_tpc
        tokens_per_task = tokens_saved / max(n_tasks, 1)

        rows.append({
            "top_k":                 top_k,
            "n_tasks":               n_tasks,
            "n_steps":               n_steps,
            "total_retrieval_attempts": int(total_attempts),
            "total_dupes_blocked":   int(total_dupes),
            "dup_rate_pct":          round(dup_rate * 100, 1),
            "mean_tokens_per_chunk": round(mean_tpc, 1),
            "total_tokens_saved":    round(tokens_saved, 0),
            "tokens_saved_per_task": round(tokens_per_task, 0),
            "dupes_per_task":        round(total_dupes / max(n_tasks, 1), 1),
        })
    return pd.DataFrame(rows).sort_values("top_k")


def compute_per_step_dedup(df: pd.DataFrame) -> pd.DataFrame:
    """Mean duplicate blocks per step number, by top_k."""
    return (
        df.groupby(["top_k", "step"])["n_skipped_dup"]
        .mean()
        .reset_index()
        .rename(columns={"n_skipped_dup": "mean_dupes_blocked"})
    )


def plot_dedup(summary: pd.DataFrame, per_step: pd.DataFrame, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    top_ks = sorted(summary["top_k"].unique())
    colors = {5: "#2563eb", 10: "#16a34a", 20: "#f97316"}

    # --- Left: tokens saved per task by top_k ---
    ax1 = axes[0]
    bars = ax1.bar(
        [str(k) for k in top_ks],
        summary["tokens_saved_per_task"].values,
        color=[colors.get(k, "#6b7280") for k in top_ks],
        alpha=0.85, edgecolor="white",
    )
    ax1.set_xlabel("top_k", fontsize=11)
    ax1.set_ylabel("Estimated Tokens Saved per Task", fontsize=11)
    ax1.set_title("Dedup Token Savings by Retrieval Breadth", fontsize=12)
    ax1.grid(True, alpha=0.3, axis="y")
    for bar, val in zip(bars, summary["tokens_saved_per_task"].values):
        ax1.annotate(f"{val:.0f}", (bar.get_x() + bar.get_width()/2, val),
                     xytext=(0, 4), textcoords="offset points",
                     ha="center", fontsize=10, fontweight="bold")

    # Also annotate dup rate
    for i, (_, row) in enumerate(summary.iterrows()):
        ax1.annotate(f"({row['dup_rate_pct']}% dup rate)",
                     (i, row["tokens_saved_per_task"] / 2),
                     ha="center", fontsize=8, color="white", fontweight="bold")

    # --- Right: per-step dedup accumulation ---
    ax2 = axes[1]
    for top_k in top_ks:
        sub = per_step[per_step["top_k"] == top_k].sort_values("step")
        # cumulative dupes
        sub = sub.copy()
        sub["cumulative_dupes"] = sub["mean_dupes_blocked"].cumsum()
        ax2.plot(sub["step"], sub["cumulative_dupes"],
                 marker="o", linewidth=2,
                 color=colors.get(top_k, "#6b7280"),
                 label=f"top_k={top_k}")

    ax2.set_xlabel("Agent Step", fontsize=11)
    ax2.set_ylabel("Cumulative Duplicate Chunks Blocked (mean/task)", fontsize=11)
    ax2.set_title("Dedup Benefit Accumulates Over Steps", fontsize=12)
    ax2.legend(fontsize=10)
    ax2.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax2.grid(True, alpha=0.3)

    fig.suptitle("Content-Hash Deduplication Benefit — ContextRegistry",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    log.info(f"Plot saved → {out_path}")
    plt.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s: %(message)s")

    df       = load_selective_steps(STEP_FILES)
    summary  = compute_dedup_stats(df)
    per_step = compute_per_step_dedup(df)

    print("\n" + "=" * 75)
    print("DEDUP BENEFIT QUANTIFICATION")
    print("=" * 75)
    print(summary.to_string(index=False))
    print()
    print("Per-step mean duplicate blocks (cumulative):")
    pivot = per_step.pivot(index="step", columns="top_k",
                           values="mean_dupes_blocked")
    pivot.columns = [f"top_k={k}" for k in pivot.columns]
    pivot["cumsum_k5"]  = per_step[per_step["top_k"]==5]["mean_dupes_blocked"].cumsum().values  if 5  in per_step["top_k"].values else np.nan
    pivot["cumsum_k10"] = per_step[per_step["top_k"]==10]["mean_dupes_blocked"].cumsum().values if 10 in per_step["top_k"].values else np.nan
    pivot["cumsum_k20"] = per_step[per_step["top_k"]==20]["mean_dupes_blocked"].cumsum().values if 20 in per_step["top_k"].values else np.nan
    print(pivot.round(1).to_string())
    print("=" * 75)

    ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_csv  = RESULTS_DIR / f"dedup_{ts}.csv"
    out_plot = RESULTS_DIR / f"dedup_{ts}.png"
    summary.to_csv(out_csv, index=False)
    log.info(f"Saved → {out_csv}")
    plot_dedup(summary, per_step, out_plot)
