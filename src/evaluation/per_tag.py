"""
Per-Tag Retrieval Quality Breakdown.

Loads existing single-step eval CSVs and breaks results down by SO tag
(fastapi / pydantic / supabase / appwrite). Generates a summary table and
comparison plot. No new LLM calls required.

Usage:
    python -m src.evaluation.per_tag
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

# The three completed single-step eval runs (k=5, k=10, k=20) — MiniLM routing
EVAL_FILES = {
    5:  "eval_20260517_191449.csv",
    10: "eval_20260517_202227.csv",
    20: "eval_20260517_210244.csv",
}

TAG_COLORS = {
    "fastapi":  "#2563eb",
    "pydantic": "#16a34a",
    "supabase": "#f97316",
    "appwrite": "#9333ea",
}


def load_and_tag(files: dict[int, str]) -> pd.DataFrame:
    frames = []
    for top_k, fname in files.items():
        path = RESULTS_DIR / fname
        if not path.exists():
            log.warning(f"Missing: {path}")
            continue
        df = pd.read_csv(path)
        df["top_k"] = top_k
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def per_tag_summary(df: pd.DataFrame) -> pd.DataFrame:
    metrics = ["prompt_tokens", "retrieval_precision_score",
               "completion_score", "precision_score", "retrieve_ms"]
    available = [m for m in metrics if m in df.columns]
    summary = (
        df.groupby(["top_k", "condition", "tag"])[available]
        .mean()
        .round(4)
        .reset_index()
    )
    return summary


def plot_per_tag(df: pd.DataFrame, out_path: Path) -> None:
    tags = sorted(df["tag"].dropna().unique())
    top_ks = sorted(df["top_k"].unique())

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # --- Left: Retrieval precision by tag, flat vs selective, at top_k=5 ---
    ax1 = axes[0]
    df5 = df[df["top_k"] == 5].copy()

    x      = np.arange(len(tags))
    width  = 0.35
    flat5  = df5[df5["condition"] == "flat"].groupby("tag")["retrieval_precision_score"].mean()
    sel5   = df5[df5["condition"] == "selective"].groupby("tag")["retrieval_precision_score"].mean()

    bars1 = ax1.bar(x - width/2,
                    [flat5.get(t, 0) for t in tags],
                    width, label="Flat RAG", color="#94a3b8", alpha=0.85)
    bars2 = ax1.bar(x + width/2,
                    [sel5.get(t, 0) for t in tags],
                    width, label="Selective RAG", color="#2563eb", alpha=0.85)

    ax1.set_xticks(x)
    ax1.set_xticklabels(tags, fontsize=11)
    ax1.set_ylabel("Mean Retrieval Precision@5", fontsize=11)
    ax1.set_title("Retrieval Precision by Tag (top_k=5)", fontsize=12)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3, axis="y")
    for bar in list(bars1) + list(bars2):
        h = bar.get_height()
        if h > 0:
            ax1.annotate(f"{h:.3f}", xy=(bar.get_x() + bar.get_width()/2, h),
                         xytext=(0, 3), textcoords="offset points",
                         ha="center", fontsize=8)

    # --- Right: Prompt tokens by tag across k values (selective only) ---
    ax2 = axes[1]
    sel = df[df["condition"] == "selective"]
    for tag in tags:
        tag_data = sel[sel["tag"] == tag].groupby("top_k")["prompt_tokens"].mean()
        color = TAG_COLORS.get(tag, "#6b7280")
        ax2.plot(tag_data.index, tag_data.values,
                 marker="o", linewidth=2, label=tag, color=color)

    flat = df[df["condition"] == "flat"]
    for tag in tags:
        tag_data = flat[flat["tag"] == tag].groupby("top_k")["prompt_tokens"].mean()
        color = TAG_COLORS.get(tag, "#6b7280")
        ax2.plot(tag_data.index, tag_data.values,
                 marker="s", linewidth=1.5, linestyle="--", color=color, alpha=0.5)

    ax2.set_xlabel("top_k", fontsize=11)
    ax2.set_ylabel("Mean Prompt Tokens", fontsize=11)
    ax2.set_title("Prompt Tokens by Tag (solid=selective, dashed=flat)", fontsize=12)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.xaxis.set_major_locator(ticker.FixedLocator(top_ks))

    fig.suptitle("Per-Tag Retrieval Quality Breakdown", fontsize=14,
                 fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    log.info(f"Plot saved → {out_path}")
    plt.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s: %(message)s")

    df = load_and_tag(EVAL_FILES)
    summary = per_tag_summary(df)

    print("\n" + "=" * 75)
    print("PER-TAG SUMMARY (mean across all top_k values)")
    print("=" * 75)
    agg = (
        df.groupby(["condition", "tag"])[
            ["prompt_tokens", "retrieval_precision_score", "completion_score"]
        ]
        .mean()
        .round(4)
    )
    print(agg.to_string())
    print("=" * 75)

    ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_csv  = RESULTS_DIR / f"per_tag_{ts}.csv"
    out_plot = RESULTS_DIR / f"per_tag_{ts}.png"

    summary.to_csv(out_csv, index=False)
    log.info(f"Summary saved → {out_csv}")

    plot_per_tag(df, out_plot)
