"""
Routing Confidence Analysis.

Tests whether routing confidence (centroid similarity gap / entropy) predicts
when selective RAG will outperform flat RAG.

Computes per-query:
  - top1 centroid score
  - top1 - top2 gap
  - softmax entropy of centroid similarities
  - sel_comp - flat_comp outcome from existing eval CSV

Then checks correlation and finds an optimal threshold.
"""

import json
import logging
import random
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pointbiserialr

from src.evaluation.embedding_ablation import MiniLMEmbedder

ROOT        = Path(__file__).parents[2]
SO_PATH     = ROOT / "data" / "raw" / "stackoverflow" / "stackoverflow_all_raw.json"
RESULTS_DIR = ROOT / "data" / "results"

log = logging.getLogger(__name__)


def routing_features(minilm: MiniLMEmbedder, query: str) -> dict:
    emb    = minilm.embed(query)
    scores = minilm._centroid_mat @ emb          # (n_clusters,)

    sorted_scores = np.sort(scores)[::-1]
    top1  = float(sorted_scores[0])
    top2  = float(sorted_scores[1])
    gap   = top1 - top2

    # Softmax entropy
    exp_s = np.exp(scores - scores.max())
    probs = exp_s / exp_s.sum()
    entropy = float(-np.sum(probs * np.log(probs + 1e-12)))

    return {"top1": top1, "top2": top2, "gap": gap, "entropy": entropy}


def main():
    logging.basicConfig(level=logging.WARNING)

    # Load eval results (MiniLM routing, k=5)
    eval_path = RESULTS_DIR / "eval_20260517_191449.csv"
    df = pd.read_csv(eval_path)

    flat = df[df.condition == "flat"][["question_id", "query", "tag", "completion_score"]] \
             .rename(columns={"completion_score": "flat_comp"})
    sel  = df[df.condition == "selective"][["question_id", "completion_score"]] \
             .rename(columns={"completion_score": "sel_comp"})
    merged = flat.merge(sel, on="question_id").dropna()
    merged["outcome"] = (merged["sel_comp"] > merged["flat_comp"]).astype(int)
    merged["diff"]    = merged["sel_comp"] - merged["flat_comp"]

    log.info(f"Loaded {len(merged)} query pairs.")

    # Load SO data for query text
    with open(SO_PATH) as f:
        so_data = {r["question_id"]: r for r in json.load(f) if "question_id" in r}

    # Compute routing features for each query
    print("Loading MiniLM embedder...")
    minilm = MiniLMEmbedder()

    rows = []
    for _, row in merged.iterrows():
        qid   = int(row["question_id"])
        query = row["query"]
        feats = routing_features(minilm, query)
        rows.append({
            "question_id": qid,
            "tag":         row["tag"],
            "query":       query,
            "flat_comp":   row["flat_comp"],
            "sel_comp":    row["sel_comp"],
            "diff":        row["diff"],
            "outcome":     row["outcome"],
            **feats,
        })

    feat_df = pd.DataFrame(rows)

    # --- Correlation analysis ---
    print("\n" + "=" * 65)
    print("ROUTING CONFIDENCE vs OUTCOME CORRELATION")
    print("=" * 65)
    for feat in ["gap", "entropy", "top1"]:
        r, p = pointbiserialr(feat_df["outcome"], feat_df[feat])
        print(f"  {feat:<12} r={r:+.3f}  p={p:.4f}")

    # --- Threshold sweep on gap ---
    print("\n--- Gap threshold sweep (predict: use selective if gap > threshold) ---")
    print(f"  {'threshold':>10} {'accuracy':>10} {'n_sel':>8} {'n_flat':>8}")
    thresholds = np.linspace(feat_df["gap"].min(), feat_df["gap"].max(), 30)
    best_acc, best_thresh = 0, None
    for t in thresholds:
        pred_sel  = feat_df["gap"] >= t
        # If we predict selective: outcome must be 1 to be correct
        # If we predict flat:      outcome must be 0 to be correct
        correct   = ((pred_sel) & (feat_df["outcome"] == 1)) | \
                    ((~pred_sel) & (feat_df["outcome"] == 0))
        acc = correct.mean()
        n_sel  = pred_sel.sum()
        n_flat = (~pred_sel).sum()
        if acc > best_acc:
            best_acc, best_thresh = acc, t
        print(f"  {t:>10.4f} {acc:>10.3f} {n_sel:>8} {n_flat:>8}")

    print(f"\n  Best threshold: gap > {best_thresh:.4f}  →  accuracy {best_acc:.3f}")

    # --- Entropy threshold sweep ---
    print("\n--- Entropy threshold sweep (predict: use flat if entropy > threshold) ---")
    best_acc_e, best_thresh_e = 0, None
    ent_thresholds = np.linspace(feat_df["entropy"].min(), feat_df["entropy"].max(), 30)
    for t in ent_thresholds:
        pred_flat = feat_df["entropy"] >= t
        correct   = ((~pred_flat) & (feat_df["outcome"] == 1)) | \
                    ((pred_flat)  & (feat_df["outcome"] == 0))
        acc = correct.mean()
        if acc > best_acc_e:
            best_acc_e, best_thresh_e = acc, t

    print(f"  Best threshold: entropy > {best_thresh_e:.4f}  →  accuracy {best_acc_e:.3f}")

    # --- Baseline ---
    always_sel  = (feat_df["outcome"] == 1).mean()
    always_flat = (feat_df["outcome"] == 0).mean()
    print(f"\n  Baseline (always selective): {always_sel:.3f}")
    print(f"  Baseline (always flat):      {always_flat:.3f}")
    print("=" * 65)

    # --- Plot ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax1 = axes[0]
    colors = ["#dc2626" if o == 0 else "#16a34a" for o in feat_df["outcome"]]
    ax1.scatter(feat_df["gap"], feat_df["diff"], c=colors, alpha=0.6, s=30)
    ax1.axvline(best_thresh, color="black", linestyle="--", linewidth=1.5,
                label=f"Best gap threshold ({best_thresh:.3f})")
    ax1.axhline(0, color="gray", linestyle=":", linewidth=1)
    ax1.set_xlabel("Routing Gap (top1 − top2 centroid score)", fontsize=11)
    ax1.set_ylabel("sel_comp − flat_comp", fontsize=11)
    ax1.set_title("Routing Confidence vs Quality Outcome", fontsize=12)
    ax1.legend(fontsize=9)
    from matplotlib.lines import Line2D
    ax1.legend(handles=[
        Line2D([0],[0],marker='o',color='w',markerfacecolor='#16a34a',markersize=8,label='selective wins'),
        Line2D([0],[0],marker='o',color='w',markerfacecolor='#dc2626',markersize=8,label='flat wins'),
        Line2D([0],[0],color='black',linestyle='--',label=f'threshold={best_thresh:.3f}'),
    ], fontsize=9)
    ax1.grid(True, alpha=0.3)

    ax2 = axes[1]
    ax2.scatter(feat_df["entropy"], feat_df["diff"], c=colors, alpha=0.6, s=30)
    ax2.axvline(best_thresh_e, color="black", linestyle="--", linewidth=1.5,
                label=f"Best entropy threshold ({best_thresh_e:.3f})")
    ax2.axhline(0, color="gray", linestyle=":", linewidth=1)
    ax2.set_xlabel("Routing Entropy (softmax centroid scores)", fontsize=11)
    ax2.set_ylabel("sel_comp − flat_comp", fontsize=11)
    ax2.set_title("Routing Entropy vs Quality Outcome", fontsize=12)
    ax2.grid(True, alpha=0.3)

    fig.suptitle("Routing Confidence as Predictor of Selective RAG Advantage",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    out = RESULTS_DIR / "routing_confidence.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\nPlot saved → {out}")
    plt.close()

    feat_df.to_csv(RESULTS_DIR / "routing_confidence.csv", index=False)


if __name__ == "__main__":
    main()
