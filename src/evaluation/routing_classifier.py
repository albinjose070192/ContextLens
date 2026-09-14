"""
Routing Decision Classifier — Predict when selective RAG outperforms flat RAG.

Uses query-level features (no retrieval needed) to decide at query time:
  - selective RAG  (deeper, cluster-routed)
  - flat RAG       (broader, all-cluster sweep)

Features:
  - word count, char count
  - question type (how-to vs conceptual vs other)
  - number of tech terms mentioned
  - cross-domain flag (2+ distinct techs)
  - code-like terms (CamelCase, snake_case, parens)
  - routing gap + entropy (from MiniLM centroid similarities)

Labels come from existing eval CSV:
  outcome = 1 if sel_comp > flat_comp else 0  (selective better)

With only ~100 labeled queries, uses logistic regression + 5-fold stratified CV.
Reports accuracy, AUC, and feature importances.
"""

import json
import logging
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from src.evaluation.embedding_ablation import MiniLMEmbedder
from src.evaluation.routing_confidence import routing_features

ROOT        = Path(__file__).parents[2]
SO_PATH     = ROOT / "data" / "raw" / "stackoverflow" / "stackoverflow_all_raw.json"
RESULTS_DIR = ROOT / "data" / "results"

log = logging.getLogger(__name__)

# Tech term vocabulary — expand if corpus grows
TECH_TERMS = [
    "fastapi", "pydantic", "supabase", "appwrite", "sqlalchemy", "alembic",
    "uvicorn", "starlette", "oauth", "jwt", "cors", "websocket", "openapi",
    "swagger", "postgresql", "sqlite", "redis", "celery", "docker",
    "basemodel", "uploadfile", "httpexception", "apirouter", "depends",
    "background", "middleware", "async", "await", "typing", "optional",
]


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def extract_features(query: str, minilm: MiniLMEmbedder) -> dict:
    q = query.lower()
    words = q.split()

    # Basic length
    word_count = len(words)
    char_count = len(query)

    # Question type
    first = words[0] if words else ""
    how_to     = int(first in ("how", "how's", "how's"))
    conceptual = int(first in ("what", "why", "when", "which", "where", "is", "are",
                                "can", "should", "does", "do"))

    # Tech terms
    found_techs = [t for t in TECH_TERMS if t in q]
    tech_count  = len(found_techs)
    cross_tech  = int(
        sum(1 for t in ["fastapi", "pydantic", "supabase", "appwrite", "sqlalchemy"]
            if t in q) >= 2
    )

    # Code-like patterns
    has_camel    = int(bool(re.search(r'[a-z][A-Z]', query)))         # camelCase
    has_snake    = int(bool(re.search(r'[a-z]_[a-z]', query)))        # snake_case
    has_backtick = int('`' in query)
    has_parens   = int(bool(re.search(r'\w+\(', query)))              # func()
    code_signals = has_camel + has_snake + has_backtick + has_parens

    # Capitalized (non-first) words — library/class names
    non_first_caps = sum(1 for w in words[1:] if w and w[0].isupper())

    # Routing signals (MiniLM centroid similarities)
    rfeats = routing_features(minilm, query)

    return {
        "word_count":       word_count,
        "char_count":       char_count,
        "how_to":           how_to,
        "conceptual":       conceptual,
        "tech_count":       tech_count,
        "cross_tech":       cross_tech,
        "code_signals":     code_signals,
        "has_snake":        has_snake,
        "non_first_caps":   non_first_caps,
        "routing_gap":      rfeats["gap"],
        "routing_entropy":  rfeats["entropy"],
        "routing_top1":     rfeats["top1"],
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    logging.basicConfig(level=logging.WARNING)

    # Load eval results
    df = pd.read_csv(RESULTS_DIR / "eval_20260517_191449.csv")
    flat = df[df.condition == "flat"][["question_id", "query", "tag", "completion_score"]] \
             .rename(columns={"completion_score": "flat_comp"})
    sel  = df[df.condition == "selective"][["question_id", "completion_score"]] \
             .rename(columns={"completion_score": "sel_comp"})
    merged = flat.merge(sel, on="question_id").dropna()
    merged["outcome"] = (merged["sel_comp"] > merged["flat_comp"]).astype(int)

    print(f"Dataset: {len(merged)} queries  |  "
          f"selective wins: {merged['outcome'].sum()} ({merged['outcome'].mean():.1%})  |  "
          f"flat wins: {(1-merged['outcome']).sum()} ({(1-merged['outcome']).mean():.1%})")

    print("\nExtracting features...")
    minilm = MiniLMEmbedder()

    feat_rows = []
    for _, row in merged.iterrows():
        feats = extract_features(row["query"], minilm)
        feats["question_id"] = row["question_id"]
        feats["tag"]         = row["tag"]
        feats["outcome"]     = row["outcome"]
        feat_rows.append(feats)

    feat_df = pd.DataFrame(feat_rows)

    feature_cols = [
        "word_count", "char_count", "how_to", "conceptual",
        "tech_count", "cross_tech", "code_signals", "has_snake",
        "non_first_caps", "routing_gap", "routing_entropy", "routing_top1",
    ]

    X = feat_df[feature_cols].values
    y = feat_df["outcome"].values

    # ---------------------------------------------------------------------------
    # 5-fold stratified CV — logistic regression
    # ---------------------------------------------------------------------------
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf",    LogisticRegression(C=0.5, max_iter=1000, class_weight="balanced",
                                      random_state=42)),
    ])

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    accs  = cross_val_score(pipe, X, y, cv=cv, scoring="accuracy")
    aucs  = cross_val_score(pipe, X, y, cv=cv, scoring="roc_auc")
    y_pred_cv = cross_val_predict(pipe, X, y, cv=cv)
    y_prob_cv = cross_val_predict(pipe, X, y, cv=cv, method="predict_proba")[:, 1]

    print("\n" + "=" * 65)
    print("LOGISTIC REGRESSION — 5-FOLD STRATIFIED CV")
    print("=" * 65)
    print(f"  Accuracy : {accs.mean():.3f} ± {accs.std():.3f}  (folds: {accs.round(3)})")
    print(f"  ROC-AUC  : {aucs.mean():.3f} ± {aucs.std():.3f}  (folds: {aucs.round(3)})")
    print(f"\n  Baseline (always flat):      {(y==0).mean():.3f}")
    print(f"  Baseline (always selective): {(y==1).mean():.3f}")
    print(f"\n  Classification report (CV predictions):")
    print(classification_report(y, y_pred_cv, target_names=["flat", "selective"],
                                 digits=3))

    # ---------------------------------------------------------------------------
    # Feature importances (fit on full data)
    # ---------------------------------------------------------------------------
    pipe.fit(X, y)
    coefs = pipe.named_steps["clf"].coef_[0]

    print("  Feature coefficients (positive → predicts selective wins):")
    coef_df = pd.DataFrame({"feature": feature_cols, "coef": coefs}) \
                .sort_values("coef", ascending=False)
    for _, r in coef_df.iterrows():
        bar = "█" * int(abs(r.coef) * 10)
        sign = "+" if r.coef > 0 else "-"
        print(f"    {r.feature:<20} {sign}{abs(r.coef):.3f}  {bar}")
    print("=" * 65)

    # ---------------------------------------------------------------------------
    # Plot
    # ---------------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Left: feature importances
    ax1 = axes[0]
    colors = ["#16a34a" if c > 0 else "#dc2626" for c in coef_df["coef"]]
    ax1.barh(coef_df["feature"], coef_df["coef"], color=colors, alpha=0.8)
    ax1.axvline(0, color="black", linewidth=0.8)
    ax1.set_xlabel("Logistic Regression Coefficient", fontsize=11)
    ax1.set_title("Feature Importances\n(+ve → selective wins, −ve → flat wins)",
                  fontsize=12)
    ax1.grid(True, alpha=0.3, axis="x")

    # Right: predicted probability vs actual outcome
    ax2 = axes[1]
    jitter = np.random.RandomState(42).uniform(-0.04, 0.04, len(y))
    c_map  = ["#dc2626" if yi == 0 else "#16a34a" for yi in y]
    ax2.scatter(y_prob_cv, y + jitter, c=c_map, alpha=0.5, s=25)
    ax2.axvline(0.5, color="black", linestyle="--", linewidth=1,
                label="decision boundary (0.5)")
    ax2.set_xlabel("Predicted P(selective wins)", fontsize=11)
    ax2.set_ylabel("Actual outcome  (0=flat, 1=selective)", fontsize=11)
    ax2.set_title(f"CV Predicted Probability vs Actual\nAUC={aucs.mean():.3f}",
                  fontsize=12)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    fig.suptitle("Routing Decision Classifier — Can We Predict When Selective RAG Wins?",
                 fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    out_path = RESULTS_DIR / "routing_classifier.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nPlot saved → {out_path}")
    plt.close()

    feat_df.to_csv(RESULTS_DIR / "routing_classifier_features.csv", index=False)


if __name__ == "__main__":
    main()
