# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# MLFP03 — Exercise 3.6: The Model Zoo — head-to-head comparison
# ════════════════════════════════════════════════════════════════════════
#
# WHAT YOU'LL LEARN:
#   - Train all five classical models on the same data
#   - Build a fair comparison table: accuracy, F1, AUC-ROC, train time
#   - Overlay decision boundaries in 2D PCA space
#   - Publish a when-to-use guide and rank by dollar impact
#
# ESTIMATED TIME: ~30 min
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import numpy as np
from dotenv import load_dotenv
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from shared.mlfp03.ex_3 import (
    build_train_test_split,
    churn_saved_dollars,
    decision_boundary_mesh,
    fit_and_evaluate,
    OUTPUT_DIR,
    project_2d,
    RANDOM_SEED,
    save_metric_comparison,
)

load_dotenv()

# ════════════════════════════════════════════════════════════════════════
# THEORY — Model selection is multi-criteria
# ════════════════════════════════════════════════════════════════════════
# Trade across accuracy, speed, interpretability, robustness, memory.


# ════════════════════════════════════════════════════════════════════════
# TASK 2 — BUILD: assemble all 5 models
# ════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("  MLFP03 Exercise 3.6 — Classical ML Zoo — head-to-head")
print("=" * 70)

data = build_train_test_split()
X_train, X_test = data["X_train"], data["X_test"]
y_train, y_test = data["y_train"], data["y_test"]

# TODO: fill in the 5-model zoo with reasonable defaults.
# Keys (exact strings): "SVM (RBF)", "KNN (k=11)", "Naive Bayes",
# "Decision Tree", "Random Forest".
#   SVM: kernel="rbf", C=1.0, probability=True
#   KNN: n_neighbors=11, metric="euclidean"
#   Naive Bayes: GaussianNB()
#   Decision Tree: max_depth=7
#   Random Forest: n_estimators=200, max_features="sqrt", oob_score=True, n_jobs=-1
zoo = {
    "SVM (RBF)": ____,
    "KNN (k=11)": ____,
    "Naive Bayes": ____,
    "Decision Tree": ____,
    "Random Forest": ____,
}


# ════════════════════════════════════════════════════════════════════════
# TASK 3 — TRAIN: fit every model on the same data
# ════════════════════════════════════════════════════════════════════════

results: list[dict] = []
for name, est in zoo.items():
    # TODO: call fit_and_evaluate(est, X_train, y_train, X_test, y_test, name=name)
    # and append to results.
    r = ____
    results.append(r)

print("\n--- Performance table ---")
print(f"{'Model':<18} {'Accuracy':>10} {'F1':>10} {'AUC-ROC':>10} " f"{'Time (s)':>10}")
print("-" * 62)
for r in results:
    print(
        f"{r['name']:<18} {r['accuracy']:>10.4f} {r['f1']:>10.4f} "
        f"{r['auc_roc']:>10.4f} {r['train_time']:>10.4f}"
    )

ranked = sorted(results, key=lambda r: r["f1"], reverse=True)
print("\nRanking by F1:")
for i, r in enumerate(ranked, 1):
    print(f"  {i}. {r['name']} (F1={r['f1']:.4f})")


# ════════════════════════════════════════════════════════════════════════
# TASK 4 — VISUALISE: boundaries + metric comparison
# ════════════════════════════════════════════════════════════════════════

pca_bundle = project_2d(X_train, X_test)
X_train_2d = pca_bundle["X_train_2d"]

boundary_scores: dict[str, float] = {}
xx, yy = decision_boundary_mesh(X_train_2d)
grid_points = np.c_[xx.ravel(), yy.ravel()]

for name, est in zoo.items():
    if hasattr(est, "oob_score"):
        est_2d = type(est)(
            **{**est.get_params(), "oob_score": False, "n_estimators": 100}
        )
    else:
        est_2d = type(est)(**est.get_params())
    est_2d.fit(X_train_2d, y_train)
    pred_2d = est_2d.predict(pca_bundle["X_test_2d"])
    boundary_scores[name] = float((pred_2d == y_test).mean())
    Z = est_2d.predict(grid_points).reshape(xx.shape)
    print(f"  {name:<18}: 2D accuracy={boundary_scores[name]:.4f} | mesh={Z.shape}")

# TODO: build metric_dict in the shape save_metric_comparison expects:
# {model_name: {"Accuracy": ..., "F1": ..., "AUC-ROC": ...}, ...}
# then save it to "ex3_06_zoo_comparison.html".
metric_dict = ____
comparison_path = save_metric_comparison(metric_dict, "ex3_06_zoo_comparison.html")
print(f"\nSaved: {comparison_path}")

# ── Checkpoint 1 ────────────────────────────────────────────────────────
assert len(results) == 5
assert all(r["accuracy"] > 0.5 for r in results)
assert len(boundary_scores) == 5
print("\n[ok] Checkpoint 1 passed\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 5 — APPLY: when-to-use guide + dollar impact ranking
# ════════════════════════════════════════════════════════════════════════

print("\n--- Dollar impact ranking (held-out test set) ---")
print(f"{'Model':<18} {'TP':>6} {'S$ saved':>14} {'Monthly S$ @250K':>18}")
print("-" * 60)
impact_rows = []
for r in results:
    # TODO: count true positives (pred==1 and y_test==1) and compute
    # churn_saved_dollars(tp). Scale to 250K MAU.
    tp = ____
    saved = ____
    monthly_scale = saved * (250_000 / len(y_test))
    impact_rows.append(
        {
            "name": r["name"],
            "tp": tp,
            "saved": saved,
            "monthly_scale": monthly_scale,
            "f1": r["f1"],
        }
    )
    print(f"{r['name']:<18} {tp:>6} S${saved:>11,.2f} S${monthly_scale:>15,.0f}")

best_dollar = max(impact_rows, key=lambda r: r["monthly_scale"])
best_f1 = max(impact_rows, key=lambda r: r["f1"])
print(
    f"\nHighest dollar impact: {best_dollar['name']} "
    f"(S${best_dollar['monthly_scale']:,.0f}/mo)"
)
print(f"Highest F1: {best_f1['name']} (F1={best_f1['f1']:.4f})")


# ════════════════════════════════════════════════════════════════════════
# DESTINATION-FIRST CLOSE — km.diagnose
# ════════════════════════════════════════════════════════════════════════
# This lesson built five classical models from primitives — fitting each,
# timing each, building a comparison table, mapping decision boundaries.
# The kailash-ml SDK packages the entire diagnostic surface (per-class
# metrics, class-balance severity, confusion matrix, accuracy heuristics)
# into a single call.
#
# Destination-first: when the journey is internalised, the SDK is one line.

from kailash_ml import diagnose

# `kind="classical_classifier"` dispatches to the sklearn ClassifierMixin
# adapter; `data=(X, y)` is the validation pair the lesson already built.
# Use the F1 winner from the comparison above.
best_model = zoo[best_f1["name"]]
report = diagnose(
    best_model, kind="classical_classifier", data=(X_test, y_test), show=False
)
print()
print(f"  km.diagnose model    : {best_f1['name']}")
print(f"  km.diagnose metrics  : {report.metrics}")
print(f"  km.diagnose severity : {report.severity}")
print()
print("km.diagnose: 1 call -> the same diagnostic surface the lesson body")
print("hand-rolled across all 5 models. Destination-first: when the")
print("journey is internalised, the SDK is one line.")


# ════════════════════════════════════════════════════════════════════════
# REFLECTION
# ════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print(
    f"""
  [x] Trained 5 classical models on identical data and folds
  [x] Built a fair comparison table (accuracy / F1 / AUC / train time)
  [x] Mapped decision boundaries in 2D PCA space
  [x] Highest-F1 model: {best_f1['name']}
  [x] Highest-dollar-impact model: {best_dollar['name']}

  NEXT: Exercise 4 — gradient boosting (XGBoost, LightGBM, CatBoost).
"""
)
