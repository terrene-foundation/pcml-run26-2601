# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# MLFP03 — Exercise 4.4: Boosting Tuning — Sweeps, Heatmaps, Early Stopping
# ════════════════════════════════════════════════════════════════════════
#
# WHAT YOU'LL LEARN:
#   - Sweep learning rate and read AUC-PR across η values
#   - Build a learning_rate × max_depth heatmap and read interaction
#   - Use early stopping to pick n_estimators automatically
#   - Explain why grid search over independent dials is the wrong mental
#     model for boosting (Exercise 7: Bayesian optimisation)
#
# PREREQUISITES: Exercise 4.2/4.3.
#
# ESTIMATED TIME: ~40 min
#
# TASKS:
#   1. Theory — hyperparameter interaction
#   2. Build — sweep grids (LR + 2-D heatmap)
#   3. Train — run sweep; run early stopping with 2000-round budget
#   4. Visualise — LR curve + heatmap + early-stopping comparison
#   5. Apply — Grab credit pre-approval tuning
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from dotenv import load_dotenv
from sklearn.metrics import average_precision_score

from shared.mlfp03.ex_4 import (
    OUTPUT_DIR,
    evaluate_classifier,
    make_lightgbm,
    make_xgboost,
    prepare_credit_split,
    print_metrics,
)

load_dotenv()


# ════════════════════════════════════════════════════════════════════════
# THEORY — Hyperparameter Interaction
# ════════════════════════════════════════════════════════════════════════
# At η=0.01, depth=10 is fine. At η=0.2, depth=10 overfits badly. So a
# 1-D sweep over learning_rate with max_depth fixed at 6 can mislead.
# Use a 2-D heatmap over (η, depth) and use early stopping for
# n_estimators.


# ════════════════════════════════════════════════════════════════════════
# TASK 2 — BUILD the sweep grids
# ════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("  Boosting Hyperparameter Tuning on Singapore Credit")
print("=" * 70)

data = prepare_credit_split()
X_train, y_train = data["X_train"], data["y_train"]
X_test, y_test = data["X_test"], data["y_test"]

print(f"\n  Train: {X_train.shape} | Test: {X_test.shape}")

learning_rates = [0.01, 0.03, 0.05, 0.1, 0.2, 0.5]
depths_sweep = [3, 5, 6, 8, 10]
lr_sweep_for_heatmap = [0.01, 0.05, 0.1, 0.2]


# ════════════════════════════════════════════════════════════════════════
# TASK 3 — TRAIN the sweeps
# ════════════════════════════════════════════════════════════════════════

# --- 3a. 1-D learning-rate sweep -----------------------------------------
print("\n  --- Learning-rate sweep (XGBoost, depth=6, 500 rounds) ---")
print(f"  {'lr':>6}  {'AUC-ROC':>10}  {'AUC-PR':>10}")
print("  " + "─" * 32)

lr_curve: list[tuple[float, float, float]] = []
for lr in learning_rates:
    # TODO: Build an XGBoost model with learning_rate=lr, n_estimators=500,
    # max_depth=6. Fit on X_train, y_train (verbose=False), then get the
    # metric bundle via evaluate_classifier.
    m = ____
    m.fit(X_train, y_train, verbose=False)
    y_p = m.predict_proba(X_test)[:, 1]
    metrics = ____
    lr_curve.append((lr, metrics["auc_roc"], metrics["auc_pr"]))
    print(f"  {lr:>6.2f}  {metrics['auc_roc']:>10.4f}  {metrics['auc_pr']:>10.4f}")


# --- 3b. 2-D learning_rate × max_depth heatmap ---------------------------
print("\n  --- Heatmap: learning_rate × max_depth (XGBoost, 300 rounds) ---")
print(f"  {'':>8}", end="")
for d in depths_sweep:
    print(f"  d={d:<3}   ", end="")
print()
print("  " + "─" * (10 + 10 * len(depths_sweep)))

heatmap = np.zeros((len(lr_sweep_for_heatmap), len(depths_sweep)))
for i, lr in enumerate(lr_sweep_for_heatmap):
    print(f"  lr={lr:<5}", end="")
    for j, d in enumerate(depths_sweep):
        # TODO: Build XGBoost with n_estimators=300, learning_rate=lr,
        # max_depth=d. Fit and compute AUC-PR via average_precision_score.
        m = ____
        m.fit(X_train, y_train, verbose=False)
        y_p = m.predict_proba(X_test)[:, 1]
        auc_pr = ____
        heatmap[i, j] = auc_pr
        print(f"  {auc_pr:>8.4f}", end="")
    print()

best_idx = np.unravel_index(heatmap.argmax(), heatmap.shape)
best_lr = lr_sweep_for_heatmap[best_idx[0]]
best_depth = depths_sweep[best_idx[1]]
print(
    f"\n  Best combo: lr={best_lr}, depth={best_depth} (AUC-PR={heatmap[best_idx]:.4f})"
)


# --- 3c. Early stopping --------------------------------------------------
print("\n  --- Early stopping (XGBoost, 2000-round budget, η=0.05) ---")

# TODO: Build XGBoost with n_estimators=2000, learning_rate=0.05,
# max_depth=6, early_stopping_rounds=50. Fit with eval_set=[(X_test, y_test)]
# and verbose=False.
es_model = ____
____
best_iter_xgb = int(es_model.best_iteration)
es_metrics = evaluate_classifier(y_test, es_model.predict_proba(X_test)[:, 1])
print(f"  XGBoost: best_iteration = {best_iter_xgb}/2000")
print_metrics("XGB+ES", es_metrics)

import lightgbm as lgb

lgb_es = make_lightgbm(n_estimators=2000, learning_rate=0.05, max_depth=6)
lgb_es.fit(
    X_train,
    y_train,
    eval_set=[(X_test, y_test)],
    callbacks=[lgb.early_stopping(50, verbose=False)],
)
best_iter_lgb = int(lgb_es.best_iteration_)
lgb_es_metrics = evaluate_classifier(y_test, lgb_es.predict_proba(X_test)[:, 1])
print(f"  LightGBM: best_iteration = {best_iter_lgb}/2000")
print_metrics("LGB+ES", lgb_es_metrics)


# ── Checkpoint 1 ────────────────────────────────────────────────────────
assert len(lr_curve) == len(learning_rates)
assert heatmap.shape == (len(lr_sweep_for_heatmap), len(depths_sweep))
assert heatmap.max() > 0
assert best_iter_xgb < 2000
assert best_iter_lgb < 2000
print("\n[ok] Checkpoint 1 passed — sweeps + heatmap + early stopping all ran\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 4 — VISUALISE the tuning surface
# ════════════════════════════════════════════════════════════════════════

# 4a. LR curve
lr_fig = go.Figure()
lrs = [row[0] for row in lr_curve]
auc_prs = [row[2] for row in lr_curve]
lr_fig.add_trace(
    go.Scatter(x=lrs, y=auc_prs, mode="lines+markers", name="XGBoost AUC-PR")
)
lr_fig.update_layout(
    title="Learning-Rate Sensitivity — XGBoost on Singapore Credit",
    xaxis_title="learning_rate (η)",
    yaxis_title="AUC-PR",
    xaxis=dict(type="log"),
)
lr_path = OUTPUT_DIR / "ex4_04_lr_curve.html"
lr_fig.write_html(lr_path)
print(f"  Saved: {lr_path}")

# 4b. Heatmap
# TODO: Build a plotly Heatmap with z=heatmap, x=depth labels, y=lr
# labels, colorscale="Viridis".
heat_fig = go.Figure(data=____)
heat_fig.update_layout(
    title="Hyperparameter Interaction Heatmap — XGBoost (300 rounds)",
    xaxis_title="max_depth",
    yaxis_title="learning_rate",
)
heat_path = OUTPUT_DIR / "ex4_04_heatmap.html"
heat_fig.write_html(heat_path)
print(f"  Saved: {heat_path}")

# 4c. Early-stopping comparison
es_fig = go.Figure()
es_fig.add_trace(
    go.Bar(
        name="Fixed 500 rounds",
        x=["XGBoost", "LightGBM"],
        y=[lr_curve[3][2], lgb_es_metrics["auc_pr"]],
    )
)
es_fig.add_trace(
    go.Bar(
        name="Early stopping (2000 budget)",
        x=["XGBoost", "LightGBM"],
        y=[es_metrics["auc_pr"], lgb_es_metrics["auc_pr"]],
    )
)
es_fig.update_layout(
    title="Early Stopping vs Fixed Rounds — AUC-PR on Singapore Credit",
    barmode="group",
    yaxis_title="AUC-PR",
)
es_path = OUTPUT_DIR / "ex4_04_early_stopping.html"
es_fig.write_html(es_path)
print(f"  Saved: {es_path}")


# ── Checkpoint 2 ────────────────────────────────────────────────────────
assert es_metrics["auc_pr"] > 0
print("\n[ok] Checkpoint 2 passed — all tuning visualisations saved\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 5 — APPLY: Grab Credit Pre-Approval Tuning
# ════════════════════════════════════════════════════════════════════════
# SCENARIO: Grab Financial ranks ~25M users for pre-approved credit
# offers. Tuning objective is NOT "maximise AUC-PR" — it is expected
# net revenue per user subject to a default-rate ceiling. That maps to:
#   - learning_rate = 0.03 (calibration smoothness)
#   - max_depth = 6 (from the heatmap sweet spot)
#   - n_estimators = early-stopped with 3000-round budget
#   - reg_lambda = 5.0 (smoother leaf weights → better calibration)
#
# BUSINESS IMPACT: ~S$2.4B pre-approved credit/year. Moving from AUC-PR
# tuning to calibration-aware tuning lifts expected net revenue ~10-15%
# = S$50-80M/year. Exercise 7 formalises this with Bayesian optimisation.


# ════════════════════════════════════════════════════════════════════════
# DESTINATION-FIRST CLOSE — km.diagnose
# ════════════════════════════════════════════════════════════════════════
# This lesson built XGBoost from primitives — early stopping, tree depth
# tuning, business-metric optimisation. The kailash-ml SDK packages the
# diagnostic surface (per-class metrics, class-balance severity,
# confusion matrix, accuracy heuristics) into a single call.
#
# Destination-first: when the journey is internalised, the SDK is one line.

from kailash_ml import diagnose

# `kind="classical_classifier"` dispatches to the sklearn ClassifierMixin
# adapter. XGBClassifier implements the ClassifierMixin interface.
report = diagnose(
    es_model, kind="classical_classifier", data=(X_test, y_test), show=False
)
print()
print(f"  km.diagnose model    : XGBoost (early-stopped at iter {best_iter_xgb})")
print(f"  km.diagnose metrics  : {report.metrics}")
print(f"  km.diagnose severity : {report.severity}")
print()
print("km.diagnose: 1 call -> the same diagnostic surface the lesson body")
print("hand-rolled across the early-stopping sweep. Destination-first:")
print("when the journey is internalised, the SDK is one line.")


# ════════════════════════════════════════════════════════════════════════
# REFLECTION
# ════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  WHAT YOU'VE MASTERED")
print("=" * 70)
print(
    f"""
  [x] Swept learning_rate across 6 values on XGBoost
  [x] Built a 2-D learning_rate × max_depth heatmap
      (best: lr={best_lr}, depth={best_depth})
  [x] Used early stopping to pick n_estimators (XGB={best_iter_xgb},
      LGB={best_iter_lgb})
  [x] Explained why 1-D sweeps lie (interaction effects)
  [x] Mapped tuning knobs to Grab's business objective

  NEXT: Exercise 5 — class imbalance and calibration (SMOTE, focal loss,
  probability calibration) on severely imbalanced data.
"""
)
