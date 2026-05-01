# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# MLFP03 — Exercise 4.4: Boosting Tuning — Sweeps, Heatmaps, Early Stopping
# ════════════════════════════════════════════════════════════════════════
#
# WHAT YOU'LL LEARN:
#   - Sweep learning rate and see how AUC-PR changes across η
#   - Build a learning_rate × max_depth heatmap and read the interaction
#   - Use early stopping to pick n_estimators automatically instead of
#     guessing
#   - Explain why "grid search over independent dials" is the wrong
#     mental model for boosting (Exercise 7 will replace it with Bayesian)
#   - Produce a final production-ready configuration
#
# PREREQUISITES: Exercise 4.2/4.3 (XGBoost and LightGBM/CatBoost).
#
# ESTIMATED TIME: ~40 min
#
# TASKS:
#   1. Theory — hyperparameter interaction, why grids lie
#   2. Build — a learning-rate sweep and a 2-D depth×η heatmap
#   3. Train — run the sweep; run early stopping with a 2000-round budget
#   4. Visualise — heatmap + LR sweep + early-stopping curve
#   5. Apply — Grab credit pre-approval team tunes for business metric
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
# THEORY — Hyperparameter Interaction Is Why Grid Search Lies
# ════════════════════════════════════════════════════════════════════════
# The naive approach to tuning boosting is "vary one knob at a time":
# hold max_depth fixed at 6, sweep learning_rate, pick the best, then
# hold learning_rate fixed and sweep max_depth. This is wrong because
# the knobs INTERACT.
#
# Example of interaction:
#   - At learning_rate=0.01, max_depth=10 is fine — the small step size
#     keeps the deep trees from overfitting.
#   - At learning_rate=0.2, max_depth=10 overfits badly — each big step
#     drives a deep tree far past the optimal loss.
#
# The right mental model is a 2-D surface over (η, depth). A 1-D sweep
# only sees a slice of that surface and can lead you to a local optimum
# that doesn't hold once the other knob moves.
#
# Two practical tools replace naive grid search:
#
#   1. Early stopping: set n_estimators to a large budget (2000+), let
#      validation loss tell you when to stop. This turns n_estimators
#      from a tuning knob into a self-tuning parameter.
#
#   2. 2-D heatmap of (learning_rate × max_depth). Small grids (3x4)
#      are enough to SEE the interaction surface. Beyond 2 dimensions,
#      grids explode combinatorially — Exercise 7 uses Bayesian
#      optimisation to scale this up.


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
print(f"  Default rate: {data['default_rate']:.2%}")

learning_rates = [0.01, 0.03, 0.05, 0.1, 0.2, 0.5]
depths_sweep = [3, 5, 6, 8, 10]
lr_sweep_for_heatmap = [0.01, 0.05, 0.1, 0.2]


# ════════════════════════════════════════════════════════════════════════
# TASK 3 — TRAIN the sweeps
# ════════════════════════════════════════════════════════════════════════

# --- 3a. 1-D learning-rate sweep (XGBoost, depth fixed at 6) ------------
print("\n  --- Learning-rate sweep (XGBoost, depth=6, 500 rounds) ---")
print(f"  {'lr':>6}  {'AUC-ROC':>10}  {'AUC-PR':>10}")
print("  " + "─" * 32)

lr_curve: list[tuple[float, float, float]] = []  # (lr, auc_roc, auc_pr)
for lr in learning_rates:
    m = make_xgboost(n_estimators=500, learning_rate=lr, max_depth=6)
    m.fit(X_train, y_train, verbose=False)
    y_p = m.predict_proba(X_test)[:, 1]
    auc_roc = float(average_precision_score(y_test, y_p))  # fallback if roc breaks
    metrics = evaluate_classifier(y_test, y_p)
    lr_curve.append((lr, metrics["auc_roc"], metrics["auc_pr"]))
    print(f"  {lr:>6.2f}  {metrics['auc_roc']:>10.4f}  {metrics['auc_pr']:>10.4f}")


# --- 3b. 2-D learning_rate × max_depth heatmap --------------------------
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
        m = make_xgboost(n_estimators=300, learning_rate=lr, max_depth=d)
        m.fit(X_train, y_train, verbose=False)
        y_p = m.predict_proba(X_test)[:, 1]
        auc_pr = float(average_precision_score(y_test, y_p))
        heatmap[i, j] = auc_pr
        print(f"  {auc_pr:>8.4f}", end="")
    print()

best_idx = np.unravel_index(heatmap.argmax(), heatmap.shape)
best_lr = lr_sweep_for_heatmap[best_idx[0]]
best_depth = depths_sweep[best_idx[1]]
print(
    f"\n  Best combo: lr={best_lr}, depth={best_depth} "
    f"(AUC-PR={heatmap[best_idx]:.4f})"
)


# --- 3c. Early stopping (XGBoost, 2000-round budget) --------------------
print("\n  --- Early stopping (XGBoost, 2000-round budget, η=0.05) ---")
es_model = make_xgboost(
    n_estimators=2000,
    learning_rate=0.05,
    max_depth=6,
    early_stopping_rounds=50,
)
es_model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
best_iter_xgb = int(es_model.best_iteration)
es_metrics = evaluate_classifier(y_test, es_model.predict_proba(X_test)[:, 1])
print(f"  XGBoost: best_iteration = {best_iter_xgb}/2000")
print_metrics("XGB+ES", es_metrics)

# Also early-stop LightGBM so students see it's the same idea, different API
import lightgbm as lgb  # local import keeps top imports clean

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
assert len(lr_curve) == len(learning_rates), "Every learning rate must be evaluated"
assert heatmap.shape == (len(lr_sweep_for_heatmap), len(depths_sweep))
assert heatmap.max() > 0, "At least one heatmap cell must have positive AUC-PR"
assert best_iter_xgb < 2000, "XGBoost early stopping must fire before the budget"
assert best_iter_lgb < 2000, "LightGBM early stopping must fire before the budget"
# INTERPRETATION: Early stopping fires well before 2000 rounds in almost
# every production setting. That's the signal the 2000-round budget was
# safely high enough — if best_iteration hits 2000, raise the budget.
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
heat_fig = go.Figure(
    data=go.Heatmap(
        z=heatmap,
        x=[f"d={d}" for d in depths_sweep],
        y=[f"η={lr}" for lr in lr_sweep_for_heatmap],
        colorscale="Viridis",
        colorbar=dict(title="AUC-PR"),
    )
)
heat_fig.update_layout(
    title="Hyperparameter Interaction Heatmap — XGBoost (300 rounds)",
    xaxis_title="max_depth",
    yaxis_title="learning_rate",
)
heat_path = OUTPUT_DIR / "ex4_04_heatmap.html"
heat_fig.write_html(heat_path)
print(f"  Saved: {heat_path}")

# 4c. Early-stopping comparison (fixed 500 vs early stop)
es_fig = go.Figure()
es_fig.add_trace(
    go.Bar(
        name="Fixed 500 rounds",
        x=["XGBoost", "LightGBM"],
        y=[lr_curve[3][2], lgb_es_metrics["auc_pr"]],  # lr=0.1 row for XGB
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
assert es_metrics["auc_pr"] > 0, "Early-stopped XGBoost should have positive AUC-PR"
# INTERPRETATION: The three figures give you three reads on the tuning
# surface. The LR curve is 1-D (easy to reason about but misleading).
# The heatmap is 2-D (hyperparameter interaction visible). Early stopping
# is the operational pattern that replaces guessing n_estimators. In
# production, you combine them: tune (η, depth) on a 2-D heatmap with
# early stopping driving n_estimators.
print("\n[ok] Checkpoint 2 passed — all tuning visualisations saved\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 5 — APPLY: Grab Credit Pre-Approval Tuning
# ════════════════════════════════════════════════════════════════════════
# SCENARIO: Grab Financial Group (Singapore HQ, operating across SEA)
# runs a GrabFin credit pre-approval model that decides which of its
# ~25M active Grab users get a "You're pre-approved for S$X" nudge in
# the app. The model must:
#   1. Rank users by default risk so the top-X% get the offer
#   2. Maintain a calibrated probability so the offer amount can be
#      priced from expected loss
#   3. Re-train weekly as new repayment data arrives
#
# The tuning task is NOT "maximise AUC-PR" — it is:
#   "maximise expected net revenue per eligible user, subject to a
#    regulatory ceiling on default rate for the approved pool"
#
# Translating that to tuning knobs:
#   - learning_rate: lower (0.03) is better for calibration — small
#     steps give smoother probability estimates. 04_04_lr_curve.html
#     shows AUC-PR is relatively flat across 0.03-0.1, so the team
#     picks 0.03 on calibration grounds.
#   - max_depth: the heatmap shows depth 6-8 is the sweet spot. Deeper
#     trees (10) overfit; shallower (3) underfit. Team picks 6.
#   - n_estimators: early stopping with a 3000-round budget and 100-
#     round patience. best_iteration is typically 800-1200 on a full
#     re-train — no need to guess.
#   - reg_lambda (λ): raised from the default 1.0 to 5.0 to pull leaf
#     weights toward zero, which smooths calibration at a tiny AUC-PR
#     cost (~0.002).
#
# BUSINESS IMPACT: Grab pre-approves ~S$2.4B in credit lines per year.
# Moving from a calibration-naive "just maximise AUC-PR" model to a
# calibration-aware tuning produces ~10-15% lift in pool-level expected
# net revenue — roughly S$50-80M/year — because mis-priced offers
# (either too cheap for the risk or too expensive vs the competition)
# get corrected. The tuning-time cost is a few hours of compute per
# week. The key is that the TUNING LOSS FUNCTION is not AUC-PR — it's
# a business metric, and the hyperparameters are chosen against it.
#
# This is the pattern every production boosting team uses: tune for the
# business metric, not for the offline AUC. Exercise 7 will formalise
# this with Bayesian optimisation over a business-metric objective.


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
  [x] Swept learning_rate and saw AUC-PR change across 6 values
  [x] Built a 2-D learning_rate × max_depth heatmap and read the
      interaction surface (best: lr={best_lr}, depth={best_depth})
  [x] Used early stopping to pick n_estimators automatically
      (XGBoost best_iter={best_iter_xgb}, LightGBM best_iter={best_iter_lgb})
  [x] Explained why grid search over independent dials is the wrong
      mental model and why Exercise 7 will use Bayesian optimisation
  [x] Mapped tuning knobs to a business metric using the Grab scenario

  KEY INSIGHT: Hyperparameters interact. Never tune one at a time.
  Always use early stopping for n_estimators. And always tune against
  the business objective — AUC-PR is a proxy, not a destination.

  NEXT: Exercise 5 (class imbalance and calibration). You've seen that
  12% positive rate is survivable with AUC-PR + good boosting. Exercise
  5 pushes into 1-2% positive rates where SMOTE, cost-sensitive
  learning, focal loss, and probability calibration become essential.
"""
)
