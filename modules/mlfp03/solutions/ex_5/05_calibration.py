# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# MLFP03 — Exercise 5.5: Calibration (Platt + Isotonic) & Final Comparison
# ════════════════════════════════════════════════════════════════════════
#
# WHAT YOU'LL LEARN:
#   - What calibration means: p_predicted = 0.2 means 20% default in reality
#   - Platt scaling (parametric logistic post-processing)
#   - Isotonic regression (non-parametric step-function post-processing)
#   - How to read a reliability diagram and spot under/over-confidence
#   - How to pick the production-ready strategy from all prior techniques
#
# PREREQUISITES: 01-04 in this directory (probabilities saved under
# outputs/ex5_imbalance/strategy_probabilities.parquet)
# ESTIMATED TIME: ~35 min
#
# 5-PHASE STRUCTURE:
#   Theory   — calibration intuition + why it matters for loan pricing
#   Build    — wrap the cost-sensitive model in CalibratedClassifierCV
#   Train    — fit Platt and Isotonic variants
#   Visualise — reliability diagrams + final comparison table
#   Apply    — Standard Chartered SG risk-based loan pricing
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import lightgbm as lgb
import numpy as np
import plotly.graph_objects as go
import polars as pl
from dotenv import load_dotenv
from sklearn.calibration import CalibratedClassifierCV

from shared.mlfp03.ex_5 import (
    ANNUAL_APPLICATIONS,
    DEFAULT_COSTS,
    OUTPUT_DIR,
    annual_roi,
    load_credit_splits,
    load_strategy_proba,
    metrics_row,
    print_metrics_table,
    print_reliability,
    print_roi,
    reliability_bins,
    save_strategy_proba,
)

load_dotenv()


# ════════════════════════════════════════════════════════════════════════
# THEORY — What calibration means and why loan pricing requires it
# ════════════════════════════════════════════════════════════════════════
# A model output p=0.2 is CALIBRATED if, among all applicants with
# p=0.2, exactly 20% actually default. A gradient booster trained with
# cost-sensitive weights is usually a great RANKER (high AUC-PR) but a
# terrible CALIBRATOR — the output scores compress towards 0 and 1
# because of the weighted loss. Banking requires calibration for:
#
#   - LOAN PRICING: risk-based interest rates are computed as
#     rate = funding_cost + expected_loss(p) + margin. If p is
#     miscalibrated by 2x, the entire pricing curve is wrong.
#
#   - EXPECTED LOSS / ECL (IFRS 9): Stage 2 and Stage 3 provisions
#     are the sum of p*LGD*EAD over the portfolio. Miscalibrated p
#     directly misstates the provision line on the balance sheet.
#
#   - PORTFOLIO SIMULATION: stress tests model the loss distribution
#     by sampling from p. If p is uncalibrated, the 99th percentile
#     tail is wrong, and the CRO under-reserves.
#
# TWO POST-HOC CALIBRATORS:
#
#   PLATT SCALING — fits a 2-parameter logistic function
#       p_cal = 1 / (1 + exp(A * raw + B))
#     Parametric, low variance, works well with small calibration sets.
#     Assumes the miscalibration is sigmoid-shaped.
#
#   ISOTONIC REGRESSION — fits a non-decreasing step function.
#     Non-parametric, higher variance, more flexible. Needs >1000
#     calibration samples to avoid overfitting, but can correct
#     non-monotonic miscalibrations that Platt cannot.
#
# RULE OF THUMB: small calibration set -> Platt; large -> Isotonic;
# always check with a reliability diagram.


# ════════════════════════════════════════════════════════════════════════
# BUILD + TRAIN — wrap the cost-sensitive LightGBM in two calibrators
# ════════════════════════════════════════════════════════════════════════

X_train, y_train, X_test, y_test, pos_rate = load_credit_splits()

scale_weight = (1 - pos_rate) / pos_rate
base_estimator = lgb.LGBMClassifier(
    n_estimators=300,
    scale_pos_weight=scale_weight,
    random_state=42,
    verbose=-1,
)

platt = CalibratedClassifierCV(base_estimator, method="sigmoid", cv=5)
isotonic = CalibratedClassifierCV(base_estimator, method="isotonic", cv=5)

print("\n" + "=" * 70)
print("  Exercise 5.5 — Calibration (Platt + Isotonic)")
print("=" * 70)
print("  Fitting Platt scaling (5-fold CV)...")
platt.fit(X_train, y_train)
print("  Fitting Isotonic regression (5-fold CV)...")
isotonic.fit(X_train, y_train)

y_proba_platt = platt.predict_proba(X_test)[:, 1]
y_proba_iso = isotonic.predict_proba(X_test)[:, 1]
save_strategy_proba("platt_calibrated", y_proba_platt)
save_strategy_proba("isotonic_calibrated", y_proba_iso)


# ── Checkpoint 5 ────────────────────────────────────────────────────────
assert 0 <= y_proba_platt.min() and y_proba_platt.max() <= 1, "Platt out of range"
assert 0 <= y_proba_iso.min() and y_proba_iso.max() <= 1, "Isotonic out of range"
print("[ok] Checkpoint 5 — two calibrated probability vectors saved\n")


# ════════════════════════════════════════════════════════════════════════
# VISUALISE — reliability diagrams + final strategy comparison
# ════════════════════════════════════════════════════════════════════════

# Reliability bins for each variant
bins_baseline = reliability_bins(y_test, load_strategy_proba("baseline"))
bins_cost = reliability_bins(y_test, load_strategy_proba("cost_sensitive_scale"))
bins_platt = reliability_bins(y_test, y_proba_platt)
bins_iso = reliability_bins(y_test, y_proba_iso)

print_reliability("Baseline", bins_baseline)
print_reliability("Cost-sensitive", bins_cost)
print_reliability("Platt", bins_platt)
print_reliability("Isotonic", bins_iso)

# INTERPRETATION: The baseline reliability curve hugs the diagonal in
# the low-probability bins (which is most of the data) because the
# model rarely predicts high default probabilities. Cost-sensitive
# compresses probabilities towards 0.5 — terrible for pricing.
# Platt straightens the curve parametrically. Isotonic matches it
# non-parametrically and usually wins on large calibration sets.

# Final comparison table across every strategy we've trained
strategies = [
    ("Baseline (none)", "baseline"),
    ("SMOTE", "smote"),
    ("Cost-sens (scale)", "cost_sensitive_scale"),
    ("Cost-sens (matrix)", "cost_sensitive_matrix"),
    ("Focal alpha=2.0", "focal_alpha_2.0"),
    ("Cost + Platt", "platt_calibrated"),
    ("Cost + Isotonic", "isotonic_calibrated"),
]
all_rows: list[dict] = []
for display, key in strategies:
    try:
        p = load_strategy_proba(key)
    except (KeyError, FileNotFoundError):
        continue
    all_rows.append(metrics_row(display, y_test, p))

print_metrics_table(all_rows, "FINAL COMPARISON — all imbalance strategies")

final_df = pl.DataFrame(all_rows)
final_df.write_parquet(OUTPUT_DIR / "final_comparison.parquet")
print(f"\n  Saved: {OUTPUT_DIR / 'final_comparison.parquet'}")

# ── Visual: Reliability diagram (calibration curves) ─────────────────────
fig = go.Figure()
fig.add_trace(
    go.Scatter(
        x=[0, 1],
        y=[0, 1],
        mode="lines",
        name="Perfect calibration",
        line=dict(dash="dash", color="#9ca3af", width=1),
    )
)
for label, bins in [
    ("Baseline", bins_baseline),
    ("Cost-sensitive", bins_cost),
    ("Platt", bins_platt),
    ("Isotonic", bins_iso),
]:
    # bins is a polars DataFrame with columns mean_pred / empirical_rate /
    # count (see shared.mlfp03.ex_5.reliability_bins). Filter out empty
    # bins, then extract the two scatter columns.
    nonempty = bins.filter(pl.col("count") > 0)
    mean_pred = nonempty["mean_pred"].to_list()
    frac_pos = nonempty["empirical_rate"].to_list()
    fig.add_trace(go.Scatter(x=mean_pred, y=frac_pos, mode="lines+markers", name=label))
fig.update_layout(
    title="Reliability Diagram: predicted probability vs observed default rate",
    xaxis_title="Mean predicted probability",
    yaxis_title="Fraction of positives (actual default rate)",
    height=500,
    legend=dict(orientation="h", y=-0.2),
)
viz_path = OUTPUT_DIR / "ex5_05_reliability_diagram.html"
fig.write_html(str(viz_path))
print(f"  Saved: {viz_path}")

# ── Visual: Brier score comparison bar chart ─────────────────────────────
fig2 = go.Figure()
fig2.add_trace(
    go.Bar(
        x=[r["strategy"] for r in all_rows],
        y=[r["brier"] for r in all_rows],
        marker_color=[
            (
                "#10b981"
                if r["brier"] == min(rr["brier"] for rr in all_rows)
                else "#6366f1"
            )
            for r in all_rows
        ],
        text=[f"{r['brier']:.4f}" for r in all_rows],
        textposition="outside",
    )
)
fig2.update_layout(
    title="Brier Score Comparison: lower = better calibration (green = best)",
    xaxis_title="Strategy",
    yaxis_title="Brier score",
    height=450,
)
viz_path2 = OUTPUT_DIR / "ex5_05_brier_comparison.html"
fig2.write_html(str(viz_path2))
print(f"  Saved: {viz_path2}")

# Pick the production-ready strategy
best_auc_pr = max(all_rows, key=lambda r: r["auc_pr"])
best_brier = min(all_rows, key=lambda r: r["brier"])
print(
    f"\n  Best AUC-PR: {best_auc_pr['strategy']} "
    f"(AUC-PR={best_auc_pr['auc_pr']:.4f})"
)
print(f"  Best Brier:  {best_brier['strategy']} (Brier={best_brier['brier']:.4f})")


# ════════════════════════════════════════════════════════════════════════
# APPLY — Standard Chartered SG risk-based personal-loan pricing
# ════════════════════════════════════════════════════════════════════════
# SCENARIO: Standard Chartered Singapore prices every personal loan
# individually using a risk-based formula:
#
#     APR = funding_cost + expected_loss(p) * LGD + operating_margin
#
# where expected_loss(p) = p * EAD. If the probability p is
# miscalibrated by even 15%, the entire pricing curve is wrong —
# profitable customers are over-priced and churn to competitors,
# while risky customers are under-priced and blow up the book.
#
# MAS guidance (MAS Notice 1101 on credit risk models) REQUIRES
# demonstrating calibration on a holdout set before any pricing
# model goes live. Platt/Isotonic post-processing is the standard
# industry answer.
#
# Production recipe at SCB-SG:
#   1. Train a strong ranker (LightGBM with scale_pos_weight)
#   2. Post-calibrate with Isotonic (SCB has >50K calibration samples)
#   3. Tune threshold from cost matrix (exercise 5.4)
#   4. Price loans from the calibrated p using the formula above
#   5. Monitor drift quarterly (kailash-ml DriftMonitor)
#
# BUSINESS IMPACT: on a S$500M/year personal-loan book, 15%
# miscalibration translates to roughly S$12M/year in either
# over-provisioning (write-off overstated) or under-pricing
# (margin leakage). Calibration post-processing costs ~1 compute
# hour per retrain. It is the highest-ROI post-hoc step in the
# entire ML lifecycle.

# Annual ROI at the cost-matrix threshold using the calibrated probs
t_star = DEFAULT_COSTS.optimal_threshold
for display, proba in [
    ("Cost-sens (raw)", load_strategy_proba("cost_sensitive_scale")),
    ("Cost + Platt", y_proba_platt),
    ("Cost + Isotonic", y_proba_iso),
]:
    roi = annual_roi(y_test, proba, threshold=t_star, annual_volume=ANNUAL_APPLICATIONS)
    print_roi(f"{display} @ t*={t_star:.4f}", roi)


# ════════════════════════════════════════════════════════════════════════
# PRODUCTION RECOMMENDATION
# ════════════════════════════════════════════════════════════════════════
print(
    """
  Production recipe for Singapore consumer credit (100:1 cost ratio,
  ~12% default rate, ~100K annual applications):

    1. LightGBM with scale_pos_weight = (1 - pos_rate) / pos_rate
       (equivalently: sample_weight = cost matrix lookup)
    2. Calibrate with Isotonic regression (5-fold CV) — OR Platt if
       your calibration set is under 1,000 samples
    3. Tune threshold from cost matrix: t* = cost_FP / (cost_FP + cost_FN)
    4. Report AUC-PR + Brier + annual S$ savings to the risk committee
    5. Monitor drift with kailash-ml DriftMonitor quarterly

  DO NOT use SMOTE in production unless:
    - You have fewer than 500 samples
    - You have fewer than 10 features
    - You have verified it IMPROVES calibration on a holdout set
"""
)


# ════════════════════════════════════════════════════════════════════════
# DESTINATION-FIRST CLOSE — km.diagnose
# ════════════════════════════════════════════════════════════════════════
# This lesson built calibration from primitives — Platt scaling, isotonic
# regression, cost-aware threshold optimisation. The kailash-ml SDK
# packages the diagnostic surface (per-class metrics, class-balance
# severity, confusion matrix) into a single call.
#
# Destination-first: when the journey is internalised, the SDK is one line.

from kailash_ml import diagnose

# `kind="classical_classifier"` dispatches to the sklearn ClassifierMixin
# adapter. CalibratedClassifierCV implements the ClassifierMixin interface.
# Use the isotonic variant — typically the better calibrator for >1k samples.
report = diagnose(
    isotonic, kind="classical_classifier", data=(X_test, y_test), show=False
)
print()
print("  km.diagnose model    : Isotonic-calibrated LightGBM")
print(f"  km.diagnose metrics  : {report.metrics}")
print(f"  km.diagnose severity : {report.severity}")
print()
print("km.diagnose: 1 call -> the same diagnostic surface the lesson body")
print("hand-rolled across Platt + Isotonic. Destination-first: when the")
print("journey is internalised, the SDK is one line.")


# ════════════════════════════════════════════════════════════════════════
# REFLECTION
# ════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  WHAT YOU'VE MASTERED — 5.5 (and Exercise 5 as a whole)")
print("=" * 70)
print(
    """
  [x] Platt scaling: parametric logistic post-calibration
  [x] Isotonic regression: non-parametric step-function post-calibration
  [x] Read reliability diagrams to spot under/over-confidence
  [x] Final comparison across all seven strategies (baseline, SMOTE,
      cost-sens x2, focal, Platt, Isotonic)
  [x] Translated the winner into annual S$ savings at SCB pricing
  [x] Documented the production recipe for Singapore consumer credit

  WHOLE-EXERCISE INSIGHT: The winning strategy on financial tabular
  data is almost never "the clever paper trick." It's cost-sensitive
  learning + post-hoc calibration + cost-matrix threshold tuning.
  Simple, production-grade, auditable.

  NEXT: Exercise 6 adds SHAP interpretability — required for MAS
  model risk governance and the EU AI Act right-to-explanation.
"""
)
