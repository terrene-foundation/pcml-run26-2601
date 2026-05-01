# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# MLFP03 — Exercise 5.5: Calibration + Final Comparison
# ════════════════════════════════════════════════════════════════════════
#
# WHAT YOU'LL LEARN:
#   - What calibration means (p=0.2 should mean 20% default in reality)
#   - Platt scaling (logistic post-processing, small-data friendly)
#   - Isotonic regression (non-parametric step function, large-data friendly)
#   - How to read a reliability diagram
#   - Final comparison across every imbalance strategy
#
# PREREQUISITES: 01-04 in this directory
# ESTIMATED TIME: ~35 min
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import lightgbm as lgb
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
# THEORY — Calibration matters for loan pricing and IFRS 9 provisions
# ════════════════════════════════════════════════════════════════════════
# A cost-sensitive gradient booster ranks well but miscalibrates — its
# outputs compress toward 0 and 1. Banking needs calibrated p because:
#   - APR = funding_cost + p * LGD * EAD + margin  (risk-based pricing)
#   - Stage 2/3 ECL = sum(p * LGD * EAD)           (IFRS 9 provisioning)
# Platt = 2-parameter logistic. Isotonic = non-parametric step function.


# ════════════════════════════════════════════════════════════════════════
# BUILD + TRAIN — two calibrators
# ════════════════════════════════════════════════════════════════════════

X_train, y_train, X_test, y_test, pos_rate = load_credit_splits()

scale_weight = (1 - pos_rate) / pos_rate
base_estimator = lgb.LGBMClassifier(
    n_estimators=300,
    scale_pos_weight=scale_weight,
    random_state=42,
    verbose=-1,
)

# TODO: Wrap base_estimator in CalibratedClassifierCV with method="sigmoid" and cv=5
# Hint: this is Platt scaling
platt = ____

# TODO: Wrap base_estimator in CalibratedClassifierCV with method="isotonic" and cv=5
isotonic = ____

print("\n" + "=" * 70)
print("  Exercise 5.5 — Calibration (Platt + Isotonic)")
print("=" * 70)

# TODO: Fit both calibrators on X_train, y_train
____
____

y_proba_platt = platt.predict_proba(X_test)[:, 1]
y_proba_iso = isotonic.predict_proba(X_test)[:, 1]
save_strategy_proba("platt_calibrated", y_proba_platt)
save_strategy_proba("isotonic_calibrated", y_proba_iso)


# ── Checkpoint 5 ────────────────────────────────────────────────────────
assert 0 <= y_proba_platt.min() and y_proba_platt.max() <= 1, "Platt out of range"
assert 0 <= y_proba_iso.min() and y_proba_iso.max() <= 1, "Isotonic out of range"
print("[ok] Checkpoint 5 — two calibrated probability vectors saved\n")


# ════════════════════════════════════════════════════════════════════════
# VISUALISE — reliability diagrams + final comparison
# ════════════════════════════════════════════════════════════════════════

bins_baseline = reliability_bins(y_test, load_strategy_proba("baseline"))
bins_cost = reliability_bins(y_test, load_strategy_proba("cost_sensitive_scale"))
bins_platt = reliability_bins(y_test, y_proba_platt)
bins_iso = reliability_bins(y_test, y_proba_iso)

print_reliability("Baseline", bins_baseline)
print_reliability("Cost-sensitive", bins_cost)
print_reliability("Platt", bins_platt)
print_reliability("Isotonic", bins_iso)

# Final comparison across every strategy
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
    # TODO: Append a metrics_row for (display, y_test, p)
    all_rows.append(____)

print_metrics_table(all_rows, "FINAL COMPARISON — all imbalance strategies")

pl.DataFrame(all_rows).write_parquet(OUTPUT_DIR / "final_comparison.parquet")

best_auc_pr = max(all_rows, key=lambda r: r["auc_pr"])
best_brier = min(all_rows, key=lambda r: r["brier"])
print(f"\n  Best AUC-PR: {best_auc_pr['strategy']}")
print(f"  Best Brier:  {best_brier['strategy']}")


# ════════════════════════════════════════════════════════════════════════
# APPLY — Standard Chartered SG risk-based pricing
# ════════════════════════════════════════════════════════════════════════
# SCB-SG prices loans with: APR = funding_cost + p*LGD*EAD + margin.
# A 15% miscalibration on a S$500M book is ~S$12M/year in leakage.
# MAS Notice 1101 requires calibration on a holdout before go-live.

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
  Production recipe for Singapore consumer credit:
    1. LightGBM with scale_pos_weight from class prior
    2. Calibrate with Isotonic (5-fold CV) OR Platt (small data)
    3. Tune threshold from cost matrix: t* = c_FP / (c_FP + c_FN)
    4. Report AUC-PR + Brier + annual S$ savings
    5. Monitor drift with kailash-ml DriftMonitor quarterly
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
print("  WHAT YOU'VE MASTERED — 5.5 (and Exercise 5 overall)")
print("=" * 70)
print(
    """
  [x] Platt + Isotonic calibration wrappers (CalibratedClassifierCV)
  [x] Reliability diagrams for all seven strategies
  [x] Final ranked comparison and the production recipe
  [x] SCB-SG pricing implication in annual S$

  WHOLE-EXERCISE INSIGHT: The winner is almost never the clever paper
  trick. It's cost-sensitive loss + post-hoc calibration + cost-matrix
  threshold tuning. Simple, production-grade, auditable.

  NEXT: Exercise 6 — SHAP interpretability for model risk governance.
"""
)
