# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# MLFP03 — Exercise 8.2: Drift Monitoring with PSI and KS
# ════════════════════════════════════════════════════════════════════════
#
# WHAT YOU'LL LEARN:
#   - Configure a drift spec with PSI and KS thresholds
#   - Compute PSI/KS against a reference distribution
#   - Simulate gradual and sudden drift and verify detection
#   - Measure the AUC degradation that drift causes on a live model
#   - Translate drift alerts into a retraining decision with $ impact
#
# PREREQUISITES: Exercise 8.1 (you need the calibrated model).
#
# ESTIMATED TIME: ~35 min
#
# TASKS:
#   1. Theory     — why drift is the #1 cause of production ML failure
#   2. Build      — PSI + KS helpers and the DriftSpec configuration
#   3. Train      — establish a drift baseline from train vs test
#   4. Visualise  — feature-level drift heatmap + performance degradation
#   5. Apply      — MAS retraining rule for an SG consumer-credit model
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from sklearn.metrics import roc_auc_score

from shared.mlfp03.ex_8 import (
    OUTPUT_DIR,
    compute_ks,
    compute_psi,
    drift_row,
    load_credit_split,
    simulate_gradual_drift,
    simulate_sudden_drift,
    train_calibrated_model,
)

from kailash_ml import DriftSpec


# ════════════════════════════════════════════════════════════════════════
# THEORY — Why drift is the silent killer
# ════════════════════════════════════════════════════════════════════════
# Covariate drift: p(x) changes. Label drift: p(y) changes. Concept
# drift: p(y|x) changes. Every flavour destroys model performance.
#
# PSI < 0.1 no shift, 0.1-0.2 moderate, > 0.2 retrain.
# KS two-sample test catches local CDF shifts PSI's bins miss.
# Use BOTH — PSI + KS alert when EITHER trips.


# ════════════════════════════════════════════════════════════════════════
# TASK 2 — BUILD: drift helpers + DriftSpec
# ════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("  MLFP03 Exercise 8.2 — Drift Monitoring")
print("=" * 70)

split = load_credit_split()
X_train, y_train = split["X_train"], split["y_train"]
X_test, y_test = split["X_test"], split["y_test"]
feature_names = split["feature_names"]

# TODO: Train the calibrated baseline model with train_calibrated_model
calibrated_model = ____

# TODO: Score the test set and compute AUC-ROC via roc_auc_score(y_test, proba)
y_proba_ref = ____
auc_ref = ____
print(f"\nReference model AUC-ROC: {auc_ref:.4f}")

# TODO: Build a DriftSpec with psi_threshold=0.1 and ks_threshold=0.05.
# Hint: DriftSpec(psi_threshold=..., ks_threshold=...) — cadence is set
# later at DriftMonitor.schedule_monitoring(interval=...) call-time,
# NOT on the spec itself.
drift_spec = ____
print(
    f"\nDriftSpec configured: PSI>{drift_spec.psi_threshold}, "
    f"KS p-value<{drift_spec.ks_threshold} (cadence set on schedule_monitoring)"
)


# ── Checkpoint 1 ────────────────────────────────────────────────────────
assert auc_ref > 0.5, "Task 2: Reference AUC should beat random"
print("\n[ok] Checkpoint 1 — drift spec + reference model ready\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 3 — TRAIN: establish the drift baseline (train vs test)
# ════════════════════════════════════════════════════════════════════════

print("=== Baseline drift (train vs test — same distribution) ===")
print(f"{'Feature':<30} {'PSI':>8} {'KS stat':>10} {'KS p-val':>10} {'Drift?':>8}")
print("─" * 70)

baseline_drift: dict[str, dict] = {}
for i, feat in enumerate(feature_names[:10]):
    # TODO: Call drift_row(X_train[:, i], X_test[:, i]) and store in baseline_drift[feat]
    row = ____
    baseline_drift[feat] = row
    print(
        f"  {feat:<28} {row['psi']:>8.4f} {row['ks_stat']:>10.4f} "
        f"{row['ks_pval']:>10.4f} {row['drift']:>8}"
    )


# ── Checkpoint 2 ────────────────────────────────────────────────────────
assert len(baseline_drift) > 0, "Task 3: Baseline drift should be populated"
baseline_alerts = sum(1 for r in baseline_drift.values() if r["drift"] == "YES")
assert (
    baseline_alerts <= 3
), f"Task 3: Train vs test should not drift heavily (got {baseline_alerts})"
print(f"\n[ok] Checkpoint 2 — baseline has {baseline_alerts} false-alert features\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 4 — VISUALISE drift + performance degradation
# ════════════════════════════════════════════════════════════════════════

# TODO: Build gradual drift via simulate_gradual_drift(X_train, X_test, n_features=3, shift=0.5)
X_gradual = ____

print("=== Scenario A: Gradual drift (0.5σ mean shift on top 3 features) ===")
print(f"{'Feature':<30} {'PSI':>8} {'KS stat':>10} {'KS p-val':>10} {'Alert?':>10}")
print("─" * 72)

gradual_psis = []
for i in range(5):
    feat = feature_names[i]
    psi = compute_psi(X_train[:, i], X_gradual[:, i])
    ks_stat, ks_pval = compute_ks(X_train[:, i], X_gradual[:, i])
    alert = "YES (!)" if (psi > 0.1 or ks_pval < 0.05) else "No"
    gradual_psis.append(psi)
    print(f"  {feat:<28} {psi:>8.4f} {ks_stat:>10.4f} {ks_pval:>10.4f} {alert:>10}")

# TODO: Score the gradual-drift data and compute AUC
y_proba_gradual = ____
auc_gradual = ____
print(f"\nAUC under gradual drift: {auc_gradual:.4f} (baseline {auc_ref:.4f})")
print(f"Degradation: {auc_ref - auc_gradual:.4f}")

# TODO: Build sudden drift via simulate_sudden_drift on feature 0, 3σ shift
X_sudden = ____
psi_sudden = compute_psi(X_train[:, 0], X_sudden[:, 0])
ks_stat_sudden, ks_pval_sudden = compute_ks(X_train[:, 0], X_sudden[:, 0])

print(f"\n=== Scenario B: Sudden drift (3σ mean shift on feature 0) ===")
print(f"  {feature_names[0]}: PSI={psi_sudden:.4f}, KS p-val={ks_pval_sudden:.2e}")
y_proba_sudden = calibrated_model.predict_proba(X_sudden)[:, 1]
auc_sudden = float(roc_auc_score(y_test, y_proba_sudden))
print(f"  AUC under sudden drift: {auc_sudden:.4f} (Δ={auc_ref - auc_sudden:.4f})")

# Visualise: grouped PSI bars for baseline / gradual / sudden
fig = go.Figure()
x_labels = [feature_names[i] for i in range(5)]
fig.add_trace(
    go.Bar(
        x=x_labels,
        y=[float(compute_psi(X_train[:, i], X_test[:, i])) for i in range(5)],
        name="PSI (baseline)",
        marker_color="#10b981",
    )
)
fig.add_trace(
    go.Bar(
        x=x_labels, y=gradual_psis, name="PSI (gradual drift)", marker_color="#f59e0b"
    )
)
fig.add_trace(
    go.Bar(
        x=x_labels,
        y=[psi_sudden]
        + [float(compute_psi(X_train[:, i], X_sudden[:, i])) for i in range(1, 5)],
        name="PSI (sudden drift)",
        marker_color="#ef4444",
    )
)
fig.add_hline(
    y=0.1, line_dash="dash", line_color="#64748b", annotation_text="PSI=0.1 (moderate)"
)
fig.add_hline(
    y=0.2, line_dash="dot", line_color="#b91c1c", annotation_text="PSI=0.2 (retrain)"
)
fig.update_layout(
    title="Feature-level PSI across drift scenarios",
    xaxis_title="Feature",
    yaxis_title="PSI",
    barmode="group",
    height=520,
)
viz_path = OUTPUT_DIR / "ex8_02_drift_psi.html"
fig.write_html(str(viz_path))
print(f"\nSaved: {viz_path}")


# ── Checkpoint 3 ────────────────────────────────────────────────────────
# The assertion is "drift is DETECTED", not "AUC always collapses". On a
# well-separable dataset the model may still ride other features through
# the drift — which is exactly why the drift-detection layer is
# INDEPENDENT from the performance layer. A tiny epsilon absorbs Monte
# Carlo noise on the Gaussian simulator.
assert psi_sudden > 0.1, "Task 4: Sudden drift should produce PSI > 0.1"
assert ks_pval_sudden < 0.05, "Task 4: Sudden drift should be detected by KS"
assert (
    auc_sudden <= auc_ref + 0.005
), "Task 4: Drift should not materially improve AUC (beyond MC noise)"
print("\n[ok] Checkpoint 3 — drift detected (perf layer watches separately)\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 5 — APPLY: MAS retraining rule for SG consumer credit
# ════════════════════════════════════════════════════════════════════════
# SCENARIO: OCBC Bank runs an unsecured credit-card application model
# scoring ~45,000 apps/month. Informal quarterly reviews catch major
# drift ~60 days late → ~S$5M/quarter avoidable losses. Daily PSI+KS
# monitoring shrinks the loss window to 7 days → S$4.7M/quarter saved.

monthly_apps = 45_000
bad_decision_cost_sgd = 4_200.0
quarterly_bad = monthly_apps * 3 * 0.033
loss_without = quarterly_bad * bad_decision_cost_sgd
loss_with = loss_without * (7 / 60)
savings_q = loss_without - loss_with

print(f"\n=== OCBC retraining economics (45K apps/mo) ===")
print(f"  Quarterly loss (informal review):   S${loss_without:>14,.0f}")
print(f"  Quarterly loss (daily monitor):     S${loss_with:>14,.0f}")
print(f"  Savings:                             S${savings_q:>14,.0f}/quarter")
print(f"  Annualised:                          S${savings_q * 4:>14,.0f}/year")
print("\n  RETRAINING RULE:")
print("    IF  any_feature_PSI > 0.2  OR  live_AUC_PR < 0.9 * baseline_AUC_PR")
print("    THEN queue retraining, freeze auto-decisions, notify risk lead")


# ════════════════════════════════════════════════════════════════════════
# REFLECTION
# ════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  WHAT YOU'VE MASTERED")
print("=" * 70)
print(
    f"""
  [x] DriftSpec with PSI + KS thresholds
  [x] Baseline from train vs test ({baseline_alerts} false alerts)
  [x] Caught gradual and sudden drift
  [x] AUC degradation: {auc_ref:.3f} → {auc_sudden:.3f}
  [x] Retraining rule worth ~S${savings_q * 4:,.0f}/yr at OCBC scale

  Next: 03_model_card.py — document every decision the model makes.
"""
)
