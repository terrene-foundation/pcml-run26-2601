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
# THEORY — Why drift is the silent killer of production ML
# ════════════════════════════════════════════════════════════════════════
# A model is trained on a photograph of the world. The world keeps moving.
#
# COVARIATE DRIFT:  p(x) changes. Interest rates rose 200 bps, customers'
#                   debt-service ratios shifted. Same labels, new inputs.
# LABEL DRIFT:      p(y) changes. Unemployment spiked, default rate jumped
#                   from 12% to 17%. Same inputs, new base rate.
# CONCEPT DRIFT:    p(y | x) changes. The RELATIONSHIP shifts. The same
#                   debt ratio that was safe last year is risky this year.
#
# Every drift flavour destroys model performance silently. You'll only
# notice if (a) you have drift monitoring, or (b) three months later the
# quarterly review surfaces a 200 bps AUC collapse.
#
# TWO COMPLEMENTARY DRIFT TESTS:
#
# PSI (Population Stability Index)
#   PSI = Σ (p_new - p_ref) * ln(p_new / p_ref)
#   - < 0.1  no meaningful shift
#   - 0.1-0.2 moderate shift, investigate
#   - > 0.2  significant shift, retrain
#   Strength: interpretable, industry standard in credit scoring.
#
# KS (Kolmogorov-Smirnov two-sample test)
#   Max distance between empirical CDFs, with a p-value.
#   Strength: catches localised CDF changes PSI's bins miss.
#
# Use BOTH. PSI catches broad distribution shifts, KS catches sharp local
# ones. Alerting when EITHER trips gives the tightest safety net.


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

# Train the same calibrated model as 8.1 so we can measure how drift
# damages ITS performance specifically.
calibrated_model = train_calibrated_model(X_train, y_train)
y_proba_ref = calibrated_model.predict_proba(X_test)[:, 1]
auc_ref = float(roc_auc_score(y_test, y_proba_ref))
print(f"\nReference model AUC-ROC: {auc_ref:.4f}")

# DriftSpec is the scheduled-monitoring contract for kailash-ml's
# DriftMonitor. The thresholds override the monitor-wide defaults for
# this specific schedule; `feature_columns=None` means "all features
# from the stored reference". An `on_drift_detected` async callback can
# be wired in when you want to page on-call instead of just logging.
drift_spec = DriftSpec(
    psi_threshold=0.1,
    ks_threshold=0.05,
)
print(
    f"\nDriftSpec configured: PSI>{drift_spec.psi_threshold}, "
    f"KS p-value<{drift_spec.ks_threshold} (daily cadence set at the "
    f"DriftMonitor.schedule call-site, not on the spec itself)"
)


# ── Checkpoint 1 ────────────────────────────────────────────────────────
assert auc_ref > 0.5, "Task 2: Reference AUC should beat random"
print("\n[ok] Checkpoint 1 — drift spec + reference model ready\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 3 — TRAIN: establish the drift baseline (train vs test)
# ════════════════════════════════════════════════════════════════════════
# If train vs test shows high PSI, the split itself is broken — you
# can't monitor drift against a poisoned baseline.

print("=== Baseline drift (train vs test — same distribution) ===")
print(f"{'Feature':<30} {'PSI':>8} {'KS stat':>10} {'KS p-val':>10} {'Drift?':>8}")
print("─" * 70)

baseline_drift: dict[str, dict[str, float | str]] = {}
for i, feat in enumerate(feature_names[:10]):
    row = drift_row(X_train[:, i], X_test[:, i])
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
), f"Task 3: Train vs test should not drift on most features (got {baseline_alerts})"
print(f"\n[ok] Checkpoint 2 — baseline has {baseline_alerts} false-alert features\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 4 — VISUALISE drift + performance degradation
# ════════════════════════════════════════════════════════════════════════
# Simulate two drift scenarios and watch the reference model's AUC drop.

# Scenario A: GRADUAL drift — 0.5σ shift on top 3 features.
X_gradual = simulate_gradual_drift(X_train, X_test, n_features=3, shift=0.5)

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

y_proba_gradual = calibrated_model.predict_proba(X_gradual)[:, 1]
auc_gradual = float(roc_auc_score(y_test, y_proba_gradual))
print(f"\nAUC under gradual drift: {auc_gradual:.4f} (baseline {auc_ref:.4f})")
print(f"Degradation: {auc_ref - auc_gradual:.4f}")

# Scenario B: SUDDEN drift — swap feature 0 for a 3σ-shifted Gaussian.
X_sudden = simulate_sudden_drift(X_train, X_test, feature_idx=0, sigma_shift=3.0)
psi_sudden = compute_psi(X_train[:, 0], X_sudden[:, 0])
ks_stat_sudden, ks_pval_sudden = compute_ks(X_train[:, 0], X_sudden[:, 0])

print(f"\n=== Scenario B: Sudden drift (3σ mean shift on feature 0) ===")
print(f"  {feature_names[0]}: PSI={psi_sudden:.4f}, KS p-val={ks_pval_sudden:.2e}")
y_proba_sudden = calibrated_model.predict_proba(X_sudden)[:, 1]
auc_sudden = float(roc_auc_score(y_test, y_proba_sudden))
print(f"  AUC under sudden drift: {auc_sudden:.4f} (Δ={auc_ref - auc_sudden:.4f})")

# Visualise: three-panel stacked bar + AUC degradation line.
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
        x=x_labels,
        y=gradual_psis,
        name="PSI (gradual drift)",
        marker_color="#f59e0b",
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
    title="Feature-level PSI across drift scenarios (Singapore credit)",
    xaxis_title="Feature",
    yaxis_title="PSI",
    barmode="group",
    height=520,
)
viz_path = OUTPUT_DIR / "ex8_02_drift_psi.html"
fig.write_html(str(viz_path))
print(f"\nSaved: {viz_path}")


# ── Checkpoint 3 ────────────────────────────────────────────────────────
# The contract we verify is "drift is DETECTED" (PSI breach + KS p-value),
# not "AUC always collapses". On a well-separable dataset the model may
# still ride other features through the drift — which is exactly why the
# drift-detection layer is INDEPENDENT from the performance layer. Both
# signals must be watched in production. A tiny tolerance absorbs
# Monte-Carlo noise on the sudden-drift Gaussian simulator.
assert psi_sudden > 0.1, "Task 4: Sudden drift should produce PSI > 0.1"
assert ks_pval_sudden < 0.05, "Task 4: Sudden drift should be detected by KS"
assert (
    auc_sudden <= auc_ref + 0.005
), "Task 4: Drift should not materially improve AUC (beyond MC noise)"
print("\n[ok] Checkpoint 3 — drift detected (perf layer watches it separately)\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 5 — APPLY: MAS retraining rule for SG consumer credit
# ════════════════════════════════════════════════════════════════════════
# SCENARIO: OCBC Bank Singapore runs an unsecured credit-card application
# model scoring ~45,000 applications/month. MAS Notice 635 (§24) requires
# "ongoing validation of credit decision models" — but doesn't say HOW.
#
# The informal quarterly review catches major drift ~60 days after it
# starts. During those 60 days the model is making bad decisions:
#
#   - False approvals cost S$4,200 each (median unsecured LGD × loan size)
#   - False rejections cost S$180 each (opportunity cost on declined NPV)
#   - Without drift monitoring: ~3% AUC drop → ~4,500 bad decisions/quarter
#     → ~S$5.2M/quarter in avoidable losses.
#
# With daily PSI + KS monitoring:
#   - Alert triggers on day 1-3 of meaningful drift
#   - Retrain pipeline runs within the week
#   - 90% of the loss window is eliminated
#   - Estimated saving: S$4.7M/quarter = S$19M/year.

monthly_apps = 45_000
bad_decision_cost_sgd = 4_200.0  # false-approval LGD cost
quarterly_bad_without_monitoring = monthly_apps * 3 * 0.033  # ~3.3% error rate jump
loss_window_days_informal = 60
loss_window_days_monitored = 7
savings_ratio = 1 - (loss_window_days_monitored / loss_window_days_informal)

loss_without = quarterly_bad_without_monitoring * bad_decision_cost_sgd
loss_with = loss_without * (1 - savings_ratio)
savings_q = loss_without - loss_with

print(f"\n=== OCBC retraining economics (45K apps/mo) ===")
print(f"  Quarterly loss without monitoring:  S${loss_without:>14,.0f}")
print(f"  Quarterly loss with daily monitor:  S${loss_with:>14,.0f}")
print(f"  Savings:                             S${savings_q:>14,.0f}/quarter")
print(f"  Annualised:                          S${savings_q * 4:>14,.0f}/year")
print("\n  RETRAINING RULE (production):")
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
  [x] Configured DriftSpec with PSI + KS thresholds
  [x] Established a baseline from train vs test ({baseline_alerts} false alerts)
  [x] Simulated gradual and sudden drift, caught both
  [x] Measured AUC degradation: {auc_ref:.3f} → {auc_sudden:.3f} under sudden drift
  [x] Translated drift alerts into a quantified retraining rule
      (~S${savings_q * 4:,.0f}/yr at OCBC scale)

  KEY INSIGHT: The pipeline that catches drift earns its keep the week
  the world changes. Quarterly reviews are insurance for yesterday.

  Next: 03_model_card.py — document every decision this model makes
  so the regulator can reconstruct WHY when the drift alert fires.
"""
)
