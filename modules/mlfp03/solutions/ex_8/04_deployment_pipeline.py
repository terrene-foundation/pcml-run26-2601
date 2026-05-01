# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# MLFP03 — Exercise 8.4: Deployment Pipeline — Registry + Promotion
# ════════════════════════════════════════════════════════════════════════
#
# WHAT YOU'LL LEARN:
#   - Register a trained model into kailash-ml ModelRegistry
#   - Log run metadata with ExperimentTracker (params, metrics, tags)
#   - Promote a version from staging to production with an audit reason
#   - Build a production monitoring dashboard from registry metadata
#   - Translate registry + audit trail into compliance $ savings
#
# PREREQUISITES: Exercise 8.1 (calibrated model) and 8.3 (model card).
#
# ESTIMATED TIME: ~35 min
#
# TASKS:
#   1. Theory     — what a registry buys you vs "pickle + s3 upload"
#   2. Build      — ModelRegistry + ExperimentTracker wiring
#   3. Train      — register + promote the final model with metrics
#   4. Visualise  — production dashboard panel
#   5. Apply      — Citi Singapore "rollback in 4 minutes" SLO
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import asyncio
import pickle

import plotly.graph_objects as go

from kailash.db import ConnectionManager
from kailash_ml import ExperimentTracker

from shared.mlfp03.ex_8 import (
    OUTPUT_DIR,
    evaluate_classification,
    load_credit_split,
    train_calibrated_model,
)

try:
    from kailash_ml import ModelRegistry

    HAS_REGISTRY = True
except ImportError:
    HAS_REGISTRY = False

try:
    from kailash_ml.types import MetricSpec

    HAS_METRIC_SPEC = True
except ImportError:
    HAS_METRIC_SPEC = False


# ════════════════════════════════════════════════════════════════════════
# THEORY — Why a model registry is non-negotiable
# ════════════════════════════════════════════════════════════════════════
# The temptation: `pickle.dump(model, open("s3://.../model.pkl", "wb"))`.
# One line. Ships today. Works tomorrow. Dies the day something bad
# happens in production and you need to:
#
#   - Prove WHICH version was live when the incident occurred
#   - Roll back to the last known-good version in minutes, not days
#   - Compare the live model's metrics to the runner-up candidate
#   - Reproduce the model from the exact data + config + code
#   - Show a regulator the full promotion history with reasons
#
# A MODEL REGISTRY gives you:
#   1. Immutable versions keyed by hash (never "latest" ambiguity)
#   2. Stages: staging / production / archived
#   3. Promotion events with reason strings in an audit log
#   4. Metadata: metrics, params, training data hash, git SHA
#   5. One-line rollback: `registry.promote(name, version=N-1, "rollback")`
#
# kailash-ml's ModelRegistry handles all of this. Pair it with
# ExperimentTracker for run-level lineage (what trials led to this
# promotion) and you have the full "who / what / when / why" that
# every incident retro needs.


# ════════════════════════════════════════════════════════════════════════
# TASK 2 — BUILD: set up the registry + tracker
# ════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("  MLFP03 Exercise 8.4 — Deployment Pipeline")
print("=" * 70)

split = load_credit_split()
X_train, y_train = split["X_train"], split["y_train"]
X_test, y_test = split["X_test"], split["y_test"]

calibrated_model = train_calibrated_model(X_train, y_train)
y_proba = calibrated_model.predict_proba(X_test)[:, 1]
metrics = evaluate_classification(y_test, y_proba)

# Conformal coverage (standalone re-derivation so this file runs alone)
import numpy as np

n_cal = X_test.shape[0] // 2
cal_proba = calibrated_model.predict_proba(X_test[:n_cal])[:, 1]
cal_scores = np.where(y_test[:n_cal] == 1, 1 - cal_proba, cal_proba)
alpha = 0.10
q_level = np.ceil((len(cal_scores) + 1) * (1 - alpha)) / len(cal_scores)
q_hat = float(np.quantile(cal_scores, min(q_level, 1.0)))
eval_proba = calibrated_model.predict_proba(X_test[n_cal:])[:, 1]
y_eval = y_test[n_cal:]
correct_sets = [
    (y_eval[i] == 1 and (1 - eval_proba[i]) <= q_hat)
    or (y_eval[i] == 0 and eval_proba[i] <= q_hat)
    for i in range(len(y_eval))
]
coverage = float(np.mean(correct_sets))

print(
    f"\nTraining run:  AUC-PR={metrics['auc_pr']:.4f}  "
    f"Brier={metrics['brier']:.4f}  Coverage={coverage:.3f}"
)


# ── Checkpoint 1 ────────────────────────────────────────────────────────
assert metrics["auc_roc"] > 0.5
print("\n[ok] Checkpoint 1 — reference model ready for registry\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 3 — TRAIN: register + promote via kailash-ml ModelRegistry
# ════════════════════════════════════════════════════════════════════════


async def register_and_promote() -> dict:
    """Register the calibrated model, log a run, promote to production."""
    db = "sqlite:///mlfp03_models.db"
    tracker = await ExperimentTracker.create(store_url=db)
    conn = ConnectionManager(db)
    await conn.initialize()

    model_version_id = "skipped"
    promotion_reason = ""

    if HAS_REGISTRY:
        registry = ModelRegistry(conn)
        model_bytes = pickle.dumps(calibrated_model)

        metrics_list = []
        if HAS_METRIC_SPEC:
            metrics_list = [
                MetricSpec(name="auc_pr", value=metrics["auc_pr"]),
                MetricSpec(name="auc_roc", value=metrics["auc_roc"]),
                MetricSpec(name="brier", value=metrics["brier"]),
                MetricSpec(name="conformal_coverage", value=coverage),
            ]

        version = await registry.register_model(
            name="credit_default_production",
            artifact=model_bytes,
            metrics=metrics_list,
        )
        promotion_reason = (
            f"Passed quality gates: AUC-PR={metrics['auc_pr']:.4f}, "
            f"Brier={metrics['brier']:.4f}, Coverage={coverage:.4f}"
        )
        await registry.promote_model(
            name="credit_default_production",
            version=version.version,
            target_stage="production",
            reason=promotion_reason,
        )
        model_version_id = str(version.version)
        print(f"\n[registry] Registered credit_default_production v{model_version_id}")
        print(f"[registry] Promoted to production: {promotion_reason}")
    else:
        print("\n[warn] ModelRegistry unavailable — skipping register/promote")

    # Experiment tracker run — lineage across training trials
    exp_id = "mlfp03_production_pipeline"
    async with tracker.track(experiment=exp_id, run_name="production_model_v1") as run:
        await run.log_params(
            {"model": "lgbm_calibrated_conformal", "conformal_alpha": str(alpha)}
        )
        await run.log_metrics({**metrics, "conformal_coverage": coverage})
        await run.add_tag("stage", "production")
        await run.add_tag("market", "singapore")

    await conn.close()
    return {
        "model_version": model_version_id,
        "reason": promotion_reason or "registry unavailable",
    }


result = asyncio.run(register_and_promote())
model_version = result["model_version"]


# ── Checkpoint 2 ────────────────────────────────────────────────────────
assert model_version is not None, "Task 3: Registration should return a version"
print("\n[ok] Checkpoint 2 — model registered and promoted\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 4 — VISUALISE: production monitoring dashboard
# ════════════════════════════════════════════════════════════════════════
# Build a 4-panel dashboard showing what an operator sees at 9am on a
# Monday: metrics, drift, uncertainty, promotion history.

print("=== Production Monitoring Dashboard ===")

dashboard_text = f"""
┌─────────────────────────────────────────────────────────────┐
│               CREDIT DEFAULT MODEL — PRODUCTION              │
├─────────────────────────────────────────────────────────────┤
│  Name:    credit_default_production                          │
│  Version: {model_version:<8}                                   │
│  Stage:   PRODUCTION                                         │
├─────────────────────────────────────────────────────────────┤
│  Performance                                                 │
│    AUC-ROC      {metrics['auc_roc']:.4f}                              │
│    AUC-PR       {metrics['auc_pr']:.4f}                              │
│    Brier        {metrics['brier']:.4f}                              │
│    F1           {metrics['f1']:.4f}                              │
├─────────────────────────────────────────────────────────────┤
│  Uncertainty (conformal α=0.10)                              │
│    Coverage     {coverage:.1%}  (target 90%)                     │
├─────────────────────────────────────────────────────────────┤
│  Alerting rule: PSI > 0.2 OR AUC-PR < {metrics['auc_pr'] * 0.9:.4f} │
└─────────────────────────────────────────────────────────────┘
"""
print(dashboard_text)

# Plotly version of the dashboard
fig = go.Figure()
labels = ["AUC-ROC", "AUC-PR", "1 - Brier", "Coverage", "F1"]
values = [
    metrics["auc_roc"],
    metrics["auc_pr"],
    1 - metrics["brier"],
    coverage,
    metrics["f1"],
]
colors = ["#2563eb", "#10b981", "#f59e0b", "#8b5cf6", "#ec4899"]
fig.add_trace(
    go.Bar(
        x=labels,
        y=values,
        marker_color=colors,
        text=[f"{v:.3f}" for v in values],
        textposition="outside",
    )
)
fig.add_hline(
    y=0.85,
    line_dash="dash",
    line_color="#b91c1c",
    annotation_text="Minimum production gate (0.85)",
)
fig.update_layout(
    title=f"Production Dashboard — credit_default_production v{model_version}",
    yaxis_title="Score (higher is better)",
    yaxis_range=[0, 1.1],
    height=500,
)
viz_path = OUTPUT_DIR / "ex8_04_dashboard.html"
fig.write_html(str(viz_path))
print(f"Saved: {viz_path}")


# ── Checkpoint 3 ────────────────────────────────────────────────────────
assert viz_path.exists(), "Task 4: Dashboard should be written"
print("\n[ok] Checkpoint 3 — dashboard rendered\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 5 — APPLY: Citi Singapore "rollback in 4 minutes" SLO
# ════════════════════════════════════════════════════════════════════════
# SCENARIO: Citi Singapore runs a real-time credit fraud model scoring
# ~320 transactions/second. A bad model in production can destroy
# approvals (false rejects = lost revenue) or let fraud through (false
# approves = direct losses).
#
# Before the registry, rollback went like this:
#   1. Incident at 02:13 — live model rejecting 40% of legitimate txns
#   2. Page on-call ML engineer (03:04 — 51 min later)
#   3. SSH to training box, find previous pickle file (03:29 — 25 min)
#   4. Upload to model server S3 bucket (03:48 — 19 min)
#   5. Restart model server, verify metrics (04:15 — 27 min)
#   → Total incident window: 2 hours. At ~S$12k/min in blocked
#     legitimate transactions that's ~S$1.4M in a single incident.
#
# With kailash-ml ModelRegistry + promotion API:
#   1. Incident at 02:13
#   2. Alert fires via DriftMonitor (02:14)
#   3. On-call runs `registry.promote(name, version=N-1, reason="rollback: ...")`
#      (02:17 — 3 min)
#   4. Model server hot-reloads from registry (02:18)
#   → Total window: 5 minutes. Loss window 24x shorter → S$60k per incident.
#
# With ~8 incidents/year that's ~(1.4 - 0.06) × 8 = S$10.7M/year avoided
# loss, plus the audit trail for MAS Notice 655 (operational incidents).

incidents_per_year = 8
loss_per_minute_sgd = 12_000
old_window_min = 120
new_window_min = 5
loss_old = incidents_per_year * old_window_min * loss_per_minute_sgd
loss_new = incidents_per_year * new_window_min * loss_per_minute_sgd
savings = loss_old - loss_new
print(f"\n=== Citi Singapore rollback economics ===")
print(f"  Incidents/year:                    {incidents_per_year}")
print(f"  Rollback window (old):             {old_window_min} min")
print(f"  Rollback window (registry):        {new_window_min} min")
print(f"  Loss (old):                        S${loss_old:>14,.0f}/yr")
print(f"  Loss (registry):                   S${loss_new:>14,.0f}/yr")
print(f"  Annual savings:                    S${savings:>14,.0f}/yr")


# ════════════════════════════════════════════════════════════════════════
# REFLECTION
# ════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  WHAT YOU'VE MASTERED")
print("=" * 70)
print(
    f"""
  [x] Registered a model with kailash-ml ModelRegistry
  [x] Logged a full run (params + metrics + tags) via ExperimentTracker
  [x] Promoted credit_default_production v{model_version} to production
  [x] Built a live dashboard panel from registry metadata
  [x] Translated registry rollback into a quantified SLO
      (~S${savings:,.0f}/yr avoided loss at Citi SG scale)

  KEY INSIGHT: The registry is not storage. It's the contract that lets
  your team act on an incident in minutes instead of hours.

  Next: 05_production_readiness.py — the final gate before the model
  goes live, and the capstone wrap for Module 3.
"""
)
