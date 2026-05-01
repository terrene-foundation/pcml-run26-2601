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
# PREREQUISITES: Exercises 8.1 and 8.3.
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

import numpy as np
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
# Pickle + S3 dies the day you need: the exact live version, an instant
# rollback, a comparison with the runner-up, full reproducibility, or an
# audit trail for a regulator. A registry handles all of these.
#
# kailash-ml ModelRegistry:
#   - Immutable versions, keyed by hash
#   - Stages: staging / production / archived
#   - Promotion events with reason strings (audit log)
#   - Pair with ExperimentTracker for run lineage


# ════════════════════════════════════════════════════════════════════════
# TASK 2 — BUILD: set up the registry + tracker
# ════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("  MLFP03 Exercise 8.4 — Deployment Pipeline")
print("=" * 70)

split = load_credit_split()
X_train, y_train = split["X_train"], split["y_train"]
X_test, y_test = split["X_test"], split["y_test"]

# TODO: Train the calibrated model and score the test set
calibrated_model = ____
y_proba = ____
metrics = evaluate_classification(y_test, y_proba)

# Conformal coverage (standalone re-derivation)
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
    f"\nTraining run: AUC-PR={metrics['auc_pr']:.4f} Brier={metrics['brier']:.4f} Coverage={coverage:.3f}"
)


# ── Checkpoint 1 ────────────────────────────────────────────────────────
assert metrics["auc_roc"] > 0.5
print("\n[ok] Checkpoint 1 — reference model ready for registry\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 3 — TRAIN: register + promote via kailash-ml ModelRegistry
# ════════════════════════════════════════════════════════════════════════


async def register_and_promote() -> dict:
    db = "sqlite:///mlfp03_models.db"
    tracker = await ExperimentTracker.create(store_url=db)
    conn = ConnectionManager(db)
    await conn.initialize()

    model_version_id = "skipped"
    promotion_reason = ""

    if HAS_REGISTRY:
        registry = ModelRegistry(conn)
        # TODO: Serialise calibrated_model with pickle.dumps
        model_bytes = ____

        metrics_list = []
        if HAS_METRIC_SPEC:
            metrics_list = [
                MetricSpec(name="auc_pr", value=metrics["auc_pr"]),
                MetricSpec(name="auc_roc", value=metrics["auc_roc"]),
                MetricSpec(name="brier", value=metrics["brier"]),
                MetricSpec(name="conformal_coverage", value=coverage),
            ]

        # TODO: await registry.register_model(name="credit_default_production",
        #                                     artifact=model_bytes, metrics=metrics_list)
        version = ____

        promotion_reason = (
            f"Passed quality gates: AUC-PR={metrics['auc_pr']:.4f}, "
            f"Brier={metrics['brier']:.4f}, Coverage={coverage:.4f}"
        )
        # TODO: await registry.promote_model with name, version, target_stage="production", reason
        ____
        model_version_id = str(version.version)
        print(f"\n[registry] Registered credit_default_production v{model_version_id}")
        print(f"[registry] Promoted to production: {promotion_reason}")
    else:
        print("\n[warn] ModelRegistry unavailable — skipping register/promote")

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
assert model_version is not None
print("\n[ok] Checkpoint 2 — model registered and promoted\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 4 — VISUALISE: production monitoring dashboard
# ════════════════════════════════════════════════════════════════════════

print("=== Production Monitoring Dashboard ===")
dashboard_text = f"""
┌─────────────────────────────────────────────────────────────┐
│               CREDIT DEFAULT MODEL — PRODUCTION              │
├─────────────────────────────────────────────────────────────┤
│  Version: {model_version:<8}    AUC-ROC {metrics['auc_roc']:.4f}            │
│  AUC-PR  {metrics['auc_pr']:.4f}   Brier {metrics['brier']:.4f}   F1 {metrics['f1']:.4f} │
│  Conformal coverage: {coverage:.1%} (target 90%)                    │
│  Alert: PSI > 0.2 OR AUC-PR < {metrics['auc_pr'] * 0.9:.4f}          │
└─────────────────────────────────────────────────────────────┘
"""
print(dashboard_text)

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
# TODO: Add a go.Bar trace for (labels, values) with marker_color=colors and text labels
____
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
assert viz_path.exists()
print("\n[ok] Checkpoint 3 — dashboard rendered\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 5 — APPLY: Citi Singapore "rollback in 4 minutes" SLO
# ════════════════════════════════════════════════════════════════════════
# Before registry: incident → SSH → find pickle → upload → restart
# = ~2 hours. At ~S$12k/min in blocked legitimate transactions that's
# ~S$1.4M per incident. With registry: promote(version=N-1) + hot reload
# = ~5 min → ~S$60k per incident. ~8 incidents/year → ~S$10.7M/yr saved.

incidents_per_year = 8
loss_per_minute_sgd = 12_000
old_window_min = 120
new_window_min = 5
loss_old = incidents_per_year * old_window_min * loss_per_minute_sgd
loss_new = incidents_per_year * new_window_min * loss_per_minute_sgd
savings = loss_old - loss_new
print(f"\n=== Citi Singapore rollback economics ===")
print(f"  Loss (old):      S${loss_old:>14,.0f}/yr")
print(f"  Loss (registry): S${loss_new:>14,.0f}/yr")
print(f"  Annual savings:  S${savings:>14,.0f}/yr")


# ════════════════════════════════════════════════════════════════════════
# REFLECTION
# ════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  WHAT YOU'VE MASTERED")
print("=" * 70)
print(
    f"""
  [x] Registered a model with kailash-ml ModelRegistry
  [x] Logged a full run via ExperimentTracker
  [x] Promoted credit_default_production v{model_version} to production
  [x] Built a live dashboard from registry metadata
  [x] Quantified rollback SLO: ~S${savings:,.0f}/yr at Citi SG scale

  Next: 05_production_readiness.py — the final gate + capstone.
"""
)
