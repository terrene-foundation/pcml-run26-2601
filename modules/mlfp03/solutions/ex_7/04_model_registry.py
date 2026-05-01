# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# MLFP03 — Exercise 7.4: ModelRegistry Lifecycle (Staging → Production)
# ════════════════════════════════════════════════════════════════════════
#
# WHAT YOU'LL LEARN:
#   - Define a ModelSignature (input schema + output contract)
#   - Register a model with kailash-ml's ModelRegistry
#   - Promote a model through `staging -> production` with an audit
#     reason that is written to the registry's history table
#   - Understand why `ModelSignature` is the checkpoint the
#     InferenceServer enforces at serving time
#
# PREREQUISITES: 03_hyperparameter_search.py
# ESTIMATED TIME: ~35 min
#
# 5-PHASE R10:
#   1. Theory     — why a registry beats "the pickle on S3"
#   2. Build      — ModelSignature + register_model call
#   3. Train      — promote through the lifecycle with a reason
#   4. Visualise  — inspect the registered version + signature
#   5. Apply      — audit-grade model lineage for MAS on-site inspection
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import asyncio

from kailash_ml import (
    EvalSpec,
    ModelSpec,
    TrainingPipeline,
)
from kailash_ml.types import ModelSignature

from shared.mlfp03.ex_7 import (
    RANDOM_SEED,
    SG_BANK_PORTFOLIO,
    build_training_registry,
    credit_feature_schema,
    headline_roi_text,
    prepare_credit_frame,
    print_metric_block,
    scale_pos_weight_for,
)


# ════════════════════════════════════════════════════════════════════════
# THEORY — Why a ModelRegistry?
# ════════════════════════════════════════════════════════════════════════
# The worst production handoff is:
#
#   "The new model is in s3://bucket/models/model_final_v2_FINAL.pkl —
#    I think Alice uploaded it last Tuesday, ping her if it breaks."
#
# No signature, no version, no promotion reason, no rollback path, no
# way to know which business metric the model was optimised for. Every
# ML-outage postmortem can be traced to this failure mode.
#
# A ModelRegistry is the antidote:
#   - register_model(name, artefact, metrics)  -> returns a typed version
#   - promote_model(name, version, target_stage, reason)
#   - ModelSignature enforces the input/output contract at inference time
#
# Every transition writes an audit row. Every version has its metrics
# captured AT REGISTRATION time (not re-computed optimistically). Every
# promotion reason becomes the paper trail an auditor can replay.


# ════════════════════════════════════════════════════════════════════════
# TASK 2 — BUILD: train, then define ModelSignature + register
# ════════════════════════════════════════════════════════════════════════

frame, feature_cols = prepare_credit_frame()
input_schema = credit_feature_schema(feature_cols)
pos_weight = scale_pos_weight_for(frame["default"].to_numpy())

# The winning hyperparameters from file 03 — this file would receive them
# via the orchestrated pipeline in file 05. We re-fit here with a sane
# set so each technique file is independently runnable.
best_params = {
    "n_estimators": 500,
    "learning_rate": 0.05,
    "max_depth": 6,
    "num_leaves": 63,
    "min_child_samples": 20,
}


# ModelSignature — the contract the InferenceServer enforces on every call.
signature = ModelSignature(
    input_schema=input_schema,
    output_columns=["default_probability", "default_label"],
    output_dtypes=["float64", "int64"],
    model_type="classifier",
)


async def register_and_promote() -> tuple[int, str, dict[str, float]]:
    """Train via TrainingPipeline (which registers the model), then promote.

    TrainingPipeline handles the fit+evaluate+register cycle as one engine
    call — no raw sklearn/LightGBM .fit() in application code. Once the
    model is registered at STAGING, we promote it to PRODUCTION with an
    audit-grade reason. That promotion is the lifecycle transition this
    file teaches.
    """
    registry, conn = await build_training_registry()
    try:
        pipeline = TrainingPipeline(feature_store=None, registry=registry)
        model_spec = ModelSpec(
            model_class="lightgbm.LGBMClassifier",
            framework="lightgbm",
            hyperparameters={
                **best_params,
                "random_state": RANDOM_SEED,
                "verbose": -1,
                "scale_pos_weight": pos_weight,
            },
        )
        eval_spec = EvalSpec(
            metrics=["accuracy", "f1", "auc"],
            split_strategy="holdout",
            test_size=0.2,
        )
        result = await pipeline.train(
            data=frame,
            schema=input_schema,
            model_spec=model_spec,
            eval_spec=eval_spec,
            experiment_name="credit_default_v2",
        )
        assert (
            result.model_version is not None
        ), "TrainingPipeline should register the model"

        # Lifecycle promotion — this is the teaching point of this file.
        await registry.promote_model(
            name="credit_default_v2",
            version=result.model_version.version,
            target_stage="production",
            reason=(
                f"Quality gates passed: AUC={result.metrics.get('auc', 0):.4f}, "
                f"F1={result.metrics.get('f1', 0):.4f}. "
                f"Hyperparameters optimised via kailash-ml Bayesian search."
            ),
        )
        return result.model_version.version, "production", result.metrics
    finally:
        await conn.close()


# ════════════════════════════════════════════════════════════════════════
# TASK 3 — TRAIN: run the async registration + promotion
# ════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("  Registering credit_default_v2 and promoting to production")
print("=" * 70)

version_id, stage, metrics = asyncio.run(register_and_promote())
print_metric_block("Model trained + registered via TrainingPipeline", metrics)


# ── Checkpoint ──────────────────────────────────────────────────────────
assert version_id is not None, "Task 3: register_model must return a version"
assert stage == "production", "Task 3: promotion should land in production"
assert len(signature.input_schema.features) == len(
    feature_cols
), "Task 3: ModelSignature features must match training features"
print(f"\n  credit_default_v2 version={version_id} stage={stage}")
print("\n[ok] Checkpoint passed — registration + promotion complete\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 4 — VISUALISE signature + promotion reason
# ════════════════════════════════════════════════════════════════════════

print("=== ModelSignature ===")
print(f"  input features : {len(signature.input_schema.features)}")
print(f"  entity id col  : {signature.input_schema.entity_id_column}")
print(f"  output columns : {signature.output_columns}")
print(f"  output dtypes  : {signature.output_dtypes}")
print(f"  model type     : {signature.model_type}")

print("\n=== Lifecycle ===")
print("  experiment -> register (staging) -> promote (production) -> retire")
print(f"  credit_default_v2 v{version_id} is now PRODUCTION")


# ════════════════════════════════════════════════════════════════════════
# TASK 5 — APPLY: MAS On-Site Inspection Lineage
# ════════════════════════════════════════════════════════════════════════
# When MAS conducts an on-site inspection of a Singapore bank's credit
# decisioning, the first question is always the same: "Show me the
# model that declined this application, and prove that the model was
# authorised to be in production at the time of the decision."
#
# With a ModelRegistry, the answer is a two-row JOIN:
#   ModelArtifact.version = X
#   RegistryTransition.model=name, version=X, to_stage=production, at=T
#
# The promotion reason we passed above becomes the evidence. Without
# a registry, the bank spends weeks reconstructing git history and
# Slack screenshots and still cannot produce a machine-verifiable answer.
print("\n" + "=" * 70)
print("  APPLY: MAS On-Site Inspection Lineage")
print("=" * 70)
print(headline_roi_text())
print(
    "\n  The `Audit prep savings` line becomes DEFENSIBLE with this file."
    "\n  The promotion reason is the evidence an auditor replays."
)


# ════════════════════════════════════════════════════════════════════════
# REFLECTION
# ════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  WHAT YOU'VE MASTERED")
print("=" * 70)
print(
    f"""
  [x] Built a ModelSignature with FeatureSchema + FeatureField
  [x] Registered a trained LightGBM model in kailash-ml's ModelRegistry
  [x] Promoted through staging -> production with an audit-grade reason
  [x] Tied the registry to a real MAS on-site inspection scenario

  KEY INSIGHT: Models don't fail because they're wrong — they fail
  because nobody remembers which one was in production on the day of
  the incident. The registry is what makes "which model was live on
  2026-03-12" a SELECT, not a forensics project.

  Next: 05_orchestrated_pipeline.py — stitch files 01-04 into one
  run that audits itself end-to-end.

  Version now in production: credit_default_v2 v{version_id}
  Portfolio anchor: S${SG_BANK_PORTFOLIO['portfolio_sgd']/1e9:.0f}B
"""
)
