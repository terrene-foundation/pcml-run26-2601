# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# MLFP03 — Exercise 7.4: ModelRegistry Lifecycle via TrainingPipeline
# ════════════════════════════════════════════════════════════════════════
#
# WHAT YOU'LL LEARN:
#   - Drive fit + evaluate + register as ONE TrainingPipeline.train() call
#   - Define a ModelSignature (input schema + output contract)
#   - Promote a registered model through `staging -> production` with an
#     audit reason that is written to the registry's history table
#   - Understand why TrainingPipeline owns the lifecycle AND registration
#     so raw lightgbm.fit() stays out of application code
#
# PREREQUISITES: 03_hyperparameter_search.py
# ESTIMATED TIME: ~35 min
#
# 5-PHASE R10:
#   1. Theory     — why a registry beats "the pickle on S3"
#   2. Build      — ModelSignature + ModelSpec + EvalSpec
#   3. Train      — TrainingPipeline.train() + promote_model
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
# THEORY — Why TrainingPipeline + ModelRegistry together?
# ════════════════════════════════════════════════════════════════════════
# Every ML-outage postmortem traces to "the pickle on S3 with no
# signature, no version, no promotion reason." The registry is the
# antidote, and TrainingPipeline is the engine that puts models INTO
# the registry automatically: `pipeline.train()` runs fit + evaluate
# AND calls `registry.register_model(...)` for you. All you have to do
# is promote the result through the lifecycle.


# ════════════════════════════════════════════════════════════════════════
# TASK 2 — BUILD: frame, schema, signature, winning hyperparameters
# ════════════════════════════════════════════════════════════════════════

frame, feature_cols = prepare_credit_frame()
input_schema = credit_feature_schema(feature_cols)
pos_weight = scale_pos_weight_for(frame["default"].to_numpy())

# The winning hyperparameters from file 03 — each technique file is
# independently runnable, so we re-use a sane set here.
best_params = {
    "n_estimators": 500,
    "learning_rate": 0.05,
    "max_depth": 6,
    "num_leaves": 63,
    "min_child_samples": 20,
}


# TODO: build the ModelSignature — outputs are default_probability + default_label
# Hint: output_columns=["default_probability", "default_label"]
#       output_dtypes=["float64", "int64"], model_type="classifier"
signature = ModelSignature(
    input_schema=input_schema,
    output_columns=____,
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
        # TODO: create a TrainingPipeline bound to the registry (no feature_store)
        # Hint: TrainingPipeline(feature_store=None, registry=registry)
        pipeline = ____

        # TODO: build a ModelSpec from best_params + pos_weight + RANDOM_SEED
        # Hint: framework="lightgbm", model_class="lightgbm.LGBMClassifier"
        model_spec = ModelSpec(
            model_class=____,
            framework=____,
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

        # TODO: run pipeline.train — it fits, evaluates, AND registers.
        # Hint: experiment_name="credit_default_v2"
        result = await pipeline.train(
            data=frame,
            schema=input_schema,
            model_spec=model_spec,
            eval_spec=eval_spec,
            experiment_name=____,
        )
        assert (
            result.model_version is not None
        ), "TrainingPipeline should register the model"

        # TODO: promote the registered model from staging to production with
        # an audit-grade reason. This is the pedagogical teaching point.
        # Hint: await registry.promote_model(name=..., version=..., target_stage="production", reason="...")
        await registry.promote_model(
            name="credit_default_v2",
            version=result.model_version.version,
            target_stage=____,
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
# TASK 3 — TRAIN: run the async training + registration + promotion
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
# With TrainingPipeline writing to a ModelRegistry, the answer is a
# two-row JOIN:
#   ModelArtifact.version = X
#   RegistryTransition.model=name, version=X, to_stage=production, at=T
#
# The promotion reason we passed above becomes the evidence an auditor
# replays. Without a registry, the bank spends weeks reconstructing git
# history and Slack screenshots.
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
  [x] Used kailash-ml TrainingPipeline for fit + evaluate + register in ONE call
  [x] Built a ModelSignature with FeatureSchema + FeatureField
  [x] Promoted through staging -> production with an audit-grade reason
  [x] Tied the registry to a real MAS on-site inspection scenario

  KEY INSIGHT: Models don't fail because they're wrong — they fail
  because nobody remembers which one was in production on the day of
  the incident. TrainingPipeline + ModelRegistry is what makes
  "which model was live on 2026-03-12" a SELECT, not a forensics project.

  Next: 05_orchestrated_pipeline.py — stitch files 01-04 into one
  run that audits itself end-to-end.

  Version now in production: credit_default_v2 v{version_id}
  Portfolio anchor: S${SG_BANK_PORTFOLIO['portfolio_sgd']/1e9:.0f}B
"""
)
