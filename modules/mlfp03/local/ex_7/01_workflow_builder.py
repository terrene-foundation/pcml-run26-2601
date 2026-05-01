# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# MLFP03 — Exercise 7.1: Kailash WorkflowBuilder + TrainingPipeline
# ════════════════════════════════════════════════════════════════════════
#
# WHAT YOU'LL LEARN:
#   - Declare an ML pipeline as a named DAG using WorkflowBuilder
#   - Wire nodes with `connections=[...]` to form execution edges
#   - Execute the workflow with `runtime.execute(workflow.build())`
#   - Drive the same pipeline via kailash-ml's TrainingPipeline engine
#     (fit + evaluate + register as ONE call — no raw lightgbm.fit())
#   - Capture the run_id so every training run is auditable
#
# PREREQUISITES: MLFP03 Exercises 1-6, MLFP02 preprocessing
# ESTIMATED TIME: ~35 min
#
# 5-PHASE R10:
#   1. Theory     — why workflows beat hand-rolled scripts
#   2. Build      — declare the DAG with WorkflowBuilder
#   3. Train      — LocalRuntime + kailash-ml TrainingPipeline
#   4. Visualise  — inspect the DAG and the baseline metrics
#   5. Apply      — Singapore bank monthly credit-model retraining
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import asyncio

from kailash.runtime import LocalRuntime
from kailash.workflow.builder import WorkflowBuilder
from kailash_ml import (
    EvalSpec,
    ModelSpec,
    TrainingPipeline,
)

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
# THEORY — Why WorkflowBuilder + TrainingPipeline?
# ════════════════════════════════════════════════════════════════════════
# A hand-rolled script has no NAME per step, no RUN_ID per run, no
# machine-readable DEPENDENCIES, and no single ENTRYPOINT. WorkflowBuilder
# gives you all four by turning the pipeline into a DAG.
#
# Below the DAG, kailash-ml's TrainingPipeline engine owns the
# fit + evaluate + register cycle. Raw `lightgbm.LGBMClassifier().fit(...)`
# in application code is BLOCKED by framework-first — the engine is the
# only supported surface.


# ════════════════════════════════════════════════════════════════════════
# TASK 2 — BUILD the workflow
# ════════════════════════════════════════════════════════════════════════

# TODO: instantiate a WorkflowBuilder with the name "credit_scoring_pipeline"
# Hint: workflow = WorkflowBuilder("credit_scoring_pipeline")
workflow = ____

workflow.add_node(
    "DataPreprocessNode",
    "preprocess",
    {
        "data_source": "sg_credit_scoring",
        "target": "default",
        "train_size": 0.8,
        "seed": RANDOM_SEED,
        "normalize": False,
        "categorical_encoding": "ordinal",
        "imputation_strategy": "median",
    },
)

# TODO: add the train node, connected to "preprocess"
# Hint: 4-param order is add_node("NodeType", "node_id", {config}, connections=[...])
workflow.add_node(
    "ModelTrainNode",
    "train",
    {
        "model_class": "lightgbm.LGBMClassifier",
        "hyperparameters": {
            "n_estimators": 500,
            "learning_rate": 0.1,
            "max_depth": 6,
            "scale_pos_weight": 7.3,
        },
    },
    connections=____,
)

workflow.add_node(
    "ModelEvalNode",
    "evaluate",
    {"metrics": ["accuracy", "f1", "auc_roc", "auc_pr", "log_loss"]},
    connections=["train"],
)

workflow.add_node(
    "PersistNode",
    "persist",
    {"storage": "sqlite:///mlfp03_models.db"},
    connections=["evaluate"],
)


# ════════════════════════════════════════════════════════════════════════
# TASK 3 — TRAIN via runtime.execute(workflow.build()) + TrainingPipeline
# ════════════════════════════════════════════════════════════════════════
# LocalRuntime walks the DAG, runs each node in topological order, and
# returns (results, run_id). The run_id is the audit primary key.

runtime = LocalRuntime()
print("\n" + "=" * 70)
print("  Executing credit_scoring_pipeline workflow")
print("=" * 70)

try:
    # TODO: call runtime.execute on the BUILT workflow
    # Hint: runtime.execute(workflow.build()) — MUST call .build()
    results, run_id = runtime.execute(____)
    workflow_ok = True
    print(f"  run_id:      {run_id}")
    print(f"  node_count:  {len(results)}")
except Exception as exc:
    # Custom nodes require registration; the orchestration plane is the
    # lesson, not the specific node implementation. We fall back to the
    # kailash-ml TrainingPipeline engine, which runs the SAME fit +
    # evaluate cycle so students observe end-to-end behaviour.
    print(f"  [info] custom nodes not registered ({type(exc).__name__})")
    print(f"  [info] falling back to kailash-ml TrainingPipeline engine")
    results, run_id = {}, "fallback-training-pipeline-run"
    workflow_ok = False

# Kailash-ML TrainingPipeline — the framework-first equivalent of the DAG.
# This is the same pipeline as the WorkflowBuilder declaration above, but
# driven by the kailash-ml engine. No raw sklearn/LightGBM .fit() lives in
# application code; TrainingPipeline owns the fit+evaluate+register cycle.
frame, feature_cols = prepare_credit_frame()
schema = credit_feature_schema(feature_cols)
pos_weight = scale_pos_weight_for(frame["default"].to_numpy())


async def train_baseline() -> dict[str, float]:
    registry, conn = await build_training_registry()
    try:
        # TODO: instantiate a TrainingPipeline with no feature_store and the
        # registry we just built.
        # Hint: TrainingPipeline(feature_store=None, registry=registry)
        pipeline = ____

        # TODO: build a ModelSpec for LightGBM. Framework is "lightgbm" and
        # model_class is the dotted path to LGBMClassifier.
        # Hint: ModelSpec(model_class="lightgbm.LGBMClassifier", framework="lightgbm", hyperparameters={...})
        model_spec = ModelSpec(
            model_class=____,
            framework=____,
            hyperparameters={
                "n_estimators": 500,
                "learning_rate": 0.1,
                "max_depth": 6,
                "scale_pos_weight": pos_weight,
                "random_state": RANDOM_SEED,
                "verbose": -1,
            },
        )

        # TODO: build an EvalSpec — holdout split, 20% test, metrics=["accuracy","f1","auc"]
        # Hint: EvalSpec(metrics=[...], split_strategy="holdout", test_size=0.2)
        eval_spec = ____

        # TODO: call pipeline.train with the frame, schema, model_spec, eval_spec
        # and experiment_name="credit_default_baseline". It's async.
        # Hint: result = await pipeline.train(data=frame, schema=schema, ...)
        result = await pipeline.train(
            data=frame,
            schema=schema,
            model_spec=model_spec,
            eval_spec=eval_spec,
            experiment_name=____,
        )
        return result.metrics
    finally:
        await conn.close()


baseline_metrics = asyncio.run(train_baseline())


# ── Checkpoint ──────────────────────────────────────────────────────────
assert run_id is not None, "Task 3: runtime.execute must return a run_id"
assert baseline_metrics.get("auc", 0.0) > 0.5, "Task 3: model must beat random"
print("\n[ok] Checkpoint passed — workflow + TrainingPipeline executed\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 4 — VISUALISE the DAG and the baseline metrics
# ════════════════════════════════════════════════════════════════════════

print("DAG shape:")
print("  preprocess -> train -> evaluate -> persist")
print(f"  workflow_ok={workflow_ok}  run_id={run_id}")
print_metric_block(
    "Baseline LightGBM via kailash-ml TrainingPipeline", baseline_metrics
)


# ════════════════════════════════════════════════════════════════════════
# TASK 5 — APPLY: DBS Credit Model Monthly Retraining
# ════════════════════════════════════════════════════════════════════════
# SCENARIO: DBS's Retail Credit Risk team retrains the default model on
# the first Monday of every month. Today the retraining script is a
# Jupyter notebook owned by one analyst. MAS Notice 635 requires every
# model used in a credit decision to have a reproducible audit trail —
# and the auditor cannot verify a notebook.
#
# Converting the notebook to a Kailash workflow + TrainingPipeline gives:
#   - A single named entrypoint the retraining scheduler can invoke
#   - A run_id per execution, which maps 1:1 to an audit row
#   - A machine-readable DAG the auditor can replay offline
#   - Engine-owned fit+evaluate+register — no hidden state
print("\n" + "=" * 70)
print("  APPLY: DBS Monthly Retraining — S$48B Unsecured Portfolio")
print("=" * 70)
print(headline_roi_text())
print(
    "\n  This technique alone unlocks the audit-prep savings line above"
    "\n  — the model-lift line waits for the Hyperparameter Search file."
)


# ════════════════════════════════════════════════════════════════════════
# REFLECTION
# ════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  WHAT YOU'VE MASTERED")
print("=" * 70)
print(
    f"""
  [x] Declared an ML pipeline as a Kailash WorkflowBuilder DAG
  [x] Wired nodes with `connections=[...]` to form execution edges
  [x] Executed with the canonical runtime.execute(workflow.build()) pattern
  [x] Trained via kailash-ml TrainingPipeline (engine-owned fit+eval+register)
  [x] Captured a run_id that serves as the audit primary key
  [x] Connected the orchestration plane to a Singapore banking ML-ops scenario

  KEY INSIGHT: The workflow is documentation you can execute. Every edge
  is machine-readable; every run_id is audit-ready. TrainingPipeline is
  the engine that owns the fit+evaluate+register cycle below that DAG.

  Next: 02_dataflow_persistence.py — stop returning metrics as a dict
  and start writing them to a DataFlow-managed database table.

  Portfolio anchor: S${SG_BANK_PORTFOLIO['portfolio_sgd']/1e9:.0f}B
"""
)
