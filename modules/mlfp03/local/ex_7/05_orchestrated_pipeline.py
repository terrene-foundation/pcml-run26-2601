# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# MLFP03 — Exercise 7.5: Orchestrated Pipeline (Workflow + DataFlow +
#                        Hyperparameter Search + Model Registry)
# ════════════════════════════════════════════════════════════════════════
#
# WHAT YOU'LL LEARN:
#   - Chain every previous technique (workflow, persistence, search,
#     registry) into one reproducible run
#   - Build a branching workflow with a conditional quality gate
#   - Write an audit trail per run_id covering every stage transition
#   - Verify reproducibility: same seed + same code -> same final metrics
#
# PREREQUISITES: 01-04 of this exercise
# ESTIMATED TIME: ~45 min
#
# 5-PHASE R10:
#   1. Theory     — why reproducibility is the ultimate ML-ops contract
#   2. Build      — the orchestrated pipeline with a branching workflow
#   3. Train      — run end-to-end: preprocess -> search -> register -> promote
#   4. Visualise  — print the audit trail and the reproducibility check
#   5. Apply      — full Singapore banking ML-ops ROI line
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import asyncio
import json
import uuid

import numpy as np

from dataflow import DataFlow
from kailash.runtime import LocalRuntime
from kailash.workflow.builder import WorkflowBuilder
from kailash_ml import HyperparameterSearch, TrainingPipeline
from kailash_ml.engines.hyperparameter_search import (
    ParamDistribution,
    SearchConfig,
    SearchSpace,
)
from kailash_ml.engines.training_pipeline import EvalSpec, ModelSpec
from kailash_ml.interop import to_sklearn_input

from shared.mlfp03.ex_7 import (
    DB_URL,
    RANDOM_SEED,
    SG_BANK_PORTFOLIO,
    audit_trail_row,
    build_training_registry,
    compute_classification_metrics,
    credit_feature_schema,
    headline_roi_text,
    prepare_credit_frame,
    print_metric_block,
    scale_pos_weight_for,
)


# ════════════════════════════════════════════════════════════════════════
# THEORY — Reproducibility as the ML-ops contract
# ════════════════════════════════════════════════════════════════════════
# You have four pieces:
#   1. A workflow DAG           (file 01)
#   2. A persistence layer      (file 02)
#   3. A hyperparameter search  (file 03)
#   4. A model registry         (file 04)
#
# An orchestrated pipeline is what ties them together so the answer to
# "can we rebuild exactly the model we shipped on March 12?" is YES.
# The contract is:
#   - Same input data + same seed + same code = same output
#   - Every stage writes an audit row tagged with the same run_id
#   - Every transition has a stage-to-stage edge the auditor can replay
#   - A branching quality gate decides register vs retrain
#
# Reproducibility is not a nice-to-have. Under MAS Notice 635, it IS
# the burden of proof. If you cannot reproduce yesterday's model, you
# cannot defend yesterday's decisions.


# ════════════════════════════════════════════════════════════════════════
# TASK 2 — BUILD: DAG + branching workflow + DataFlow models
# ════════════════════════════════════════════════════════════════════════

RUN_ID = str(uuid.uuid4())
db = DataFlow(DB_URL)


@db.model
class PipelineAuditEntry:
    """One row per stage transition for an orchestrated pipeline run.

    Modern DataFlow: `@db.model` treats plain annotations as the schema
    and auto-generates the `id` primary key. No `field(primary_key=True)`
    declarations are needed (or permitted).
    """

    # TODO: Declare the 4 columns this audit table needs.
    # Hint: id (int), run_id (str), stage (str), detail (str) — plain annotations.
    id: int
    run_id: ____
    stage: ____
    detail: ____


branching_workflow = WorkflowBuilder("credit_scoring_orchestrated")
branching_workflow.add_node(
    "DataPreprocessNode",
    "preprocess",
    {"data_source": "sg_credit_scoring", "target": "default"},
)
branching_workflow.add_node(
    "ModelTrainNode",
    "train_primary",
    {"model_class": "lightgbm.LGBMClassifier"},
    connections=["preprocess"],
)
branching_workflow.add_node(
    "ModelEvalNode",
    "evaluate",
    {"metrics": ["auc_pr", "brier_score"]},
    connections=["train_primary"],
)
branching_workflow.add_node(
    "ConditionalNode",
    "quality_gate",
    {
        "condition": "auc_pr > 0.5",
        "true_output": "register",
        "false_output": "retrain",
    },
    connections=["evaluate"],
)
branching_workflow.add_node(
    "PersistNode",
    "register",
    {"storage": DB_URL, "stage": "staging"},
    connections=["quality_gate"],
)


runtime = LocalRuntime()


# ════════════════════════════════════════════════════════════════════════
# TASK 3 — TRAIN: run the full pipeline
# ════════════════════════════════════════════════════════════════════════
# We execute the branching workflow for its declarative value (the DAG
# is what the auditor reads), then run the imperative pipeline that
# actually trains, searches, registers, and promotes — every step
# writing into the PipelineAuditEntry table under the same run_id.


async def orchestrated_run() -> dict:
    audit: list[dict] = []

    # Modern DataFlow: @db.model auto-migrates the PipelineAuditEntry
    # table on first use — no explicit db.initialize() handshake needed.
    async def log_stage(stage: str, detail: str) -> None:
        row = audit_trail_row(stage=stage, detail=detail, run_id=RUN_ID)
        audit.append(row)
        await db.express.create("PipelineAuditEntry", row)

    # Stage 1: declarative DAG execution. Custom nodes remain
    # unregistered in the course env — the DAG is recorded for the
    # auditor, the imperative path below does the real work. Catch the
    # expected NodeNotFoundError and log it as "declarative-only".
    try:
        _, wf_run_id = runtime.execute(branching_workflow.build())
        await log_stage("workflow.run", f"runtime.execute ok wf_run_id={wf_run_id}")
    except Exception as exc:
        await log_stage(
            "workflow.run",
            f"declarative-only (custom nodes unregistered): {type(exc).__name__}",
        )

    # Stage 2: preprocess
    frame, feature_cols = prepare_credit_frame()
    schema = credit_feature_schema(feature_cols)
    X_raw, y_raw, _ = to_sklearn_input(
        frame, feature_columns=feature_cols, target_column="default"
    )
    assert y_raw is not None, "target_column='default' must yield labels"
    y_all = np.asarray(y_raw)
    pos_weight = scale_pos_weight_for(y_all)
    await log_stage(
        "preprocess",
        f"rows={len(y_all)} features={len(feature_cols)} pos_weight={pos_weight:.3f}",
    )

    # Stage 3: Bayesian hyperparameter search via TrainingPipeline —
    # no raw .fit() in user code, the engine owns the trial loop.
    registry, conn = await build_training_registry()
    try:
        pipeline = TrainingPipeline(feature_store=None, registry=registry)
        searcher = HyperparameterSearch(pipeline=pipeline)

        base_model_spec = ModelSpec(
            model_class="lightgbm.LGBMClassifier",
            framework="lightgbm",
            hyperparameters={
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
        # TODO: Declare the search space — same 5 hyperparameters as ex_7/03.
        # Hint: ParamDistribution("name", "int_uniform"|"log_uniform", low=..., high=...)
        search_space = SearchSpace(
            params=[
                ParamDistribution("n_estimators", ____, low=____, high=____),
                ParamDistribution("learning_rate", ____, low=____, high=____),
                ParamDistribution("max_depth", ____, low=____, high=____),
                ParamDistribution("num_leaves", ____, low=____, high=____),
                ParamDistribution("min_child_samples", ____, low=____, high=____),
            ]
        )
        # TODO: Configure the Bayesian search (same pattern as ex_7/03).
        # Hint: SearchConfig(strategy=..., n_trials=..., metric_to_optimize=..., direction=...)
        search_config = SearchConfig(
            strategy=____,
            n_trials=____,
            metric_to_optimize=____,
            direction=____,
            register_best=False,
        )

        search_result = await searcher.search(
            data=frame,
            schema=schema,
            base_model_spec=base_model_spec,
            search_space=search_space,
            config=search_config,
            eval_spec=eval_spec,
            experiment_name=f"credit_default_orchestrated_{RUN_ID[:8]}",
        )
        best_params = dict(search_result.best_params)
        best_score = float(search_result.best_metrics.get("auc", 0.0))
        await log_stage(
            "hyperparameter_search",
            f"bayesian n_trials=20 best_auc={best_score:.4f}",
        )

        # Stage 4: train final, evaluate (engine-driven, registered)
        final_spec = ModelSpec(
            model_class="lightgbm.LGBMClassifier",
            framework="lightgbm",
            hyperparameters={
                **base_model_spec.hyperparameters,
                **best_params,
            },
        )
        # TODO: Train the final model via pipeline.train() with the winning params.
        # Hint: pipeline.train(data=..., schema=..., model_spec=..., eval_spec=..., experiment_name=...)
        final_result = await pipeline.train(
            data=____,
            schema=____,
            model_spec=____,
            eval_spec=____,
            experiment_name=____,
        )
        engine_metrics = dict(final_result.metrics)
        await log_stage(
            "evaluate",
            f"auc={engine_metrics.get('auc', 0.0):.4f} f1={engine_metrics.get('f1', 0.0):.4f}",
        )

        # Compute the rich metric block (AUC-PR, log-loss, Brier) that
        # the engine evaluator does not emit for the search-optimised
        # metric set, by re-fitting once on a holdout split and routing
        # through the shared compute_classification_metrics helper.
        import lightgbm as lgb

        X_all = np.asarray(X_raw)
        n_test = int(0.2 * len(y_all))
        X_train, X_test = X_all[:-n_test], X_all[-n_test:]
        y_train, y_test = y_all[:-n_test], y_all[-n_test:]
        report_model = lgb.LGBMClassifier(**final_spec.hyperparameters)
        report_model.fit(X_train, y_train)
        y_pred = np.asarray(report_model.predict(X_test))
        y_proba = np.asarray(report_model.predict_proba(X_test))[:, 1]
        metrics = compute_classification_metrics(y_test, y_pred, y_proba)

        # Stage 5: quality gate
        gate_passes = metrics["auc_pr"] > 0.5
        await log_stage(
            "quality_gate", f"auc_pr>0.5 -> {'register' if gate_passes else 'retrain'}"
        )

        version_id: int | None = None
        if gate_passes:
            # Stage 6: persist evaluation (results store)
            await db.express.create(
                "PipelineAuditEntry",
                audit_trail_row(
                    stage="persist.evaluation",
                    detail=json.dumps({k: round(v, 6) for k, v in metrics.items()}),
                    run_id=RUN_ID,
                ),
            )

            # Stage 7: TrainingPipeline already registered the final
            # model at STAGING. Promote it to PRODUCTION with an
            # audit-grade reason — this is the registry transition the
            # file teaches.
            assert (
                final_result.model_version is not None
            ), "TrainingPipeline should register the final model"
            version_id = final_result.model_version.version
            await log_stage("register", f"credit_default_v2 v{version_id} in staging")
            # TODO: Promote the model from staging to production with an audit reason.
            # Hint: registry.promote_model(name=..., version=..., target_stage=..., reason=...)
            await registry.promote_model(
                name=____,
                version=____,
                target_stage=____,
                reason=____,
            )
            await log_stage(
                "promote",
                f"credit_default_v2 v{version_id} staging->production",
            )

        # Stage 8: reproducibility check — re-fit the reporting model
        # with the same seed and compare AUC-PR drift. This is the
        # regulator-facing proof that the pipeline is deterministic.
        repro_model = lgb.LGBMClassifier(**final_spec.hyperparameters)
        repro_model.fit(X_train, y_train)
        y_proba_repro = np.asarray(repro_model.predict_proba(X_test))[:, 1]
        repro_metrics = compute_classification_metrics(
            y_test, np.asarray(repro_model.predict(X_test)), y_proba_repro
        )
        # TODO: Compute the AUC-PR drift between the original and reproduced model.
        # Hint: abs(repro_metrics["auc_pr"] - metrics["auc_pr"])
        drift = ____
        await log_stage("reproducibility", f"drift_auc_pr={drift:.6f} (must be <1e-3)")

        return {
            "metrics": metrics,
            "repro_metrics": repro_metrics,
            "best_params": best_params,
            "best_cv_score": best_score,
            "version_id": version_id,
            "drift": drift,
            "audit": audit,
            "report_model": report_model,
            "X_test": X_test,
            "y_test": y_test,
        }
    finally:
        await conn.close()


print("\n" + "=" * 70)
print(f"  Orchestrated pipeline run — run_id={RUN_ID}")
print("=" * 70)

orchestration = asyncio.run(orchestrated_run())


# ── Checkpoint ──────────────────────────────────────────────────────────
assert (
    orchestration["drift"] < 1e-3
), f"Task 3: same seed must reproduce same metrics (drift={orchestration['drift']:.6f})"
assert (
    orchestration["metrics"]["auc_pr"] > 0.5
), "Task 3: final model must clear the quality gate"
assert len(orchestration["audit"]) >= 6, "Task 3: audit trail must record every stage"
print("\n[ok] Checkpoint passed — orchestrated pipeline + reproducibility verified\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 4 — VISUALISE the audit trail + reproducibility certificate
# ════════════════════════════════════════════════════════════════════════

print("=== Audit Trail ===")
for row in orchestration["audit"]:
    print(f"  [{row['stage']:<22}] {row['detail']}")

print_metric_block("Final metrics (orchestrated run)", orchestration["metrics"])
print_metric_block("Reproducibility re-run metrics", orchestration["repro_metrics"])
print(f"\n  AUC-PR drift: {orchestration['drift']:.6f}  " f"(threshold: 1e-3 — PASS)")
print(
    f"\nPipeline DAG:"
    f"\n  preprocess -> hyperparameter_search -> train -> evaluate"
    f"\n       -> quality_gate -> [register -> promote]"
)


# ════════════════════════════════════════════════════════════════════════
# TASK 5 — APPLY: Full Singapore Banking ML-Ops ROI
# ════════════════════════════════════════════════════════════════════════
# This file unlocks BOTH lines of the `headline_roi_text()` table at once:
#   - Loss avoided (from the Bayesian lift, file 03)
#   - Audit prep savings (from persistence + registry, files 02 + 04)
# plus a reproducibility certificate that is the regulator's gold
# standard for model-risk management.
print("\n" + "=" * 70)
print("  APPLY: End-to-End Credit Risk ML Ops (S$48B Portfolio)")
print("=" * 70)
print(headline_roi_text())
print(
    f"\n  Plus: a reproducibility certificate for run_id={RUN_ID[:8]}"
    f"\n  that maps 1:1 to the PipelineAuditEntry rows above."
)


# ════════════════════════════════════════════════════════════════════════
# DESTINATION-FIRST CLOSE — km.diagnose
# ════════════════════════════════════════════════════════════════════════
# This orchestrated capstone wired WorkflowBuilder, DataFlow @db.model,
# HyperparameterSearch (Bayesian), TrainingPipeline, ModelRegistry, and a
# branching quality gate from primitives — every stage tagged to one
# run_id, every transition written to an audit table, every metric
# reproduced under the same seed. The kailash-ml SDK packages the
# diagnostic surface (per-class metrics, class-balance severity, confusion
# matrix) into a single call — the foundation every regulator-facing
# audit gate evaluates against.
#
# Destination-first: when the journey is internalised, the SDK is one line.

from kailash_ml import diagnose

report_model = orchestration["report_model"]
X_test = orchestration["X_test"]
y_test = orchestration["y_test"]

# `kind="classical_classifier"` dispatches to the sklearn ClassifierMixin
# adapter. The pipeline's fitted LightGBM classifier implements the
# interface — exactly the model that passed the AUC-PR quality gate above.
report = diagnose(
    report_model,
    kind="classical_classifier",
    data=(X_test, y_test),
    show=False,
)
print()
print("  km.diagnose model    : orchestrated credit-default classifier")
print(f"  km.diagnose metrics  : {report.metrics}")
print(f"  km.diagnose severity : {report.severity}")
print()
print("km.diagnose: 1 call -> the diagnostic surface every audit gate")
print("evaluates against. Destination-first: when the journey is")
print("internalised, the SDK is one line.")


# ════════════════════════════════════════════════════════════════════════
# REFLECTION
# ════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  WHAT YOU'VE MASTERED")
print("=" * 70)
print(
    f"""
  [x] Tied WorkflowBuilder, DataFlow, HyperparameterSearch, and
      ModelRegistry into ONE orchestrated Kailash pipeline
  [x] Wrote every stage transition into a DataFlow-managed audit table
  [x] Branched on a quality gate (AUC-PR > 0.5 -> register, else retrain)
  [x] Proved reproducibility: same seed + same code -> drift < 1e-3
  [x] Mapped the pipeline to a full Singapore banking ML-ops ROI

  KEY INSIGHT: Orchestration is not about running scripts in order — it
  is about producing an audit trail the regulator can replay. Every
  stage is named, every run is tagged, every transition is a row in a
  table that survives the analyst who wrote the pipeline.

  This exercise is complete. Next up in MLFP03: Exercise 8 brings in
  conformal prediction, DriftMonitor, and a production monitoring
  dashboard — the operational layer above everything you just built.

  Version promoted to production: credit_default_v2 v{orchestration['version_id']}
  Portfolio anchor: S${SG_BANK_PORTFOLIO['portfolio_sgd']/1e9:.0f}B
"""
)
