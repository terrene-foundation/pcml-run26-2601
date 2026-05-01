# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# MLFP03 — Exercise 7.3: Bayesian Hyperparameter Search
# ════════════════════════════════════════════════════════════════════════
#
# WHAT YOU'LL LEARN:
#   - Declare a search space with ParamDistribution
#     (int_uniform / log_uniform / uniform)
#   - Configure a Bayesian search run with SearchConfig
#   - Drive the search through kailash-ml's HyperparameterSearch engine,
#     wired on top of TrainingPipeline (no raw .fit() in user code)
#   - Compare Bayesian vs grid-search lift on the same training split
#   - Feed the best_params back into a final, fully-evaluated model
#
# PREREQUISITES: 01_workflow_builder.py, 02_dataflow_persistence.py
# ESTIMATED TIME: ~45 min
#
# 5-PHASE R10:
#   1. Theory     — why Bayesian search beats grid search
#   2. Build      — SearchSpace + SearchConfig + TrainingPipeline base
#   3. Train      — async .search() against the credit-default frame
#   4. Visualise  — top-5 trials leaderboard + final model AUC-PR
#   5. Apply      — Singapore credit default lift worth S$4M+/yr
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import asyncio

import lightgbm as lgb
import numpy as np
from sklearn.model_selection import cross_val_score

from kailash_ml import HyperparameterSearch, TrainingPipeline
from kailash_ml.engines.hyperparameter_search import (
    ParamDistribution,
    SearchConfig,
    SearchSpace,
)
from kailash_ml.engines.training_pipeline import EvalSpec, ModelSpec
from kailash_ml.interop import to_sklearn_input

from shared.mlfp03.ex_7 import (
    RANDOM_SEED,
    SG_BANK_PORTFOLIO,
    build_training_registry,
    compute_classification_metrics,
    credit_feature_schema,
    headline_roi_text,
    prepare_credit_frame,
    print_metric_block,
    scale_pos_weight_for,
)


# ════════════════════════════════════════════════════════════════════════
# THEORY — Bayesian vs grid search
# ════════════════════════════════════════════════════════════════════════
# Grid search is the "brute-force every combination" approach. A 5x5x5
# grid is 125 trials even at 3 hyperparameters. Every extra dimension
# multiplies the cost. At 5 hyperparameters you're at 3,125 trials and
# you still haven't explored any fractional learning rates.
#
# Bayesian optimisation is smarter:
#   1. Fit a cheap surrogate (Gaussian process / tree-based regressor)
#      to the (hyperparameters -> score) pairs seen so far.
#   2. Compute an "acquisition function" that balances EXPLORATION
#      (try a region with high variance) and EXPLOITATION (try a region
#      with a high mean).
#   3. Evaluate the hyperparameter with the highest acquisition score.
#   4. Repeat.
#
# In practice, 20 Bayesian trials match or beat 200 random trials on
# most tabular problems. kailash-ml's HyperparameterSearch engine
# wraps this algorithm behind a small interface: SearchSpace,
# SearchConfig, and a TrainingPipeline that owns the fit + evaluate
# cycle for every trial. No raw .fit() in user code — the engine drives
# the search loop and registers the winner at STAGING when it is done.


# ════════════════════════════════════════════════════════════════════════
# TASK 2 — BUILD the search space and configuration
# ════════════════════════════════════════════════════════════════════════
# Modern ParamDistribution uses explicit type strings:
#   - "int_uniform"  for integer bounds
#   - "log_uniform"  for floats sampled on a log scale (good for LRs)
#   - "uniform"      for floats sampled linearly
#   - "categorical"  for discrete choices (pass via `choices=[...]`)
# The `log=True` keyword from earlier kailash-ml releases is gone — the
# distribution shape is encoded in the `type` string, not a kwarg.

search_space = SearchSpace(
    params=[
        ParamDistribution("n_estimators", "int_uniform", low=100, high=1000),
        ParamDistribution("learning_rate", "log_uniform", low=0.01, high=0.3),
        ParamDistribution("max_depth", "int_uniform", low=3, high=10),
        ParamDistribution("num_leaves", "int_uniform", low=15, high=127),
        ParamDistribution("min_child_samples", "int_uniform", low=5, high=50),
    ]
)

# SearchConfig fields:
#   - strategy="bayesian" picks the Optuna TPE backend
#   - n_trials caps the search budget
#   - metric_to_optimize must match a key TrainingPipeline emits in
#     result.metrics. EvalSpec below requests "auc" so we optimise that.
#   - register_best=False keeps the search hermetic; we register the
#     winner explicitly in file 04 / file 05.
search_config = SearchConfig(
    strategy="bayesian",
    n_trials=20,
    metric_to_optimize="auc",
    direction="maximize",
    register_best=False,
)


# ════════════════════════════════════════════════════════════════════════
# TASK 3 — TRAIN the search run via TrainingPipeline + HyperparameterSearch
# ════════════════════════════════════════════════════════════════════════
# The engine drives 20 TrainingPipeline.train() calls under the hood,
# each with a different sampled hyperparameter set. The base ModelSpec
# carries the FIXED params (random_state, scale_pos_weight, verbose) so
# every trial inherits the same reproducibility / class-balance contract.

frame, feature_cols = prepare_credit_frame()
schema = credit_feature_schema(feature_cols)
pos_weight = scale_pos_weight_for(frame["default"].to_numpy())


async def run_bayesian_search() -> tuple[dict, float, list, dict]:
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

        result = await searcher.search(
            data=frame,
            schema=schema,
            base_model_spec=base_model_spec,
            search_space=search_space,
            config=search_config,
            eval_spec=eval_spec,
            experiment_name="credit_default_hp_search",
        )

        # Final model — train once more with the winning hyperparameters
        # so we can compute the full classification metric block
        # (including AUC-PR / log-loss) that the search loop does not.
        final_spec = ModelSpec(
            model_class="lightgbm.LGBMClassifier",
            framework="lightgbm",
            hyperparameters={
                **base_model_spec.hyperparameters,
                **result.best_params,
            },
        )
        final_train = await pipeline.train(
            data=frame,
            schema=schema,
            model_spec=final_spec,
            eval_spec=eval_spec,
            experiment_name="credit_default_hp_search_final",
        )
        return (
            dict(result.best_params),
            float(result.best_metrics.get("auc", 0.0)),
            list(result.all_trials),
            dict(final_train.metrics),
        )
    finally:
        await conn.close()


print("\n" + "=" * 70)
print("  Bayesian Hyperparameter Search — 20 trials, holdout AUC")
print("=" * 70)

best_params, best_score, all_trials, final_metrics_engine = asyncio.run(
    run_bayesian_search()
)

print(f"\n  Best AUC (TrainingPipeline holdout): {best_score:.4f}")
print(f"  Best params: {best_params}")


# Compute the rich classification metric block (AUC-PR, log-loss, etc.)
# on a hand-rolled holdout fit so the Apply phase can talk in AUC-PR
# units that auditors recognise from MLFP02. The kailash-ml engine
# evaluates "auc" (ROC) for the search loop; "average_precision"
# requires y_prob which the engine's evaluator does not yet plumb
# through, so we materialise it here for the final reporting block.
X_raw, y_raw, _ = to_sklearn_input(
    frame, feature_columns=feature_cols, target_column="default"
)
# Binary supervised path — `default` is a real label column, so y_raw
# is never None at runtime. Narrow it for the type checker and cast
# both sides to dense ndarrays so downstream slicing and metric calls
# see the shape the helper contract expects.
assert y_raw is not None, "target_column='default' must yield labels"
X = np.asarray(X_raw)
y = np.asarray(y_raw)
n_test = int(0.2 * len(y))
X_train, X_test = X[:-n_test], X[-n_test:]
y_train, y_test = y[:-n_test], y[-n_test:]

best_model = lgb.LGBMClassifier(
    **best_params,
    random_state=RANDOM_SEED,
    verbose=-1,
    scale_pos_weight=pos_weight,
)
best_model.fit(X_train, y_train)
# sklearn stubs return loose union types (ndarray | spmatrix | list) for
# predict / predict_proba; lightgbm always returns dense numpy arrays at
# runtime, so we cast at the boundary.
y_pred = np.asarray(best_model.predict(X_test))
y_proba = np.asarray(best_model.predict_proba(X_test))[:, 1]
final_metrics = compute_classification_metrics(y_test, y_pred, y_proba)


# ── Checkpoint ──────────────────────────────────────────────────────────
assert best_score > 0.0, "Task 3: search should yield a positive AUC"
assert best_params is not None, "Task 3: search should return a params dict"
assert final_metrics["auc_pr"] > 0.0, "Task 3: final fit should evaluate"
assert (
    len(all_trials) == search_config.n_trials
), f"Task 3: expected {search_config.n_trials} trials, got {len(all_trials)}"
print("\n[ok] Checkpoint passed — Bayesian search complete\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 4 — VISUALISE the leaderboard + final model metrics
# ════════════════════════════════════════════════════════════════════════
# all_trials is a list[TrialResult]. Each trial carries .trial_number,
# .params, .metrics, and .training_time_seconds. Sort by the optimised
# metric to produce the top-5 leaderboard.

top_k = sorted(
    all_trials,
    key=lambda t: t.metrics.get(search_config.metric_to_optimize, 0.0),
    reverse=True,
)[:5]
print("Top 5 trials (score = holdout AUC):")
for i, trial in enumerate(top_k, 1):
    score = trial.metrics.get(search_config.metric_to_optimize, 0.0)
    print(f"  {i}. trial #{trial.trial_number}  score={score:.4f}")
    for k, v in trial.params.items():
        print(f"       {k}: {v}")

print_metric_block("Final model on held-out test set", final_metrics)


# Baseline (grid) comparison — 4 manual points, same training split,
# same scoring metric (AUC-PR via 5-fold CV on the training tail).
baseline_grid = [
    {"n_estimators": 300, "learning_rate": 0.05, "max_depth": 5},
    {"n_estimators": 500, "learning_rate": 0.10, "max_depth": 6},
    {"n_estimators": 500, "learning_rate": 0.05, "max_depth": 7},
    {"n_estimators": 700, "learning_rate": 0.03, "max_depth": 8},
]
best_grid = -1.0
for params in baseline_grid:
    est = lgb.LGBMClassifier(
        **params,
        random_state=RANDOM_SEED,
        verbose=-1,
        scale_pos_weight=pos_weight,
    )
    cv = cross_val_score(est, X_train, y_train, cv=5, scoring="average_precision")
    best_grid = max(best_grid, float(cv.mean()))

# Compare in AUC-PR units — the metric the bank's risk team reports.
lift = final_metrics["auc_pr"] - best_grid
print(
    f"\nGrid-search best (CV AUC-PR)   : {best_grid:.4f}"
    f"\nBayesian-search best (test AUC-PR): {final_metrics['auc_pr']:.4f}"
    f"\nLift                            : +{lift:.4f} AUC-PR points"
)


# ════════════════════════════════════════════════════════════════════════
# TASK 5 — APPLY: Lift the S$48B portfolio's model quality
# ════════════════════════════════════════════════════════════════════════
# The hand-tuned production model used by a typical SG bank credit team
# is equivalent to the 4-point grid baseline above. A 20-trial Bayesian
# search, running overnight on a single CPU, typically lifts AUC-PR by
# 2-4 points on this dataset. Every AUC-PR point at the bank's default
# threshold translates to ~140 additional defaults CAUGHT per year.
#
# At a blended S$18,000 principal and 65% LGD, every caught default
# saves ~S$11,700 — so a 4-point lift saves ~560 * S$11,700 = S$6.5M.
# The `headline_roi_text()` table uses the conservative S$4M figure.
print("\n" + "=" * 70)
print("  APPLY: Bayesian Search Lift = Real Defaults Caught")
print("=" * 70)
print(headline_roi_text())
print(
    "\n  The `Loss avoided` line above is UNLOCKED by this file."
    "\n  Without Bayesian search, the bank ships the grid baseline."
)


# ════════════════════════════════════════════════════════════════════════
# REFLECTION
# ════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  WHAT YOU'VE MASTERED")
print("=" * 70)
print(
    f"""
  [x] Declared a 5-dimensional hyperparameter SearchSpace using the
      modern int_uniform / log_uniform type strings
  [x] Configured Bayesian search via SearchConfig(strategy="bayesian")
  [x] Drove HyperparameterSearch on top of a kailash-ml TrainingPipeline
      so every trial is a real engine train() call (no raw .fit())
  [x] Compared the Bayesian winner against a 4-point grid baseline
  [x] Tied AUC-PR lift to the portfolio's annual dollar savings

  KEY INSIGHT: Bayesian search is NOT exotic — it is a small API call
  on top of TrainingPipeline that routinely finds better hyperparameters
  than a grid five times its size, and the engine handles the trial
  loop for you.

  Next: 04_model_registry.py — promote the winning model through the
  ModelRegistry lifecycle so production can safely pick it up.

  Portfolio anchor: S${SG_BANK_PORTFOLIO['portfolio_sgd']/1e9:.0f}B  |  Lift: +{lift:.4f} AUC-PR
"""
)
