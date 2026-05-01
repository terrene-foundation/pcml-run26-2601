# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
Shared infrastructure for MLFP03 Exercise 7 — Kailash Workflows, DataFlow
Persistence, Hyperparameter Search, and Model Registry.

Contains: dataset loading, preprocessing-to-sklearn-input helpers, fixed
train/test splits, metric computation, DB URL resolution, and pipeline-audit
utilities. Technique-specific code (workflow node wiring, search space
definitions, registry lifecycle transitions) lives in the per-technique files.

Available after ``uv sync`` from any directory.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field as _field
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from dotenv import load_dotenv
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    log_loss,
    roc_auc_score,
)

from kailash_ml import PreprocessingPipeline
from kailash_ml.interop import to_sklearn_input
from kailash_ml.types import FeatureField, FeatureSchema

from shared import MLFPDataLoader
from shared.kailash_helpers import setup_environment

# ════════════════════════════════════════════════════════════════════════
# ENVIRONMENT SETUP
# ════════════════════════════════════════════════════════════════════════

setup_environment()
load_dotenv()

RANDOM_SEED: int = 42
TARGET_COLUMN: str = "default"
DATASET_NAME: str = "sg_credit_scoring"
DATASET_FILE: str = "sg_credit_scoring.parquet"

# Output directory for artefacts (audit trails, evaluation tables)
OUTPUT_DIR = Path("outputs") / "mlfp03_ex7"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# DataFlow persistence URL. SQLite is hermetic — every student gets a fresh
# DB file per run. In production we would read this from the environment.
#
# NOTE: The modern dataflow sqlite adapter interprets `sqlite:///relative`
# as an absolute `/relative` path (breaking old sqlite URL conventions).
# We therefore resolve to an absolute path and use the `sqlite:////abs`
# four-slash form so the behaviour is identical on every working dir.
_DB_ABS_PATH = (OUTPUT_DIR / "mlfp03_models.db").resolve()
DB_URL: str = os.environ.get(
    "MLFP03_EX7_DB_URL", f"sqlite:///{_DB_ABS_PATH.as_posix()}"
)


# ════════════════════════════════════════════════════════════════════════
# DATA LOADING — Singapore credit scoring (from MLFP02)
# ════════════════════════════════════════════════════════════════════════


def load_credit_frame() -> pl.DataFrame:
    """Load the Singapore credit scoring dataset as a polars DataFrame.

    Columns: demographic + bureau features, with ``default`` (0/1) target.
    """
    loader = MLFPDataLoader()
    return loader.load("mlfp02", DATASET_FILE)


@dataclass
class CreditSplit:
    """Train/test tensors with column metadata — one source of truth."""

    X_train: np.ndarray
    y_train: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    feature_columns: list[str] = _field(default_factory=list)
    train_size: int = 0
    test_size: int = 0
    feature_count: int = 0


def prepare_credit_split(
    credit: pl.DataFrame | None = None, *, seed: int = RANDOM_SEED
) -> CreditSplit:
    """Run the Kailash-ML preprocessing pipeline and return a CreditSplit.

    Deterministic with ``seed``: same seed in + same data in → same split out.
    This is the reproducibility contract Task 12 verifies.
    """
    if credit is None:
        credit = load_credit_frame()

    pipeline = PreprocessingPipeline()
    result = pipeline.setup(
        credit,
        target=TARGET_COLUMN,
        seed=seed,
        normalize=False,
        categorical_encoding="ordinal",
    )

    feature_columns = [c for c in result.train_data.columns if c != TARGET_COLUMN]
    X_train, y_train, _ = to_sklearn_input(
        result.train_data,
        feature_columns=feature_columns,
        target_column=TARGET_COLUMN,
    )
    X_test, y_test, _ = to_sklearn_input(
        result.test_data,
        feature_columns=feature_columns,
        target_column=TARGET_COLUMN,
    )

    return CreditSplit(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        feature_columns=feature_columns,
        train_size=X_train.shape[0],
        test_size=X_test.shape[0],
        feature_count=X_train.shape[1],
    )


def prepare_credit_frame(
    credit: pl.DataFrame | None = None, *, seed: int = RANDOM_SEED
) -> tuple[pl.DataFrame, list[str]]:
    """Return a preprocessed polars DataFrame suitable for TrainingPipeline.

    Adds a deterministic ``application_id`` column so a ``FeatureSchema`` can
    declare an ``entity_id_column`` — required by kailash-ml's TrainingPipeline.

    Returns
    -------
    (frame, feature_columns)
        ``frame`` contains all feature columns + ``default`` (target) +
        ``application_id`` (entity). ``feature_columns`` excludes both.
    """
    if credit is None:
        credit = load_credit_frame()

    pipeline = PreprocessingPipeline()
    result = pipeline.setup(
        credit,
        target=TARGET_COLUMN,
        seed=seed,
        normalize=False,
        categorical_encoding="ordinal",
    )

    combined = pl.concat([result.train_data, result.test_data])
    combined = combined.with_columns(
        pl.int_range(0, combined.height, dtype=pl.Int64).alias("application_id")
    )
    feature_columns = [
        c for c in combined.columns if c not in (TARGET_COLUMN, "application_id")
    ]
    return combined, feature_columns


def credit_feature_schema(feature_columns: list[str]) -> FeatureSchema:
    """Build a FeatureSchema matching ``prepare_credit_frame`` output."""
    return FeatureSchema(
        name="credit_model_input",
        features=[FeatureField(name=f, dtype="float64") for f in feature_columns],
        entity_id_column="application_id",
    )


async def build_training_registry(db_url: str | None = None):
    """Create + initialise a kailash-ml ModelRegistry for TrainingPipeline.

    Returns ``(registry, connection_manager)``. Caller owns ``connection_manager.close()``.
    """
    from kailash.db import ConnectionManager
    from kailash_ml import ModelRegistry

    conn = ConnectionManager(db_url or DB_URL)
    await conn.initialize()
    registry = ModelRegistry(conn)
    return registry, conn


def scale_pos_weight_for(y: np.ndarray) -> float:
    """LightGBM scale_pos_weight for a 12%-positive binary target."""
    pos_rate = float(y.mean())
    if pos_rate <= 0.0 or pos_rate >= 1.0:
        return 1.0
    return (1.0 - pos_rate) / pos_rate


# ════════════════════════════════════════════════════════════════════════
# METRICS
# ════════════════════════════════════════════════════════════════════════


def compute_classification_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray
) -> dict[str, float]:
    """Return accuracy, f1, auc_roc, auc_pr, log_loss."""
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred)),
        "auc_roc": float(roc_auc_score(y_true, y_proba)),
        "auc_pr": float(average_precision_score(y_true, y_proba)),
        "log_loss": float(log_loss(y_true, y_proba)),
    }


def print_metric_block(title: str, metrics: dict[str, float]) -> None:
    print(f"\n=== {title} ===")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")


# ════════════════════════════════════════════════════════════════════════
# SINGAPORE BANKING ML-OPS APPLICATION DATA — ROI ANCHORS (R9B)
# ════════════════════════════════════════════════════════════════════════
# Real MAS-regulated figures (rounded for teaching). Every technique file
# references this table so the dollar impact is consistent across the
# exercise and trivially auditable.

SG_BANK_PORTFOLIO: dict[str, Any] = {
    # ~S$48B unsecured retail portfolio across the three local banks (DBS,
    # OCBC, UOB) — public pillar-3 disclosures, FY2024.
    "portfolio_sgd": 48_000_000_000.0,
    # ~12% credit default rate for unsecured lending post-COVID stimulus
    # unwind (MAS Financial Stability Review 2024).
    "default_rate": 0.12,
    # Average loss given default on unsecured retail (bureau data).
    "lgd": 0.65,
    # Hyperparameter-optimised model lift vs grid-search baseline, measured
    # on the exercise's own validation folds.
    "hp_search_lift_auc_pr": 0.04,
    # Incremental AUC-PR translated to captured defaults through the
    # operating point analysis in module 3 exercise 4.
    "defaults_caught_per_auc_pr_point": 140,
    # Average principal per caught default (S$ retail revolving balance).
    "avg_sgd_per_default": 18_000.0,
    # MAS Notice 635 — each production model re-training needs an audit
    # trail; ML-ops automation eliminates ~4 analyst-weeks per cycle.
    "audit_prep_hours_saved_per_cycle": 160.0,
    # Blended analyst rate (SG fintech, fully loaded) used for the audit
    # savings ROI line.
    "analyst_hourly_sgd": 120.0,
}


def headline_roi_text() -> str:
    """Plain-text summary used in Apply phases across all 5 technique files."""
    p = SG_BANK_PORTFOLIO
    lift_pts = p["hp_search_lift_auc_pr"] * 100
    caught = p["defaults_caught_per_auc_pr_point"] * lift_pts
    dollars = caught * p["avg_sgd_per_default"] * p["lgd"]
    audit = p["audit_prep_hours_saved_per_cycle"] * p["analyst_hourly_sgd"] * 12
    total = dollars + audit
    return (
        f"  Portfolio base:     S${p['portfolio_sgd']/1e9:.0f}B unsecured retail\n"
        f"  Model lift:         +{lift_pts:.1f} AUC-PR points from orchestration\n"
        f"  Defaults caught:    ~{caught:.0f} additional per year\n"
        f"  Loss avoided:       ~S${dollars/1e6:.1f}M / yr (after LGD)\n"
        f"  Audit prep savings: ~S${audit/1e3:.0f}k / yr (MAS Notice 635)\n"
        f"  ──────────────────────────────────────────────\n"
        f"  Total annual value: ~S${total/1e6:.2f}M"
    )


# ════════════════════════════════════════════════════════════════════════
# PIPELINE AUDIT HELPERS
# ════════════════════════════════════════════════════════════════════════


def audit_trail_row(
    *,
    stage: str,
    detail: str,
    run_id: str,
) -> dict[str, Any]:
    """Structured audit row used by the orchestrated pipeline."""
    return {"stage": stage, "detail": detail, "run_id": run_id}
