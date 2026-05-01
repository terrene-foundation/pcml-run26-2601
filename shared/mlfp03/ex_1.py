# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
Shared infrastructure for MLFP03 Exercise 1 — Feature Engineering
and Feature Selection on ICU data.

Contains:
    - ICU multi-table loading with temporal casts
    - Point-in-time feature builders (vitals, medications, labs)
    - ExperimentTracker / ConnectionManager setup
    - Shared prep helpers for feature-selection methods
    - Plotting helpers for feature rankings

Technique-specific code (mutual_info, RFE, Lasso paths, schema
validation) does NOT belong here — it lives in the per-technique files.
"""
from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from dotenv import load_dotenv

from kailash.db import ConnectionManager
from kailash_ml import DataExplorer, ExperimentTracker

from shared.data_loader import MLFPDataLoader
from shared.kailash_helpers import setup_environment


# ════════════════════════════════════════════════════════════════════════
# ENVIRONMENT
# ════════════════════════════════════════════════════════════════════════

setup_environment()
load_dotenv()

OUTPUT_DIR = Path("outputs") / "mlfp03_ex1_features"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

EXPERIMENT_DB = "sqlite:///mlfp03_experiments.db"
EXPERIMENT_NAME = "mlfp03_healthcare_features"

_DT_FMT = "%Y-%m-%d %H:%M:%S"

VITAL_COLS: list[str] = [
    "heart_rate",
    "systolic_bp",
    "diastolic_bp",
    "temperature",
    "spo2",
    "respiratory_rate",
]


# ════════════════════════════════════════════════════════════════════════
# DATA LOADING — ICU multi-table
# ════════════════════════════════════════════════════════════════════════


def load_icu_tables() -> dict[str, pl.DataFrame]:
    """Load all five ICU tables and cast timestamp columns to datetime.

    Returns a dict with keys: patients, admissions, vitals (long format),
    medications, labs.

    Vitals are returned in LONG format (columns: admission_id, patient_id,
    timestamp, vital_name, value) — the monolithic exercise unpivots them
    inline, but every technique file wants the long form.
    """
    loader = MLFPDataLoader()
    patients = loader.load("mlfp02", "icu_patients.parquet")
    admissions = loader.load("mlfp02", "icu_admissions.parquet")
    vitals = loader.load("mlfp02", "icu_vitals.parquet")
    medications = loader.load("mlfp02", "icu_medications.parquet")
    labs = loader.load("mlfp02", "icu_labs.parquet")

    # Cast timestamps — polars reads them as strings from the parquet
    admissions = admissions.with_columns(
        pl.col("admit_time").str.to_datetime(_DT_FMT),
        pl.col("discharge_time").str.to_datetime(_DT_FMT),
    )
    medications = medications.with_columns(
        pl.col("start_time").str.to_datetime(_DT_FMT),
        pl.col("end_time").str.to_datetime(_DT_FMT),
    )
    if "timestamp" in labs.columns and labs["timestamp"].dtype == pl.String:
        labs = labs.with_columns(pl.col("timestamp").str.to_datetime(_DT_FMT))

    # Vitals: attach patient_id via admissions join, cast timestamp, unpivot
    vitals = vitals.join(
        admissions.select(["admission_id", "patient_id"]),
        on="admission_id",
        how="left",
    ).with_columns(pl.col("timestamp").str.to_datetime(_DT_FMT))

    present = [c for c in VITAL_COLS if c in vitals.columns]
    if present:
        vitals = vitals.unpivot(
            present,
            index=["admission_id", "patient_id", "timestamp"],
            variable_name="vital_name",
            value_name="value",
        )

    return {
        "patients": patients,
        "admissions": admissions,
        "vitals": vitals,
        "medications": medications,
        "labs": labs,
    }


# ════════════════════════════════════════════════════════════════════════
# FEATURE BUILDERS — point-in-time aggregates
# ════════════════════════════════════════════════════════════════════════


def build_vital_features(
    vitals: pl.DataFrame, admissions: pl.DataFrame
) -> pl.DataFrame:
    """Aggregate long-format vitals per admission with temporal correctness.

    Only uses vital readings recorded BETWEEN admit_time and discharge_time
    for each admission. Returns one row per admission with columns:
        {vital}_{mean,std,min,max,range,trend,count,cv}
    """
    # Vitals already carries admission_id and patient_id. Join only to pull
    # in the admit/discharge window; drop admissions' patient_id on the
    # way in to avoid duplicate columns.
    filtered = vitals.join(
        admissions.select("admission_id", "admit_time", "discharge_time"),
        on="admission_id",
        how="inner",
    ).filter(
        (pl.col("timestamp") >= pl.col("admit_time"))
        & (pl.col("timestamp") <= pl.col("discharge_time"))
    )

    names = filtered["vital_name"].unique().to_list()
    aggs: list[pl.DataFrame] = []
    for vital in names:
        agg = (
            filtered.filter(pl.col("vital_name") == vital)
            .group_by("admission_id")
            .agg(
                pl.col("value").mean().alias(f"{vital}_mean"),
                pl.col("value").std().alias(f"{vital}_std"),
                pl.col("value").min().alias(f"{vital}_min"),
                pl.col("value").max().alias(f"{vital}_max"),
                (pl.col("value").max() - pl.col("value").min()).alias(f"{vital}_range"),
                (pl.col("value").last() - pl.col("value").first()).alias(
                    f"{vital}_trend"
                ),
                pl.col("value").count().alias(f"{vital}_count"),
                (pl.col("value").std() / pl.col("value").mean()).alias(f"{vital}_cv"),
            )
        )
        aggs.append(agg)

    # Merge vital aggregates via full-outer join with coalesced key.
    out = aggs[0]
    for a in aggs[1:]:
        out = out.join(a, on="admission_id", how="full", coalesce=True)
    return out


def build_medication_features(
    medications: pl.DataFrame, admissions: pl.DataFrame
) -> pl.DataFrame:
    """Flag high-risk medications and count distinct drugs per admission."""
    return (
        medications.join(
            admissions.select(
                "patient_id", "admission_id", "admit_time", "discharge_time"
            ),
            on="admission_id",
            how="inner",
        )
        .filter(
            (pl.col("start_time") >= pl.col("admit_time"))
            & (pl.col("start_time") <= pl.col("discharge_time"))
        )
        .group_by("admission_id")
        .agg(
            pl.col("drug_name").n_unique().alias("n_unique_medications"),
            pl.col("drug_name").count().alias("n_medication_doses"),
            pl.col("drug_name")
            .str.contains("(?i)vasopressor|norepinephrine|dopamine")
            .any()
            .alias("received_vasopressors"),
            pl.col("drug_name")
            .str.contains("(?i)antibiotic|vancomycin|meropenem")
            .any()
            .alias("received_antibiotics"),
            pl.col("drug_name")
            .str.contains("(?i)propofol|midazolam|fentanyl")
            .any()
            .alias("received_sedation"),
        )
    )


def build_lab_features(labs: pl.DataFrame, admissions: pl.DataFrame) -> pl.DataFrame:
    """Aggregate lab results per admission with abnormal-flag counts."""
    return (
        labs.join(
            admissions.select("admission_id", "admit_time", "discharge_time"),
            on="admission_id",
            how="inner",
        )
        .filter(
            (pl.col("timestamp") >= pl.col("admit_time"))
            & (pl.col("timestamp") <= pl.col("discharge_time"))
        )
        .group_by("admission_id")
        .agg(
            pl.col("test_name").n_unique().alias("n_unique_labs"),
            pl.col("value").count().alias("n_lab_results"),
            (pl.col("flag") != "normal").sum().alias("n_abnormal_labs"),
            pl.col("value").mean().alias("lab_value_mean"),
            pl.col("value").std().alias("lab_value_std"),
        )
    )


def build_full_feature_frame(tables: dict[str, pl.DataFrame]) -> pl.DataFrame:
    """End-to-end feature matrix used by every technique file.

    Composes patients + admissions with vital / medication / lab / derived /
    interaction features. Every technique file calls this so the starting
    feature matrix is identical — only the SELECTION method differs.
    """
    patients = tables["patients"]
    admissions = tables["admissions"]

    patient_admissions = patients.join(admissions, on="patient_id", how="inner")

    features = patient_admissions.clone()
    vf = build_vital_features(tables["vitals"], admissions)
    features = features.join(vf, on="admission_id", how="left")

    # Vital stat columns (std, trend, cv, ...) are legitimately null
    # when a patient has only 1 reading for a given vital, or when a
    # vital was never sampled. Fill with 0 so downstream selection
    # methods (sklearn) see no nulls.
    vital_stat_cols = [
        c
        for c in features.columns
        if any(
            c.endswith(f"_{s}")
            for s in ("mean", "std", "min", "max", "range", "trend", "count", "cv")
        )
    ]
    features = features.with_columns(
        *[pl.col(c).fill_null(0.0) for c in vital_stat_cols]
    )

    mf = build_medication_features(tables["medications"], admissions)
    features = features.join(mf, on="admission_id", how="left")

    lf = build_lab_features(tables["labs"], admissions)
    features = features.join(lf, on="admission_id", how="left")

    # Derived features
    features = features.with_columns(
        (pl.col("n_abnormal_labs") / pl.col("n_lab_results").clip(lower_bound=1)).alias(
            "abnormal_lab_ratio"
        ),
        (pl.col("n_medication_doses") / pl.col("los_days").clip(lower_bound=1)).alias(
            "medication_intensity"
        ),
        (pl.col("n_lab_results") / pl.col("los_days").clip(lower_bound=1)).alias(
            "lab_intensity"
        ),
        (pl.col("n_unique_medications") > 10).alias("polypharmacy_flag"),
    )

    # Null fills for patients without meds / labs
    fill_int = [
        "n_unique_medications",
        "n_medication_doses",
        "n_unique_labs",
        "n_lab_results",
        "n_abnormal_labs",
    ]
    fill_bool = [
        "received_vasopressors",
        "received_antibiotics",
        "received_sedation",
        "polypharmacy_flag",
    ]
    fill_float = [
        "abnormal_lab_ratio",
        "medication_intensity",
        "lab_intensity",
        "lab_value_mean",
        "lab_value_std",
    ]
    features = features.with_columns(
        *[pl.col(c).fill_null(0) for c in fill_int if c in features.columns],
        *[pl.col(c).fill_null(False) for c in fill_bool if c in features.columns],
        *[pl.col(c).fill_null(0.0) for c in fill_float if c in features.columns],
    )

    # Interactions (clinical domain knowledge)
    cols = features.columns
    exprs: list[pl.Expr] = []
    if "heart_rate_mean" in cols and "systolic_bp_mean" in cols:
        exprs.append(
            (
                pl.col("heart_rate_mean")
                / pl.col("systolic_bp_mean").clip(lower_bound=1)
            ).alias("shock_index")
        )
    if "systolic_bp_mean" in cols and "diastolic_bp_mean" in cols:
        exprs.append(
            ((pl.col("systolic_bp_mean") + 2 * pl.col("diastolic_bp_mean")) / 3).alias(
                "map_mean"
            )
        )
    if "temperature_mean" in cols and "heart_rate_mean" in cols:
        exprs.append(
            (pl.col("temperature_mean") * pl.col("heart_rate_mean")).alias(
                "fever_tachycardia"
            )
        )
    exprs.append(
        (pl.col("medication_intensity") * pl.col("abnormal_lab_ratio")).alias(
            "treatment_burden_score"
        )
    )
    features = features.with_columns(*exprs)

    # Fill any nulls introduced by the interactions
    for name in (
        "shock_index",
        "map_mean",
        "fever_tachycardia",
        "treatment_burden_score",
    ):
        if name in features.columns:
            features = features.with_columns(pl.col(name).fill_null(0.0))

    return features


# ════════════════════════════════════════════════════════════════════════
# SELECTION INPUT PREP
# ════════════════════════════════════════════════════════════════════════


def prepare_selection_inputs(
    features: pl.DataFrame,
) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Return (feature_cols, X, y_binary) for every feature-selection method.

    - Drops ID columns and the target from the feature matrix
    - Coerces bool/int/float columns only (selection methods require numeric)
    - Replaces NaN / inf with bounded numbers
    - Builds a binary target: mortality if present, otherwise los_days > median
    """
    id_cols = {"patient_id", "admission_id", "admit_time", "discharge_time"}
    target_col = "mortality" if "mortality" in features.columns else "los_days"
    exclude = id_cols | {target_col}

    numeric_dtypes = {pl.Float64, pl.Float32, pl.Int64, pl.Int32, pl.Boolean}
    feature_cols = [
        c
        for c in features.columns
        if c not in exclude and features[c].dtype in numeric_dtypes
    ]

    X = features.select(feature_cols).to_numpy().astype(np.float64)
    X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)

    if target_col == "mortality":
        y = features["mortality"].to_numpy().astype(np.float64).ravel()
    else:
        median_los = features["los_days"].median()
        y = (
            (features["los_days"] > median_los)
            .cast(pl.Int32)
            .to_numpy()
            .ravel()
            .astype(np.float64)
        )
    y_binary = (
        (y > np.median(y)).astype(int) if target_col != "mortality" else y.astype(int)
    )
    return feature_cols, X, y_binary


# ════════════════════════════════════════════════════════════════════════
# EXPERIMENT TRACKING
# ════════════════════════════════════════════════════════════════════════


async def setup_tracking() -> tuple[ConnectionManager, ExperimentTracker, str]:
    """Initialize ConnectionManager + ExperimentTracker (kailash-ml 1.1.1).

    Every technique file in ex_1 logs into the same experiment so selection
    runs are directly comparable. The tracker is constructed via the async
    factory; the experiment is auto-created on first ``tracker.track(...)``.
    """
    tracker = await ExperimentTracker.create(store_url=EXPERIMENT_DB)
    conn = ConnectionManager(EXPERIMENT_DB)
    await conn.initialize()
    return conn, tracker, EXPERIMENT_NAME


def setup_tracking_sync() -> tuple[ConnectionManager, ExperimentTracker, str]:
    """Sync wrapper for setup_tracking — convenience for non-async files."""
    return asyncio.run(setup_tracking())


async def log_selection_run(
    tracker: ExperimentTracker,
    experiment_id: str,
    *,
    run_name: str,
    method: str,
    selected_features: list[str],
    total_features: int,
    extra_params: dict[str, str] | None = None,
    extra_metrics: dict[str, float] | None = None,
) -> str:
    """Log a feature-selection run to ExperimentTracker. Returns run id."""
    params = {
        "method": method,
        "n_features_total": str(total_features),
        "n_features_selected": str(len(selected_features)),
        "selected_features": ",".join(selected_features[:30]),
    }
    if extra_params:
        params.update(extra_params)
    metrics = {
        "n_features_selected": float(len(selected_features)),
        "selection_ratio": float(len(selected_features)) / max(1, total_features),
    }
    if extra_metrics:
        metrics.update(extra_metrics)

    async with tracker.track(experiment=experiment_id, run_name=run_name) as run:
        await run.log_params(params)
        await run.log_metrics(metrics)
        await run.add_tag("domain", "clinical")
        await run.add_tag("selection_family", method)
        run_id = run.run_id
    return run_id


# ════════════════════════════════════════════════════════════════════════
# REPORTING HELPERS
# ════════════════════════════════════════════════════════════════════════


def print_ranking(
    title: str, ranking: list[tuple[str, float]], *, top: int = 15
) -> None:
    """Print a ranked feature list with a simple ASCII bar chart."""
    print(f"\n=== {title} ===")
    print(f"{'Feature':<35} {'Score':>10}")
    print("-" * 48)
    if not ranking:
        print("  (empty ranking)")
        return
    max_score = max(abs(s) for _, s in ranking[:top]) or 1.0
    for name, score in ranking[:top]:
        bar_len = int(abs(score) / max_score * 20)
        bar = "#" * bar_len
        print(f"  {name:<33} {score:>10.4f}  {bar}")


def save_ranking_csv(
    ranking: list[tuple[str, float]], filename: str, score_col: str = "score"
) -> Path:
    """Persist a ranking as CSV into OUTPUT_DIR. Returns the file path."""
    path = OUTPUT_DIR / filename
    pl.DataFrame(
        {"feature": [n for n, _ in ranking], score_col: [s for _, s in ranking]}
    ).write_csv(path)
    return path
