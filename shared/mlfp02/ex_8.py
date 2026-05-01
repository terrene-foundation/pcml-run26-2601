# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
Shared infrastructure for MLFP02 Exercise 8 — FeatureStore + Feature Engineering.

Contains: HDB resale data loading, FeatureStore / ExperimentTracker setup
through kailash-ml, and OLS-from-scratch helpers reused across the four
R10 technique files:

    01_feature_schema.py        — FeatureSchema v1 + validation
    02_point_in_time.py         — Leakage prevention + temporal correctness
    03_rolling_features.py      — FeatureSchema v2 + group_by_dynamic
    04_modeling_with_features.py — Regression + hypothesis tests + Bayes

Technique-specific logic (schema construction, rolling window design,
coefficient interpretation) belongs in the per-technique files. This
module only owns infrastructure and reusable numeric helpers.
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from shared import MLFPDataLoader
from shared.kailash_helpers import setup_environment

setup_environment()

# ════════════════════════════════════════════════════════════════════════
# PATHS
# ════════════════════════════════════════════════════════════════════════

OUTPUT_DIR = Path("outputs") / "mlfp02_ex8"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FEATURE_STORE_URL = "sqlite:///mlfp02_ex8_features.db"
FEATURE_TABLE_PREFIX = "kml_feat_"
EXPERIMENT_NAME = "mlfp02_ex8_hdb_features"


# ════════════════════════════════════════════════════════════════════════
# DATA LOADING — HDB resale flats (data.gov.sg)
# ════════════════════════════════════════════════════════════════════════


def load_hdb_resale() -> pl.DataFrame:
    """Load HDB resale transactions with a parsed transaction_date column.

    The raw file stores `month` as "YYYY-MM"; we convert it to a polars
    Date so every downstream technique can sort, filter, and roll on a
    real temporal axis without string parsing.
    """
    loader = MLFPDataLoader()
    hdb = loader.load("mlfp01", "hdb_resale.parquet")
    hdb = hdb.with_columns(
        pl.col("month").str.to_date("%Y-%m").alias("transaction_date")
    )
    return hdb


# ════════════════════════════════════════════════════════════════════════
# FEATURE STORE + EXPERIMENT TRACKER — kailash-ml wiring
# ════════════════════════════════════════════════════════════════════════


async def setup_feature_store() -> tuple[Any, Any, Any, bool]:
    """Create (conn, FeatureStore, ExperimentTracker, has_backend) for kailash-ml 1.1.1.

    Returns ``has_backend=False`` if the infrastructure is unavailable.
    Callers handle the degraded path by running the Polars-only versions
    of each operation.

    Note: the first tuple element is now a ``ConnectionManager`` rather than
    the old ``StoreFactory`` — kailash-ml's ExperimentTracker no longer
    accepts a positional store object; it constructs its own through the
    ``store_url`` factory. We still return a ConnectionManager so FeatureStore
    has the connection it needs.
    """
    try:
        from kailash.db import ConnectionManager
        from kailash_ml import ExperimentTracker, FeatureStore

        conn = ConnectionManager(FEATURE_STORE_URL)
        await conn.initialize()
        fs = FeatureStore(conn, table_prefix=FEATURE_TABLE_PREFIX)
        tracker = await ExperimentTracker.create(store_url=FEATURE_STORE_URL)
        return conn, fs, tracker, True
    except Exception as exc:  # noqa: BLE001 — degrade gracefully
        print(
            f"  [warn] FeatureStore backend unavailable "
            f"({type(exc).__name__}: {exc})"
        )
        return None, None, None, False


# ════════════════════════════════════════════════════════════════════════
# FEATURE COMPUTATION — v1 (basic property) and v2 (rolling market)
# ════════════════════════════════════════════════════════════════════════


def compute_v1_features(df: pl.DataFrame) -> pl.DataFrame:
    """Compute version-1 HDB property features from raw transactions.

    Produces: storey_midpoint, price_per_sqm, remaining_lease_years,
    transaction_id (row index). These are the base features v2 extends.
    """
    return df.with_columns(
        (
            (
                pl.col("storey_range").str.extract(r"(\d+)", 1).cast(pl.Float64)
                + pl.col("storey_range").str.extract(r"TO (\d+)", 1).cast(pl.Float64)
            )
            / 2
        ).alias("storey_midpoint"),
        (pl.col("resale_price") / pl.col("floor_area_sqm")).alias("price_per_sqm"),
        (99 - (pl.col("transaction_date").dt.year() - pl.col("lease_commence_date")))
        .cast(pl.Float64)
        .alias("remaining_lease_years"),
    ).with_row_index("transaction_id")


def compute_v2_features(df: pl.DataFrame) -> pl.DataFrame:
    """Compute v2 features = v1 + rolling town-level market context.

    Uses polars ``group_by_dynamic`` on ``transaction_date`` bucketed by
    month, then a 6-month rolling window per town. The first six months
    per town have nulls (warm-up period) — callers must ``drop_nulls``
    before modelling.
    """
    result = compute_v1_features(df).sort("transaction_date")

    town_stats = (
        result.group_by_dynamic("transaction_date", every="1mo", group_by="town")
        .agg(
            pl.col("resale_price").median().alias("monthly_median"),
            pl.col("resale_price").count().alias("monthly_volume"),
        )
        .sort("town", "transaction_date")
    )

    town_stats = town_stats.with_columns(
        pl.col("monthly_median")
        .rolling_mean(window_size=6)
        .over("town")
        .alias("town_median_price"),
        pl.col("monthly_volume")
        .rolling_sum(window_size=6)
        .over("town")
        .alias("town_transaction_volume"),
        (
            (pl.col("monthly_median") - pl.col("monthly_median").shift(6).over("town"))
            / pl.col("monthly_median").shift(6).over("town")
            * 100
        ).alias("town_price_trend"),
    )

    result = result.join(
        town_stats.select(
            "town",
            "transaction_date",
            "town_median_price",
            "town_transaction_volume",
            "town_price_trend",
        ),
        on=["town", "transaction_date"],
        how="left",
    )
    return result


# ════════════════════════════════════════════════════════════════════════
# POINT-IN-TIME RETRIEVAL HELPERS
# ════════════════════════════════════════════════════════════════════════


def as_of(
    df: pl.DataFrame, cutoff: datetime, date_col: str = "transaction_date"
) -> pl.DataFrame:
    """Return rows strictly before ``cutoff`` — the Polars-only PIT path.

    When FeatureStore is unavailable, every technique falls back to this
    helper so the leakage-prevention lesson still runs end-to-end.
    """
    return df.filter(pl.col(date_col) < pl.lit(cutoff.date()))


# ════════════════════════════════════════════════════════════════════════
# FROM-SCRATCH OLS HELPERS — reused across techniques 3 and 4
# ════════════════════════════════════════════════════════════════════════

FEATURE_LIST: list[str] = [
    "floor_area_sqm",
    "storey_midpoint",
    "remaining_lease_years",
    "town_median_price",
    "town_price_trend",
]


def prepare_design_matrix(
    df: pl.DataFrame,
    feature_list: list[str] = FEATURE_LIST,
    target: str = "resale_price",
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Drop nulls, build ``[1, X]`` design matrix, return ``(X, y, names)``."""
    model_data = df.drop_nulls(subset=[*feature_list, target])
    X_raw = model_data.select(feature_list).to_numpy().astype(np.float64)
    y = model_data[target].to_numpy().astype(np.float64)
    X = np.column_stack([np.ones(len(y)), X_raw])
    names = ["intercept", *feature_list]
    return X, y, names


def fit_ols(X: np.ndarray, y: np.ndarray) -> dict[str, Any]:
    """Fit OLS from scratch and return a dict with betas, SEs, t, p, R²."""
    from scipy import stats as sp_stats  # local import — optional at module load

    n, k = X.shape
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    y_hat = X @ beta
    resid = y - y_hat

    ssr = float(np.sum(resid**2))
    sst = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ssr / sst
    adj_r2 = 1.0 - (1.0 - r2) * (n - 1) / (n - k)
    rmse = float(np.sqrt(ssr / n))

    sigma_sq = ssr / (n - k)
    xtx_inv = np.linalg.inv(X.T @ X)
    se = np.sqrt(sigma_sq * np.diag(xtx_inv))
    t_stat = beta / se
    p_val = 2.0 * (1.0 - sp_stats.t.cdf(np.abs(t_stat), df=n - k))

    sse = float(np.sum((y_hat - y.mean()) ** 2))
    f_stat = (sse / (k - 1)) / (ssr / (n - k))
    f_p = 1.0 - sp_stats.f.cdf(f_stat, dfn=k - 1, dfd=n - k)

    return {
        "n": n,
        "k": k,
        "beta": beta,
        "se": se,
        "t": t_stat,
        "p": p_val,
        "y_hat": y_hat,
        "resid": resid,
        "r2": float(r2),
        "adj_r2": float(adj_r2),
        "rmse": rmse,
        "f_stat": float(f_stat),
        "f_p": float(f_p),
    }


def normal_normal_posterior(
    beta_hat: float,
    se_hat: float,
    mu_prior: float = 0.0,
    sigma_prior: float = 10_000.0,
) -> dict[str, float]:
    """Normal-Normal conjugate posterior for a single OLS coefficient."""
    prec_prior = 1.0 / sigma_prior**2
    prec_data = 1.0 / se_hat**2
    prec_post = prec_prior + prec_data
    mu_post = (mu_prior * prec_prior + beta_hat * prec_data) / prec_post
    sigma_post = float(np.sqrt(1.0 / prec_post))
    return {
        "mu_post": float(mu_post),
        "sigma_post": sigma_post,
        "ci_low": float(mu_post - 1.96 * sigma_post),
        "ci_high": float(mu_post + 1.96 * sigma_post),
    }


# ════════════════════════════════════════════════════════════════════════
# FEATURE SCHEMA BUILDERS — kailash-ml FeatureSchema / FeatureField
# ════════════════════════════════════════════════════════════════════════


def build_schema_v1() -> Any:
    """Return the FeatureSchema v1 definition (basic property features)."""
    from kailash_ml.types import FeatureField, FeatureSchema

    return FeatureSchema(
        name="hdb_property_features",
        features=[
            FeatureField(
                name="floor_area_sqm",
                dtype="float64",
                nullable=False,
                description="Floor area in square metres",
            ),
            FeatureField(
                name="remaining_lease_years",
                dtype="float64",
                nullable=False,
                description="Remaining lease in years",
            ),
            FeatureField(
                name="storey_midpoint",
                dtype="float64",
                nullable=False,
                description="Midpoint of storey range",
            ),
            FeatureField(
                name="price_per_sqm",
                dtype="float64",
                nullable=False,
                description="Transaction price per square metre",
            ),
        ],
        entity_id_column="transaction_id",
        timestamp_column="transaction_date",
        version=1,
    )


def build_schema_v2() -> Any:
    """Return FeatureSchema v2 = v1 + three rolling market-context fields."""
    from kailash_ml.types import FeatureField, FeatureSchema

    v1 = build_schema_v1()
    return FeatureSchema(
        name="hdb_property_features",
        features=[
            *v1.features,
            FeatureField(
                name="town_median_price",
                dtype="float64",
                nullable=True,
                description="Median price in town (trailing 6 months)",
            ),
            FeatureField(
                name="town_transaction_volume",
                dtype="int64",
                nullable=True,
                description="Transaction count in town (trailing 6 months)",
            ),
            FeatureField(
                name="town_price_trend",
                dtype="float64",
                nullable=True,
                description="6-month price change % in town",
            ),
        ],
        entity_id_column="transaction_id",
        timestamp_column="transaction_date",
        version=2,
    )
