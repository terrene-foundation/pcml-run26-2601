# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# MLFP01 — Exercise 8: Data Cleaning and End-to-End Project
# ════════════════════════════════════════════════════════════════════════
#
# WHAT YOU'LL LEARN:
#   After completing this exercise, you will be able to:
#   - Build a complete data pipeline from raw data to clean output
#   - Profile raw data, translate alerts into cleaning actions, and verify
#   - Use PreprocessingPipeline for automated encoding, scaling, imputation
#   - Structure a multi-stage pipeline using all three M1 Kailash engines
#   - Measure data quality improvement quantitatively (alerts before/after)
#
# PREREQUISITES: Complete Exercises 1-7 (all of Module 1).
#
# ESTIMATED TIME: ~150-180 min (capstone exercise — the longest in M1)
#
# TASKS:
#   1.  Load and inspect messy Singapore taxi trip data
#   2.  Manual quality analysis — range checks, null patterns
#   3.  Profile raw data with DataExplorer
#   4.  Translate alerts into a cleaning action plan
#   5.  Clean the data — GPS, fare, duration filters
#   6.  Engineer temporal features (hour, weekday, peak period)
#   7.  Engineer spatial features (haversine distance, speed)
#   8.  PreprocessingPipeline — model-ready features
#   9.  Visualise key patterns with ModelVisualizer
#   10. Re-profile cleaned data and generate quality report
#
# DATASET: Singapore taxi trip data (deliberately messy)
#   Source: Singapore Land Transport Authority (LTA) / synthetic extension
#   Quality issues by design:
#     - GPS coordinates outside Singapore's bounding box
#     - Negative and extreme fare outliers
#     - Zero-length and unrealistically long trips
#     - Missing pickup/dropoff coordinates
#     - Schema drift across collection years
#
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import asyncio
import os

import polars as pl
from kailash_ml import DataExplorer, ModelVisualizer, PreprocessingPipeline
from kailash_ml import AlertConfig

from shared import MLFPDataLoader


# ── Data Loading ──────────────────────────────────────────────────────
loader = MLFPDataLoader()
taxi_raw = loader.load("mlfp01", "sg_taxi_trips.parquet")

print("=" * 60)
print("  MLFP01 Exercise 8: Data Cleaning and End-to-End Project")
print("=" * 60)
print(f"\n  Data loaded: sg_taxi_trips.parquet")
print(f"    {taxi_raw.height:,} rows | {taxi_raw.width} columns")
print(f"  You're ready to start!\n")


# ══════════════════════════════════════════════════════════════════════
# TASK 1: Load and inspect — understand the mess before touching it
# ══════════════════════════════════════════════════════════════════════
# Rule: never clean data blindly. First describe what is wrong, then
# decide what to do about each problem.

print("=== Raw Taxi Trip Data ===")
print(f"Shape: {taxi_raw.shape}")
print(f"Columns: {taxi_raw.columns}")
print(f"\nData types:")
for col, dtype in zip(taxi_raw.columns, taxi_raw.dtypes):
    print(f"  {col:>30}: {dtype}")

print(f"\nFirst 5 rows:")
print(taxi_raw.head(5))

# Basic statistics
print(f"\n=== describe() ===")
print(taxi_raw.describe())

# ── Checkpoint 1 ─────────────────────────────────────────────────────
assert taxi_raw.height > 0, "Raw taxi dataset is empty"
assert taxi_raw.width >= 3, "Should have at least 3 columns"
print("\n✓ Checkpoint 1 passed — raw data loaded and inspected\n")


# ══════════════════════════════════════════════════════════════════════
# TASK 2: Manual quality analysis — ranges, nulls, suspicious values
# ══════════════════════════════════════════════════════════════════════

# --- 2a: Null counts ---
print("=== Null Analysis ===")
null_cols = []
for col in taxi_raw.columns:
    nc = taxi_raw[col].null_count()
    if nc > 0:
        pct = nc / taxi_raw.height
        null_cols.append({"column": col, "nulls": nc, "pct": pct})
        print(f"  {col:>30}: {nc:>8,} nulls ({pct:>6.1%})")

if not null_cols:
    print("  No null values found!")

# --- 2b: Numeric range check ---
numeric_dtypes = (pl.Float64, pl.Float32, pl.Int64, pl.Int32)
numeric_cols = [
    c for c, d in zip(taxi_raw.columns, taxi_raw.dtypes) if d in numeric_dtypes
]

print(f"\n=== Numeric Column Ranges ===")
print(f"  {'Column':>30} {'Min':>12} {'Max':>12} {'Mean':>12} {'Nulls':>8}")
print(f"  {'─' * 76}")
for col in numeric_cols:
    series = taxi_raw[col].drop_nulls()
    if series.len() > 0:
        print(
            f"  {col:>30} {series.min():>12.3g} {series.max():>12.3g} "
            f"{series.mean():>12.3g} {taxi_raw[col].null_count():>8,}"
        )

# --- 2c: Identify specific quality issues ---
lat_cols = [c for c in taxi_raw.columns if "lat" in c.lower()]
lng_cols = [c for c in taxi_raw.columns if "lng" in c.lower() or "lon" in c.lower()]

SG_LAT_MIN, SG_LAT_MAX = 1.15, 1.47
SG_LNG_MIN, SG_LNG_MAX = 103.60, 104.05

quality_issues = []

# Check GPS bounds
for lat_col in lat_cols:
    series = taxi_raw[lat_col].drop_nulls()
    out_of_bounds = series.filter((series < SG_LAT_MIN) | (series > SG_LAT_MAX)).len()
    if out_of_bounds > 0:
        quality_issues.append(
            f"GPS: {lat_col} has {out_of_bounds:,} out-of-bounds values"
        )

for lng_col in lng_cols:
    series = taxi_raw[lng_col].drop_nulls()
    out_of_bounds = series.filter((series < SG_LNG_MIN) | (series > SG_LNG_MAX)).len()
    if out_of_bounds > 0:
        quality_issues.append(
            f"GPS: {lng_col} has {out_of_bounds:,} out-of-bounds values"
        )

# Check fare
if "fare" in taxi_raw.columns:
    neg_fares = taxi_raw.filter(pl.col("fare") < 0).height
    zero_fares = taxi_raw.filter(pl.col("fare") == 0).height
    if neg_fares > 0:
        quality_issues.append(f"Fare: {neg_fares:,} negative fares")
    if zero_fares > 0:
        quality_issues.append(f"Fare: {zero_fares:,} zero fares")

# Check trip duration
if "trip_duration_sec" in taxi_raw.columns:
    zero_dur = taxi_raw.filter(pl.col("trip_duration_sec") <= 0).height
    very_long = taxi_raw.filter(pl.col("trip_duration_sec") > 10_800).height
    if zero_dur > 0:
        quality_issues.append(f"Duration: {zero_dur:,} zero/negative durations")
    if very_long > 0:
        quality_issues.append(f"Duration: {very_long:,} trips > 3 hours")

print(f"\n=== Identified Quality Issues ({len(quality_issues)}) ===")
for i, issue in enumerate(quality_issues, 1):
    print(f"  {i}. {issue}")
# INTERPRETATION: Ranges reveal red flags:
# - Negative fares: impossible — data entry errors
# - Lat outside [1.15, 1.47]: outside Singapore
# - Duration = 0: meter malfunction or cancelled trip

# ── Checkpoint 2 ─────────────────────────────────────────────────────
assert len(numeric_cols) > 0, "Should have numeric columns"
print("\n✓ Checkpoint 2 passed — manual quality analysis complete\n")


# ══════════════════════════════════════════════════════════════════════
# TASK 3: Profile raw data with DataExplorer
# ══════════════════════════════════════════════════════════════════════


async def profile_raw_data():
    """Profile raw taxi data and return alerts."""
    alert_config = AlertConfig(
        high_null_pct_threshold=0.02,
        skewness_threshold=2.0,
        high_cardinality_ratio=0.80,
        zero_pct_threshold=0.10,
        high_correlation_threshold=0.90,
    )

    explorer = DataExplorer(alert_config=alert_config)
    sample_size = min(200_000, taxi_raw.height)
    taxi_sample = taxi_raw.sample(n=sample_size, seed=42)

    print(f"\n=== DataExplorer Profile (n={sample_size:,}) ===")
    profile = await explorer.profile(taxi_sample)

    print(f"Rows: {profile.n_rows}  Columns: {profile.n_columns}")
    print(f"Duplicates: {profile.duplicate_count} ({profile.duplicate_pct:.1%})")

    # Categorise alerts
    alert_categories: dict[str, list] = {}
    for alert in profile.alerts:
        alert_type = alert["type"]
        if alert_type not in alert_categories:
            alert_categories[alert_type] = []
        alert_categories[alert_type].append(alert)

    print(f"\n--- Alert Summary ({len(profile.alerts)} total) ---")
    for alert_type, alerts in sorted(alert_categories.items()):
        print(f"  {alert_type}: {len(alerts)} alerts")
        for alert in alerts[:3]:
            col = alert.get("column", "N/A")
            severity = alert["severity"]
            print(f"    [{severity.upper()}] {col}")

    return profile, alert_categories


profile_raw, alert_categories = asyncio.run(profile_raw_data())

# ── Checkpoint 3 ─────────────────────────────────────────────────────
assert profile_raw is not None
assert profile_raw.n_rows > 0
print("\n✓ Checkpoint 3 passed — DataExplorer profile complete\n")


# ══════════════════════════════════════════════════════════════════════
# TASK 4: Translate alerts into a cleaning action plan
# ══════════════════════════════════════════════════════════════════════

cleaning_plan: list[dict] = []

for alert in profile_raw.alerts:
    col = alert.get("column", "N/A")
    alert_type = alert["type"]

    if alert_type == "high_nulls":
        cleaning_plan.append(
            {
                "action": f"Handle nulls in '{col}'",
                "method": "Drop rows if critical (coordinates) or impute (numeric)",
                "priority": "high",
            }
        )
    elif alert_type == "high_skewness":
        cleaning_plan.append(
            {
                "action": f"Investigate outliers in '{col}'",
                "method": "Check min/max, apply domain filters, consider log transform",
                "priority": "high",
            }
        )
    elif alert_type == "high_zeros":
        cleaning_plan.append(
            {
                "action": f"Verify zeros in '{col}'",
                "method": "Determine if zeros are real measurements or missing data",
                "priority": "medium",
            }
        )
    elif alert_type == "duplicates":
        cleaning_plan.append(
            {
                "action": "Remove duplicate rows",
                "method": "df.unique() before modelling",
                "priority": "medium",
            }
        )
    elif alert_type == "high_correlation":
        cleaning_plan.append(
            {
                "action": f"Review collinearity: {col}",
                "method": "Consider dropping one of the highly correlated features",
                "priority": "low",
            }
        )

print(f"=== Cleaning Action Plan ({len(cleaning_plan)} actions) ===")
for i, plan in enumerate(cleaning_plan, 1):
    print(f"  {i}. [{plan['priority'].upper():>6}] {plan['action']}")
    print(f"             Method: {plan['method']}")

# ── Checkpoint 4 ─────────────────────────────────────────────────────
assert isinstance(cleaning_plan, list)
print("\n✓ Checkpoint 4 passed — cleaning action plan created\n")


# ══════════════════════════════════════════════════════════════════════
# TASK 5: Clean the data — GPS, fare, duration filters
# ══════════════════════════════════════════════════════════════════════
# Each cleaning step maps to a specific alert or observation above.

taxi_clean = taxi_raw.clone()
rows_before = taxi_clean.height
cleaning_log: list[str] = []

# --- 5a: Remove GPS coordinates outside Singapore ---
for lat_col in lat_cols:
    before = taxi_clean.height
    taxi_clean = taxi_clean.filter(
        pl.col(lat_col).is_null()
        | ((pl.col(lat_col) >= SG_LAT_MIN) & (pl.col(lat_col) <= SG_LAT_MAX))
    )
    removed = before - taxi_clean.height
    if removed > 0:
        msg = f"GPS filter ({lat_col}): removed {removed:,} out-of-bounds rows"
        cleaning_log.append(msg)
        print(msg)

for lng_col in lng_cols:
    before = taxi_clean.height
    taxi_clean = taxi_clean.filter(
        pl.col(lng_col).is_null()
        | ((pl.col(lng_col) >= SG_LNG_MIN) & (pl.col(lng_col) <= SG_LNG_MAX))
    )
    removed = before - taxi_clean.height
    if removed > 0:
        msg = f"GPS filter ({lng_col}): removed {removed:,} out-of-bounds rows"
        cleaning_log.append(msg)
        print(msg)

# --- 5b: Remove fare outliers ---
if "fare" in taxi_clean.columns:
    # Remove negative and zero fares
    before = taxi_clean.height
    taxi_clean = taxi_clean.filter(pl.col("fare") > 0)
    removed = before - taxi_clean.height
    if removed > 0:
        msg = f"Non-positive fare filter: removed {removed:,} rows"
        cleaning_log.append(msg)
        print(msg)

    # Cap at 99.9th percentile
    fare_p999 = taxi_clean["fare"].quantile(0.999)
    before = taxi_clean.height
    taxi_clean = taxi_clean.filter(pl.col("fare") <= fare_p999)
    removed = before - taxi_clean.height
    if removed > 0:
        msg = f"Extreme fare cap (>{fare_p999:.0f}): removed {removed:,} rows"
        cleaning_log.append(msg)
        print(msg)

# --- 5c: Remove duration anomalies ---
if "trip_duration_sec" in taxi_clean.columns:
    # Too short: < 60 seconds
    before = taxi_clean.height
    taxi_clean = taxi_clean.filter(pl.col("trip_duration_sec") > 60)
    removed = before - taxi_clean.height
    if removed > 0:
        msg = f"Short trip filter (<60s): removed {removed:,} rows"
        cleaning_log.append(msg)
        print(msg)

    # Too long: > 3 hours
    before = taxi_clean.height
    taxi_clean = taxi_clean.filter(pl.col("trip_duration_sec") <= 10_800)
    removed = before - taxi_clean.height
    if removed > 0:
        msg = f"Long trip filter (>3h): removed {removed:,} rows"
        cleaning_log.append(msg)
        print(msg)

# --- 5d: Drop rows missing critical coordinates ---
critical_cols = lat_cols + lng_cols
if critical_cols:
    before = taxi_clean.height
    taxi_clean = taxi_clean.drop_nulls(subset=critical_cols)
    removed = before - taxi_clean.height
    if removed > 0:
        msg = f"Null coordinate filter: removed {removed:,} rows"
        cleaning_log.append(msg)
        print(msg)

# --- 5e: Remove duplicates ---
before = taxi_clean.height
taxi_clean = taxi_clean.unique()
removed = before - taxi_clean.height
if removed > 0:
    msg = f"Deduplication: removed {removed:,} duplicate rows"
    cleaning_log.append(msg)
    print(msg)

retention_pct = taxi_clean.height / rows_before * 100
print(f"\n=== Cleaning Summary ===")
print(
    f"  Rows: {rows_before:,} -> {taxi_clean.height:,} ({retention_pct:.1f}% retained)"
)
print(f"  Steps applied: {len(cleaning_log)}")
for step in cleaning_log:
    print(f"    - {step}")

# ── Checkpoint 5 ─────────────────────────────────────────────────────
assert taxi_clean.height > 0, "Cleaning removed all rows"
assert taxi_clean.height <= taxi_raw.height, "Cleaning should not add rows"
if "fare" in taxi_clean.columns:
    assert (taxi_clean["fare"] > 0).all(), "All fares should be positive"
print("\n✓ Checkpoint 5 passed — data cleaned and validated\n")


# ══════════════════════════════════════════════════════════════════════
# TASK 6: Engineer temporal features
# ══════════════════════════════════════════════════════════════════════

# --- 6a: Parse datetime columns ---
datetime_cols = [
    c for c in taxi_clean.columns if "time" in c.lower() or "date" in c.lower()
]
for col in datetime_cols:
    if taxi_clean[col].dtype == pl.Utf8:
        taxi_clean = taxi_clean.with_columns(pl.col(col).str.to_datetime().alias(col))

# --- 6b: Extract pickup time features ---
pickup_col = next(
    (
        c
        for c in taxi_clean.columns
        if "pickup" in c.lower() and ("time" in c.lower() or "date" in c.lower())
    ),
    None,
)

if pickup_col:
    taxi_clean = taxi_clean.with_columns(
        pl.col(pickup_col).dt.hour().alias("hour_of_day"),
        pl.col(pickup_col).dt.weekday().alias("day_of_week"),
        pl.col(pickup_col).dt.month().alias("month"),
        pl.col(pickup_col).dt.day().alias("day_of_month"),
        (pl.col(pickup_col).dt.weekday() >= 5).alias("is_weekend"),
    )

    # --- 6c: Peak-hour classification ---
    taxi_clean = taxi_clean.with_columns(
        pl.when((pl.col("hour_of_day") >= 7) & (pl.col("hour_of_day") <= 9))
        .then(pl.lit("morning_peak"))
        .when((pl.col("hour_of_day") >= 17) & (pl.col("hour_of_day") <= 20))
        .then(pl.lit("evening_peak"))
        .when((pl.col("hour_of_day") >= 22) | (pl.col("hour_of_day") <= 5))
        .then(pl.lit("late_night"))
        .otherwise(pl.lit("off_peak"))
        .alias("time_period")
    )

    # --- 6d: Day type classification ---
    taxi_clean = taxi_clean.with_columns(
        pl.when(pl.col("day_of_week") >= 5)
        .then(pl.lit("weekend"))
        .when(pl.col("day_of_week") == 4)
        .then(pl.lit("friday"))
        .otherwise(pl.lit("weekday"))
        .alias("day_type")
    )

# --- 6e: Duration features ---
if "trip_duration_sec" in taxi_clean.columns:
    taxi_clean = taxi_clean.with_columns(
        (pl.col("trip_duration_sec") / 60).alias("trip_duration_min"),
        pl.when(pl.col("trip_duration_sec") < 600)
        .then(pl.lit("short"))
        .when(pl.col("trip_duration_sec") < 1800)
        .then(pl.lit("medium"))
        .otherwise(pl.lit("long"))
        .alias("trip_length_category"),
    )

temporal_cols = [
    c
    for c in taxi_clean.columns
    if c
    in (
        "hour_of_day",
        "day_of_week",
        "month",
        "is_weekend",
        "time_period",
        "day_type",
        "trip_duration_min",
        "trip_length_category",
        "day_of_month",
    )
]
print(f"=== Temporal Features ({len(temporal_cols)}) ===")
if temporal_cols:
    print(taxi_clean.select(temporal_cols).head(5))

# ── Checkpoint 6 ─────────────────────────────────────────────────────
if pickup_col:
    assert "hour_of_day" in taxi_clean.columns
    assert "time_period" in taxi_clean.columns
    assert "day_type" in taxi_clean.columns
    time_periods = set(taxi_clean["time_period"].unique().to_list())
    assert "morning_peak" in time_periods
print("\n✓ Checkpoint 6 passed — temporal features engineered\n")


# ══════════════════════════════════════════════════════════════════════
# TASK 7: Engineer spatial features
# ══════════════════════════════════════════════════════════════════════

# --- 7a: Haversine distance ---
if lat_cols and lng_cols and len(lat_cols) >= 2 and len(lng_cols) >= 2:
    pickup_lat, dropoff_lat = lat_cols[0], lat_cols[1]
    pickup_lng, dropoff_lng = lng_cols[0], lng_cols[1]

    _RAD = pl.lit(3.141592653589793 / 180)

    taxi_clean = taxi_clean.with_columns(
        (
            2
            * 6371
            * (
                (
                    ((pl.col(dropoff_lat) - pl.col(pickup_lat)) * _RAD / 2).sin().pow(2)
                    + (pl.col(pickup_lat) * _RAD).cos()
                    * (pl.col(dropoff_lat) * _RAD).cos()
                    * ((pl.col(dropoff_lng) - pl.col(pickup_lng)) * _RAD / 2)
                    .sin()
                    .pow(2)
                )
                .sqrt()
                .arcsin()
            )
        ).alias("haversine_km")
    )

    # --- 7b: Average speed ---
    if "trip_duration_sec" in taxi_clean.columns:
        taxi_clean = taxi_clean.with_columns(
            (pl.col("haversine_km") / (pl.col("trip_duration_sec") / 3600)).alias(
                "avg_speed_kmh"
            )
        )

        # Remove impossible speeds (> 120 km/h in Singapore)
        before = taxi_clean.height
        taxi_clean = taxi_clean.filter(
            (pl.col("avg_speed_kmh") > 0) & (pl.col("avg_speed_kmh") <= 120)
        )
        removed = before - taxi_clean.height
        if removed > 0:
            print(f"Speed filter (>120 km/h): removed {removed:,} rows")

    # --- 7c: Distance category ---
    taxi_clean = taxi_clean.with_columns(
        pl.when(pl.col("haversine_km") < 3)
        .then(pl.lit("short_distance"))
        .when(pl.col("haversine_km") < 8)
        .then(pl.lit("medium_distance"))
        .when(pl.col("haversine_km") < 15)
        .then(pl.lit("long_distance"))
        .otherwise(pl.lit("cross_island"))
        .alias("distance_category"),
    )

    # --- 7d: Fare per km ---
    if "fare" in taxi_clean.columns:
        taxi_clean = taxi_clean.with_columns(
            (pl.col("fare") / pl.col("haversine_km")).alias("fare_per_km")
        )

        # Filter extreme fare_per_km
        fare_km_p99 = taxi_clean["fare_per_km"].quantile(0.99)
        before = taxi_clean.height
        taxi_clean = taxi_clean.filter(
            (pl.col("fare_per_km") > 0) & (pl.col("fare_per_km") <= fare_km_p99)
        )
        removed = before - taxi_clean.height
        if removed > 0:
            print(f"Fare/km filter: removed {removed:,} rows")

spatial_cols = [
    c
    for c in taxi_clean.columns
    if c in ("haversine_km", "avg_speed_kmh", "distance_category", "fare_per_km")
]
print(f"\n=== Spatial Features ({len(spatial_cols)}) ===")
if spatial_cols:
    print(taxi_clean.select(spatial_cols).head(5))

new_cols = [c for c in taxi_clean.columns if c not in taxi_raw.columns]
print(f"\nTotal new features: {len(new_cols)}")

# ── Checkpoint 7 ─────────────────────────────────────────────────────
if "haversine_km" in taxi_clean.columns:
    assert (taxi_clean["haversine_km"].drop_nulls() >= 0).all()
if "avg_speed_kmh" in taxi_clean.columns:
    assert (taxi_clean["avg_speed_kmh"].drop_nulls() <= 120).all()
print("\n✓ Checkpoint 7 passed — spatial features engineered\n")


# ══════════════════════════════════════════════════════════════════════
# TASK 8: PreprocessingPipeline — model-ready features
# ══════════════════════════════════════════════════════════════════════
# PreprocessingPipeline automates: train/test split, imputation,
# scaling, categorical encoding.

# --- 8a: Select feature columns ---
exclude = set(["fare", "fare_per_km"] + datetime_cols + lat_cols + lng_cols)
feature_cols = [
    c
    for c in taxi_clean.columns
    if c not in exclude
    and taxi_clean[c].dtype
    in (
        pl.Float64,
        pl.Float32,
        pl.Int64,
        pl.Int32,
        pl.Utf8,
        pl.Boolean,
        pl.Categorical,
    )
]

# --- 8b: Cast strings to Categorical ---
for col in feature_cols:
    if taxi_clean[col].dtype == pl.Utf8:
        taxi_clean = taxi_clean.with_columns(pl.col(col).cast(pl.Categorical))

# --- 8c: Run PreprocessingPipeline ---
taxi_sample = taxi_clean.sample(n=min(50_000, taxi_clean.height), seed=42)
result = None

if "fare" in taxi_sample.columns:
    pipeline_cols = [c for c in feature_cols if c in taxi_sample.columns] + ["fare"]
    pipeline_df = taxi_sample.select(pipeline_cols)

    pipeline = PreprocessingPipeline()
    result = pipeline.setup(
        data=pipeline_df,
        target="fare",
        train_size=0.8,
        seed=42,
        normalize=True,
        categorical_encoding="onehot",
        imputation_strategy="median",
    )

    print(f"=== PreprocessingPipeline Result ===")
    print(result.summary)
    print(f"  Task type:     {result.task_type}")
    print(f"  Train shape:   {result.train_data.shape}")
    print(f"  Test shape:    {result.test_data.shape}")
    print(f"  Numeric feats: {len(result.numeric_columns)}")
    print(f"  Cat feats:     {len(result.categorical_columns)}")

    # --- 8d: Inspect the processed features ---
    print(f"\n  Numeric columns: {result.numeric_columns[:10]}...")
    print(f"  Categorical columns: {result.categorical_columns[:10]}...")

    # --- 8e: Verify train/test split ---
    total_rows = result.train_data.shape[0] + result.test_data.shape[0]
    train_pct = result.train_data.shape[0] / total_rows * 100
    print(f"\n  Train: {result.train_data.shape[0]:,} ({train_pct:.0f}%)")
    print(f"  Test:  {result.test_data.shape[0]:,} ({100 - train_pct:.0f}%)")
else:
    print("\n'fare' column not found — skipping PreprocessingPipeline")

# ── Checkpoint 8 ─────────────────────────────────────────────────────
if result is not None:
    assert result.task_type == "regression"
    total_rows = result.train_data.shape[0] + result.test_data.shape[0]
    assert abs(total_rows - taxi_sample.height) <= 1
print("\n✓ Checkpoint 8 passed — PreprocessingPipeline complete\n")


# ══════════════════════════════════════════════════════════════════════
# TASK 9: Visualise key patterns with ModelVisualizer
# ══════════════════════════════════════════════════════════════════════

viz = ModelVisualizer()
os.makedirs("charts", exist_ok=True)
viz_files: list[str] = []

# --- 9a: Fare distribution (cleaned) ---
if "fare" in taxi_clean.columns:
    clean_fares = taxi_clean["fare"].drop_nulls().to_list()
    fig_dist = viz.feature_distribution(
        values=clean_fares[:50_000],
        feature_name="Fare (S$) — Cleaned",
    )
    fig_dist.update_layout(title="Taxi Fare Distribution (After Cleaning)")
    fig_dist.write_html("charts/ex8_fare_distribution.html")
    viz_files.append("charts/ex8_fare_distribution.html")
    print("Saved: charts/ex8_fare_distribution.html")

# --- 9b: Trip metrics by time period ---
if "time_period" in taxi_clean.columns and "fare" in taxi_clean.columns:
    periods = ["morning_peak", "evening_peak", "off_peak", "late_night"]
    time_metrics: dict[str, dict[str, float]] = {}

    for period in periods:
        subset = taxi_clean.filter(pl.col("time_period") == period)
        if subset.height > 0:
            metrics_dict: dict[str, float] = {
                "avg_fare": float(subset["fare"].mean() or 0),
                "trip_count": float(subset.height),
            }
            if "haversine_km" in subset.columns:
                metrics_dict["avg_distance_km"] = float(
                    subset["haversine_km"].mean() or 0
                )
            if "avg_speed_kmh" in subset.columns:
                metrics_dict["avg_speed_kmh"] = float(
                    subset["avg_speed_kmh"].mean() or 0
                )
            time_metrics[period] = metrics_dict

    if time_metrics:
        fig_periods = viz.metric_comparison(time_metrics)
        fig_periods.update_layout(title="Trip Metrics by Time Period")
        fig_periods.write_html("charts/ex8_time_period_metrics.html")
        viz_files.append("charts/ex8_time_period_metrics.html")
        print("Saved: charts/ex8_time_period_metrics.html")

# --- 9c: Hourly trip volume ---
if "hour_of_day" in taxi_clean.columns:
    hourly = (
        taxi_clean.group_by("hour_of_day")
        .agg(pl.len().alias("trip_count"))
        .sort("hour_of_day")
    )

    fig_hourly = viz.training_history(
        metrics={"Trip Volume": hourly["trip_count"].to_list()},
        x_label="Hour of Day",
        y_label="Number of Trips",
    )
    fig_hourly.update_layout(title="Taxi Trip Volume by Hour of Day")
    fig_hourly.write_html("charts/ex8_hourly_volume.html")
    viz_files.append("charts/ex8_hourly_volume.html")
    print("Saved: charts/ex8_hourly_volume.html")

# --- 9d: Distance distribution ---
if "haversine_km" in taxi_clean.columns:
    distances = taxi_clean["haversine_km"].drop_nulls().to_list()
    fig_dist_km = viz.feature_distribution(
        values=distances[:50_000],
        feature_name="Trip Distance (km)",
    )
    fig_dist_km.update_layout(title="Trip Distance Distribution")
    fig_dist_km.write_html("charts/ex8_distance_distribution.html")
    viz_files.append("charts/ex8_distance_distribution.html")
    print("Saved: charts/ex8_distance_distribution.html")

# --- 9e: Day type comparison ---
if "day_type" in taxi_clean.columns and "fare" in taxi_clean.columns:
    day_metrics = {}
    for dt in ["weekday", "friday", "weekend"]:
        subset = taxi_clean.filter(pl.col("day_type") == dt)
        if subset.height > 0:
            day_metrics[dt] = {
                "avg_fare": float(subset["fare"].mean() or 0),
                "trips": float(subset.height),
            }
    if day_metrics:
        fig_day = viz.metric_comparison(day_metrics)
        fig_day.update_layout(title="Trip Metrics: Weekday vs Friday vs Weekend")
        fig_day.write_html("charts/ex8_day_type_comparison.html")
        viz_files.append("charts/ex8_day_type_comparison.html")
        print("Saved: charts/ex8_day_type_comparison.html")

# --- 9f: Feature list (if pipeline ran) ---
if result is not None:
    all_features = result.numeric_columns + result.categorical_columns
    feat_metrics = {
        f: {"Weight": 1.0 / max(len(all_features), 1)} for f in all_features[:20]
    }
    fig_feats = viz.metric_comparison(feat_metrics)
    fig_feats.update_layout(title="Pipeline Features (Equal Weight Placeholder)")
    fig_feats.write_html("charts/ex8_feature_list.html")
    viz_files.append("charts/ex8_feature_list.html")
    print("Saved: charts/ex8_feature_list.html")

# ── Checkpoint 9 ─────────────────────────────────────────────────────
assert len(viz_files) > 0, "Should have saved at least one chart"
for f in viz_files:
    assert os.path.exists(f), f"Missing: {f}"
print(f"\n✓ Checkpoint 9 passed — {len(viz_files)} visualisation files saved\n")


# ══════════════════════════════════════════════════════════════════════
# TASK 10: Re-profile cleaned data and generate quality report
# ══════════════════════════════════════════════════════════════════════


async def profile_and_report():
    """Re-profile cleaned data, compare to raw, and generate report."""
    explorer = DataExplorer(
        alert_config=AlertConfig(
            high_null_pct_threshold=0.01,
            skewness_threshold=2.0,
        )
    )

    sample = taxi_clean.sample(n=min(200_000, taxi_clean.height), seed=42)
    profile_clean = await explorer.profile(sample)

    # --- Compare alerts ---
    print(f"\n=== Data Quality: Before vs After Cleaning ===")
    print(f"  Rows before:   {taxi_raw.height:,}")
    print(f"  Rows after:    {taxi_clean.height:,}")
    print(f"  Retention:     {taxi_clean.height / taxi_raw.height:.1%}")
    print(f"  Alerts before: {len(profile_raw.alerts)}")
    print(f"  Alerts after:  {len(profile_clean.alerts)}")

    alert_reduction = len(profile_raw.alerts) - len(profile_clean.alerts)
    if alert_reduction > 0:
        print(f"  Improvement:   {alert_reduction} fewer alerts")
    elif alert_reduction == 0:
        print(f"  No change in alert count")
    else:
        print(f"  Warning: {abs(alert_reduction)} MORE alerts after cleaning")

    # Remaining alerts
    if profile_clean.alerts:
        print(f"\n  Remaining alerts ({len(profile_clean.alerts)}):")
        for alert in profile_clean.alerts:
            print(
                f"    [{alert['severity'].upper()}] {alert['type']}: "
                f"{alert.get('column', 'N/A')}"
            )
    else:
        print(f"\n  No remaining alerts — data quality confirmed clean.")

    # --- Generate HTML report ---
    report_html = await explorer.to_html(
        sample,
        title="Singapore Taxi Trips — Cleaned Data Profile",
    )
    with open("ex8_taxi_profile_clean.html", "w") as f:
        f.write(report_html)
    print(f"\nSaved: ex8_taxi_profile_clean.html")

    return profile_clean


try:
    profile_clean = asyncio.run(profile_and_report())
except Exception as exc:
    print(f"\n[ERROR] Post-cleaning profile failed: {exc}")
    raise


# ── Checkpoint 10 ────────────────────────────────────────────────────
assert profile_clean is not None
assert os.path.exists("ex8_taxi_profile_clean.html")
print("\n✓ Checkpoint 10 passed — post-cleaning profile and report complete\n")


# ── Pipeline summary ─────────────────────────────────────────────────
print(f"\n{'═' * 65}")
print(f"  END-TO-END PIPELINE SUMMARY")
print(f"{'═' * 65}")
print(f"  Stage 1  Load:       {taxi_raw.height:,} rows, {taxi_raw.width} cols")
print(f"  Stage 2  Inspect:    {len(quality_issues)} quality issues identified")
print(f"  Stage 3  Profile:    {len(profile_raw.alerts)} alerts from DataExplorer")
print(f"  Stage 4  Plan:       {len(cleaning_plan)} cleaning actions defined")
print(
    f"  Stage 5  Clean:      {taxi_clean.height:,} rows retained ({retention_pct:.1f}%)"
)
print(f"  Stage 6  Temporal:   {len(temporal_cols)} temporal features")
print(f"  Stage 7  Spatial:    {len(spatial_cols)} spatial features")
if result is not None:
    print(
        f"  Stage 8  Pipeline:   {result.train_data.shape[0]:,} train / "
        f"{result.test_data.shape[0]:,} test"
    )
print(f"  Stage 9  Visualise:  {len(viz_files)} charts saved")
print(f"  Stage 10 Verify:     {len(profile_clean.alerts)} alerts remaining")
print(f"{'═' * 65}")

print(
    "\n✓ Exercise 8 complete — full pipeline: load -> profile -> clean -> "
    "engineer -> preprocess -> visualise -> verify"
)


# ══════════════════════════════════════════════════════════════════════
# REFLECTION
# ══════════════════════════════════════════════════════════════════════
print("═" * 60)
print("  WHAT YOU'VE MASTERED")
print("═" * 60)
print(
    """
  ✓ End-to-end thinking: load -> inspect -> profile -> plan -> clean
    -> engineer -> preprocess -> visualise -> verify
  ✓ Manual quality checks: null counts, range validation, domain rules
  ✓ DataExplorer: automated profiling with typed, actionable alerts
  ✓ Action planning: translating alerts into specific cleaning steps
  ✓ Domain-aware cleaning: GPS bounding boxes, fare ranges, duration limits
  ✓ Feature engineering:
    - Temporal: hour, weekday, peak period, day type, duration category
    - Spatial: haversine distance, average speed, distance category
    - Derived: fare per km
  ✓ PreprocessingPipeline: standardise, encode, impute, and split in one call
  ✓ ModelVisualizer: fare distributions, hourly patterns, segment comparisons
  ✓ Quality measurement: alert count before vs after as a cleaning KPI
  ✓ Three engines together: DataExplorer + PreprocessingPipeline +
    ModelVisualizer — the full M1 toolkit

  MODULE 1 COMPLETE — you've gone from raw CSV to model-ready data.

  NEXT — MODULE 2: Feature Engineering and Experiment Design
  In M2, you'll move beyond data exploration into systematic feature
  construction. You'll learn:
    - FeatureEngineer: automated feature generation (interactions,
      polynomials, target encoding, lag features) at scale
    - FeatureStore: versioning and retrieving feature sets
    - ExperimentTracker: logging runs, parameters, and metrics
  The taxi trip data you cleaned here will become a feature store entry.
"""
)
