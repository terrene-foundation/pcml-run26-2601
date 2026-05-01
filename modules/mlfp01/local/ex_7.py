# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# MLFP01 — Exercise 7: Automated Data Profiling
# ════════════════════════════════════════════════════════════════════════
#
# WHAT YOU'LL LEARN:
#   After completing this exercise, you will be able to:
#   - Run automated data profiling on any dataset using DataExplorer
#   - Configure AlertConfig thresholds for data quality rules
#   - Compare two datasets and identify distribution differences
#   - Handle errors gracefully with try/except
#   - Use async functions and asyncio.run() in a real pipeline context
#
# PREREQUISITES: Complete Exercise 6 first (all of Exercises 1-6).
#
# ESTIMATED TIME: ~150-180 min
#
# TASKS:
#   1.  Load and inspect Singapore economic indicators
#   2.  Merge datasets with different reporting frequencies
#   3.  Configure AlertConfig with domain-appropriate thresholds
#   4.  Run async profiling and interpret each alert type
#   5.  Column-level profile deep dive
#   6.  Spearman correlation analysis
#   7.  Compare pre-COVID vs COVID-era data quality
#   8.  try/except — error handling patterns
#   9.  Generate self-contained HTML profiling report
#   10. Data quality scorecard and action plan
#
# DATASET: Three Singapore economic time-series datasets (deliberately messy):
#   - sg_cpi.csv:         Monthly Consumer Price Index (data.gov.sg / SingStat)
#   - sg_employment.csv:  Quarterly labour market statistics (MOM)
#   - sg_fx_rates.csv:    Daily SGD exchange rates (MAS)
#   Quality issues by design: mixed granularity, forward-fill gaps,
#   COVID-era outliers, and near-zero values in some trade-flow columns.
#
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import asyncio

import polars as pl
from kailash_ml import DataExplorer, ModelVisualizer
from kailash_ml import AlertConfig

from shared import MLFPDataLoader

viz = ModelVisualizer()


# ── Data Loading ──────────────────────────────────────────────────────
loader = MLFPDataLoader()
cpi = loader.load("mlfp01", "sg_cpi.csv")
employment = loader.load("mlfp01", "sg_employment.csv")
fx_rates = loader.load("mlfp01", "sg_fx_rates.csv")

print("=" * 60)
print("  MLFP01 Exercise 7: Automated Data Profiling")
print("=" * 60)
print(f"\n  Data loaded:")
print(f"    sg_cpi.csv        ({cpi.height:,} rows — monthly)")
print(f"    sg_employment.csv ({employment.height:,} rows — quarterly)")
print(f"    sg_fx_rates.csv   ({fx_rates.height:,} rows — daily)")
print(f"  You're ready to start!\n")


# ══════════════════════════════════════════════════════════════════════
# TASK 1: Inspect each dataset independently
# ══════════════════════════════════════════════════════════════════════
# Always profile individual tables before merging — that way you know
# which quality issues originate where.

# --- 1a: CPI data ---
print("=== CPI Data (Monthly) ===")
print(f"Shape: {cpi.shape}")
print(f"Columns: {cpi.columns}")
print(f"Date range: {cpi['date'].min()} to {cpi['date'].max()}")
print(f"Dtypes: {cpi.dtypes}")
print("Nulls per column:")
for col in cpi.columns:
    null_count = cpi[col].null_count()
    if null_count > 0:
        print(f"  {col}: {null_count} ({null_count / cpi.height:.1%})")
print(cpi.head(5))

# --- 1b: Employment data ---
print("\n=== Employment Data (Quarterly) ===")
print(f"Shape: {employment.shape}")
print(f"Columns: {employment.columns}")
print(f"Dtypes: {employment.dtypes}")
print(employment.head(5))

# --- 1c: FX rates data ---
print("\n=== FX Rates Data (Daily) ===")
print(f"Shape: {fx_rates.shape}")
print(f"Columns: {fx_rates.columns}")
print(f"Dtypes: {fx_rates.dtypes}")
print(fx_rates.head(5))

# --- 1d: Granularity comparison ---
print(f"\n=== Granularity Comparison ===")
print(f"  CPI (monthly):    {cpi.height:>6,} rows")
print(
    f"  Employment (qtr): {employment.height:>6,} rows  (~{cpi.height / max(employment.height, 1):.1f}x fewer)"
)
print(
    f"  FX rates (daily): {fx_rates.height:>6,} rows  (~{fx_rates.height / max(cpi.height, 1):.1f}x more)"
)
# INTERPRETATION: Three datasets, three granularities. Before merging,
# think about which frequency to target. We'll align to monthly.

# ── Checkpoint 1 ─────────────────────────────────────────────────────
assert cpi.height > 0, "CPI dataset is empty"
assert employment.height > 0, "Employment dataset is empty"
assert fx_rates.height > 0, "FX rates dataset is empty"
assert "date" in cpi.columns, "CPI should have a 'date' column"
assert employment.height < cpi.height, "Quarterly should have fewer rows than monthly"
assert fx_rates.height > cpi.height, "Daily should have more rows than monthly"
print("\n✓ Checkpoint 1 passed — all three economic datasets inspected\n")


# ══════════════════════════════════════════════════════════════════════
# TASK 2: Merge datasets with different granularities
# ══════════════════════════════════════════════════════════════════════
# Strategy: align everything to monthly frequency.
#   - CPI:        already monthly — use as-is
#   - Employment: quarterly -> forward-fill across 3 months
#   - FX rates:   daily -> aggregate to monthly mean

# --- 2a: Parse date columns (each has a DIFFERENT format) ---
# CPI: mixed formats ("01/2000", "2000-02", "201108")
cpi = cpi.with_columns(
    pl.col("date")
    .str.replace(r"^(\d{2})/(\d{4})$", "$2-$1-01")
    .str.replace(r"^(\d{4})(\d{2})$", "$1-$2-01")
    .str.replace(r"^(\d{4})-(\d{2})$", "$1-$2-01")
    .str.to_date("%Y-%m-%d")
    .alias("date")
)


# Employment: quarterly ("2000 Q1") -> quarter start month
def quarter_to_date(q_str: str) -> str:
    """Convert '2000 Q1' to '2000-01-01'."""
    parts = q_str.split()
    year = parts[0]
    q = int(parts[1][1])
    month = {1: "01", 2: "04", 3: "07", 4: "10"}[q]
    return f"{year}-{month}-01"


employment = employment.with_columns(
    pl.col("quarter")
    .map_elements(quarter_to_date, return_dtype=pl.String)
    .str.to_date("%Y-%m-%d")
    .alias("date")
)

# FX rates: YYYY-MM-DD — may be auto-detected
if fx_rates["date"].dtype == pl.String:
    fx_rates = fx_rates.with_columns(pl.col("date").str.to_date("%Y-%m-%d"))

# --- 2b: Truncate to first-of-month ---
cpi = cpi.with_columns(pl.col("date").dt.truncate("1mo").alias("month_date"))
employment = employment.with_columns(
    pl.col("date").dt.truncate("1mo").alias("month_date")
)

# --- 2c: Build monthly spine ---
date_range = pl.date_range(
    cpi["month_date"].min(),
    cpi["month_date"].max(),
    interval="1mo",
    eager=True,
)
monthly_spine = pl.DataFrame({"month_date": date_range})

# --- 2d: Forward-fill quarterly employment ---
employment_monthly = (
    monthly_spine.join(employment.drop("date"), on="month_date", how="left")
    .sort("month_date")
    .with_columns(
        [
            pl.col(c).forward_fill()
            for c in employment.columns
            if c not in ("date", "month_date")
        ]
    )
)

# --- 2e: Aggregate daily FX to monthly mean ---
fx_monthly = (
    fx_rates.with_columns(pl.col("date").dt.truncate("1mo").alias("month_date"))
    .group_by("month_date")
    .agg([pl.col(c).mean() for c in fx_rates.columns if c != "date"])
    .sort("month_date")
)

# --- 2f: Merge all three ---
economic = (
    cpi.join(employment_monthly, on="month_date", how="left", suffix="_emp")
    .join(fx_monthly, on="month_date", how="left", suffix="_fx")
    .sort("month_date")
)

print(f"\n=== Merged Economic Dataset ===")
print(f"Shape: {economic.shape}")
print(f"Date range: {economic['month_date'].min()} to {economic['month_date'].max()}")

null_summary = []
for col in economic.columns:
    nc = economic[col].null_count()
    if nc > 0:
        null_summary.append({"column": col, "nulls": nc, "pct": nc / economic.height})
if null_summary:
    print("\nNull summary after merge:")
    for ns in null_summary:
        print(f"  {ns['column']}: {ns['nulls']} ({ns['pct']:.1%})")

# ── Checkpoint 2 ─────────────────────────────────────────────────────
assert economic.height > 0, "Merged dataset is empty"
assert (
    economic.height == monthly_spine.height
), f"Should have {monthly_spine.height} rows, got {economic.height}"
assert "month_date" in economic.columns
print("\n✓ Checkpoint 2 passed — datasets merged to monthly frequency\n")


# ══════════════════════════════════════════════════════════════════════
# TASK 3: Configure AlertConfig with domain-appropriate thresholds
# ══════════════════════════════════════════════════════════════════════
# AlertConfig controls when DataExplorer raises a warning.
# Economic time-series has different characteristics than typical ML data.

# TODO: Create an AlertConfig for economic time-series data
# Use these domain-tuned thresholds:
#   high_correlation_threshold = 0.95  (macro indicators are naturally correlated)
#   high_null_pct_threshold    = 0.10  (forward-fill leaves some edge nulls)
#   constant_threshold         = 1
#   high_cardinality_ratio     = 0.95
#   skewness_threshold         = 3.0   (only flag extreme COVID/GFC shocks)
#   zero_pct_threshold         = 0.30
#   imbalance_ratio_threshold  = 0.05
#   duplicate_pct_threshold    = 0.05
alert_config = AlertConfig(
    high_correlation_threshold=____,  # Hint: 0.95
    high_null_pct_threshold=____,  # Hint: 0.10
    constant_threshold=____,  # Hint: 1
    high_cardinality_ratio=____,  # Hint: 0.95
    skewness_threshold=____,  # Hint: 3.0
    zero_pct_threshold=____,  # Hint: 0.30
    imbalance_ratio_threshold=____,  # Hint: 0.05
    duplicate_pct_threshold=____,  # Hint: 0.05
)

print(f"=== Custom AlertConfig ===")
print(f"  Correlation threshold: {alert_config.high_correlation_threshold}")
print(f"  Null threshold:        {alert_config.high_null_pct_threshold:.0%}")
print(f"  Skewness threshold:    {alert_config.skewness_threshold}")
print(f"  Zero threshold:        {alert_config.zero_pct_threshold:.0%}")
print(f"  Cardinality ratio:     {alert_config.high_cardinality_ratio}")
# INTERPRETATION: Each threshold is a deliberate choice. Raising the
# correlation threshold to 0.95 means structurally correlated macro
# indicators won't trigger false alarms. The skewness threshold of 3.0
# means only extreme COVID/GFC shocks get flagged.

# ── Checkpoint 3 ─────────────────────────────────────────────────────
assert alert_config.high_correlation_threshold == 0.95
assert alert_config.skewness_threshold == 3.0
assert alert_config.high_null_pct_threshold == 0.10
print("\n✓ Checkpoint 3 passed — AlertConfig configured\n")


# ══════════════════════════════════════════════════════════════════════
# TASK 4: Run async profiling and interpret alerts
# ══════════════════════════════════════════════════════════════════════
# DataExplorer.profile() is async — it runs inside asyncio.run().


def _interpret_alert(alert_type: str, column: str, value) -> str:
    """Provide domain-specific interpretation of a DataExplorer alert."""
    interpretations = {
        "high_nulls": (
            f"Column '{column}' has {value:.1%} missing values. "
            "For forward-filled quarterly data, edge nulls are expected."
        ),
        "constant": (
            f"Column '{column}' has <=1 unique value — likely a pipeline failure."
        ),
        "high_skewness": (
            f"Column '{column}' skewness={value:.2f}. "
            "Crisis-period outliers cause this. Consider log-transform or winsorisation."
        ),
        "high_zeros": (
            f"Column '{column}' has {value:.1%} zero values. "
            "Check whether zeros are real measurements or missing-data coded as zero."
        ),
        "high_cardinality": (
            f"Column '{column}' cardinality ratio={value:.3f}. "
            "If this is a date column, expected. If categorical, consider binning."
        ),
        "high_correlation": (
            f"Columns {column} have |correlation|={value:.3f}. "
            "Expected for macro indicators — flag for VIF check before regression."
        ),
        "duplicates": (
            f"Dataset has {value:.1%} duplicate rows. "
            "Check whether month_date is truly unique."
        ),
        "imbalanced": (
            f"Column '{column}' minority class at {value:.1%}. "
            "Rare events (recessions) are naturally rare in economic data."
        ),
    }
    return interpretations.get(
        alert_type, f"Alert type '{alert_type}' — review manually."
    )


async def profile_economic_data():
    """Profile the merged economic dataset."""
    # TODO: Create a DataExplorer passing your configured alert_config
    explorer = DataExplorer(alert_config=____)  # Hint: alert_config

    print("\n=== Running DataExplorer Profile ===")
    # TODO: Await explorer.profile() on the economic DataFrame
    profile = await explorer.profile(____)  # Hint: economic

    # Top-level summary
    print(f"Rows: {profile.n_rows}  Columns: {profile.n_columns}")
    print(f"Duplicates: {profile.duplicate_count} ({profile.duplicate_pct:.1%})")
    print(f"Type summary: {profile.type_summary}")

    # Alerts
    print(f"\n--- Alerts ({len(profile.alerts)}) ---")
    for alert in profile.alerts:
        alert_type = alert["type"]
        col = alert.get("column", alert.get("columns", "N/A"))
        value = alert.get("value", "N/A")
        severity = alert["severity"]
        interpretation = _interpret_alert(alert_type, col, value)

        print(f"\n  [{severity.upper()}] {alert_type}")
        print(f"    Column: {col}")
        print(f"    Value:  {value}")
        print(f"    Why:    {interpretation}")

    return profile


# ══════════════════════════════════════════════════════════════════════
# TASK 5: Column-level profile deep dive
# ══════════════════════════════════════════════════════════════════════


async def column_deep_dive(profile):
    """Analyse column-level statistics from the profile."""
    print("\n--- Column Profiles ---")

    numeric_profiles = []
    categorical_profiles = []

    for col in profile.columns:
        if col.inferred_type == "numeric":
            numeric_profiles.append(col)
            print(
                f"  {col.name}: {col.inferred_type} | "
                f"mean={col.mean:.3g}, std={col.std:.3g}, "
                f"nulls={col.null_pct:.1%}, skew={col.skewness:.2f}"
            )
        else:
            categorical_profiles.append(col)
            print(
                f"  {col.name}: {col.inferred_type} | "
                f"unique={col.unique_count}, nulls={col.null_pct:.1%}"
            )

    # --- Identify most problematic columns ---
    print(f"\n--- Column Quality Ranking ---")
    # Score: higher = more problematic
    scored = []
    for col in numeric_profiles:
        score = 0
        if col.null_pct > 0.05:
            score += 2
        if abs(col.skewness) > 2:
            score += 1
        if col.std == 0:
            score += 3  # constant column
        scored.append((col.name, score, col.null_pct, col.skewness))

    scored.sort(key=lambda x: x[1], reverse=True)
    print(f"  {'Column':<25} {'Score':>6} {'Nulls':>8} {'Skew':>8}")
    print(f"  {'─' * 50}")
    for name, score, nulls, skew in scored[:10]:
        print(f"  {name:<25} {score:>6} {nulls:>7.1%} {skew:>8.2f}")

    return numeric_profiles, categorical_profiles


# ══════════════════════════════════════════════════════════════════════
# TASK 6: Spearman correlation analysis
# ══════════════════════════════════════════════════════════════════════


async def spearman_analysis(profile):
    """Analyse Spearman rank correlations from the profile."""
    if not profile.spearman_matrix:
        print("\n--- No Spearman matrix available ---")
        return

    print("\n--- Top Spearman Correlations (|r| > 0.8) ---")
    seen: set[tuple[str, str]] = set()
    corrs = []
    for col_a, row in profile.spearman_matrix.items():
        for col_b, corr in row.items():
            if (
                col_a != col_b
                and corr is not None
                and (col_b, col_a) not in seen
                and abs(corr) > 0.8
            ):
                seen.add((col_a, col_b))
                corrs.append((col_a, col_b, corr))
    corrs.sort(key=lambda x: abs(x[2]), reverse=True)

    for col_a, col_b, corr in corrs[:15]:
        direction = "positive" if corr > 0 else "negative"
        strength = "strong" if abs(corr) > 0.9 else "moderate"
        print(f"  {col_a} <-> {col_b}: {corr:.3f}  ({strength} {direction})")
    # INTERPRETATION: Spearman measures rank-order correlation — captures
    # monotonic relationships that Pearson misses. |r| > 0.8 between two
    # predictors signals multicollinearity risk for regression models.

    # --- Count high-correlation pairs per column ---
    col_corr_counts: dict[str, int] = {}
    for col_a, col_b, _ in corrs:
        col_corr_counts[col_a] = col_corr_counts.get(col_a, 0) + 1
        col_corr_counts[col_b] = col_corr_counts.get(col_b, 0) + 1

    if col_corr_counts:
        print(f"\n--- Columns with Most High Correlations ---")
        for col, count in sorted(
            col_corr_counts.items(), key=lambda x: x[1], reverse=True
        )[:5]:
            print(f"  {col}: {count} pairs with |r| > 0.8")


# ══════════════════════════════════════════════════════════════════════
# TASK 7: Compare pre-COVID vs COVID-era data quality
# ══════════════════════════════════════════════════════════════════════


async def compare_periods():
    """Compare pre-COVID vs COVID-era economic data quality."""
    explorer = DataExplorer(alert_config=alert_config)

    # TODO: Create a date cutoff for March 2020 using pl.date()
    covid_cutoff = pl.date(____, ____, ____)  # Hint: 2020, 3, 1
    pre_covid = economic.filter(pl.col("month_date") < covid_cutoff)
    during_covid = economic.filter(pl.col("month_date") >= covid_cutoff)

    print(f"\n=== Period Comparison ===")
    print(f"Pre-COVID months:  {pre_covid.height}")
    print(f"COVID-era months:  {during_covid.height}")

    # TODO: Await explorer.compare() with the two period DataFrames
    comparison = await explorer.compare(____, ____)  # Hint: pre_covid, during_covid

    # Access comparison result — handle both dict and object return types
    if isinstance(comparison, dict):
        shape_cmp = comparison.get("shape_comparison", "N/A")
        shared_cols = comparison.get("shared_columns", [])
        col_deltas = comparison.get("column_deltas", [])
    else:
        shape_cmp = getattr(comparison, "shape_comparison", "N/A")
        shared_cols = getattr(comparison, "shared_columns", [])
        col_deltas = getattr(comparison, "column_deltas", [])

    print(f"\nShape comparison: {shape_cmp}")
    print(f"Shared columns:   {len(shared_cols)}")

    # Sort by absolute mean delta
    print("\n--- Column Deltas (largest mean shifts) ---")
    deltas = sorted(
        col_deltas,
        key=lambda d: abs(
            d.get("mean_delta", 0)
            if isinstance(d, dict)
            else getattr(d, "mean_delta", 0)
        ),
        reverse=True,
    )
    for delta in deltas[:10]:
        col = (
            delta.get("column", "?")
            if isinstance(delta, dict)
            else getattr(delta, "column", "?")
        )
        mean_delta = (
            delta.get("mean_delta", 0)
            if isinstance(delta, dict)
            else getattr(delta, "mean_delta", 0)
        )
        std_delta = (
            delta.get("std_delta", 0)
            if isinstance(delta, dict)
            else getattr(delta, "std_delta", 0)
        )
        print(f"  {col}: mean delta={mean_delta:+,.3g}  std delta={std_delta:+,.3g}")

    # --- Classify the impact ---
    print(f"\n--- COVID Impact Classification ---")
    for delta in deltas[:10]:
        col = delta.get("column", "?")
        md = delta.get("mean_delta", 0)
        sd = delta.get("std_delta", 0)
        if abs(md) < 0.01 and abs(sd) < 0.01:
            impact = "minimal"
        elif sd > 0:
            impact = "more volatile during COVID"
        else:
            impact = "less volatile during COVID"
        if md > 0:
            direction = "higher"
        elif md < 0:
            direction = "lower"
        else:
            direction = "unchanged"
        print(f"  {col:<30} {direction:<12} {impact}")
    # INTERPRETATION: Columns with the largest mean_delta experienced the
    # biggest distributional shift. Positive = COVID-era average is higher.
    # Higher std_delta = more volatile during COVID.

    return comparison


# ══════════════════════════════════════════════════════════════════════
# TASK 8: try/except — error handling patterns
# ══════════════════════════════════════════════════════════════════════
# try/except is Python's error handling mechanism. Without it, an error
# crashes your entire program. With it, you can handle the error
# gracefully and continue.

# --- 8a: Basic try/except ---
print("\n=== Error Handling Patterns ===")

# Division by zero
try:
    result = 100 / 0
except ZeroDivisionError as e:
    print(f"  Caught ZeroDivisionError: {e}")
    result = 0
print(f"  result = {result}")

# Invalid type conversion
try:
    number = int("not_a_number")
except ValueError as e:
    print(f"  Caught ValueError: {e}")
    number = 0
print(f"  number = {number}")


# --- 8b: Multiple except blocks ---
def safe_parse_date(date_str: str) -> str:
    """Try multiple date formats and return the first that works."""
    formats = ["%Y-%m-%d", "%d/%m/%Y", "%Y-%m", "%m/%Y"]
    for fmt in formats:
        try:
            parsed = pl.Series([date_str]).str.to_date(fmt)[0]
            return str(parsed)
        except Exception:
            continue
    return f"UNPARSEABLE: {date_str}"


# Test the parser
test_dates = ["2023-01-15", "15/01/2023", "2023-01", "01/2023", "garbage"]
print(f"\n  Date parsing results:")
for d in test_dates:
    print(f"    {d!r:>20} -> {safe_parse_date(d)}")


# --- 8c: try/except/else/finally ---
def load_and_describe(module: str, filename: str) -> dict | None:
    """Attempt to load a file and return its description.

    - try:     the operation that might fail
    - except:  what to do if it fails
    - else:    what to do if it succeeds (no error)
    - finally: what to do ALWAYS, regardless of success or failure
    """
    try:
        df = loader.load(module, filename)
    except FileNotFoundError:
        print(f"    File not found: {module}/{filename}")
        return None
    except Exception as e:
        print(f"    Unexpected error: {type(e).__name__}: {e}")
        return None
    else:
        # Only runs if try succeeded
        return {"shape": df.shape, "columns": df.columns}
    finally:
        # Always runs — useful for cleanup
        print(f"    Attempted load of {module}/{filename}")


print(f"\n  File loading with error handling:")
result_good = load_and_describe("mlfp01", "sg_cpi.csv")
result_bad = load_and_describe("mlfp01", "nonexistent_file.csv")
print(f"  Good result: {result_good}")
print(f"  Bad result:  {result_bad}")


# --- 8d: Raising your own exceptions ---
def validate_threshold(value: float, name: str) -> None:
    """Validate that a threshold is between 0 and 1."""
    if not 0 <= value <= 1:
        raise ValueError(f"{name} must be between 0 and 1, got {value}")


try:
    validate_threshold(0.5, "correlation_threshold")
    print(f"\n  validate_threshold(0.5): passed")
    validate_threshold(1.5, "correlation_threshold")
except ValueError as e:
    print(f"  validate_threshold(1.5): {e}")

# ── Checkpoint 8 ─────────────────────────────────────────────────────
assert result == 0, "Division by zero should be caught"
assert result_good is not None, "Good file should load"
assert result_bad is None, "Missing file should return None"
print("\n✓ Checkpoint 8 passed — error handling patterns working\n")


# ══════════════════════════════════════════════════════════════════════
# TASK 9: Generate self-contained HTML report
# ══════════════════════════════════════════════════════════════════════


async def generate_report():
    """Generate a single-file HTML profiling report."""
    explorer = DataExplorer(alert_config=alert_config)

    # TODO: Await explorer.to_html() passing the economic DataFrame and a title
    report_html = await explorer.to_html(
        ____,  # Hint: economic
        title=____,  # Hint: "Singapore Economic Indicators — Data Profile"
    )

    report_path = "ex7_economic_profile_report.html"
    with open(report_path, "w") as f:
        f.write(report_html)
    print(f"\nSaved: {report_path}")

    # Also generate per-column visualisations using the profile
    print("--- Generating Visualisations ---")
    numeric_cols = [
        c
        for c, t in zip(economic.columns, economic.dtypes)
        if t in (pl.Float64, pl.Int64, pl.Float32, pl.Int32)
    ]
    for col in numeric_cols[:6]:
        fig = viz.histogram(economic, column=col, title=f"Distribution: {col}")
        filename = f"ex7_{col}_dist.html"
        fig.write_html(filename)
        print(f"  Saved: {filename}")


# ══════════════════════════════════════════════════════════════════════
# TASK 10: Data quality scorecard and action plan
# ══════════════════════════════════════════════════════════════════════


async def build_scorecard(profile, comparison):
    """Build a data quality scorecard from profiling results."""
    print(f"\n{'═' * 65}")
    print(f"  DATA QUALITY SCORECARD")
    print(f"{'═' * 65}")

    # --- Completeness score ---
    total_cells = profile.n_rows * profile.n_columns
    null_cells = sum(col.null_pct * profile.n_rows for col in profile.columns)
    completeness = (total_cells - null_cells) / total_cells * 100
    print(f"  Completeness:  {completeness:.1f}%  (non-null cells / total cells)")

    # --- Uniqueness score ---
    uniqueness = (1 - profile.duplicate_pct) * 100
    print(f"  Uniqueness:    {uniqueness:.1f}%  (non-duplicate rows / total rows)")

    # --- Alert score ---
    alert_count = len(profile.alerts)
    severity_counts = {}
    for alert in profile.alerts:
        sev = alert["severity"]
        severity_counts[sev] = severity_counts.get(sev, 0) + 1

    print(f"  Alerts:        {alert_count} total")
    for sev, count in sorted(severity_counts.items()):
        print(f"    {sev}: {count}")

    # --- Overall grade ---
    # TODO: Fill in the grade thresholds
    # Grade A: completeness >= 95, uniqueness >= 99, alert_count <= 3
    # Grade B: completeness >= 90, uniqueness >= 95, alert_count <= 6
    # Grade C: completeness >= 80, uniqueness >= 90
    # Grade D: otherwise
    if (
        completeness >= ____ and uniqueness >= ____ and alert_count <= ____
    ):  # Hint: 95, 99, 3
        grade = "A"
    elif (
        completeness >= ____ and uniqueness >= ____ and alert_count <= ____
    ):  # Hint: 90, 95, 6
        grade = "B"
    elif completeness >= ____ and uniqueness >= ____:  # Hint: 80, 90
        grade = "C"
    else:
        grade = "D"
    print(f"\n  Overall Grade: {grade}")

    # --- Action plan ---
    print(f"\n--- Recommended Actions ---")
    actions = []
    if completeness < 95:
        actions.append(
            "1. Investigate and handle missing values (impute or document gaps)"
        )
    if uniqueness < 99:
        actions.append("2. Deduplicate rows before modelling")
    if any(a["type"] == "high_skewness" for a in profile.alerts):
        actions.append("3. Apply log-transform or winsorisation to skewed columns")
    if any(a["type"] == "high_correlation" for a in profile.alerts):
        actions.append(
            "4. Check for multicollinearity — consider dropping redundant features"
        )
    if any(a["type"] == "constant" for a in profile.alerts):
        actions.append("5. Remove constant columns (zero information content)")
    if any(a["type"] == "high_zeros" for a in profile.alerts):
        actions.append(
            "6. Verify zero-heavy columns — are zeros real or missing-as-zero?"
        )
    if not actions:
        actions.append("Data quality is good — no critical actions needed.")
    for action in actions:
        print(f"  {action}")

    # --- COVID impact summary ---
    if comparison:
        n_shifted = sum(
            1
            for d in comparison.get("column_deltas", [])
            if abs(d.get("mean_delta", 0)) > 0.1
        )
        print(f"\n--- COVID Impact ---")
        print(f"  Columns with significant distributional shift: {n_shifted}")
        print(f"  Recommendation: Consider training separate models for pre/post-COVID")

    print(f"{'═' * 65}")
    return grade


# ── Run all async tasks ───────────────────────────────────────────────
async def main():
    profile = await profile_economic_data()
    numeric_profiles, cat_profiles = await column_deep_dive(profile)
    await spearman_analysis(profile)
    comparison = await compare_periods()
    await generate_report()
    grade = await build_scorecard(profile, comparison)
    return profile, comparison, grade


# TODO: Run the main() coroutine using asyncio.run()
try:
    profile, comparison, grade = asyncio.run(____)  # Hint: main()

    # ── Checkpoint 4 (profiling) ─────────────────────────────────────
    assert profile is not None, "profile should not be None"
    assert profile.n_rows == economic.height
    print("\n✓ Checkpoint 4 passed — DataExplorer profiling complete")

    # ── Checkpoint 5 (column analysis) ───────────────────────────────
    assert profile.columns is not None
    print("✓ Checkpoint 5 passed — column-level analysis complete")

    # ── Checkpoint 6 (Spearman) ──────────────────────────────────────
    print("✓ Checkpoint 6 passed — Spearman analysis complete")

    # ── Checkpoint 7 (comparison) ────────────────────────────────────
    assert comparison is not None
    assert "column_deltas" in comparison
    print("✓ Checkpoint 7 passed — period comparison complete")

    # ── Checkpoint 9 (report) ────────────────────────────────────────
    import os

    assert os.path.exists("ex7_economic_profile_report.html")
    print("✓ Checkpoint 9 passed — HTML report generated")

    # ── Checkpoint 10 (scorecard) ────────────────────────────────────
    assert grade in ("A", "B", "C", "D")
    print("✓ Checkpoint 10 passed — quality scorecard complete\n")

except Exception as exc:
    print(f"\n[ERROR] Profiling failed: {exc}")
    raise


# ══════════════════════════════════════════════════════════════════════
# REFLECTION
# ══════════════════════════════════════════════════════════════════════
print("═" * 60)
print("  WHAT YOU'VE MASTERED")
print("═" * 60)
print(
    """
  ✓ DataExplorer: one call to profile an entire dataset
  ✓ AlertConfig: tuning thresholds for your specific domain
  ✓ Alert interpretation: mapping each alert type to a cleaning action
  ✓ Column-level profiling: mean, std, skewness, null rates per column
  ✓ Spearman correlation: rank-based correlation for non-linear patterns
  ✓ compare(): detecting distribution drift between time periods
  ✓ try/except: handling errors gracefully without crashing
  ✓ try/except/else/finally: the full error handling pattern
  ✓ Raising exceptions: validating inputs with custom error messages
  ✓ to_html(): generating shareable profiling reports
  ✓ Data quality scorecard: completeness, uniqueness, alert grading
  ✓ Async: async def, await, asyncio.run() for parallel column profiling

  NEXT: In Exercise 8, you'll put all of M1 together in one end-to-end
  pipeline — load messy taxi trip data, profile it with DataExplorer,
  clean it based on the alerts, engineer features, prepare with
  PreprocessingPipeline, visualise patterns, then re-profile to confirm
  quality improvement. This is the capstone of Module 1.
"""
)
