# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# MLFP02 — Exercise 7.1: CUPED Variance Reduction
# ════════════════════════════════════════════════════════════════════════
#
# WHAT YOU'LL LEARN:
#   - Derive and implement CUPED using pre-experiment covariates
#   - Quantify variance reduction from the pre-post correlation (rho^2)
#   - Extend CUPED to multiple covariates via multivariate regression
#   - Apply stratified CUPED to detect heterogeneous treatment effects
#   - Log results to ExperimentTracker for reproducibility
#
# PREREQUISITES: Exercises 3-4 — hypothesis testing, p-values, SRM
# ESTIMATED TIME: ~45 min
#
# TASKS:
#   1. Load experiment data, SRM check
#   2. Standard A/B baseline (no CUPED)
#   3. Single-covariate CUPED: derive theta, adjust Y, verify reduction
#   4. Multi-covariate CUPED: multivariate regression
#   5. Stratified CUPED: segment-level treatment effects
#   6. Visualise and log results
#
# THEORY (CUPED):
#   Y_adj = Y - theta*(X - E[X])  where theta = Cov(Y,X)/Var(X)
#   Var(Y_adj) = Var(Y)(1 - rho^2) where rho = Cor(Y,X)
#
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import asyncio

import numpy as np
import plotly.graph_objects as go
from kailash.db import ConnectionManager
from kailash_ml import ExperimentTracker

from shared.mlfp02.ex_7 import (
    OUTPUT_DIR,
    compute_srm,
    get_covariate_arrays,
    get_revenue_arrays,
    load_experiment,
    multi_cov_cuped,
    naive_ab,
    print_banner,
    single_cov_cuped,
    split_groups,
    stratified_cuped,
    stratify_by_covariate,
    variance_reduction,
)


# ════════════════════════════════════════════════════════════════════════
# THEORY — Why Variance Reduction Matters for A/B Tests
# ════════════════════════════════════════════════════════════════════════
# An A/B test's power depends on the standard error of the treatment
# effect estimate. Smaller SE = narrower CI = faster decisions.
#
# CUPED (Controlled-experiment Using Pre-Experiment Data) exploits the
# correlation between pre-experiment behaviour (X) and the outcome (Y).
# If a user spent $100 before the experiment, they will probably spend
# around $100 during — this predictable portion is noise we can remove.
#
# Y_adj = Y - theta*(X - E[X])
# theta = Cov(Y, X) / Var(X)  — the optimal noise-removal coefficient
# Var(Y_adj) = Var(Y) * (1 - rho^2)
#
# With rho = 0.7, CUPED removes 49% of variance — equivalent to
# doubling your sample size for free!
#
# WHY THIS MATTERS: At Shopee (Singapore), CUPED reduced experiment
# duration from 14 days to 7 days by shrinking confidence intervals —
# letting product teams ship features twice as fast.


# ════════════════════════════════════════════════════════════════════════
# TASK 1 — Load Data and SRM Check
# ════════════════════════════════════════════════════════════════════════

print_banner("MLFP02 Exercise 7.1: CUPED Variance Reduction")

experiment = load_experiment()
print(f"\n  Data loaded: experiment_data.parquet")
print(f"  Shape: {experiment.shape}")
print(f"  Columns: {experiment.columns}")
print(experiment.head(5))

control, treatment = split_groups(experiment)
n_c, n_t = control.height, treatment.height
print(f"\nControl: {n_c:,} | Treatment: {n_t:,}")

# SRM check
srm_p = compute_srm(n_c, n_t)
print(f"\nSRM check: p={srm_p:.6f} — {'OK' if srm_p > 0.01 else 'SRM DETECTED'}")

# Explore pre-experiment covariates
for col in ["revenue", "pre_metric_value", "metric_value"]:
    if col in experiment.columns:
        vals = experiment[col].drop_nulls()
        print(f"  {col}: mean={vals.mean():.2f}, std={vals.std():.2f}, n={vals.len()}")

# ── Checkpoint 1 ─────────────────────────────────────────────────────
assert 0 <= srm_p <= 1, "SRM p-value must be valid"
print("\n>>> Checkpoint 1 passed -- SRM check and data exploration completed\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 2 — Standard Analysis Baseline (No CUPED)
# ════════════════════════════════════════════════════════════════════════

y_c, y_t = get_revenue_arrays(control, treatment)
baseline = naive_ab(y_c, y_t)

print(f"\n=== Standard Analysis (no CUPED) ===")
print(f"Control mean: ${baseline['mean_c']:.2f}")
print(f"Treatment mean: ${baseline['mean_t']:.2f}")
print(
    f"Lift: ${baseline['lift']:.2f} ({baseline['lift'] / baseline['mean_c']:.2%} relative)"
)
print(f"SE: ${baseline['se']:.2f}")
print(f"95% CI: [${baseline['ci_lo']:.2f}, ${baseline['ci_hi']:.2f}]")
print(f"CI width: ${baseline['ci_hi'] - baseline['ci_lo']:.2f}")
print(f"p-value: {baseline['p_value']:.6f}")
# INTERPRETATION: The naive analysis uses only experiment-period data.
# It ignores that some users are naturally high-spenders — CUPED
# removes this baseline noise by leveraging pre-experiment data.

# ── Checkpoint 2 ─────────────────────────────────────────────────────
assert baseline["se"] > 0, "SE must be positive"
assert baseline["ci_lo"] < baseline["ci_hi"], "CI lower must be below upper"
print("\n>>> Checkpoint 2 passed -- standard analysis baseline established\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 3 — Single-Covariate CUPED
# ════════════════════════════════════════════════════════════════════════
# CUPED: Y_adj = Y - theta*(X - E[X])
# theta = Cov(Y, X) / Var(X) — the optimal coefficient
# Var(Y_adj) = Var(Y)(1 - rho^2) where rho = Cor(Y, X)

x_c, x_t = get_covariate_arrays(control, treatment)
cuped = single_cov_cuped(y_c, y_t, x_c, x_t)
vr = variance_reduction(baseline["se"], cuped["se"])

print(f"\n=== Single-Covariate CUPED ===")
print(f"Pre-post correlation (rho): {cuped['rho']:.4f}")
print(f"theta (optimal coefficient): {cuped['theta']:.4f}")
print(f"Theoretical variance reduction: {cuped['theoretical_reduction']:.1%}")
print(f"Actual variance reduction: {vr['variance_reduction']:.1%}")
print(f"CI width reduction: {vr['ci_width_reduction']:.1%}")
print(f"\nCUPED lift: ${cuped['lift']:.2f}")
print(f"SE (naive): ${baseline['se']:.2f} -> SE (CUPED): ${cuped['se']:.2f}")
ci_w_naive = baseline["ci_hi"] - baseline["ci_lo"]
ci_w_cuped = cuped["ci_hi"] - cuped["ci_lo"]
print(f"CI width (naive): ${ci_w_naive:.2f} -> CI width (CUPED): ${ci_w_cuped:.2f}")
print(f"95% CI: [${cuped['ci_lo']:.2f}, ${cuped['ci_hi']:.2f}]")
print(f"p-value: {cuped['p_value']:.6f} (was {baseline['p_value']:.6f})")

# Verify CUPED does not bias the point estimate
print(f"\n--- Bias Check ---")
print(f"Naive lift: ${baseline['lift']:.4f}")
print(f"CUPED lift: ${cuped['lift']:.4f}")
print(f"Difference: ${cuped['lift'] - baseline['lift']:.4f}")
print(
    f"CUPED is {'unbiased' if abs(cuped['lift'] - baseline['lift']) < 2 * baseline['se'] else 'BIASED -- investigate'}"
)
# INTERPRETATION: CUPED reduces variance by rho^2. The point estimate is
# unbiased because E[X - E[X]] = 0. The only change is precision —
# you get the same answer, just with a tighter confidence interval.
# This is equivalent to collecting {1/(1-rho**2):.1f}x more data.

# ── Checkpoint 3 ─────────────────────────────────────────────────────
assert 0 <= abs(cuped["rho"]) <= 1, "Correlation must be between -1 and 1"
assert cuped["se"] <= baseline["se"] * 1.01, "CUPED SE must be <= naive SE"
assert (
    abs(vr["variance_reduction"] - cuped["theoretical_reduction"]) < 0.1
), "Actual reduction should approximate theoretical rho^2"
print("\n>>> Checkpoint 3 passed -- CUPED variance reduction verified\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 4 — Multi-Covariate CUPED
# ════════════════════════════════════════════════════════════════════════
# With multiple pre-experiment features, use multivariate regression
# to compute the optimal adjustment: Y_adj = Y - X*theta where
# theta = (X'X)^-1 X'Y (regression coefficients from Y on pre-covariates)

print(f"\n=== Multi-Covariate CUPED ===")

# Use pre_metric_value and metric_value as covariates
pre_features = ["pre_metric_value"]
if "metric_value" in control.columns:
    pre_features.append("metric_value")

X_c_multi = control.select(pre_features).to_numpy().astype(np.float64)
X_t_multi = treatment.select(pre_features).to_numpy().astype(np.float64)

multi = multi_cov_cuped(y_c, y_t, X_c_multi, X_t_multi)
vr_multi = variance_reduction(baseline["se"], multi["se"])

print(f"Covariates: {pre_features}")
print(f"Multi-covariate theta: {multi['theta']}")
print(
    f"Variance reduction: {vr_multi['variance_reduction']:.1%} (single-cov: {vr['variance_reduction']:.1%})"
)
print(
    f"SE: ${multi['se']:.2f} (single: ${cuped['se']:.2f}, naive: ${baseline['se']:.2f})"
)
print(f"CI: [${multi['ci_lo']:.2f}, ${multi['ci_hi']:.2f}]")
# INTERPRETATION: Multiple covariates can capture more variance than
# a single one. The improvement depends on how much additional
# predictive power the extra covariates provide.

# ── Checkpoint 4 ─────────────────────────────────────────────────────
assert multi["se"] <= baseline["se"] * 1.01, "Multi-CUPED SE should be <= naive"
print("\n>>> Checkpoint 4 passed -- multi-covariate CUPED completed\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 5 — Stratified CUPED: Heterogeneous Treatment Effects
# ════════════════════════════════════════════════════════════════════════
# Do different user segments respond differently to treatment?
# Stratify by pre-experiment spending level and apply CUPED within strata.

print(f"\n=== Stratified CUPED ===")

strata = stratify_by_covariate(x_c, x_t)
strat_results = stratified_cuped(y_c, y_t, x_c, x_t, strata)

print(
    f"{'Stratum':<20} {'n_ctrl':>8} {'n_treat':>8} {'Lift':>10} {'SE':>8} {'p-value':>10}"
)
print("-" * 68)
for name, r in strat_results.items():
    print(
        f"{name:<20} {r['n_ctrl']:>8,} {r['n_treat']:>8,} "
        f"${r['lift']:>8.2f} ${r['se']:>6.2f} {r['p_value']:>10.6f}"
    )
# INTERPRETATION: If high spenders respond differently to treatment
# than low spenders, a one-size-fits-all analysis masks the heterogeneity.
# Stratified CUPED reveals these differences while maintaining precision.

# ── Checkpoint 5 ─────────────────────────────────────────────────────
assert len(strat_results) >= 2, "Should have at least 2 strata"
print("\n>>> Checkpoint 5 passed -- stratified CUPED completed\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 6 — Visualise: Naive vs CUPED Confidence Intervals
# ════════════════════════════════════════════════════════════════════════

fig = go.Figure()
methods = ["Naive", "CUPED (1-cov)", "CUPED (multi)"]
ses = [baseline["se"], cuped["se"], multi["se"]]
lifts = [baseline["lift"], cuped["lift"], multi["lift"]]
for m, s, l in zip(methods, ses, lifts):
    lo, hi = l - 1.96 * s, l + 1.96 * s
    fig.add_trace(
        go.Scatter(
            x=[lo, l, hi],
            y=[m] * 3,
            mode="markers+lines",
            name=m,
            marker={"size": [8, 12, 8]},
        )
    )
fig.add_vline(x=0, line_dash="dot", line_color="red")
fig.update_layout(
    title="Confidence Intervals: Naive vs CUPED",
    xaxis_title="Treatment Effect ($)",
)
out_path = OUTPUT_DIR / "cuped_comparison.html"
fig.write_html(str(out_path))
print(f"\nSaved: {out_path}")


# ════════════════════════════════════════════════════════════════════════
# APPLY — Shopee Singapore: Faster Experiment Decisions
# ════════════════════════════════════════════════════════════════════════
# Scenario: Shopee's experimentation platform runs ~200 A/B tests per
# quarter. Typical experiment duration is 14 days with standard analysis.
#
# With CUPED (rho ~0.7), the variance reduction of ~49% means confidence
# intervals shrink by ~30%. This lets teams reach statistical significance
# in ~7 days instead of 14 — effectively doubling experiment throughput.
#
# Business impact:
#   - 200 experiments/quarter -> 400 experiments/quarter
#   - Faster iteration on product features, pricing, UX
#   - At ~S$50K opportunity cost per delayed experiment week,
#     CUPED saves ~S$5M/year in faster decision-making

print(f"\n--- Singapore Application: E-Commerce Experiment Velocity ---")
eff_mult = vr["effective_sample_multiplier"]
speedup = 1 - 1 / eff_mult if eff_mult > 1 else 0
print(f"Effective sample multiplier: {eff_mult:.1f}x")
print(f"Experiment duration reduction: {speedup:.0%}")
print(f"If base duration is 14 days -> CUPED duration: {14 / eff_mult:.0f} days")
print(f"Quarterly experiments (200 base): {200 * eff_mult:.0f} with CUPED")
savings = 200 * speedup * 50_000
print(f"Estimated annual savings: S${savings:,.0f}")


# ════════════════════════════════════════════════════════════════════════
# LOG — ExperimentTracker
# ════════════════════════════════════════════════════════════════════════


async def log_cuped_results():
    db = "sqlite:///mlfp02_experiments.db"
    tracker = await ExperimentTracker.create(store_url=db)
    conn = ConnectionManager(db)
    await conn.initialize()

    exp_id = "mlfp02_ex7_cuped"

    async with tracker.track(experiment=exp_id, run_name="cuped_analysis") as run:
        await run.log_params(
            {
                "cuped_covariate": "pre_metric_value",
                "cuped_theta": str(float(cuped["theta"])),
                "cuped_rho": str(float(cuped["rho"])),
                "n_covariates_multi": str(len(pre_features)),
            }
        )
        await run.log_metrics(
            {
                "lift_naive": float(baseline["lift"]),
                "lift_cuped": float(cuped["lift"]),
                "se_naive": float(baseline["se"]),
                "se_cuped": float(cuped["se"]),
                "variance_reduction": float(vr["variance_reduction"]),
                "ci_width_reduction": float(vr["ci_width_reduction"]),
                "p_naive": float(baseline["p_value"]),
                "p_cuped": float(cuped["p_value"]),
            }
        )
    print(f"\nLogged CUPED experiment run")
    await conn.close()


try:
    asyncio.run(log_cuped_results())
except Exception as e:
    print(f"  [Skipped: ExperimentTracker logging ({type(e).__name__}: {e})]")

# ── Checkpoint 6 ─────────────────────────────────────────────────────
print("\n>>> Checkpoint 6 passed -- visualisation and logging complete\n")


# ══════════════════════════════════════════════════════════════════════
# REFLECTION
# ══════════════════════════════════════════════════════════════════════
print("=" * 70)
print("  WHAT YOU'VE MASTERED")
print("=" * 70)
print(
    f"""
  - CUPED: Y_adj = Y - theta*(X - E[X]), theta = Cov(Y,X)/Var(X)
  - Variance reduction: Var(Y_adj) = Var(Y)(1 - rho^2)
  - Multi-covariate CUPED: multivariate regression for adjustment
  - Stratified CUPED: heterogeneous treatment effects by segment
  - Bias check: CUPED preserves the point estimate, only reduces SE

  NEXT: In 02_bayesian_ab.py, you'll learn to compute P(B > A)
  and make ship/continue/hold decisions using expected loss.
"""
)

print("\n>>> Exercise 7.1 complete -- CUPED Variance Reduction")
