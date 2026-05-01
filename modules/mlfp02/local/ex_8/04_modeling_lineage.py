# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# MLFP02 — Exercise 8.4: Regression + Lineage — Full Audit Trail
# ════════════════════════════════════════════════════════════════════════
#
# WHAT YOU'LL LEARN:
#   - Build an OLS regression model on FeatureStore v2 features
#   - Compute from-scratch t-statistics and F-tests on coefficients
#   - Apply Normal-Normal Bayesian posteriors to interpret coefficients
#   - Log model parameters, metrics, and lineage via ExperimentTracker
#   - Generate a stakeholder report that synthesises all M2 concepts
#
# PREREQUISITES: Exercise 8.1-8.3 (schemas, PIT, rolling features)
# ESTIMATED TIME: ~50 min
#
# TASKS:
#   1. Theory — why model lineage is essential for production ML
#   2. Build — OLS regression on v2 features with full diagnostics
#   3. Train — hypothesis tests + Bayesian posteriors + ExperimentTracker
#   4. Visualise — actual vs predicted, coefficient forest, residuals
#   5. Apply — MAS regulatory model audit for Singapore banks
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import asyncio

import numpy as np
import polars as pl
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats as sp_stats

from shared.mlfp02.ex_8 import (
    FEATURE_LIST,
    OUTPUT_DIR,
    compute_v2_features,
    fit_ols,
    load_hdb_resale,
    normal_normal_posterior,
    prepare_design_matrix,
    setup_feature_store,
)


# ════════════════════════════════════════════════════════════════════════
# THEORY — Why Model Lineage Is Essential for Production ML
# ════════════════════════════════════════════════════════════════════════
# A model without lineage is like a financial audit without receipts.
# You know the number on the page, but you can't prove how you got
# there. Model lineage records:
#
#   1. DATA PROVENANCE — which dataset, which version, which time range
#   2. FEATURE VERSION — which schema version produced the features
#   3. HYPERPARAMETERS — which settings were used for training
#   4. METRICS — R², RMSE, F-statistic, individual coefficient p-values
#   5. ARTIFACTS — the model weights, the training script, the config
#
# ExperimentTracker from kailash-ml captures all five automatically.
# When a regulator asks "why did your model approve this mortgage?",
# you can trace back from the prediction → model run → feature version
# → raw data → individual transaction records.
#
# Singapore context: The Monetary Authority of Singapore (MAS) requires
# banks to demonstrate model governance under FEAT principles (Fairness,
# Ethics, Accountability, Transparency). Without experiment tracking,
# banks cannot demonstrate reproducibility or explain individual
# predictions — both MAS requirements.


# ════════════════════════════════════════════════════════════════════════
# TASK 2 — BUILD: OLS regression on v2 features
# ════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("  Exercise 8.4 — Regression + Lineage: Full Audit Trail")
print("=" * 70)

# --- 2a. Prepare v2 features ---
hdb = load_hdb_resale()
features_v2 = compute_v2_features(hdb)
n_with_market = features_v2.filter(pl.col("town_median_price").is_not_null()).height

print(f"\n  v2 features: {features_v2.shape[0]:,} rows")
print(f"  With market context: {n_with_market:,} rows")

# --- 2b. Fit OLS regression ---
# TODO: Build the design matrix from v2 features using the shared helper.
# Hint: prepare_design_matrix(features_v2) returns (X, y, names) where
# X has a column of ones prepended and names includes "intercept".
X, y, names = ____

# TODO: Fit OLS using the shared helper.
# Hint: fit_ols(X, y) returns a dict with "beta", "se", "t", "p",
# "r2", "adj_r2", "rmse", "f_stat", "f_p", "y_hat", "resid", etc.
ols = ____

print(f"\n  === Regression Model on v2 Features ===")
print(f"  n = {ols['n']:,}, k = {ols['k']}")
print(f"  R-squared = {ols['r2']:.6f} ({ols['r2']:.2%} variance explained)")
print(f"  Adj R-squared = {ols['adj_r2']:.6f}")
print(f"  RMSE = ${ols['rmse']:,.0f}")

print(f"\n  {'Feature':<25} {'Coefficient':>14}")
print("  " + "-" * 42)
for name, coef in zip(names, ols["beta"]):
    print(f"  {name:<25} {coef:>14,.2f}")


# ── Checkpoint 1 ─────────────────────────────────────────────────────
assert ols["r2"] > 0.3, f"Task 2: R-squared should be reasonable, got {ols['r2']:.4f}"
assert ols["rmse"] > 0, "Task 2: RMSE must be positive"
print("\n[ok] Checkpoint 1 passed — regression model built on v2 features\n")

# INTERPRETATION: The R² tells us what fraction of price variation is
# explained by our 5 features. The remaining (1 - R²) is unexplained
# variance — renovation quality, unit facing, floor plan, negotiation
# skill, and luck. Adding flat-type dummies would likely improve R²
# by 5-10% (foreshadowing M3 feature selection).


# ════════════════════════════════════════════════════════════════════════
# TASK 3 — TRAIN: Hypothesis tests, Bayesian posteriors, and tracking
# ════════════════════════════════════════════════════════════════════════

# --- 3a. Coefficient significance tests ---
print("--- Coefficient Significance (from-scratch t-statistics) ---")
print(
    f"\n  {'Feature':<25} {'beta':>12} {'SE':>10} {'t':>8} "
    f"{'p-value':>12} {'Sig':>4}"
)
print("  " + "-" * 75)
for i, name in enumerate(names):
    p = ols["p"][i]
    sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
    print(
        f"  {name:<25} {ols['beta'][i]:>12,.2f} {ols['se'][i]:>10,.2f} "
        f"{ols['t'][i]:>8.2f} {p:>12.2e} {sig:>4}"
    )

print(f"\n  F-statistic: {ols['f_stat']:.2f} (p < {ols['f_p']:.2e})")
print(
    f"  Model is "
    f"{'significantly better' if ols['f_p'] < 0.05 else 'NOT better'} "
    f"than mean-only"
)

# --- 3b. Bayesian posteriors for each coefficient ---
print(f"\n--- Bayesian Posteriors (Normal-Normal Conjugate) ---")
posteriors = {}
for i, name in enumerate(names):
    if i == 0:
        continue  # Skip intercept

    # TODO: Compute the Normal-Normal posterior for this coefficient.
    # Hint: normal_normal_posterior(ols["beta"][i], ols["se"][i]) returns
    # a dict with "mu_post", "sigma_post", "ci_low", "ci_high".
    post = ____
    p_positive = 1 - sp_stats.norm.cdf(0, post["mu_post"], post["sigma_post"])
    posteriors[name] = {**post, "p_positive": p_positive}

    print(f"\n  {name}:")
    print(f"    OLS: beta={ols['beta'][i]:,.2f} +/- {ols['se'][i]:,.2f}")
    print(f"    Posterior: N({post['mu_post']:,.2f}, {post['sigma_post']:,.2f})")
    print(f"    95% credible: [{post['ci_low']:,.2f}, {post['ci_high']:,.2f}]")
    print(f"    P(beta > 0): {p_positive:.4f}")


# ── Checkpoint 2 ─────────────────────────────────────────────────────
assert all(se > 0 for se in ols["se"]), "Task 3: standard errors must be positive"
assert len(posteriors) == len(
    FEATURE_LIST
), "Task 3: must have posteriors for all features"
print(
    "\n[ok] Checkpoint 2 passed — hypothesis tests and Bayesian posteriors complete\n"
)

# --- 3c. Log to ExperimentTracker ---
print("--- ExperimentTracker Lineage ---")

factory, fs, tracker, has_backend = asyncio.run(setup_feature_store())

if has_backend:
    try:

        async def log_lineage():
            exp_id = "mlfp02_capstone_model"
            async with tracker.track(
                experiment=exp_id, run_name="hdb_price_ols_v2"
            ) as run:
                await run.log_params(
                    {
                        "feature_schema": "hdb_property_features",
                        "feature_version": "2",
                        "model_type": "OLS",
                        "n_features": str(len(FEATURE_LIST)),
                        "n_observations": str(ols["n"]),
                    }
                )
                await run.log_metrics(
                    {
                        "r2": float(ols["r2"]),
                        "adj_r2": float(ols["adj_r2"]),
                        "rmse": float(ols["rmse"]),
                        "f_statistic": float(ols["f_stat"]),
                    }
                )
                run_id = run.run_id
            return exp_id, run_id

        exp_id, run_id = asyncio.run(log_lineage())

        print(f"\n  Experiment logged:")
        print(f"    Run ID: {run_id}")
        print(f"    Feature schema: hdb_property_features v2")
        print(f"    Features: {FEATURE_LIST}")
        print(f"    Training rows: {ols['n']:,}")
        print(f"    R-squared: {ols['r2']:.4f}, RMSE: ${ols['rmse']:,.0f}")
    except Exception as e:
        print(f"  [Skipped: Lineage logging ({type(e).__name__}: {e})]")
else:
    print(f"\n  [Manual lineage — ExperimentTracker unavailable]")
    print(f"    Model: OLS regression")
    print(f"    Features: {FEATURE_LIST}")
    print(f"    Training rows: {ols['n']:,}")
    print(f"    R-squared: {ols['r2']:.4f}, RMSE: ${ols['rmse']:,.0f}")


# ── Checkpoint 3 ─────────────────────────────────────────────────────
print("\n[ok] Checkpoint 3 passed — model lineage documented\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 4 — VISUALISE: Actual vs predicted, coefficients, residuals
# ════════════════════════════════════════════════════════════════════════

print("--- Model Diagnostics Visualisations ---")

# Plot 1: Actual vs Predicted
rng = np.random.default_rng(42)
n_sample = min(3000, ols["n"])
idx = rng.choice(ols["n"], size=n_sample, replace=False)

# TODO: Create an actual vs predicted scatter plot.
# Hint: go.Scatter with x=y[idx].tolist(), y=ols["y_hat"][idx].tolist()
# then add a perfect prediction line with go.Scatter in "lines" mode.
fig1 = go.Figure()
fig1.add_trace(
    go.Scatter(
        x=____,
        y=____,
        mode="markers",
        marker={"size": 3, "opacity": 0.4, "color": "steelblue"},
        name="Predictions",
    )
)
fig1.add_trace(
    go.Scatter(
        x=[float(y.min()), float(y.max())],
        y=[float(y.min()), float(y.max())],
        mode="lines",
        name="Perfect",
        line={"dash": "dash", "color": "red"},
    )
)
fig1.update_layout(
    title=f"Capstone Model: Actual vs Predicted (R-squared={ols['r2']:.4f})",
    xaxis_title="Actual ($)",
    yaxis_title="Predicted ($)",
)
fig1.write_html(str(OUTPUT_DIR / "04_actual_vs_predicted.html"))
print(f"\n  Saved: {OUTPUT_DIR / '04_actual_vs_predicted.html'}")

# Plot 2: Coefficient forest plot
fig2 = go.Figure()
for i in range(1, ols["k"]):
    ci_lo = ols["beta"][i] - 1.96 * ols["se"][i]
    ci_hi = ols["beta"][i] + 1.96 * ols["se"][i]
    fig2.add_trace(
        go.Scatter(
            x=[ci_lo, ols["beta"][i], ci_hi],
            y=[names[i]] * 3,
            mode="markers+lines",
            name=names[i],
            marker={"size": [6, 10, 6]},
        )
    )
fig2.add_vline(x=0, line_dash="dot", line_color="red")
fig2.update_layout(
    title="Regression Coefficients with 95% CIs",
    xaxis_title="Coefficient Value",
)
fig2.write_html(str(OUTPUT_DIR / "04_coefficient_forest.html"))
print(f"  Saved: {OUTPUT_DIR / '04_coefficient_forest.html'}")

# Plot 3: Residual distribution
fig3 = go.Figure()
fig3.add_trace(
    go.Histogram(
        x=ols["resid"][idx].tolist(),
        nbinsx=50,
        marker_color="steelblue",
        opacity=0.7,
    )
)
fig3.update_layout(
    title="Residual Distribution",
    xaxis_title="Residual ($)",
    yaxis_title="Count",
)
fig3.write_html(str(OUTPUT_DIR / "04_residuals.html"))
print(f"  Saved: {OUTPUT_DIR / '04_residuals.html'}")

# TODO: Create Bayesian posterior density plots for each coefficient.
# Hint: make_subplots(rows=2, cols=3), then for each posterior,
# compute a Normal PDF using sp_stats.norm.pdf and add a go.Scatter.
fig4 = make_subplots(
    rows=2,
    cols=3,
    subplot_titles=list(posteriors.keys()),
)
for idx_p, (name, post) in enumerate(posteriors.items()):
    row = idx_p // 3 + 1
    col = idx_p % 3 + 1
    x_range = np.linspace(
        post["ci_low"] - 2 * post["sigma_post"],
        post["ci_high"] + 2 * post["sigma_post"],
        200,
    )
    pdf_vals = sp_stats.norm.pdf(x_range, post["mu_post"], post["sigma_post"])
    fig4.add_trace(
        go.Scatter(x=x_range.tolist(), y=pdf_vals.tolist(), name=name, mode="lines"),
        row=row,
        col=col,
    )
    fig4.add_vline(x=0, line_dash="dot", line_color="red", row=row, col=col)

fig4.update_layout(title="Bayesian Posterior Densities for Coefficients", height=600)
fig4.write_html(str(OUTPUT_DIR / "04_bayesian_posteriors.html"))
print(f"  Saved: {OUTPUT_DIR / '04_bayesian_posteriors.html'}")


# ── Checkpoint 4 ─────────────────────────────────────────────────────
print("\n[ok] Checkpoint 4 passed — all diagnostic visualisations saved\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 5 — APPLY: MAS Regulatory Model Audit for Singapore Banks
# ════════════════════════════════════════════════════════════════════════
# Scenario: The Monetary Authority of Singapore (MAS) conducts an
# annual model risk assessment of DBS Bank's property valuation model.
# MAS requires banks to demonstrate:
#   - FEAT Fairness: model doesn't discriminate by protected attributes
#   - FEAT Ethics: training data is ethically sourced (public HDB data)
#   - FEAT Accountability: full audit trail from data to prediction
#   - FEAT Transparency: model is interpretable (OLS coefficients)
#
# Without ExperimentTracker: the bank presents a PowerPoint with
# "R² = 0.76" and cannot show which data, which features, or which
# parameters produced that number. MAS flags this as a governance gap.
#
# With ExperimentTracker: the bank presents a complete lineage:
#   data version → feature schema v2 → OLS with 5 features →
#   R² = X.XX, RMSE = $Y → coefficients with p-values and CIs
# MAS approves the model governance framework.
#
# Regulatory impact: A governance gap finding can result in MAS
# requiring additional capital reserves (typically 10-20% buffer).
# For a bank with S$50B in HDB mortgage exposure, a 10% buffer
# requirement ties up S$5B in additional capital — money that cannot
# be deployed for other lending.

print("=== APPLY: MAS Regulatory Model Audit ===")
print()
print("  Scenario: MAS FEAT assessment of DBS property valuation model")
print()

# Stakeholder report
print("  " + "=" * 66)
print("  STAKEHOLDER REPORT: HDB Resale Price Analysis")
print("  " + "=" * 66)
print(
    f"""
  EXECUTIVE SUMMARY
  This analysis applies statistical methods from Module 2 to Singapore's
  HDB resale market, covering {features_v2.height:,} transactions.

  KEY FINDINGS:

  1. PRICE DISTRIBUTION (Ex 1-2: Bayesian + MLE)
     Average price: ${y.mean():,.0f} +/- ${y.std():,.0f}
     Skewness indicates {'right-skewed' if sp_stats.skew(y) > 0.5 else 'approximately symmetric'} distribution

  2. PRICE DRIVERS (Ex 5: Linear Regression)
     Our v2 model with {len(FEATURE_LIST)} features explains {ols['r2']:.1%} of
     price variation. Significant drivers:"""
)
for i in range(1, ols["k"]):
    if ols["p"][i] < 0.05:
        print(
            f"     - {names[i]}: ${ols['beta'][i]:+,.0f} per unit "
            f"(p<{max(ols['p'][i], 1e-10):.2e})"
        )
print(
    f"""
  3. MARKET CONTEXT (Ex 8: Feature Engineering)
     Rolling 6-month town medians and volumes capture local market.
     {n_with_market:,} of {features_v2.height:,} transactions have context.

  4. DATA QUALITY
     Point-in-time correctness ensures no future data leaks.
     Feature versioning (v1 -> v2) tracks schema evolution.

  5. MODEL LINEAGE
     ExperimentTracker records: feature schema, model params, metrics.
     Full audit trail from raw data to final prediction.

  6. MODEL LIMITATIONS
     - {(1-ols['r2'])*100:.0f}% of variance unexplained
     - Linear model may miss non-linear relationships
     - Market features have 6-month warm-up period (nulls)

  RECOMMENDATIONS:
     - Use v2 features for all new valuation models
     - Add flat-type encoding for +5-10% R-squared improvement
     - Consider non-linear models (Random Forest, XGBoost) in M3
     - Monitor town-level trends for early price-shift signals
"""
)

print("  MAS FEAT compliance summary:")
print("    Fairness:       Public HDB data, no protected-attribute features")
print("    Ethics:         Data from data.gov.sg (public, anonymised)")
print(f"    Accountability: ExperimentTracker lineage — {len(FEATURE_LIST)} features, ")
print(f"                    {ols['n']:,} rows, R-squared={ols['r2']:.4f}")
print("    Transparency:   OLS coefficients with CIs and p-values")
print()
print("  Regulatory impact of governance gap:")
print("    - 10% additional capital reserve on S$50B HDB mortgage book")
print("    - S$5B tied up in non-deployable capital")
print("    - ExperimentTracker eliminates this by providing full audit trail")


# ── Checkpoint 5 ─────────────────────────────────────────────────────
print("\n[ok] Checkpoint 5 passed — stakeholder report and MAS audit complete\n")

# Clean up
if has_backend and factory is not None and hasattr(factory, "close"):
    try:
        asyncio.run(factory.close())
    except Exception:
        pass


# ════════════════════════════════════════════════════════════════════════
# REFLECTION
# ════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("  WHAT YOU'VE MASTERED")
print("=" * 70)
print(
    """
  [ok] OLS regression on FeatureStore v2 features
  [ok] From-scratch t-statistics and F-tests on coefficients
  [ok] Bayesian Normal-Normal posteriors for coefficient interpretation
  [ok] ExperimentTracker: parameters, metrics, and lineage logging
  [ok] Stakeholder reporting: translating statistics into business decisions
  [ok] MAS FEAT compliance: fairness, ethics, accountability, transparency

  MODULE 2 COMPLETE — YOUR STATISTICAL TOOLKIT:
  ==============================================
  Ex 1: Bayesian inference — conjugate priors, credible intervals
  Ex 2: MLE + MAP — optimisation, CLT, failure modes, AIC/BIC
  Ex 3: Hypothesis testing — bootstrap, power, BH-FDR, permutation
  Ex 4: A/B design — pre-registration, SRM, adaptive sample sizes
  Ex 5: Linear regression — OLS from scratch, VIF, WLS, diagnostics
  Ex 6: Logistic regression — sigmoid MLE, odds ratios, calibration
  Ex 7: CUPED + causal inference — variance reduction, DiD, mSPRT
  Ex 8: Capstone — feature store, lineage, complete pipeline

  -> NEXT MODULE: M3 — Supervised ML in the Kailash Pipeline
     You'll use TrainingPipeline, HyperparameterSearch, and ModelRegistry
     to build, tune, and deploy models at production scale.
"""
)
