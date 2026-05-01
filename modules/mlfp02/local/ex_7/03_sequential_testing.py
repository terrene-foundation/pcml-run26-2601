# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# MLFP02 — Exercise 7.3: Sequential Testing with mSPRT
# ════════════════════════════════════════════════════════════════════════
#
# WHAT YOU'LL LEARN:
#   - Implement sequential testing with mSPRT (always-valid p-values)
#   - Demonstrate the peeking problem with simulation
#   - Compare fixed vs sequential p-values over time
#   - Understand why standard p-values fail under continuous monitoring
#   - Log sequential results to ExperimentTracker
#
# PREREQUISITES: Exercise 7.1-7.2 (CUPED, Bayesian concepts)
# ESTIMATED TIME: ~45 min
#
# TASKS:
#   1. Load experiment data and compute baseline SE
#   2. Sequential testing: mSPRT always-valid p-values day by day
#   3. Peeking problem simulation: inflated Type I error
#   4. Compare fixed vs sequential p-value trajectories
#   5. Visualise both analyses
#   6. Apply to Singapore ride-hailing scenario
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
    get_revenue_arrays,
    load_experiment,
    msprt_sequential_pvalues,
    naive_ab,
    print_banner,
    simulate_peeking,
    split_groups,
)


# ════════════════════════════════════════════════════════════════════════
# THEORY — The Peeking Problem and Sequential Testing
# ════════════════════════════════════════════════════════════════════════
# Standard p-values are designed for a SINGLE look at the data. If you
# peek at your experiment 20 times and stop when p < 0.05, the actual
# false positive rate jumps from 5% to ~64%.
#
# Why? Each peek is an independent test. With 20 independent tests at
# alpha=0.05, the probability of at LEAST ONE false positive is:
#   1 - (1-0.05)^20 = 64%
#
# mSPRT (mixture Sequential Probability Ratio Test) provides "always-
# valid" p-values that remain correct no matter when you look. The
# trade-off: mSPRT p-values are more conservative — they need more
# data to reach significance. But they never lie.
#
# Analogy: Fixed p-values are like a bathroom scale that only gives the
# right weight if you step on it exactly once. Step on it 20 times and
# take the lowest reading? You'll think you lost weight. mSPRT is a
# scale that gives the correct reading no matter how many times you
# step on it.
#
# WHY THIS MATTERS: At Grab (Singapore), experiments run continuously
# and dashboards update hourly. Product managers peek daily. Without
# sequential testing, ~30% of "significant" results were false positives
# that reverted after full rollout.


# ════════════════════════════════════════════════════════════════════════
# TASK 1 — Load Data and Compute Baseline
# ════════════════════════════════════════════════════════════════════════

print_banner("MLFP02 Exercise 7.3: Sequential Testing (mSPRT)")

# TODO: Load experiment data, split groups, extract revenue arrays,
# and compute the naive A/B baseline.
# Hint: load_experiment(), split_groups(), get_revenue_arrays(), naive_ab()
experiment = ____
control, treatment = ____
y_c, y_t = ____
baseline = ____
se_naive = baseline["se"]

print(f"  Data loaded: {experiment.shape[0]:,} rows")
print(f"  Baseline SE: ${se_naive:.2f}")
print(f"  Baseline lift: ${baseline['lift']:.2f}")

# ── Checkpoint 1 ─────────────────────────────────────────────────────
assert se_naive > 0, "Baseline SE must be positive"
print("\n>>> Checkpoint 1 passed -- data loaded and baseline computed\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 2 — Sequential Testing: mSPRT Day by Day
# ════════════════════════════════════════════════════════════════════════

print(f"\n=== Sequential Testing (mSPRT) ===")

# TODO: Compute sequential p-values using the shared helper.
# The mSPRT hyperparameter tau_sq is typically set to se_naive**2.
# Hint: msprt_sequential_pvalues(experiment, tau_sq=tau_sq)
#   returns a list of dicts with day, n, lift, p_fixed, p_sequential.
tau_sq = ____
sequential_results = ____

print(f"{'Day':>4} {'n':>8} {'Lift':>10} {'p (fixed)':>12} {'p (mSPRT)':>12}")
print("-" * 52)
step = max(1, len(sequential_results) // 10)
for r in sequential_results[::step]:
    print(
        f"{r['day']:>4} {int(r['n']):>8,} ${r['lift']:>8.2f} "
        f"{r['p_fixed']:>12.6f} {r['p_sequential']:>12.6f}"
    )

early_sig_fixed = sum(1 for r in sequential_results if r["p_fixed"] < 0.05)
early_sig_seq = sum(1 for r in sequential_results if r["p_sequential"] < 0.05)
print(f"\nDays with p < 0.05 (fixed):      {early_sig_fixed}/{len(sequential_results)}")
print(f"Days with p < 0.05 (sequential): {early_sig_seq}/{len(sequential_results)}")

# ── Checkpoint 2 ─────────────────────────────────────────────────────
assert len(sequential_results) > 0, "Must have sequential results"
for r in sequential_results:
    assert 0 <= r["p_sequential"] <= 1, "Sequential p-values must be valid"
print("\n>>> Checkpoint 2 passed -- sequential testing completed\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 3 — Peeking Problem Simulation
# ════════════════════════════════════════════════════════════════════════
# Simulate experiments with NO real effect to show how peeking inflates
# the false positive rate.

print(f"\n=== Peeking Problem Simulation ===")

# TODO: Run the peeking simulation using the shared helper.
# Hint: simulate_peeking(n_sims=1000, n_per_sim=2000, n_checks=20, seed=42)
#   returns a dict with rate_no_peek, rate_fixed_peek, theoretical_inflated_rate.
peek_results = ____

print(f"Simulations: {int(peek_results['n_sims']):,} (all with NO real effect)")
print(f"Peeks per experiment: {int(peek_results['n_checks'])}")
print(f"\nFalse positive rates:")
print(f"  No peeking (test at end):   {peek_results['rate_no_peek']:.1%} (target: 5%)")
print(
    f"  Peeking with fixed p:       {peek_results['rate_fixed_peek']:.1%} (inflated!)"
)
print(
    f"  Expected with {int(peek_results['n_checks'])} peeks:     "
    f"~{peek_results['theoretical_inflated_rate']*100:.0f}% (theory)"
)

# ── Checkpoint 3 ─────────────────────────────────────────────────────
assert (
    peek_results["rate_fixed_peek"] > peek_results["rate_no_peek"]
), "Peeking must inflate false positive rate"
print("\n>>> Checkpoint 3 passed -- peeking problem demonstrated\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 4 — Visualise: Fixed vs Sequential p-Values Over Time
# ════════════════════════════════════════════════════════════════════════

fig = go.Figure()
days_seq = [r["day"] for r in sequential_results]
fig.add_trace(
    go.Scatter(
        x=days_seq,
        y=[r["p_fixed"] for r in sequential_results],
        name="Fixed p-value",
    )
)
fig.add_trace(
    go.Scatter(
        x=days_seq,
        y=[r["p_sequential"] for r in sequential_results],
        name="mSPRT p-value",
    )
)
fig.add_hline(y=0.05, line_dash="dash", annotation_text="alpha=0.05")
fig.update_layout(
    title="Sequential Testing: Fixed vs mSPRT p-values",
    xaxis_title="Day",
    yaxis_title="p-value",
    yaxis_type="log",
)
out_path = OUTPUT_DIR / "sequential_pvalues.html"
fig.write_html(str(out_path))
print(f"\nSaved: {out_path}")

# Peeking problem visualisation
fig2 = go.Figure()
categories = ["No peeking", "Peeking (fixed p)", "Theory (20 peeks)"]
rates = [
    peek_results["rate_no_peek"],
    peek_results["rate_fixed_peek"],
    peek_results["theoretical_inflated_rate"],
]
colours = ["green", "red", "orange"]
fig2.add_trace(
    go.Bar(
        x=categories,
        y=rates,
        marker_color=colours,
        text=[f"{r:.1%}" for r in rates],
        textposition="auto",
    )
)
fig2.add_hline(y=0.05, line_dash="dash", annotation_text="Nominal alpha=5%")
fig2.update_layout(
    title="Peeking Problem: False Positive Rate Inflation",
    yaxis_title="False Positive Rate",
    yaxis_tickformat=".0%",
)
out_path2 = OUTPUT_DIR / "peeking_problem.html"
fig2.write_html(str(out_path2))
print(f"Saved: {out_path2}")


# ════════════════════════════════════════════════════════════════════════
# APPLY — Grab Singapore: Continuous Experiment Monitoring
# ════════════════════════════════════════════════════════════════════════
# Scenario: Grab runs ~50 experiments simultaneously on their ride-
# hailing platform. Dashboards update hourly, and product managers
# check results daily — effectively peeking 30 times per month.
#
# Without sequential testing:
#   - 50 experiments x 64% false positive rate = ~32 false positives
#   - Each false positive ships a non-effect -> wasted eng time
#   - Cost: ~S$100K per false positive (development, rollback, re-test)
#
# With mSPRT:
#   - 50 experiments x 5% false positive rate = ~2.5 false positives
#   - Trade-off: experiments take ~20% longer to reach significance

print(f"\n--- Singapore Application: Ride-Hailing Experiment Monitoring ---")
n_experiments = 50
fp_peeking = n_experiments * peek_results["rate_fixed_peek"]
fp_msprt = n_experiments * 0.05
cost_per_fp = 100_000
print(f"Concurrent experiments: {n_experiments}")
print(f"False positives (peeking): ~{fp_peeking:.0f}")
print(f"False positives (mSPRT):   ~{fp_msprt:.0f}")
print(f"Cost per false positive: S${cost_per_fp:,}")
print(f"Annual waste (peeking): S${fp_peeking * cost_per_fp * 12:,.0f}")
print(f"Annual waste (mSPRT):   S${fp_msprt * cost_per_fp * 12:,.0f}")
print(f"Annual savings: S${(fp_peeking - fp_msprt) * cost_per_fp * 12:,.0f}")


# ════════════════════════════════════════════════════════════════════════
# LOG — ExperimentTracker
# ════════════════════════════════════════════════════════════════════════


async def log_sequential_results():
    db = "sqlite:///mlfp02_experiments.db"
    tracker = await ExperimentTracker.create(store_url=db)
    conn = ConnectionManager(db)
    await conn.initialize()

    exp_id = "mlfp02_ex7_sequential_testing"

    async with tracker.track(experiment=exp_id, run_name="msprt_analysis") as run:
        await run.log_params(
            {
                "sequential_method": "mSPRT",
                "tau_sq": str(float(tau_sq)),
                "n_peek_sims": "1000",
                "n_checks": "20",
            }
        )
        await run.log_metrics(
            {
                "days_sig_fixed": float(early_sig_fixed),
                "days_sig_sequential": float(early_sig_seq),
                "fp_rate_no_peek": float(peek_results["rate_no_peek"]),
                "fp_rate_peeking": float(peek_results["rate_fixed_peek"]),
                "fp_rate_theoretical": float(peek_results["theoretical_inflated_rate"]),
            }
        )
    print(f"\nLogged sequential testing run")
    await conn.close()


try:
    asyncio.run(log_sequential_results())
except Exception as e:
    print(f"  [Skipped: ExperimentTracker logging ({type(e).__name__}: {e})]")

# ── Checkpoint 4 ─────────────────────────────────────────────────────
print("\n>>> Checkpoint 4 passed -- visualisation and logging complete\n")


# ══════════════════════════════════════════════════════════════════════
# REFLECTION
# ══════════════════════════════════════════════════════════════════════
print("=" * 70)
print("  WHAT YOU'VE MASTERED")
print("=" * 70)
print(
    f"""
  - mSPRT: always-valid p-values for safe experiment monitoring
  - Peeking problem: {int(peek_results['n_checks'])} peeks inflates alpha from 5% to ~{peek_results['rate_fixed_peek']:.0%}
  - Fixed vs sequential p-value trajectories
  - tau_sq hyperparameter: set to baseline SE^2
  - Why dashboards with live p-values need sequential methods

  NEXT: In 04_diff_in_diff.py, you'll learn to estimate causal effects
  from observational data when randomisation is not possible.
"""
)

print("\n>>> Exercise 7.3 complete -- Sequential Testing with mSPRT")
