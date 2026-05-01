# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# MLFP02 — Exercise 7.4: Difference-in-Differences (DiD)
# ════════════════════════════════════════════════════════════════════════
#
# WHAT YOU'LL LEARN:
#   - Implement DiD for causal inference from observational data
#   - Test the parallel trends assumption that underlies DiD
#   - Understand when DiD is appropriate vs randomised experiments
#   - Visualise the counterfactual and treatment effect
#   - Synthesise CUPED, Bayesian, sequential, and DiD into a decision
#     framework for choosing the right causal inference method
#
# PREREQUISITES: Exercise 7.1-7.3
# ESTIMATED TIME: ~45 min
#
# TASKS:
#   1. Simulate Singapore HDB cooling-measure data
#   2. Compute the DiD estimate and standard error
#   3. Test the parallel trends assumption (bootstrap)
#   4. Visualise DiD with counterfactual
#   5. Apply to Singapore property policy evaluation
#   6. Synthesise all causal inference methods
#
# THEORY (DiD):
#   ATT = (Y_treat_post - Y_treat_pre) - (Y_ctrl_post - Y_ctrl_pre)
#   Key assumption: without treatment, both groups would have followed
#   parallel trends.
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
    diff_in_diff,
    parallel_trends_test,
    print_banner,
    simulate_hdb_cooling_measures,
)


# ════════════════════════════════════════════════════════════════════════
# THEORY — When Randomisation Is Impossible
# ════════════════════════════════════════════════════════════════════════
# You cannot randomise government policy. You cannot randomly assign
# stamp duty to some districts and not others (well, you could, but
# no government would agree). Yet policymakers still need to know:
# "Did the cooling measure actually reduce HDB prices?"
#
# Difference-in-Differences (DiD) answers this by comparing:
#   - How the TREATED group changed (Central HDB prices: pre vs post)
#   - How the CONTROL group changed (Non-Central HDB prices: pre vs post)
#   - The DIFFERENCE of these differences isolates the treatment effect
#
# The key assumption is PARALLEL TRENDS: without the policy, Central
# and Non-Central prices would have moved in the same direction by the
# same amount. If Central was already declining before the policy, DiD
# attributes the decline to the policy when it was already happening.
#
# Analogy: Two runners are jogging side by side at the same pace.
# One drinks an energy drink (treatment). If they speed up relative
# to the other runner, the energy drink had an effect. But if they
# were already faster BEFORE the drink, you cannot attribute the
# difference to the drink — that violates parallel trends.
#
# WHY THIS MATTERS: Singapore's property cooling measures (Additional
# Buyer's Stamp Duty, loan-to-value limits) are evaluated using DiD
# by MAS and URA to determine whether to tighten, relax, or maintain.


# ════════════════════════════════════════════════════════════════════════
# TASK 1 — Simulate Singapore HDB Cooling Measures
# ════════════════════════════════════════════════════════════════════════

print_banner("MLFP02 Exercise 7.4: Difference-in-Differences")

cells = simulate_hdb_cooling_measures(n_per_cell=500, seed=99)

print(f"\n  Scenario: Stamp duty increase in Central Singapore")
print(f"  Treatment: Central HDB transactions (hit by policy)")
print(f"  Control:   Non-Central HDB transactions (exempt)")
print(f"  Samples per cell: 500")

# ── Checkpoint 1 ─────────────────────────────────────────────────────
assert all(len(v) == 500 for v in cells.values()), "All cells must have 500 samples"
print("\n>>> Checkpoint 1 passed -- HDB data simulated\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 2 — Compute DiD Estimate
# ════════════════════════════════════════════════════════════════════════
# ATT = (Y_treat_post - Y_treat_pre) - (Y_ctrl_post - Y_ctrl_pre)

did = diff_in_diff(cells)

print(f"\n=== Difference-in-Differences ===")
print(f"\n{'Group':<15} {'Pre-policy':>14} {'Post-policy':>14} {'Delta':>14}")
print("-" * 60)
print(
    f"{'Central':<15} ${did['y_treat_pre']:>12,.0f} ${did['y_treat_post']:>12,.0f} "
    f"${did['y_treat_post'] - did['y_treat_pre']:>+12,.0f}"
)
print(
    f"{'Non-Central':<15} ${did['y_ctrl_pre']:>12,.0f} ${did['y_ctrl_post']:>12,.0f} "
    f"${did['y_ctrl_post'] - did['y_ctrl_pre']:>+12,.0f}"
)
print(f"\nDiD estimate (policy effect): ${did['did_estimate']:,.0f}")
print(f"SE: ${did['se']:,.0f}")
print(f"95% CI: [${did['ci_lo']:,.0f}, ${did['ci_hi']:,.0f}]")
print(f"p-value: {did['p_value']:.4f}")
# INTERPRETATION: DiD removes time-invariant confounders by differencing
# pre and post periods. The assumption is that without the policy,
# Central and Non-Central would have followed parallel trends. The
# negative DiD estimate means the policy reduced Central prices
# relative to what they would have been.

# ── Checkpoint 2 ─────────────────────────────────────────────────────
assert did["se"] > 0, "DiD SE must be positive"
print("\n>>> Checkpoint 2 passed -- DiD analysis completed\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 3 — Parallel Trends Test
# ════════════════════════════════════════════════════════════════════════
# DiD validity requires parallel trends in the pre-period.
# Test: are the pre-period trends in treatment and control similar?

print(f"\n=== Parallel Trends Test ===")

pt = parallel_trends_test(seed=99)

print(f"Pre-period trends:")
print(f"  Central slope:     ${pt['slope_central']:,.0f}/period")
print(f"  Non-Central slope: ${pt['slope_noncentral']:,.0f}/period")
print(f"  Slope difference:  ${pt['slope_diff']:,.0f}/period")
print(f"  Bootstrap p-value: {pt['bootstrap_p']:.4f}")
if pt["passes"]:
    print(f"  Parallel trends assumption HOLDS (cannot reject equal slopes)")
else:
    print(f"  Parallel trends assumption VIOLATED -- DiD may be biased")

# ── Checkpoint 3 ─────────────────────────────────────────────────────
assert isinstance(pt["passes"], (bool, np.bool_)), "Parallel trends must return bool"
print("\n>>> Checkpoint 3 passed -- parallel trends test completed\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 4 — Visualise: DiD with Counterfactual
# ════════════════════════════════════════════════════════════════════════

fig = go.Figure()
fig.add_trace(
    go.Scatter(
        x=["Pre", "Post"],
        y=[did["y_treat_pre"], did["y_treat_post"]],
        name="Central (treated)",
        line={"color": "red"},
        mode="lines+markers",
    )
)
fig.add_trace(
    go.Scatter(
        x=["Pre", "Post"],
        y=[did["y_ctrl_pre"], did["y_ctrl_post"]],
        name="Non-Central (control)",
        line={"color": "blue"},
        mode="lines+markers",
    )
)
# Counterfactual: what Central would have been without the policy
counterfactual = did["y_treat_pre"] + (did["y_ctrl_post"] - did["y_ctrl_pre"])
fig.add_trace(
    go.Scatter(
        x=["Pre", "Post"],
        y=[did["y_treat_pre"], counterfactual],
        name="Counterfactual (Central without policy)",
        line={"dash": "dot", "color": "red"},
        mode="lines+markers",
    )
)
# Annotate the DiD
fig.add_annotation(
    x="Post",
    y=(did["y_treat_post"] + counterfactual) / 2,
    text=f"DiD = ${did['did_estimate']:,.0f}",
    showarrow=True,
    arrowhead=2,
)
fig.update_layout(
    title="Difference-in-Differences: Singapore HDB Cooling Measures",
    yaxis_title="Mean HDB Price (S$)",
)
out_path = OUTPUT_DIR / "did_visualization.html"
fig.write_html(str(out_path))
print(f"\nSaved: {out_path}")

# Parallel trends visualisation
fig2 = go.Figure()
time_pts = list(range(len(pt["pre_central"])))
fig2.add_trace(
    go.Scatter(
        x=time_pts,
        y=pt["pre_central"],
        name="Central (pre-period)",
        line={"color": "red"},
        mode="lines+markers",
    )
)
fig2.add_trace(
    go.Scatter(
        x=time_pts,
        y=pt["pre_noncentral"],
        name="Non-Central (pre-period)",
        line={"color": "blue"},
        mode="lines+markers",
    )
)
fig2.update_layout(
    title=f"Parallel Trends Test (bootstrap p={pt['bootstrap_p']:.3f})",
    xaxis_title="Pre-Period",
    yaxis_title="Mean HDB Price (S$)",
)
out_path2 = OUTPUT_DIR / "parallel_trends.html"
fig2.write_html(str(out_path2))
print(f"Saved: {out_path2}")


# ════════════════════════════════════════════════════════════════════════
# APPLY — MAS/URA Policy Evaluation: Additional Buyer's Stamp Duty
# ════════════════════════════════════════════════════════════════════════
# Scenario: Singapore's Additional Buyer's Stamp Duty (ABSD) was
# introduced in December 2011 and has been revised multiple times.
# MAS and URA evaluate its impact using DiD:
#
#   Treatment: property types affected by ABSD (e.g., private condos)
#   Control: property types exempt (e.g., HDB resale for first-time buyers)
#   Pre: 12 months before each ABSD revision
#   Post: 12 months after each ABSD revision
#
# The DiD estimate tells policymakers whether the ABSD actually
# cooled prices or if prices were already declining. This informs
# whether to tighten (raise rates), maintain, or relax the measure.
#
# Business impact:
#   - S$1.2T residential property market
#   - 1% pricing correction = S$12B market impact
#   - Getting the DiD wrong means either over-cooling (market freeze)
#     or under-cooling (bubble continues)

print(f"\n--- Singapore Application: Property Cooling Measure Evaluation ---")
market_size = 1_200_000_000_000  # S$1.2T
price_effect_pct = abs(did["did_estimate"]) / did["y_treat_pre"]
market_impact = market_size * price_effect_pct
print(f"Singapore residential property market: S${market_size / 1e12:.1f}T")
print(f"Estimated price effect: {price_effect_pct:.1%}")
print(f"Market impact: S${market_impact / 1e9:.1f}B")
print(
    f"Policy conclusion: {'ABSD effective' if did['p_value'] < 0.05 else 'ABSD effect not significant'}"
)


# ════════════════════════════════════════════════════════════════════════
# TASK 5 — Causal Inference Decision Framework
# ════════════════════════════════════════════════════════════════════════
# Synthesise all four methods into a decision tree.

print(f"\n{'='*70}")
print(f"CAUSAL INFERENCE DECISION FRAMEWORK")
print(f"{'='*70}")
print(
    """
When to use each method:

  CUPED: You have an RCT AND pre-experiment data.
    -> Reduces CI width (free precision gain)
    -> Unbiased: same point estimate, just tighter

  Bayesian A/B: You want P(B > A) instead of "is p < 0.05?"
    -> Directly answers "should we ship?"
    -> Expected loss quantifies cost of being wrong

  Sequential (mSPRT): You need to monitor experiments safely.
    -> Fixed p-values inflate Type I error when peeking
    -> mSPRT: always-valid, correct alpha at any stopping time

  DiD: Randomisation is impossible (policy evaluation).
    -> Requires parallel trends assumption
    -> Less precise than RCT but works with observational data
"""
)


# ════════════════════════════════════════════════════════════════════════
# LOG — ExperimentTracker
# ════════════════════════════════════════════════════════════════════════


async def log_did_results():
    db = "sqlite:///mlfp02_experiments.db"
    tracker = await ExperimentTracker.create(store_url=db)
    conn = ConnectionManager(db)
    await conn.initialize()

    exp_id = "mlfp02_ex7_diff_in_diff"

    async with tracker.track(experiment=exp_id, run_name="did_hdb_cooling") as run:
        await run.log_params(
            {
                "did_treatment": "Central Singapore HDB",
                "did_control": "Non-Central Singapore HDB",
                "n_per_cell": "500",
                "parallel_trends_method": "bootstrap",
            }
        )
        await run.log_metrics(
            {
                "did_estimate": float(did["did_estimate"]),
                "did_se": float(did["se"]),
                "did_p_value": float(did["p_value"]),
                "parallel_trends_p": float(pt["bootstrap_p"]),
                "parallel_trends_passes": float(pt["passes"]),
            }
        )
    print(f"\nLogged DiD experiment run")
    await conn.close()


try:
    asyncio.run(log_did_results())
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
    """
  - DiD: ATT = (treat_post - treat_pre) - (ctrl_post - ctrl_pre)
  - Parallel trends: bootstrap test for the key DiD assumption
  - Counterfactual reasoning: what WOULD have happened without treatment
  - Singapore policy evaluation: ABSD impact on HDB prices
  - Decision framework: CUPED vs Bayesian vs Sequential vs DiD

  COMPLETE: You now have four causal inference tools:
    1. CUPED — precision gain for randomised experiments
    2. Bayesian A/B — probability-based decisions with expected loss
    3. Sequential testing — safe experiment monitoring
    4. DiD — causal inference from observational data

  NEXT: In Exercise 8 (Module 2 Capstone), you'll build a complete
  statistical analysis pipeline from data to stakeholder report.
"""
)

print("\n>>> Exercise 7.4 complete -- Difference-in-Differences")
