# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# MLFP02 — Exercise 7.2: Bayesian A/B Testing
# ════════════════════════════════════════════════════════════════════════
#
# WHAT YOU'LL LEARN:
#   - Compute P(treatment > control | data) using posterior distributions
#   - Calculate expected loss for decision-making under uncertainty
#   - Apply a ship/continue/hold decision framework
#   - Understand when Bayesian beats frequentist A/B testing
#   - Log Bayesian results to ExperimentTracker
#
# PREREQUISITES: Exercise 7.1 (CUPED) — you need CUPED-adjusted arrays
# ESTIMATED TIME: ~35 min
#
# TASKS:
#   1. Load data and apply CUPED adjustment
#   2. Compute Bayesian posterior for the treatment effect
#   3. Expected loss analysis (both directions)
#   4. Decision framework: ship / continue / hold
#   5. Visualise posterior distribution
#   6. Apply to Singapore fintech scenario
#
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import asyncio

import numpy as np
import plotly.graph_objects as go
from kailash.db import ConnectionManager
from kailash_ml import ExperimentTracker
from scipy import stats

from shared.mlfp02.ex_7 import (
    OUTPUT_DIR,
    bayesian_decision,
    bayesian_decision_rule,
    get_covariate_arrays,
    get_revenue_arrays,
    load_experiment,
    print_banner,
    single_cov_cuped,
    split_groups,
)


# ════════════════════════════════════════════════════════════════════════
# THEORY — Why Bayesian A/B Testing?
# ════════════════════════════════════════════════════════════════════════
# Frequentist A/B answers: "If there were no effect, how unlikely is
# this data?" — that is a p-value. But what product teams actually want
# is: "Given the data, what is the probability that B is better than A?"
#
# Bayesian analysis provides:
#   P(treatment > control | data) — direct probability of improvement
#   Expected loss — the average revenue you lose by choosing wrong
#
# The expected loss is particularly powerful: if P(B > A) = 75% but
# the expected loss of choosing B is only $0.02/user, you can ship
# confidently. If P(B > A) = 95% but expected loss is $5/user, you
# should collect more data.
#
# WHY THIS MATTERS: At GrabPay (Singapore), the product team uses
# expected loss rather than p-values for payment flow experiments,
# because the cost of a wrong decision (friction in checkout) is
# directly quantifiable in S$/transaction.


# ════════════════════════════════════════════════════════════════════════
# TASK 1 — Load Data and Apply CUPED
# ════════════════════════════════════════════════════════════════════════

print_banner("MLFP02 Exercise 7.2: Bayesian A/B Testing")

# TODO: Load experiment, split groups, extract revenue and covariate arrays.
# Hint: Use load_experiment(), split_groups(), get_revenue_arrays(),
#   get_covariate_arrays() from the shared module.
experiment = ____
control, treatment = ____
y_c, y_t = ____
x_c, x_t = ____

# TODO: Apply CUPED to get adjusted arrays.
# Hint: single_cov_cuped(y_c, y_t, x_c, x_t) returns a dict with
#   y_c_adj, y_t_adj, lift, rho, etc.
cuped = ____
y_c_adj = cuped["y_c_adj"]
y_t_adj = cuped["y_t_adj"]
lift_adj = cuped["lift"]

print(f"  Data loaded, CUPED applied (rho={cuped['rho']:.3f})")
print(f"  CUPED-adjusted lift: ${lift_adj:.2f}")

# ── Checkpoint 1 ─────────────────────────────────────────────────────
assert y_c_adj is not None and len(y_c_adj) > 0, "CUPED adjustment must produce data"
print("\n>>> Checkpoint 1 passed -- data loaded and CUPED applied\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 2 — Bayesian Posterior for Treatment Effect
# ════════════════════════════════════════════════════════════════════════
# Using normal approximation on CUPED-adjusted arrays.
# Posterior: lift ~ Normal(lift_adj, se_lift)

# TODO: Compute the Bayesian decision metrics.
# Hint: bayesian_decision(y_c_adj, y_t_adj, lift_adj, practical_threshold=1.0)
#   returns a dict with prob_treatment_better, prob_practical,
#   expected_loss_treatment, expected_loss_control, se_lift, ci_lo, ci_hi.
bayes = ____

print(f"\n=== Bayesian A/B Test ===")
print(
    f"P(treatment > control): {bayes['prob_treatment_better']:.4f} "
    f"({bayes['prob_treatment_better']:.1%})"
)
print(
    f"P(treatment > control by >$1): {bayes['prob_practical']:.4f} "
    f"({bayes['prob_practical']:.1%})"
)
print(f"Expected loss (choose treatment): ${bayes['expected_loss_treatment']:.2f}/user")
print(f"Expected loss (choose control):   ${bayes['expected_loss_control']:.2f}/user")
print(f"95% credible interval: [${bayes['ci_lo']:.2f}, ${bayes['ci_hi']:.2f}]")

# ── Checkpoint 2 ─────────────────────────────────────────────────────
assert 0 <= bayes["prob_treatment_better"] <= 1, "Probability must be valid"
assert bayes["expected_loss_treatment"] >= 0, "Expected loss must be non-negative"
print("\n>>> Checkpoint 2 passed -- Bayesian posterior computed\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 3 — Expected Loss Analysis
# ════════════════════════════════════════════════════════════════════════
# Expected loss quantifies the cost of being wrong.
# E[loss | choose treatment] = E[max(control - treatment, 0)]
# E[loss | choose control]   = E[max(treatment - control, 0)]

print(f"\n=== Expected Loss Analysis ===")
print(f"If we ship treatment and it is worse:")
print(f"  Average loss per user: ${bayes['expected_loss_treatment']:.4f}")
print(f"If we keep control and treatment is actually better:")
print(f"  Average loss per user: ${bayes['expected_loss_control']:.4f}")
print(
    f"\nLoss ratio (control/treatment): "
    f"{bayes['expected_loss_control'] / max(bayes['expected_loss_treatment'], 1e-9):.1f}x"
)

# ── Checkpoint 3 ─────────────────────────────────────────────────────
assert bayes["expected_loss_control"] >= 0, "Expected loss must be non-negative"
print("\n>>> Checkpoint 3 passed -- expected loss analysis complete\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 4 — Decision Framework: Ship / Continue / Hold
# ════════════════════════════════════════════════════════════════════════

# TODO: Apply the decision rule using the shared helper.
# Hint: bayesian_decision_rule(prob_better, expected_loss_treatment)
#   returns one of "SHIP ...", "CONTINUE ...", "HOLD ...".
decision = ____

print(f"\n=== Decision Framework ===")
print(f"P(treatment better): {bayes['prob_treatment_better']:.1%}")
print(f"Expected loss: ${bayes['expected_loss_treatment']:.2f}/user")
print(f"\nDecision: {decision}")
print(f"\nDecision rules:")
print(f"  SHIP:     P > 95% AND expected loss < $0.50/user")
print(f"  CONTINUE: P > 80% (promising but need more data)")
print(f"  HOLD:     P <= 80% (insufficient evidence)")

# ── Checkpoint 4 ─────────────────────────────────────────────────────
assert decision in [
    "SHIP — high confidence + low expected loss",
    "CONTINUE — promising but need more data",
    "HOLD — insufficient evidence",
], "Decision must be one of the three options"
print("\n>>> Checkpoint 4 passed -- decision framework applied\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 5 — Visualise: Posterior Distribution
# ════════════════════════════════════════════════════════════════════════

# TODO: Create a plotly figure showing the posterior distribution of the
# treatment effect. Add vertical lines at 0 (no effect) and $1
# (practical threshold).
# Hint: Use stats.norm.pdf(x, loc=lift_adj, scale=bayes["se_lift"])
x_range = np.linspace(
    lift_adj - 4 * bayes["se_lift"], lift_adj + 4 * bayes["se_lift"], 200
)
pdf_vals = ____

fig = go.Figure()
fig.add_trace(
    go.Scatter(x=x_range, y=pdf_vals, mode="lines", name="Posterior", fill="tozeroy")
)
fig.add_vline(x=0, line_dash="dot", line_color="red", annotation_text="No effect")
fig.add_vline(
    x=1.0,
    line_dash="dash",
    line_color="green",
    annotation_text="Practical threshold ($1)",
)
fig.add_vline(
    x=lift_adj,
    line_dash="solid",
    line_color="blue",
    annotation_text=f"Estimated lift: ${lift_adj:.2f}",
)
fig.update_layout(
    title="Posterior Distribution of Treatment Effect",
    xaxis_title="Treatment Effect ($)",
    yaxis_title="Density",
)
out_path = OUTPUT_DIR / "bayesian_posterior.html"
fig.write_html(str(out_path))
print(f"\nSaved: {out_path}")


# ════════════════════════════════════════════════════════════════════════
# APPLY — GrabPay Singapore: Payment Flow Experiments
# ════════════════════════════════════════════════════════════════════════
# Scenario: GrabPay tests a new checkout flow. The metric is revenue
# per transaction. With ~2M transactions/day, even small lifts matter.
#
# Traditional approach: "Is p < 0.05?" — binary, ignores magnitude.
# Bayesian approach: "P(B > A) = 87%, expected loss = $0.03/txn"
#   -> At 2M txns/day, choosing wrong costs $60K/day
#   -> But choosing right gains $200K/day
#   -> Ship: expected gain far exceeds expected loss

print(f"\n--- Singapore Application: Payment Flow Experiment ---")
daily_txns = 2_000_000
daily_gain = daily_txns * lift_adj * bayes["prob_treatment_better"]
daily_loss = daily_txns * bayes["expected_loss_treatment"]
print(f"Daily transactions: {daily_txns:,}")
print(f"Expected daily gain from shipping: S${daily_gain:,.0f}")
print(f"Expected daily loss if wrong: S${daily_loss:,.0f}")
print(f"Net expected daily value: S${daily_gain - daily_loss:,.0f}")
print(f"Annualised net value: S${(daily_gain - daily_loss) * 365:,.0f}")


# ════════════════════════════════════════════════════════════════════════
# LOG — ExperimentTracker
# ════════════════════════════════════════════════════════════════════════


async def log_bayesian_results():
    db = "sqlite:///mlfp02_experiments.db"
    tracker = await ExperimentTracker.create(store_url=db)
    conn = ConnectionManager(db)
    await conn.initialize()

    exp_id = "mlfp02_ex7_bayesian_ab"

    async with tracker.track(experiment=exp_id, run_name="bayesian_decision") as run:
        await run.log_params(
            {
                "method": "bayesian_normal_approx",
                "practical_threshold": "1.0",
                "decision": decision,
            }
        )
        await run.log_metrics(
            {
                "prob_treatment_better": float(bayes["prob_treatment_better"]),
                "prob_practical": float(bayes["prob_practical"]),
                "expected_loss_treatment": float(bayes["expected_loss_treatment"]),
                "expected_loss_control": float(bayes["expected_loss_control"]),
                "lift_cuped": float(lift_adj),
            }
        )
    print(f"\nLogged Bayesian experiment run")
    await conn.close()


try:
    asyncio.run(log_bayesian_results())
except Exception as e:
    print(f"  [Skipped: ExperimentTracker logging ({type(e).__name__}: {e})]")

# ── Checkpoint 5 ─────────────────────────────────────────────────────
print("\n>>> Checkpoint 5 passed -- visualisation and logging complete\n")


# ══════════════════════════════════════════════════════════════════════
# REFLECTION
# ══════════════════════════════════════════════════════════════════════
print("=" * 70)
print("  WHAT YOU'VE MASTERED")
print("=" * 70)
print(
    """
  - Bayesian A/B: P(treatment > control) and expected loss
  - Decision framework: ship (>95% + low loss) / continue / hold
  - Expected loss quantifies the cost of being wrong in $/user
  - Posterior credible interval vs frequentist confidence interval
  - When to ship with moderate confidence (low expected loss)

  NEXT: In 03_sequential_testing.py, you'll learn to safely monitor
  experiments without inflating Type I error, using mSPRT.
"""
)

print("\n>>> Exercise 7.2 complete -- Bayesian A/B Testing")
