# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# MLFP03 — Exercise 6.5: Fairness Audit (Disparate Impact, Equalized Odds,
#                                          Impossibility Theorem)
# ════════════════════════════════════════════════════════════════════════
#
# WHAT YOU'LL LEARN:
#   - Measure the disparate impact ratio (4/5 rule)
#   - Measure equalized odds (TPR and FPR parity across groups)
#   - Understand the Chouldechova/Kleinberg impossibility theorem
#   - Produce a regulatory-grade fairness audit report
#   - Apply: MAS Fair Dealing Guidelines annual disclosure
#
# PREREQUISITES: 01_shap_global.py
# ESTIMATED TIME: ~40 min
#
# TASKS:
#   1. Theory — multiple incompatible fairness definitions
#   2. Build — group splitting, disparate impact, equalized odds
#   3. Train — AUDIT the trained model
#   4. Visualise — per-group rate table + per-group TPR/FPR
#   5. Apply — MAS Fair Dealing report for SG Retail Bank
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import numpy as np
from dotenv import load_dotenv
from sklearn.metrics import confusion_matrix

from shared.mlfp03.ex_6 import (
    PROTECTED_CANDIDATES,
    build_shap_explainer,
    feature_index,
    print_section,
    rank_features_by_mean_abs_shap,
    synthetic_group_split,
)

load_dotenv()


# ════════════════════════════════════════════════════════════════════════
# THEORY — Three Incompatible Fairness Definitions
# ════════════════════════════════════════════════════════════════════════
#   1. Demographic parity: equal selection rates across groups
#   2. Equalized odds:     equal TPR and FPR across groups
#   3. Calibration:        predicted probs reliable per group
#
# Chouldechova (2017) + Kleinberg et al. (2016): when base rates differ,
# the three are MATHEMATICALLY INCOMPATIBLE. Any production model MUST
# pick one criterion and document the tradeoff.


# ════════════════════════════════════════════════════════════════════════
# TASK 2 — BUILD group-level audit machinery
# ════════════════════════════════════════════════════════════════════════

bundle = build_shap_explainer()
model = bundle["model"]
X_test = bundle["X_test"]
y_test = bundle["y_test"]
y_pred = bundle["y_pred"]
y_proba = bundle["y_proba"]
feature_names: list[str] = bundle["feature_names"]
shap_vals = bundle["shap_vals"]


def disparate_impact_by_attribute(
    X: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    attr_idx: int,
    attr_name: str,
) -> dict[str, dict]:
    """Compute per-value approval rate and disparate impact ratio."""
    attr_vals = X[:, attr_idx]
    unique_vals = np.unique(attr_vals[~np.isnan(attr_vals)])
    group_rates: dict[str, dict] = {}
    if len(unique_vals) > 10:
        return group_rates
    for val in sorted(unique_vals):
        mask = attr_vals == val
        n_group = int(mask.sum())
        if n_group < 10:
            continue
        # TODO: compute approval_rate = fraction of y_pred[mask] equal to 0
        approval_rate = ____
        default_rate = float(y_proba[mask].mean())
        group_rates[str(val)] = {
            "n": n_group,
            "approval_rate": approval_rate,
            "mean_default_prob": default_rate,
        }
    return group_rates


def equalized_odds_two_groups(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    group_a: np.ndarray,
    group_b: np.ndarray,
) -> dict[str, dict]:
    """Compute TPR and FPR for two boolean group masks."""
    out: dict[str, dict] = {}
    for name, mask in [("A", group_a), ("B", group_b)]:
        y_g = y_true[mask]
        p_g = y_pred[mask]
        if y_g.sum() > 0 and (y_g == 0).sum() > 0:
            tn, fp, fn, tp = confusion_matrix(y_g, p_g).ravel()
            # TODO: compute TPR = tp / (tp + fn), FPR = fp / (fp + tn)
            tpr = ____
            fpr = ____
        else:
            tpr, fpr = 0.0, 0.0
        out[name] = {"n": int(mask.sum()), "TPR": float(tpr), "FPR": float(fpr)}
    return out


# ════════════════════════════════════════════════════════════════════════
# TASK 3 — AUDIT the trained model
# ════════════════════════════════════════════════════════════════════════

print_section("Fairness Audit: Disparate Impact by Protected Attribute")

protected_in_model = [f for f in PROTECTED_CANDIDATES if f in feature_names]
di_results: dict[str, dict[str, dict]] = {}

if protected_in_model:
    print(f"Protected attributes in model: {protected_in_model}")
    for attr in protected_in_model:
        attr_idx = feature_index(feature_names, attr)
        rates = disparate_impact_by_attribute(X_test, y_pred, y_proba, attr_idx, attr)
        if len(rates) >= 2:
            di_results[attr] = rates
            majority_val = max(rates, key=lambda v: rates[v]["n"])
            majority_rate = rates[majority_val]["approval_rate"]
            print(f"\n  --- {attr} (majority = {majority_val}) ---")
            print(f"  {'Value':>10} {'N':>6} {'Approval':>10} {'Disp.Impact':>14}")
            print("  " + "─" * 44)
            for val, info in sorted(rates.items()):
                di = info["approval_rate"] / max(majority_rate, 0.001)
                flag = " < 0.8 FAIL" if di < 0.8 else ""
                print(
                    f"  {val:>10} {info['n']:>6} "
                    f"{info['approval_rate']:>10.3f} {di:>14.3f}{flag}"
                )
else:
    print("No protected attributes in feature set; using synthetic split.")
    group_a, group_b, median_val = synthetic_group_split(X_test, feature_idx=0)
    approval_a = float((y_pred[group_a] == 0).mean())
    approval_b = float((y_pred[group_b] == 0).mean())
    di_ratio = approval_a / max(approval_b, 0.001)
    print(f"  Feature: {feature_names[0]} (median split={median_val:.3f})")
    print(f"  Group A: approval={approval_a:.3f} (n={int(group_a.sum())})")
    print(f"  Group B: approval={approval_b:.3f} (n={int(group_b.sum())})")
    print(f"  DIR:     {di_ratio:.3f}  " f"({'PASS' if di_ratio >= 0.8 else 'FAIL'})")


# ── Equalized odds on synthetic split ─────────────────────────────────
print_section("Fairness Audit: Equalized Odds (synthetic split)", char="─")
group_a, group_b, median_val = synthetic_group_split(X_test, feature_idx=0)
# TODO: call equalized_odds_two_groups with y_test, y_pred, group_a, group_b
eo = ____
print(f"  Group A: n={eo['A']['n']} TPR={eo['A']['TPR']:.3f} FPR={eo['A']['FPR']:.3f}")
print(f"  Group B: n={eo['B']['n']} TPR={eo['B']['TPR']:.3f} FPR={eo['B']['FPR']:.3f}")
tpr_gap = abs(eo["A"]["TPR"] - eo["B"]["TPR"])
fpr_gap = abs(eo["A"]["FPR"] - eo["B"]["FPR"])
print(f"  |TPR_A - TPR_B| = {tpr_gap:.3f}")
print(f"  |FPR_A - FPR_B| = {fpr_gap:.3f}")


# ── Impossibility theorem ─────────────────────────────────────────────
print_section("Impossibility Theorem", char="─")
print(
    """
  When base rates differ between groups you CANNOT simultaneously satisfy
  demographic parity, equalized odds, and calibration. Pick one.
"""
)
base_rate_a = float(y_test[group_a].mean())
base_rate_b = float(y_test[group_b].mean())
print(f"  Base rate A: {base_rate_a:.3f}")
print(f"  Base rate B: {base_rate_b:.3f}")


# ── SHAP contribution of protected attributes ─────────────────────────
importance_ranking = rank_features_by_mean_abs_shap(shap_vals, feature_names)

print_section("SHAP Contribution of Protected Attributes", char="─")
if protected_in_model:
    for attr in protected_in_model:
        attr_idx_audit = feature_index(feature_names, attr)
        attr_shap = shap_vals[:, attr_idx_audit]
        rank = [n for n, _ in importance_ranking].index(attr) + 1
        print(
            f"  {attr}: mean|SHAP|={np.abs(attr_shap).mean():.4f}  "
            f"rank #{rank}/{len(feature_names)}"
        )
else:
    print("  No explicit protected attributes (proxies may still encode them).")

# ── Checkpoint ──────────────────────────────────────────────────────────
assert eo["A"]["n"] > 0 and eo["B"]["n"] > 0, "Task 3: both groups must be non-empty"
print("\n[ok] Checkpoint — fairness audit complete\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 4 — Print the full audit report
# ════════════════════════════════════════════════════════════════════════

print_section("FAIRNESS AUDIT REPORT — Credit Default Model")
print(
    f"""
AUDIT SUMMARY:
  1. Disparate impact:   {len(di_results)} protected attrs reviewed
  2. Equalized odds:     |TPR gap|={tpr_gap:.3f}, |FPR gap|={fpr_gap:.3f}
  3. SHAP protected-attribute rank: completed
  4. Impossibility theorem: acknowledged

RECOMMENDATIONS:
  a. Monitor DIR quarterly (threshold 0.8)
  b. Document which fairness criterion is prioritised and why
  c. SHAP explanation for EVERY declined application
  d. Review proxy variable effects (zip, education)
"""
)


# ════════════════════════════════════════════════════════════════════════
# TASK 5 — APPLY: MAS Fair Dealing Compliance Report
# ════════════════════════════════════════════════════════════════════════
# MAS 2024 Fair Dealing Guidelines require an annual fairness disclosure
# per algorithmic credit model: DIR table, TPR/FPR parity, written
# justification, escalation path for DIR < 0.8.
#
# BUSINESS IMPACT: One avoided MAS enforcement incident = ~30x payback
# on the S$16,000 annual audit cost.


# ════════════════════════════════════════════════════════════════════════
# DESTINATION-FIRST CLOSE — km.diagnose
# ════════════════════════════════════════════════════════════════════════
# This lesson built a fairness audit from primitives — disparate impact
# ratio, equal opportunity difference, predictive parity — to internalise
# the regulatory framing. The kailash-ml SDK packages the standard
# diagnostic surface (per-class metrics, severity heuristics, confusion
# matrix) into a single call; group-conditional fairness metrics layer on
# top of that base.
#
# Destination-first: when the journey is internalised, the SDK is one line.

from kailash_ml import diagnose

# `kind="classical_classifier"` dispatches to the sklearn ClassifierMixin
# adapter. The fairness audit's underlying classifier is the bundle's
# `model` already loaded for the audit.
report = diagnose(model, kind="classical_classifier", data=(X_test, y_test), show=False)
print()
print("  km.diagnose model    : audited credit-default classifier")
print(f"  km.diagnose metrics  : {report.metrics}")
print(f"  km.diagnose severity : {report.severity}")
print()
print("km.diagnose: 1 call for the base diagnostic surface; the fairness")
print("metrics above sit on top of it. Destination-first: when the")
print("journey is internalised, the SDK is one line.")


# ════════════════════════════════════════════════════════════════════════
# REFLECTION
# ════════════════════════════════════════════════════════════════════════
print_section("WHAT YOU'VE MASTERED")
print(
    """
  [x] Measured disparate impact ratio across protected groups
  [x] Measured TPR and FPR parity (equalized odds)
  [x] Explained the Chouldechova/Kleinberg impossibility theorem
  [x] Audited SHAP contribution of protected attributes
  [x] Produced a regulatory-grade fairness audit report
  [x] Mapped the pipeline to MAS Fair Dealing annual disclosure

  KEY INSIGHT: Fairness is a FAMILY of incompatible definitions. Measure
  all three, document the tradeoff, make the decision legible to auditors.

  END OF EXERCISE 6.
"""
)
