# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# MLFP05 — Exercise 7, Part 3: Data Efficiency Experiment
# ════════════════════════════════════════════════════════════════════════
#
# WHAT YOU'LL LEARN:
#   After completing this section, you will be able to:
#   - Run a controlled data efficiency experiment: train a transfer
#     model on 10%, 25%, 50%, and 100% of training data
#   - Quantify how transfer learning bends the labelling cost curve
#   - Plot data efficiency curves comparing transfer vs from-scratch
#   - Answer the business question: "How many images do we need to label?"
#   - Calculate cost savings in a real Singapore business scenario
#
# PREREQUISITES: Part 1 (baseline), Part 2 (transfer learning).
# ESTIMATED TIME: ~25 min
#
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import asyncio
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

import torchvision

from shared.mlfp05.ex_7 import (
    BATCH_SIZE,
    EPOCHS,
    N_CLASSES,
    OUTPUT_DIR,
    count_params,
    device,
    init_engines,
    load_cifar10,
)


# ════════════════════════════════════════════════════════════════════════
# THEORY — The Labelling Bottleneck
# ════════════════════════════════════════════════════════════════════════
# In production ML, the biggest cost isn't compute — it's labelled data.
#
# Consider what labelling costs in practice:
#   - Simple image classification: S$0.50-1.00 per image
#   - Medical image annotation: S$5-20 per image (specialist required)
#   - Autonomous driving frames: S$50-200 per frame (3D bounding boxes)
#
# If you need 50,000 labelled images at S$1 each, that's S$50,000 just
# for data — before you've written a single line of code.
#
# Transfer learning bends this cost curve. By reusing features from a
# pre-trained model, you can achieve 80-90% of the full-data accuracy
# with only 10-25% of the labelled data. This experiment quantifies
# exactly where that sweet spot is.
#
# The key question for any ML project: "What's the minimum amount of
# labelled data that gives us acceptable accuracy?" This experiment
# gives you the data to answer it.
# ════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("  PART 3: Data Efficiency Experiment")
print("=" * 70)


# ════════════════════════════════════════════════════════════════════════
# TASK 1 — Load data and set up engines
# ════════════════════════════════════════════════════════════════════════

train_set, val_set, train_loader, val_loader = load_cifar10()
conn, tracker, exp_name, registry, has_registry = init_engines()


# ════════════════════════════════════════════════════════════════════════
# TASK 2 — Define model builders
# ════════════════════════════════════════════════════════════════════════


def build_transfer_resnet(n_classes: int = N_CLASSES) -> nn.Module:
    """Frozen ResNet-18 backbone + fresh classification head."""
    # TODO: Load pre-trained ResNet-18, freeze backbone, replace fc head
    # Steps:
    #   1. Load weights: torchvision.models.ResNet18_Weights.IMAGENET1K_V1
    #   2. Create model with those weights (try/except for offline fallback)
    #   3. Freeze all params: for p in model.parameters(): p.requires_grad = False
    #   4. Replace model.fc with nn.Linear(model.fc.in_features, n_classes)
    # Hint: Same pattern as Part 2's build_transfer_resnet
    ____


def build_scratch_cnn(n_classes: int = N_CLASSES) -> nn.Module:
    """From-scratch CNN baseline."""
    # TODO: Build a 3-layer CNN identical to Part 1
    # Hint: nn.Sequential with Conv2d->BN->ReLU->Pool blocks
    #       then AdaptiveAvgPool2d(1)->Flatten->Dropout(0.3)->Linear(128, n_classes)
    ____


# ════════════════════════════════════════════════════════════════════════
# TASK 3 — Run the data efficiency experiment
# ════════════════════════════════════════════════════════════════════════
# For each data fraction (10%, 25%, 50%, 100%), we train BOTH a transfer
# model and a from-scratch model, then compare. This gives us two
# curves that show exactly where transfer learning helps most.

DATA_FRACTIONS = [0.10, 0.25, 0.50, 1.0]
EFF_EPOCHS = 4  # Shorter training for sub-experiments

transfer_results: dict[float, float] = {}
scratch_results: dict[float, float] = {}

rng = np.random.default_rng(42)


async def _run_efficiency_trial(
    frac: float,
    model_builder,
    model_name: str,
) -> tuple[float, int]:
    """Train one model on a fraction of data, return (accuracy, n_samples)."""
    # TODO: Implement the efficiency trial
    # Steps:
    #   1. Compute n_samples = int(len(train_set) * frac)
    #   2. Sample random indices: rng.choice(len(train_set), size=n_samples, replace=False).tolist()
    #   3. Create Subset(train_set, indices) and DataLoader
    #   4. Build model, move to device, get trainable params
    #   5. Create Adam optimizer with lr=1e-3
    #   6. Log to ExperimentTracker with run_name=f"{model_name}_{int(frac*100)}pct"
    #   7. Train for EFF_EPOCHS: forward -> cross_entropy -> backward -> step
    #   8. Evaluate on full val_loader: count correct/total
    #   9. Return (accuracy, n_samples)
    # Hint: async with tracker.track(experiment=exp_name, run_name=run_name) as run:
    # Hint: await run.log_params({...}), await run.log_metric("val_acc", acc)
    n_samples = int(len(train_set) * frac)
    ____

    return ____, n_samples


print("\n" + "=" * 70)
print("  DATA EFFICIENCY EXPERIMENT")
print("=" * 70)

for frac in DATA_FRACTIONS:
    # Transfer model
    t_acc, n_samples = asyncio.run(
        _run_efficiency_trial(frac, build_transfer_resnet, "transfer")
    )
    transfer_results[frac] = t_acc

    # From-scratch model
    s_acc, _ = asyncio.run(_run_efficiency_trial(frac, build_scratch_cnn, "scratch"))
    scratch_results[frac] = s_acc

    print(
        f"  {frac * 100:5.0f}% data ({n_samples:>5,} samples): "
        f"transfer={t_acc:.4f}  scratch={s_acc:.4f}  "
        f"gap={t_acc - s_acc:+.4f}"
    )

# ── Checkpoint 1 ─────────────────────────────────────────────────────
assert len(transfer_results) == len(
    DATA_FRACTIONS
), "Should have transfer results for all fractions"
assert len(scratch_results) == len(
    DATA_FRACTIONS
), "Should have scratch results for all fractions"
assert (
    transfer_results[0.10] > 0.15
), f"Transfer with 10% data should beat random (acc={transfer_results[0.10]:.3f})"
# INTERPRETATION: Transfer learning shows diminishing returns as data
# increases — the gap between 10% and 100% is smaller than the scratch
# model's gap. Pre-trained features already capture general visual
# patterns, so additional data helps but isn't as critical.
print("\n--- Checkpoint 1 passed --- efficiency experiment complete\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 4 — Visualise: Data efficiency curves
# ════════════════════════════════════════════════════════════════════════
# The money chart: accuracy vs percentage of training data for both
# approaches. This is the chart you show to the VP of Engineering.

fracs = sorted(transfer_results.keys())
transfer_accs_by_frac = [transfer_results[f] for f in fracs]
scratch_accs_by_frac = [scratch_results[f] for f in fracs]
pct_labels = [f * 100 for f in fracs]

# TODO: Create a Plotly figure with two traces:
#   1. Transfer learning curve (lines+markers, color="#2196F3")
#   2. From-scratch curve (lines+markers, dash="dash", color="#FF5722")
# Add annotations for the 10% data points on both curves
# Hint: fig = go.Figure()
# Hint: fig.add_trace(go.Scatter(x=pct_labels, y=transfer_accs_by_frac, ...))
# Hint: fig.add_annotation(x=10, y=transfer_results[0.10], text=..., showarrow=True)
fig = go.Figure()
____

fig.update_layout(
    title="Data Efficiency: Transfer Learning vs From-Scratch",
    xaxis_title="% of CIFAR-10 Training Data",
    yaxis_title="Validation Accuracy",
    template="plotly_white",
    legend=dict(x=0.6, y=0.15),
    width=800,
    height=500,
)

eff_path = OUTPUT_DIR / "03_data_efficiency.html"
fig.write_html(str(eff_path))
print(f"  Saved: {eff_path}")

# ── Checkpoint 2 ─────────────────────────────────────────────────────
assert (eff_path).exists(), "Data efficiency plot should be saved"
print("--- Checkpoint 2 passed --- efficiency curves plotted\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 5 — Visualise: Accuracy gap and diminishing returns
# ════════════════════════════════════════════════════════════════════════
# The gap between transfer and scratch narrows as data increases.
# This shows that transfer learning's biggest value is with LIMITED data.

# TODO: Compute gaps and create a bar chart
# Steps:
#   1. gaps = [t - s for t, s in zip(transfer_accs_by_frac, scratch_accs_by_frac)]
#   2. Create go.Figure() with go.Bar trace
#   3. x = [f"{p:.0f}%" for p in pct_labels], y = [g * 100 for g in gaps]
#   4. Color bars green if gap > 0, red otherwise
#   5. Save to OUTPUT_DIR / "03_accuracy_gap.html"
# Hint: marker_color=["#4CAF50" if g > 0 else "#F44336" for g in gaps]
gaps = [t - s for t, s in zip(transfer_accs_by_frac, scratch_accs_by_frac)]
____

gap_path = OUTPUT_DIR / "03_accuracy_gap.html"
fig_gap.write_html(str(gap_path))
print(f"  Saved: {gap_path}")


# ════════════════════════════════════════════════════════════════════════
# TASK 6 — Apply: The VP of Engineering at Grab Asks "How Many Images?"
# ════════════════════════════════════════════════════════════════════════
# SCENARIO: You're the ML lead at Grab Singapore. The VP of Engineering
# asks: "We want to build an image classifier for food delivery photos.
# How many images do we need to label? What will it cost?"
#
# You use this data efficiency experiment to answer concretely.

print("\n" + "=" * 70)
print("  APPLY: Grab Singapore — 'How many images do we need to label?'")
print("=" * 70)

COST_PER_LABEL = 0.80  # S$ per image label (food photo classification)
TOTAL_AVAILABLE = 50000  # Total unlabelled images available

# TODO: Print the cost-accuracy trade-off table
# Steps:
#   1. Loop through fracs, compute n_images and label_cost for each
#   2. Print a formatted table with columns: Data%, Images, Transfer acc,
#      Scratch acc, Label Cost, Transfer Saves
# Hint: n_images = int(TOTAL_AVAILABLE * frac)
# Hint: label_cost = n_images * COST_PER_LABEL
print(f"\n  === Cost-Accuracy Trade-off Analysis ===")
print(f"  Labelling cost: S${COST_PER_LABEL:.2f} per image")
print(f"  Unlabelled pool: {TOTAL_AVAILABLE:,} food delivery photos")
____

# TODO: Find the sweet spot where transfer reaches 90% of max accuracy
# Steps:
#   1. max_transfer_acc = transfer_results[1.0]
#   2. sweet_spot_threshold = 0.90 * max_transfer_acc
#   3. Loop through fracs to find first frac where accuracy >= threshold
#   4. Calculate savings vs labelling the full dataset
# Hint: sweet_spot_frac = None; iterate and break when found
max_transfer_acc = transfer_results[1.0]
sweet_spot_threshold = 0.90 * max_transfer_acc
sweet_spot_frac = None
____

if sweet_spot_frac is not None:
    sweet_n = int(TOTAL_AVAILABLE * sweet_spot_frac)
    sweet_cost = sweet_n * COST_PER_LABEL
    full_cost = TOTAL_AVAILABLE * COST_PER_LABEL
    savings = full_cost - sweet_cost
    print(f"\n  SWEET SPOT: {sweet_spot_frac * 100:.0f}% of data ({sweet_n:,} images)")
    print(
        f"  Reaches {transfer_results[sweet_spot_frac]:.1%} accuracy "
        f"(90% of maximum {max_transfer_acc:.1%})"
    )
    print(f"  Label cost: S${sweet_cost:,.0f} vs S${full_cost:,.0f} for full dataset")
    print(f"  SAVINGS: S${savings:,.0f}")
else:
    print(f"\n  All fractions tested achieve >=90% of maximum accuracy.")

print()
print(f"  RECOMMENDATION TO VP:")
print(
    f"  'Start with {int(TOTAL_AVAILABLE * 0.25):,} labelled images "
    f"(S${int(TOTAL_AVAILABLE * 0.25 * COST_PER_LABEL):,})."
)
print(f"   Use transfer learning with ResNet-18. If accuracy is insufficient,")
print(f"   label more images in batches of 5,000 until you reach the target.")
print(f"   Transfer learning means we never need to label all 50,000 images.'")

# ── Checkpoint 3 ─────────────────────────────────────────────────────
assert (
    sweet_spot_frac is not None or len(fracs) > 0
), "Should identify a sweet spot or have results"
# INTERPRETATION: The data efficiency curve directly answers the VP's
# question with concrete numbers: how many images to label, how much
# it costs, and where the diminishing returns kick in. This is how ML
# engineers translate technical results into business decisions.
print("\n--- Checkpoint 3 passed --- business analysis complete\n")


# ════════════════════════════════════════════════════════════════════════
# CLEANUP
# ════════════════════════════════════════════════════════════════════════
asyncio.run(conn.close())


# ════════════════════════════════════════════════════════════════════════
# REFLECTION
# ════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  PART 3 COMPLETE — What You've Learned")
print("=" * 70)
print(
    f"""
  [x] Ran data efficiency experiment across 4 data fractions
  [x] Transfer with 10% data: {transfer_results[0.10]:.1%} accuracy
  [x] Transfer with 100% data: {transfer_results[1.0]:.1%} accuracy
  [x] Scratch with 10% data: {scratch_results[0.10]:.1%} accuracy
  [x] Plotted data efficiency curves (transfer vs scratch)
  [x] Identified the sweet spot: {sweet_spot_frac * 100:.0f}% of data for 90% of max accuracy
  [x] Calculated labelling cost savings for Grab Singapore scenario

  KEY INSIGHT: Transfer learning's biggest value is with LIMITED data.
  The gap between transfer and scratch is largest at 10-25% data, then
  narrows as data increases. This means:
    - With abundant data: transfer helps but isn't critical
    - With scarce data: transfer is transformative

  THE LABELLING BOTTLENECK EQUATION:
    Cost = (images needed) x (cost per label)
    Transfer learning reduces the first term by 4-10x.
    This is often the difference between a viable project and a shelved one.

  NEXT: Part 4 introduces adapter modules — a parameter-efficient
  alternative to full fine-tuning that bridges to M6's LoRA technique.
"""
)

# ══════════════════════════════════════════════════════════════════
# DIAGNOSTIC CHECKPOINT — five instruments before Visualise
# ══════════════════════════════════════════════════════════════════
# Reference: `kailash_ml.diagnostics` (via `kailash-ml`) — see gold standard
# `solutions/ex_1/01_standard_ae.py` for the full pattern.
from kailash_ml.diagnostics import run_diagnostic_checkpoint


def _diag_loss(m, batch):
    # Training at 10%, 25%, 50%, 100% of data
    # Customise per your exercise's loss shape.
    if isinstance(batch, (tuple, list)):
        x = batch[0]
        y = batch[1] if len(batch) > 1 else None
    else:
        x, y = batch, None
    out = m(x)
    import torch.nn.functional as F
    if y is None:
        return F.mse_loss(out, x)
    return F.cross_entropy(out, y)


print("\n── Diagnostic Report (Data efficiency — how small can we go?) ──")
try:
    diag, findings = run_diagnostic_checkpoint(
        models_by_frac[1.0],
        train_loader,
        _diag_loss,
        title="Data efficiency — how small can we go?",
        n_batches=8,
        show=False,
    )
except Exception as exc:
    # Diagnostic is pedagogical — never block the exercise on it.
    print(f"[diagnostic skipped: {exc}]")

# ══════ EXPECTED OUTPUT (synthesized reference — full run produces similar pattern) ══════
# ════════════════════════════════════════════════════════════════
#   DL Diagnostics Report — Prescription Pad
# ════════════════════════════════════════════════════════════════
# [Cross-run comparison — all 4 data fractions]
# 100% data: RMS healthy, 87% val accuracy
#  50% data: RMS healthy, 84% val accuracy
#  25% data: train-val gap widening (overfit)
#  10% data: [CRITICAL] 52% val accuracy — too little data to generalise
# ════════════════════════════════════════════════════════════════
#
# STUDENT INTERPRETATION GUIDE — reading the Prescription Pad:

#  [STETHOSCOPE] The data-efficiency curve shows transfer
#     learning's power: 50% of data still gives ~97% of full
#     performance. Below 25%, diminishing returns kick in.
#     >> Decision rule: if you have >1000 labelled examples,
#        transfer learning + fine-tune works. If <500, try
#        adapter modules (ex_7/04) or few-shot methods.
#
#  [SCALING LAWS] This is the practical flipside of slide 5M
#     (scaling laws) — for downstream tasks with small data,
#     pretrained features + small fine-tune data = best ROI.


