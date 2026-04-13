# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# MLFP05 — Exercise 1.4: Sparse Autoencoder (Sparsity Penalty)
# ════════════════════════════════════════════════════════════════════════
#
# WHAT YOU'LL LEARN:
#   - Build a sparse AE with L1 penalty on hidden activations
#   - Understand WHY sparsity forces specialist neurons
#   - Visualise activation histograms proving sparsity
#   - Apply to semiconductor wafer defect detection at GlobalFoundries
#   - Quantify business impact: defect detection rate + cost savings
#
# PREREQUISITES: 03_denoising_ae.py
# ESTIMATED TIME: ~20 min
#
# TASKS:
#   1. Build Sparse AE with L1 regularisation
#   2. Train and verify sparsity in activation distribution
#   3. Visualise reconstruction + activation histograms
#   4. Apply: wafer defect detection with error heatmaps
#
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from shared.mlfp05.ex_1 import (
    INPUT_DIM,
    EPOCHS,
    OUTPUT_DIR,
    device,
    load_fashion_mnist,
    setup_engines,
    train_variant,
    show_reconstruction,
    show_activation_sparsity,
    register_model,
)


# ════════════════════════════════════════════════════════════════════════
# THEORY — Sparsity Forces Specialist Neurons
# ════════════════════════════════════════════════════════════════════════
# Instead of constraining the bottleneck SIZE, we constrain the
# ACTIVATIONS. An L1 penalty on the hidden layer forces most neurons
# to stay near zero. Only a few "specialist" neurons fire for each input.
#
# Analogy: In a bank's fraud team, you want each analyst to specialise.
# One analyst for "unusual amounts", another for "unusual times",
# another for "unusual merchants". Sparsity forces this specialisation.
# The model cannot spread the signal across all neurons equally.
#
# Biological parallel: In the visual cortex, only ~1% of neurons fire
# for any given stimulus. Sparse representations are more interpretable
# and more energy-efficient — nature discovered this long before ML.


# ════════════════════════════════════════════════════════════════════════
# TASK 1 — Load data and engines
# ════════════════════════════════════════════════════════════════════════

X_flat, X_test_flat, X_img, X_test_img, flat_loader, img_loader = load_fashion_mnist()
conn, tracker, exp_name, registry, has_registry = setup_engines()


# ════════════════════════════════════════════════════════════════════════
# TASK 2 — Build and Train Sparse AE
# ════════════════════════════════════════════════════════════════════════


class SparseAE(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 256):
        super().__init__()
        # TODO: Build encoder — nn.Sequential:
        #       Linear(input_dim, hidden_dim), ReLU,
        #       Linear(hidden_dim, 128), ReLU,
        #       Linear(128, 64)
        self.encoder = ____

        # TODO: Build decoder — mirror:
        #       Linear(64, 128), ReLU, Linear(128, hidden_dim), ReLU,
        #       Linear(hidden_dim, input_dim), Sigmoid
        self.decoder = ____

    def forward(self, x):
        # TODO: Encode then decode. Return (reconstruction, latent_code)
        ____


SPARSITY_WEIGHT = 1e-4


def sparse_ae_loss(model, xb):
    """MSE + L1 sparsity penalty on hidden activations."""
    # TODO: Forward pass to get x_hat and z
    # recon_loss = F.mse_loss(x_hat, xb)
    # sparsity_loss = SPARSITY_WEIGHT * torch.mean(torch.abs(z))
    # Return (recon_loss + sparsity_loss, {"sparsity": sparsity_loss.item()})
    ____


print("\n" + "=" * 70)
print("  Sparse AE — L1 Sparsity Penalty")
print("=" * 70)
print(f"  Sparsity weight: {SPARSITY_WEIGHT}. Most neurons should stay near zero.")

# TODO: Create SparseAE(INPUT_DIM) and train
sparse_model = ____
sparse_losses = ____


# ════════════════════════════════════════════════════════════════════════
# TASK 3 — Visualise Reconstruction + Sparsity
# ════════════════════════════════════════════════════════════════════════

# TODO: show_reconstruction and show_activation_sparsity
____
____

# ── Checkpoint ──────────────────────────────────────────────────────
assert len(sparse_losses) == EPOCHS
assert sparse_losses[-1] < sparse_losses[0]
sparse_model.eval()
with torch.no_grad():
    test_z = sparse_model.encoder(X_test_flat[:1000].to(device))
    pct_sparse = (torch.abs(test_z) < 0.1).float().mean().item()
assert pct_sparse > 0.3, f"Expected >30% activations near zero, got {pct_sparse:.1%}"
print(f"  Sparsity: {pct_sparse:.1%} of activations near zero")
print("\n--- Checkpoint passed --- sparse AE trained\n")

if has_registry:
    register_model(registry, "sparse_ae", sparse_model, sparse_losses[-1])


# ════════════════════════════════════════════════════════════════════════
# APPLY — Semiconductor Wafer Defect Detection (GlobalFoundries)
# ════════════════════════════════════════════════════════════════════════
# BUSINESS SCENARIO: You are an ML engineer at a semiconductor fab in
# Singapore (GlobalFoundries/SSMC). Visual inspection of silicon wafers
# is the quality bottleneck — manual inspection catches 82% of defects
# at 15 seconds per wafer. A missed defect costs S$5,000 in downstream
# rework. Your plant manager asks: "Can we automate inspection?"

print("\n" + "=" * 70)
print("  APPLICATION: Wafer Defect Detection (GlobalFoundries)")
print("=" * 70)

# --- Generate synthetic wafer images ---
IMG_SIZE = 64
N_GOOD = 3000
N_DEFECTIVE = 400
wafer_rng = np.random.default_rng(42)


# TODO: Implement generate_wafer_base(rng_local) -> np.ndarray (64x64)
# Create a circular wafer mask, add die grid pattern inside, add edge ring,
# add small noise. Return clipped [0, 1] image.
def generate_wafer_base(rng_local):
    ____


# TODO: Implement generate_defective_wafer(rng_local) -> (image, mask, defect_type)
# Start from base wafer, add one of: scratch, particle, edge_chip, hotspot
# Return (clipped_image, defect_location_mask, defect_type_string)
def generate_defective_wafer(rng_local):
    ____


# TODO: Generate good_wafers and defective data
good_wafers = ____
defect_data = ____
defective_wafers = np.stack([d[0] for d in defect_data])
defect_masks = np.stack([d[1] for d in defect_data])
defect_types = [d[2] for d in defect_data]

n_train_wafer = int(N_GOOD * 0.8)
train_good_tensor = torch.tensor(good_wafers[:n_train_wafer, None, :, :], device=device)
test_good_tensor = torch.tensor(good_wafers[n_train_wafer:, None, :, :], device=device)
test_defective_tensor = torch.tensor(defective_wafers[:, None, :, :], device=device)
wafer_train_loader = DataLoader(
    TensorDataset(train_good_tensor), batch_size=64, shuffle=True
)

print(f"Good wafers: {N_GOOD}, Defective: {N_DEFECTIVE}")


class SparseConvAE(nn.Module):
    def __init__(self, sparsity_weight: float = 1e-3):
        super().__init__()
        self.sparsity_weight = sparsity_weight
        # TODO: Build encoder — 3 Conv2d layers:
        #       Conv2d(1, 16, 3, stride=2, padding=1), ReLU,
        #       Conv2d(16, 32, 3, stride=2, padding=1), ReLU,
        #       Conv2d(32, 64, 3, stride=2, padding=1), ReLU
        self.encoder = ____

        # TODO: Build decoder — 3 ConvTranspose2d layers:
        #       ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1), ReLU,
        #       ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1), ReLU,
        #       ConvTranspose2d(16, 1, 3, stride=2, padding=1, output_padding=1), Sigmoid
        self.decoder = ____

    def forward(self, x):
        # TODO: Encode then decode. Return (reconstruction, latent)
        ____

    def loss(self, recon, x, z):
        # TODO: MSE + sparsity_weight * mean(abs(z))
        ____


# TODO: Create model, optimizer. Train 50 epochs on good wafers only.
wafer_model = ____
wafer_opt = ____

print("\nTraining sparse autoencoder on good wafers...")
for epoch in range(50):
    wafer_model.train()
    epoch_loss, n_batches = 0.0, 0
    for (batch,) in wafer_train_loader:
        # TODO: Forward, compute loss with wafer_model.loss(), backprop
        ____
    if (epoch + 1) % 10 == 0:
        print(f"  Epoch {epoch+1:3d}/50: loss = {epoch_loss/n_batches:.6f}")

# --- Compute reconstruction errors ---
wafer_model.eval()
with torch.no_grad():
    recon_good, _ = wafer_model(test_good_tensor)
    recon_defective, _ = wafer_model(test_defective_tensor)
    good_errors = (
        ((test_good_tensor - recon_good) ** 2).mean(dim=(1, 2, 3)).cpu().numpy()
    )
    defective_errors = (
        ((test_defective_tensor - recon_defective) ** 2)
        .mean(dim=(1, 2, 3))
        .cpu()
        .numpy()
    )
    pixel_errors_defective = (
        ((test_defective_tensor - recon_defective) ** 2).squeeze(1).cpu().numpy()
    )

print(
    f"\nReconstruction errors: Good={good_errors.mean():.6f}, Defective={defective_errors.mean():.6f}"
)
print(f"Separation: {defective_errors.mean() / good_errors.mean():.1f}x")

# --- Visualisation: Defect localisation ---
# TODO: Create 4x5 grid showing: input, reconstructed, error heatmap, ground truth
# for 5 defective wafers. Save to OUTPUT_DIR / "ex1_manufacturing_defect_localisation.png"
fig, axes = plt.subplots(4, 5, figsize=(15, 12))
____
plt.tight_layout()
plt.savefig(
    OUTPUT_DIR / "ex1_manufacturing_defect_localisation.png",
    dpi=150,
    bbox_inches="tight",
)
plt.show()

# --- Business Impact ---
WAFERS_PER_DAY = 5000
MANUAL_DETECTION_RATE = 0.82
COST_PER_MISSED_DEFECT = 5000
DEFECT_RATE = 0.03
INSPECTOR_ANNUAL_COST = 60_000
INSPECTORS_NEEDED = 4
WORKING_DAYS = 260

# TODO: Compute detection threshold and business savings
# Use the defective error distribution to find operating detection rate
all_wafer_errors = np.concatenate([good_errors, defective_errors])
all_wafer_labels = np.concatenate(
    [np.zeros(len(good_errors)), np.ones(len(defective_errors))]
)
thresholds = np.linspace(
    all_wafer_errors.min(), np.percentile(all_wafer_errors, 99.5), 200
)
detection_rates, false_alarm_rates = [], []
for t in thresholds:
    pred = all_wafer_errors > t
    tp = np.sum(pred & (all_wafer_labels == 1))
    fp = np.sum(pred & (all_wafer_labels == 0))
    fn = np.sum(~pred & (all_wafer_labels == 1))
    tn = np.sum(~pred & (all_wafer_labels == 0))
    detection_rates.append(tp / (tp + fn) if (tp + fn) > 0 else 0.0)
    false_alarm_rates.append(fp / (fp + tn) if (fp + tn) > 0 else 0.0)

detection_rates = np.array(detection_rates)
false_alarm_rates = np.array(false_alarm_rates)
target_dr = 0.94
best_idx = np.argmin(np.abs(detection_rates - target_dr))
operating_dr = detection_rates[best_idx]

# TODO: Calculate annual savings from missed defects + labour
daily_defective = int(WAFERS_PER_DAY * DEFECT_RATE)
manual_missed = daily_defective - int(daily_defective * MANUAL_DETECTION_RATE)
ae_missed = daily_defective - int(daily_defective * operating_dr)
annual_missed_manual = ____
annual_missed_ae = ____
annual_savings_defects = ____
annual_savings_labour = ____
total_annual_savings = ____

print("\n" + "=" * 64)
print("BUSINESS IMPACT SUMMARY — Semiconductor Wafer Inspection")
print("=" * 64)
print(f"\nPlant capacity: {WAFERS_PER_DAY:,} wafers/day, {DEFECT_RATE:.0%} defect rate")
print(f"Manual detection rate:   {MANUAL_DETECTION_RATE:>8.0%}")
print(f"Sparse AE detection:     {operating_dr:>8.0%}")
print(f"\nAnnual financial impact:")
print(f"  Missed defect savings:        {'S$' + f'{annual_savings_defects:,.0f}':>14}")
print(f"  Labour savings (3 FTE):       {'S$' + f'{annual_savings_labour:,.0f}':>14}")
print(f"  Total annual savings:         {'S$' + f'{total_annual_savings:,.0f}':>14}")
print("=" * 64)


# ════════════════════════════════════════════════════════════════════════
# REFLECTION
# ════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  WHAT YOU'VE MASTERED")
print("=" * 70)
print(
    """
  [x] Built a sparse AE with L1 penalty on hidden activations
  [x] Verified sparsity: >30% of activations near zero
  [x] Visualised activation histogram proving specialist neurons
  [x] Applied to semiconductor wafer defect detection
  [x] Generated error heatmaps that localise defects spatially
  [x] Quantified business impact: detection rate + S$ savings

  KEY INSIGHT: Sparsity forces each neuron to specialise. Instead
  of all neurons contributing a little to every reconstruction,
  a few "expert" neurons fire for each specific pattern. This makes
  the representation both interpretable and efficient — exactly
  what you need for defect detection where you want to know
  WHICH feature detected WHICH defect.

  Next: 05_contractive_ae.py smooths the latent space with Jacobian penalty...
"""
)
