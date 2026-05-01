# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# MLFP05 — Exercise 5.2: WGAN-GP (Wasserstein GAN with Gradient Penalty)
# ════════════════════════════════════════════════════════════════════════
#
# WHAT YOU'LL LEARN:
#   - Why vanilla GAN training is unstable (JS divergence saturation)
#   - Wasserstein distance intuition: the "earth mover's distance"
#   - Gradient penalty vs weight clipping — why GP wins
#   - Train a WGAN-GP with critic (not discriminator) architecture
#   - Compare training stability against vanilla GAN
#   - Apply privacy-preserving synthetic medical imaging for a
#     Singapore hospital under PDPA
#
# PREREQUISITES: ex_5/01_vanilla_gan.py (vanilla GAN fundamentals)
# ESTIMATED TIME: ~45 min
# DATASET: MNIST — 60,000 real 28x28 grayscale digits
#
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import asyncio
import copy

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from shared.mlfp05.ex_5 import (
    LATENT_DIM,
    OUTPUT_DIR,
    Generator,
    Discriminator,
    init_environment,
    load_mnist,
    setup_engines,
    close_engines,
    plot_image_grid,
    plot_latent_interpolation,
    plot_training_progression,
    plot_loss_curves,
)


# ════════════════════════════════════════════════════════════════════════
# PHASE 1 — THEORY: Why Vanilla GANs Fail and How Wasserstein Fixes It
# ════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("  PHASE 1 — THEORY: From BCE to Wasserstein Distance")
print("=" * 70)
print(
    """
  THE PROBLEM WITH VANILLA GANS:

  Vanilla GAN uses Jensen-Shannon (JS) divergence to measure how
  different the real and generated distributions are. JS divergence
  has a fatal flaw: when the two distributions don't overlap at all
  (which happens early in training), JS divergence is a CONSTANT.

  A constant means zero gradient. Zero gradient means the generator
  learns nothing. Training stalls or collapses.

  Think of it this way: the detective is SO good that the counterfeiter
  gets no useful feedback — just "everything you make is obviously fake."
  No signal about HOW to improve.

  THE WASSERSTEIN FIX:

  Wasserstein distance (Earth Mover's Distance) measures how much
  "work" it takes to reshape one distribution into another. Imagine
  you have two piles of dirt (real and generated distributions). The
  Wasserstein distance is the minimum total distance you'd need to
  move dirt to transform one pile into the other.

  Key advantage: Wasserstein distance is SMOOTH. Even when distributions
  don't overlap, it still provides meaningful gradients — the generator
  always knows which direction to improve.

  GRADIENT PENALTY (Gulrajani 2017):

  The original WGAN (Arjovsky 2017) enforced the Lipschitz constraint
  by clipping weights to [-c, c]. This caused "capacity underuse":
  most weights cluster at the clip boundaries, wasting the network's
  representational power.

  Gradient penalty is smarter: instead of restricting weights, it
  directly penalises the critic's gradients when they deviate from
  norm 1. This is a soft constraint that preserves the full network
  capacity while enforcing the Lipschitz condition.

  CRITIC VS DISCRIMINATOR:

  In WGAN, the "discriminator" is called a CRITIC because it outputs
  an unbounded real number (the Wasserstein estimate), not a
  probability in [0, 1]. The critic scores images — higher score
  means "more real-looking" — without the sigmoid bottleneck.

  WGAN LOSSES:
    L_critic = E[critic(fake)] - E[critic(real)] + lambda * GP
    L_G      = -E[critic(fake)]
    GP       = E[(||grad critic(interpolated)||_2 - 1)^2]
"""
)

# ════════════════════════════════════════════════════════════════════════
# PHASE 2 — BUILD: Gradient Penalty Implementation
# ════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  PHASE 2 — BUILD: WGAN-GP Architecture + Gradient Penalty")
print("=" * 70)

device = init_environment()
X_real, y_real, real_loader = load_mnist(device)
conn, tracker, exp_name, registry = setup_engines()


def gradient_penalty(
    D: nn.Module, real: torch.Tensor, fake: torch.Tensor
) -> torch.Tensor:
    """Compute gradient penalty (Gulrajani et al. 2017).

    1. Sample alpha ~ U(0, 1) per example
    2. Create interpolated images: x_hat = alpha * real + (1 - alpha) * fake
    3. Compute critic output on interpolated images
    4. Compute gradient of critic w.r.t. interpolated images
    5. Penalise when gradient norm deviates from 1

    The penalty term enforces the 1-Lipschitz constraint on the critic
    without weight clipping, preserving full network capacity.
    """
    batch = real.size(0)
    alpha = torch.rand(batch, 1, 1, 1, device=real.device)
    interp = (alpha * real + (1 - alpha) * fake).requires_grad_(True)
    d_interp = D(interp)
    grad = torch.autograd.grad(
        outputs=d_interp,
        inputs=interp,
        grad_outputs=torch.ones_like(d_interp),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    return ((grad.reshape(batch, -1).norm(2, dim=1) - 1) ** 2).mean()


# Verify gradient penalty computation
_G_test = Generator().to(device)
_D_test = Discriminator().to(device)
_z = torch.randn(4, LATENT_DIM, device=device)
_real_batch = X_real[:4]
_fake_batch = _G_test(_z)
_gp = gradient_penalty(_D_test, _real_batch, _fake_batch)

print(f"\n  Gradient penalty on test batch: {_gp.item():.4f}")
print("  (should be a positive number — penalises gradient norm != 1)")

# ── Checkpoint 1 ─────────────────────────────────────────────────────
assert _gp.item() >= 0, "Gradient penalty must be non-negative"
assert _gp.requires_grad, "GP must be differentiable (part of training graph)"
del _G_test, _D_test, _z, _real_batch, _fake_batch, _gp
print("\n--- Checkpoint 1 passed --- gradient penalty verified\n")


# ════════════════════════════════════════════════════════════════════════
# PHASE 3 — TRAIN: WGAN-GP with Critic Training
# ════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  PHASE 3 — TRAIN: WGAN-GP (5 Critic Steps per Generator Step)")
print("=" * 70)

EPOCHS = 20
LR = 1e-4
N_CRITIC = 5
GP_LAMBDA = 10.0

print(
    f"""
  WGAN-GP training differs from vanilla GAN in three key ways:
  1. Critic trains {N_CRITIC}x more often than generator
     (critic needs to be a good Wasserstein estimator)
  2. No sigmoid — critic outputs unbounded scores
  3. Gradient penalty (lambda={GP_LAMBDA}) replaces weight clipping
"""
)


async def train_wgan_gp(
    epochs: int = EPOCHS,
    lr: float = LR,
    n_critic: int = N_CRITIC,
    lam: float = GP_LAMBDA,
):
    """Train WGAN-GP with gradient penalty, logging to ExperimentTracker."""
    G = Generator().to(device)
    D = Discriminator().to(device)
    opt_g = torch.optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.9))
    opt_d = torch.optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.9))
    g_losses, d_losses = [], []
    epoch_snapshots = {}

    epoch_snapshots[0] = copy.deepcopy(G.state_dict())

    async with tracker.track(experiment=exp_name, run_name="wgan_gp") as run:
        await run.log_params(
            {
                "architecture": "WGAN_GP_MLP",
                "latent_dim": str(LATENT_DIM),
                "lr": str(lr),
                "epochs": str(epochs),
                "batch_size": "128",
                "loss_type": "Wasserstein+GP",
                "n_critic": str(n_critic),
                "gp_lambda": str(lam),
                "optimizer": "Adam(0.5,0.9)",
            }
        )

        for epoch in range(epochs):
            eg, ed = [], []
            for (real_batch,) in real_loader:
                bs = real_batch.size(0)

                # ── Train Critic (n_critic steps per G step) ─────────
                for _ in range(n_critic):
                    z = torch.randn(bs, LATENT_DIM, device=device)
                    fake = G(z).detach()
                    gp = gradient_penalty(D, real_batch, fake)
                    # Wasserstein loss: maximize E[D(real)] - E[D(fake)]
                    # Equivalent to minimize E[D(fake)] - E[D(real)]
                    loss_d = D(fake).mean() - D(real_batch).mean() + lam * gp
                    opt_d.zero_grad()
                    loss_d.backward()
                    opt_d.step()

                # ── Train Generator ──────────────────────────────────
                z = torch.randn(bs, LATENT_DIM, device=device)
                loss_g = -D(G(z)).mean()  # maximize critic score on fakes
                opt_g.zero_grad()
                loss_g.backward()
                opt_g.step()

                eg.append(loss_g.item())
                ed.append(loss_d.item())

            avg_g, avg_d = float(np.mean(eg)), float(np.mean(ed))
            g_losses.append(avg_g)
            d_losses.append(avg_d)
            await run.log_metrics({"g_loss": avg_g, "d_loss": avg_d}, step=epoch + 1)
            print(
                f"  [WGAN-GP] epoch {epoch+1:2d}/{epochs}  "
                f"critic={avg_d:.3f}  G={avg_g:.3f}"
            )

            if (epoch + 1) in {1, 5, 10, 15, 20}:
                epoch_snapshots[epoch + 1] = copy.deepcopy(G.state_dict())

        await run.log_metrics(
            {"final_g_loss": g_losses[-1], "final_d_loss": d_losses[-1]}
        )

    return G, g_losses, d_losses, epoch_snapshots


print("\n  Training WGAN-GP on full MNIST (60K images)...")
G_wgan, wgan_g_losses, wgan_d_losses, wgan_snapshots = asyncio.run(train_wgan_gp())

# ── Checkpoint 2 ─────────────────────────────────────────────────────
assert (
    len(wgan_g_losses) == EPOCHS
), f"Expected {EPOCHS} epochs, got {len(wgan_g_losses)}"
# INTERPRETATION: Unlike vanilla GAN's BCE loss, the WGAN critic loss
# approximates the Wasserstein distance — a MEANINGFUL quality metric.
# Lower critic loss = distributions are closer = better generation.
# The loss should decrease smoothly, unlike vanilla GAN's oscillation.
print("\n--- Checkpoint 2 passed --- WGAN-GP trained\n")


# ══════════════════════════════════════════════════════════════════
# DIAGNOSTIC CHECKPOINT — WGAN-GP (Wasserstein distance = smooth gradient)
# ══════════════════════════════════════════════════════════════════
# The core WGAN-GP selling point is that the critic provides a
# GRADIENT EVERYWHERE — unlike vanilla GAN where the saturating BCE
# kills Generator gradients when D wins. We expect the Blood Test
# to read healthier than 01_vanilla_gan: no vanishing, no mode
# collapse signature in the activations. The Stethoscope should
# show critic loss decreasing smoothly as the Wasserstein distance
# shrinks — NOT the oscillating adversarial dance of vanilla GAN.
from kailash_ml.diagnostics import run_diagnostic_checkpoint
import torch.nn.functional as _F


def _g_loss(m, batch):
    bs = batch[0].size(0)
    z = torch.randn(bs, LATENT_DIM, device=device)
    fake = m(z)
    return _F.mse_loss(fake, batch[0])


print("\n── Diagnostic Report (WGAN-GP Generator) ──")
g_diag, g_findings = run_diagnostic_checkpoint(
    G_wgan,
    real_loader,
    _g_loss,
    title="WGAN-GP — Generator",
    n_batches=6,
    train_losses=wgan_g_losses,
    show=False,
)

# ══════ EXPECTED OUTPUT (synthesized reference — full run produces similar pattern) ══════
# ════════════════════════════════════════════════════════════════
#   DL Diagnostics Report — Prescription Pad (WGAN-GP Generator)
# ════════════════════════════════════════════════════════════════
#   [✓] Gradient flow (HEALTHY): min RMS ~2.8e-04 across G layers.
#       Contrast 01's vanilla G at 5e-6 under D dominance.
#       The Wasserstein objective is unsaturating — gradient
#       survives even when critic perfectly separates real/fake.
#   [✓] Activations   (HEALTHY): tanh output entropy across the
#       batch is high — digits of varied shape, no mode collapse.
#       Visible signature of diverse latent -> diverse output.
#   [✓] Loss trend    (HEALTHY): critic (Wasserstein) loss
#       descends smoothly with slope -4.2e-03/epoch — this is the
#       MEANINGFUL quality metric that vanilla GAN BCE cannot give.
# ════════════════════════════════════════════════════════════════
# Final G loss ~0.18, critic loss ~-2.3 (trend: monotonically toward 0).
#
# STUDENT INTERPRETATION GUIDE — reading the WGAN-GP Prescription Pad:
#
#  [BLOOD TEST — GRADIENT EVERYWHERE] G's gradient RMS holds
#     steady ~1e-4 across training. Compare vanilla GAN (ex_5/01)
#     where the same instrument reads WARNING (RMS < 1e-5) the
#     moment D_loss approached 0. The gradient penalty is the
#     mechanism: it enforces 1-Lipschitz on the critic, which
#     bounds the gradient magnitude but also PREVENTS it from
#     vanishing. Slide 5.5 covers this — the Earth-Mover distance
#     is continuous everywhere, unlike JS divergence.
#     >> Prescription: if G RMS still drops below 1e-5 here,
#        the gradient penalty coefficient (lambda_gp=10) is too
#        low or the critic is under-trained (increase n_critic).
#
#  [X-RAY — NO MODE COLLAPSE] Activation entropy across the batch
#     confirms visual diversity. The gallery (below) should show
#     varied digits, not 50 copies of a "7". WGAN-GP's Lipschitz
#     constraint discourages the critic from sharp rejection of
#     minority modes, so G is not punished for diversity.
#     >> Prescription: persistent mode collapse even with WGAN-GP
#        means the critic architecture is too shallow OR the
#        gradient penalty is mis-implemented (check sample points
#        are interpolated between real and fake, not just on real).
#
#  [STETHOSCOPE — MEANINGFUL QUALITY METRIC] Critic loss ≈ negative
#     Wasserstein distance. It DECREASES as generation quality
#     improves — unlike BCE which oscillates. You can actually
#     monitor this for early stopping, something impossible with
#     vanilla GAN. This is a textbook plot: smooth descent, tight
#     error bars, no mode-collapse cliff.
#     >> Prescription: if critic loss oscillates like vanilla GAN,
#        check that gradient penalty is actually being applied
#        (common bug: lambda_gp=0 silent default).
#
#  FIVE-INSTRUMENT TAKEAWAY: WGAN-GP inverts every red flag of
#  vanilla GAN — gradient healthy, activations diverse, loss
#  meaningful. This is the architectural fix that unlocks
#  training stability. Move to ex_5/03 to quantify quality with
#  FID, because even WGAN-GP's loss doesn't answer "how real?"
# ════════════════════════════════════════════════════════════════════


# ════════════════════════════════════════════════════════════════════════
# PHASE 4 — VISUALISE: Quality Comparison and Stability Analysis
# ════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  PHASE 4 — VISUALISE: WGAN-GP Quality and Stability")
print("=" * 70)

# 4A: Generated image gallery
print("\n  4A: Gallery of 64 WGAN-GP generated digits")
G_wgan.eval()
with torch.no_grad():
    z_gallery = torch.randn(64, LATENT_DIM, device=device)
    gallery_images = G_wgan(z_gallery)

fig_gallery = plot_image_grid(
    gallery_images,
    title="WGAN-GP — Generated Digits (8x8 Grid)",
    save_path=str(OUTPUT_DIR / "ex_5_02_wgan_gallery.png"),
)
plt.show()

# 4B: Training progression
print("\n  4B: Training progression — smoother than vanilla GAN")
G_progression = Generator().to(device)
fig_prog = plot_training_progression(
    G_progression,
    device,
    wgan_snapshots,
    title="WGAN-GP — Training Progression",
    save_path=str(OUTPUT_DIR / "ex_5_02_wgan_progression.png"),
)
plt.show()
G_wgan.load_state_dict(wgan_snapshots[max(wgan_snapshots.keys())])
G_wgan.eval()

# 4C: Critic loss curve — should be smoother than D loss
print("\n  4C: Critic loss dynamics (should decrease smoothly)")
fig_critic = plot_loss_curves(
    wgan_g_losses,
    wgan_d_losses,
    title="WGAN-GP — Training Dynamics",
    g_label="Generator",
    d_label="Critic (Wasserstein estimate)",
    save_path=str(OUTPUT_DIR / "ex_5_02_wgan_losses.png"),
)
plt.show()

# 4D: Latent interpolation
print("\n  4D: Latent space interpolation")
fig_interp = plot_latent_interpolation(
    G_wgan,
    device,
    title="WGAN-GP — Latent Interpolation",
    save_path=str(OUTPUT_DIR / "ex_5_02_wgan_interpolation.png"),
)
plt.show()

# 4E: Side-by-side stability comparison
print("\n  4E: Stability comparison — Vanilla GAN vs WGAN-GP")
fig_compare, axes = plt.subplots(1, 2, figsize=(16, 5))
fig_compare.suptitle(
    "Training Stability: Vanilla GAN vs WGAN-GP",
    fontsize=14,
    fontweight="bold",
)

# Load vanilla GAN losses from previous run if available, otherwise note
# (In practice both files run in sequence; we re-train a small vanilla GAN
# for comparison if the previous data isn't available)
try:
    # Import from sibling module — but we'll just retrain briefly for comparison
    raise ImportError("Using local comparison")
except ImportError:
    # Quick vanilla GAN training for comparison plot
    print("  Training a brief vanilla GAN for comparison...")
    _G_cmp = Generator().to(device)
    _D_cmp = Discriminator().to(device)
    _opt_g = torch.optim.Adam(_G_cmp.parameters(), lr=2e-4, betas=(0.5, 0.999))
    _opt_d = torch.optim.Adam(_D_cmp.parameters(), lr=2e-4, betas=(0.5, 0.999))
    _bce = nn.BCEWithLogitsLoss()
    _cmp_g, _cmp_d = [], []
    for _ep in range(min(EPOCHS, 15)):
        _eg, _ed = [], []
        for (_rb,) in real_loader:
            _bs = _rb.size(0)
            _z = torch.randn(_bs, LATENT_DIM, device=device)
            _fk = _G_cmp(_z).detach()
            _ld = _bce(_D_cmp(_rb), torch.ones(_bs, 1, device=device)) + _bce(
                _D_cmp(_fk), torch.zeros(_bs, 1, device=device)
            )
            _opt_d.zero_grad()
            _ld.backward()
            _opt_d.step()
            _z = torch.randn(_bs, LATENT_DIM, device=device)
            _lg = _bce(_D_cmp(_G_cmp(_z)), torch.ones(_bs, 1, device=device))
            _opt_g.zero_grad()
            _lg.backward()
            _opt_g.step()
            _eg.append(_lg.item())
            _ed.append(_ld.item())
        _cmp_g.append(float(np.mean(_eg)))
        _cmp_d.append(float(np.mean(_ed)))
    del _G_cmp, _D_cmp, _opt_g, _opt_d

# Vanilla GAN D loss (oscillating)
van_epochs = range(1, len(_cmp_d) + 1)
axes[0].plot(van_epochs, _cmp_d, "r-", linewidth=2, alpha=0.8, label="Vanilla D Loss")
axes[0].plot(van_epochs, _cmp_g, "b-", linewidth=2, alpha=0.8, label="Vanilla G Loss")
axes[0].set_xlabel("Epoch", fontsize=12)
axes[0].set_ylabel("Loss (BCE)", fontsize=12)
axes[0].set_title("Vanilla GAN: Oscillating Losses", fontsize=13)
axes[0].legend(fontsize=11)
axes[0].grid(True, alpha=0.3)

# WGAN-GP critic loss (smooth descent)
wgan_epochs = range(1, len(wgan_d_losses) + 1)
axes[1].plot(
    wgan_epochs, wgan_d_losses, "r-", linewidth=2, alpha=0.8, label="Critic Loss"
)
axes[1].plot(wgan_epochs, wgan_g_losses, "b-", linewidth=2, alpha=0.8, label="G Loss")
axes[1].set_xlabel("Epoch", fontsize=12)
axes[1].set_ylabel("Loss (Wasserstein)", fontsize=12)
axes[1].set_title("WGAN-GP: Smooth Convergence", fontsize=13)
axes[1].legend(fontsize=11)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
fig_compare.savefig(
    str(OUTPUT_DIR / "ex_5_02_stability_comparison.png"),
    dpi=150,
    bbox_inches="tight",
)
plt.show()
print("  Stability comparison saved")

# ── Checkpoint 3 ─────────────────────────────────────────────────────
import os

assert os.path.exists(
    str(OUTPUT_DIR / "ex_5_02_wgan_gallery.png")
), "WGAN gallery should exist"
assert os.path.exists(
    str(OUTPUT_DIR / "ex_5_02_stability_comparison.png")
), "Stability comparison should exist"
print("\n--- Checkpoint 3 passed --- WGAN-GP visualisations complete\n")

# Interactive training curves with ModelVisualizer
from kailash_ml import ModelVisualizer

viz = ModelVisualizer()
fig_html = viz.training_history(
    metrics={
        "WGAN-GP G loss": wgan_g_losses,
        "WGAN-GP Critic loss": wgan_d_losses,
    },
    x_label="Epoch",
    y_label="Loss",
)
fig_html.write_html(str(OUTPUT_DIR / "ex_5_02_wgan_training.html"))
print("  Interactive training curves saved to ex_5_02_wgan_training.html")


# ════════════════════════════════════════════════════════════════════════
# PHASE 5 — APPLY: Privacy-Preserving Synthetic Medical Images at NUH
# ════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  PHASE 5 — APPLY: Synthetic Medical Images for NUH")
print("=" * 70)
print(
    """
  BUSINESS SCENARIO:
  You are an AI researcher at the National University Hospital (NUH)
  in Singapore. Your team is developing an automated chest X-ray
  screening model for tuberculosis (TB). Training requires thousands
  of annotated X-ray images, but sharing real patient scans outside
  the hospital is prohibited by PDPA and the Ministry of Health (MOH)
  data governance framework.

  Other hospitals want to train their own TB screening models but
  lack sufficient local data. NUH cannot share real patient images,
  but CAN share a trained GAN that generates synthetic X-ray-like
  images preserving the visual patterns of TB without containing
  any real patient data.

  SOLUTION: Train a WGAN-GP on NUH's X-ray dataset (kept internal).
  Share only the trained generator. Partner hospitals generate synthetic
  training data locally — no real patient data ever leaves NUH.

  WHY WGAN-GP (not vanilla GAN):
  Medical images demand training STABILITY. A mode-collapsed GAN
  that only generates one type of X-ray would train a biased screening
  model. WGAN-GP's smooth Wasserstein gradients prevent collapse,
  ensuring the synthetic dataset covers the full range of TB
  presentations (mild, moderate, severe, bilateral, unilateral).

  BUSINESS IMPACT:
  - 3 partner hospitals gain access to TB screening AI
  - Zero real patient data shared (PDPA + MOH compliant)
  - Screening model sensitivity: 89% (synthetic) vs 92% (real data)
  - Cost savings: S$1.2M/year across 3 hospitals (reduced manual reads)
  - Time to diagnosis: 48 hours → 4 hours for preliminary screen
"""
)

# Simulate the medical imaging scenario with MNIST as proxy
# (real deployment would use chest X-ray datasets like CheXpert)
print("\n  Simulating the NUH medical imaging scenario...")

# Step 1: Generate a large synthetic dataset from the trained WGAN-GP
G_wgan.eval()
n_synthetic = 10000
with torch.no_grad():
    z_medical = torch.randn(n_synthetic, LATENT_DIM, device=device)
    X_medical_synthetic = G_wgan(z_medical)

print(f"  Synthetic 'X-ray' images generated: {n_synthetic}")

# Step 2: Compare quality — real vs WGAN-GP synthetic vs vanilla GAN
# Train a quick vanilla GAN to show the quality difference
print("  Training vanilla GAN for quality comparison...")
_G_van = Generator().to(device)
_D_van = Discriminator().to(device)
_opt_gv = torch.optim.Adam(_G_van.parameters(), lr=2e-4, betas=(0.5, 0.999))
_opt_dv = torch.optim.Adam(_D_van.parameters(), lr=2e-4, betas=(0.5, 0.999))
_bce_van = nn.BCEWithLogitsLoss()
for _ep in range(10):
    for (_rb,) in real_loader:
        _bs = _rb.size(0)
        _z = torch.randn(_bs, LATENT_DIM, device=device)
        _fk = _G_van(_z).detach()
        _ld = _bce_van(_D_van(_rb), torch.ones(_bs, 1, device=device)) + _bce_van(
            _D_van(_fk), torch.zeros(_bs, 1, device=device)
        )
        _opt_dv.zero_grad()
        _ld.backward()
        _opt_dv.step()
        _z = torch.randn(_bs, LATENT_DIM, device=device)
        _lg = _bce_van(_D_van(_G_van(_z)), torch.ones(_bs, 1, device=device))
        _opt_gv.zero_grad()
        _lg.backward()
        _opt_gv.step()

_G_van.eval()
with torch.no_grad():
    X_vanilla_synthetic = _G_van(torch.randn(64, LATENT_DIM, device=device))
del _D_van, _opt_gv, _opt_dv

# Step 3: Visual comparison — Real vs Vanilla GAN vs WGAN-GP
fig_med, axes = plt.subplots(3, 8, figsize=(16, 7))
fig_med.suptitle(
    "Medical Image Quality Comparison\n"
    "Real Patient Scans vs Vanilla GAN vs WGAN-GP Synthetic",
    fontsize=14,
    fontweight="bold",
)

rng = np.random.default_rng(42)
real_sample_idx = rng.choice(len(X_real), 8, replace=False)

for col in range(8):
    # Row 1: Real images
    img = X_real[real_sample_idx[col]].squeeze().cpu().numpy()
    axes[0][col].imshow((img + 1) / 2, cmap="gray", vmin=0, vmax=1)
    axes[0][col].axis("off")
    if col == 0:
        axes[0][col].set_ylabel("Real", fontsize=12, rotation=0, labelpad=50)

    # Row 2: Vanilla GAN
    img = X_vanilla_synthetic[col].squeeze().cpu().numpy()
    axes[1][col].imshow((img + 1) / 2, cmap="gray", vmin=0, vmax=1)
    axes[1][col].axis("off")
    if col == 0:
        axes[1][col].set_ylabel("Vanilla", fontsize=12, rotation=0, labelpad=50)

    # Row 3: WGAN-GP
    img = X_medical_synthetic[col].squeeze().cpu().numpy()
    axes[2][col].imshow((img + 1) / 2, cmap="gray", vmin=0, vmax=1)
    axes[2][col].axis("off")
    if col == 0:
        axes[2][col].set_ylabel("WGAN-GP", fontsize=12, rotation=0, labelpad=50)

plt.tight_layout()
fig_med.savefig(
    str(OUTPUT_DIR / "ex_5_02_medical_comparison.png"),
    dpi=150,
    bbox_inches="tight",
)
plt.show()
print("  Medical image comparison saved")
del _G_van, X_vanilla_synthetic

# Step 4: Diagnostic model comparison — trained on real vs synthetic
print("\n  Training diagnostic models: real data vs synthetic data...")


# Simple classifier to simulate the diagnostic model
class SimpleClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        return self.net(x)


# Model A: trained on real data
model_real = SimpleClassifier().to(device)
opt_real = torch.optim.Adam(model_real.parameters(), lr=1e-3)
X_01 = (X_real + 1.0) / 2.0
for _ in range(3):
    for xb, yb in torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(X_01[:5000], y_real[:5000]),
        batch_size=256,
        shuffle=True,
    ):
        loss = torch.nn.functional.cross_entropy(model_real(xb), yb)
        opt_real.zero_grad()
        loss.backward()
        opt_real.step()

# Model B: trained on synthetic data (use generated images with pseudo-labels)
# In a real scenario, a radiologist labels a small synthetic subset
model_synth = SimpleClassifier().to(device)
opt_synth = torch.optim.Adam(model_synth.parameters(), lr=1e-3)

# Generate pseudo-labels using the real-data model
with torch.no_grad():
    synth_01 = (X_medical_synthetic + 1) / 2
    pseudo_labels = model_real(synth_01).argmax(-1)

for _ in range(3):
    for xb, yb in torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(synth_01, pseudo_labels),
        batch_size=256,
        shuffle=True,
    ):
        loss = torch.nn.functional.cross_entropy(model_synth(xb), yb)
        opt_synth.zero_grad()
        loss.backward()
        opt_synth.step()

# Evaluate both on held-out real test data
X_test_01 = X_01[50000:]
y_test = y_real[50000:]
with torch.no_grad():
    acc_real = (model_real(X_test_01).argmax(-1) == y_test).float().mean().item()
    acc_synth = (model_synth(X_test_01).argmax(-1) == y_test).float().mean().item()

print(f"\n  Diagnostic model accuracy (trained on real data):      {acc_real:.1%}")
print(f"  Diagnostic model accuracy (trained on synthetic data): {acc_synth:.1%}")
print(f"  Performance gap: {abs(acc_real - acc_synth):.1%}")

# Step 5: Stakeholder-ready summary
print("\n  ┌────────────────────────────────────────────────────────────┐")
print("  │  STAKEHOLDER SUMMARY: NUH Synthetic Medical Imaging       │")
print("  ├────────────────────────────────────────────────────────────┤")
print(f"  │  Synthetic images generated:   {n_synthetic:>8,}                  │")
print(f"  │  Real-data model accuracy:     {acc_real:>8.1%}                  │")
print(f"  │  Synthetic-data model accuracy: {acc_synth:>7.1%}                  │")
print(
    f"  │  Performance gap:              {abs(acc_real - acc_synth):>8.1%}                  │"
)
print("  │  Patient data shared:          ZERO                       │")
print("  │  PDPA compliance:              FULL                       │")
print("  │  Partner hospitals enabled:    3                          │")
print("  │  Annual cost savings:          S$1.2M (3 hospitals)       │")
print("  │  Time to preliminary screen:   48h → 4h                   │")
print("  └────────────────────────────────────────────────────────────┘")

# ── Checkpoint 4 ─────────────────────────────────────────────────────
assert os.path.exists(
    str(OUTPUT_DIR / "ex_5_02_medical_comparison.png")
), "Medical comparison should exist"
assert acc_synth > 0.3, "Synthetic-trained model should be non-trivial"
print("\n--- Checkpoint 4 passed --- NUH medical application complete\n")


# ════════════════════════════════════════════════════════════════════════
# Cleanup
# ════════════════════════════════════════════════════════════════════════
asyncio.run(close_engines(conn))


# ════════════════════════════════════════════════════════════════════════
# REFLECTION
# ════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  WHAT YOU'VE MASTERED")
print("=" * 70)
print(
    """
  WGAN-GP THEORY AND PRACTICE:
  [x] Why vanilla GAN fails: JS divergence gives zero gradient when
      distributions don't overlap (early training = no learning signal)
  [x] Wasserstein distance: "earth mover's distance" — smooth gradients
      that always point toward improvement
  [x] Gradient penalty replaces weight clipping (Gulrajani 2017):
      soft constraint preserves full network capacity
  [x] Critic (not discriminator): unbounded score, no sigmoid bottleneck
  [x] 5 critic steps per generator step for accurate Wasserstein estimation

  VISUAL INTUITION:
  [x] WGAN-GP gallery vs vanilla GAN — sharper, more diverse digits
  [x] Critic loss decreases smoothly (unlike vanilla GAN oscillation)
  [x] Side-by-side stability comparison proves WGAN-GP advantage
  [x] Latent interpolation shows continuous learned manifold

  REAL-WORLD APPLICATION:
  [x] Privacy-preserving synthetic medical images for NUH
  [x] PDPA compliance: trained generator shared, real data stays internal
  [x] Diagnostic model trained on synthetic data achieves comparable accuracy
  [x] Business impact: 3 partner hospitals, S$1.2M annual savings

  KEY INSIGHT — WHEN TO USE WGAN-GP:
  Use WGAN-GP when training stability matters (medical, financial,
  safety-critical). Vanilla GAN is simpler but unreliable. WGAN-GP's
  smooth gradients make training predictable and debuggable.

  Next: Exercise 5.3 — GAN Evaluation and Model Registry.
  How do you PROVE synthetic data is good enough for production?
  FID scores, mode coverage analysis, and quality assurance pipelines.
"""
)
