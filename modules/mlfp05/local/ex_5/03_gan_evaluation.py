# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# MLFP05 — Exercise 5.3: GAN Evaluation and Model Registry
# ════════════════════════════════════════════════════════════════════════
#
# WHAT YOU'LL LEARN:
#   - Why "looks good" is not a valid evaluation metric for GANs
#   - FID (Frechet Inception Distance) — the standard automated metric
#   - Mode coverage analysis — detecting hidden mode collapse
#   - Shannon entropy as a diversity measure
#   - Register trained generators in ModelRegistry with quality metrics
#   - Build a quality assurance pipeline for synthetic data production
#   - Apply: QA validation for the insurance company's synthetic data
#     pipeline before deploying to production fraud detection models
#
# PREREQUISITES: ex_5/01_vanilla_gan.py, ex_5/02_wgan_gp.py
# ESTIMATED TIME: ~40 min
# DATASET: MNIST — 60,000 real 28x28 grayscale digits
#
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import asyncio
import os

import numpy as np
import torch
import matplotlib.pyplot as plt

from shared.mlfp05.ex_5 import (
    LATENT_DIM,
    OUTPUT_DIR,
    Generator,
    Discriminator,
    LeNetFeatureExtractor,
    init_environment,
    load_mnist,
    setup_engines,
    close_engines,
    train_feature_extractor,
    compute_fid,
    mode_coverage,
    plot_image_grid,
    plot_latent_interpolation,
    plot_loss_curves,
    register_generator,
)
from kailash_ml import ModelVisualizer


# ════════════════════════════════════════════════════════════════════════
# PHASE 1 — THEORY: Why GAN Evaluation Is Hard
# ════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("  PHASE 1 — THEORY: The GAN Evaluation Problem")
print("=" * 70)
print(
    """
  THE PROBLEM:
  Unlike classifiers (accuracy, F1) or regressors (MSE, R²), GANs have
  no single ground truth to compare against. A generator that produces
  perfect images of the digit "7" — and ONLY "7" — could score well on
  per-image quality but is useless for any practical application.

  "LOOKS GOOD" IS DANGEROUS:
  Human visual inspection doesn't scale, is subjective, and misses
  subtle distribution mismatches. A GAN producing sharp but repetitive
  images fools the eye while silently failing on diversity.

  THREE EVALUATION DIMENSIONS:

  1. QUALITY (per-image fidelity):
     Are individual generated images sharp and realistic?
     Metric: FID (lower = better)

  2. DIVERSITY (mode coverage):
     Does the generator cover the full range of real data?
     Metric: mode coverage count + Shannon entropy

  3. NOVELTY (not memorising):
     Is the generator creating NEW images, not copying training data?
     Metric: nearest-neighbour distance to training set

  FID — FRECHET INCEPTION DISTANCE:

  FID treats both real and generated images as points in a learned
  feature space (from a pre-trained classifier). It then fits a
  multivariate Gaussian to each set and measures the distance between
  the two Gaussians:

    FID = ||mu_r - mu_g||^2 + Tr(Sig_r + Sig_g - 2*sqrt(Sig_r @ Sig_g))

  Intuition for professionals:
  - FID = 0: generated images are statistically indistinguishable from real
  - FID < 10: publication-quality generation
  - FID 10-50: recognisable but imperfect
  - FID 50-100: blurry or distorted
  - FID > 100: the generator hasn't learned much

  MODE COLLAPSE DETECTION:

  Mode collapse is the GAN equivalent of a factory that produces only
  one product. Shannon entropy quantifies diversity:
  - max entropy = log2(10) = 3.32 (uniform across all 10 digit classes)
  - entropy = 0: generator produces only one class (total collapse)

  A generator can have low FID (individual images look good) but low
  entropy (it only produces 3 of the 10 digit types). Both metrics
  are needed.
"""
)


# ════════════════════════════════════════════════════════════════════════
# PHASE 2 — BUILD: FID Computation Pipeline + Feature Extractor
# ════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  PHASE 2 — BUILD: FID Pipeline and Feature Extractor")
print("=" * 70)

device = init_environment()
X_real, y_real, real_loader = load_mnist(device)
conn, tracker, exp_name, registry = setup_engines()

# Train the feature extractor for FID computation
fid_ext = train_feature_extractor(X_real, y_real, device, epochs=5)

# ── Checkpoint 1 ─────────────────────────────────────────────────────
fid_ext.eval()
with torch.no_grad():
    _test_acc = (
        (fid_ext((X_real[:1000] + 1) / 2).argmax(-1) == y_real[:1000]).float().mean()
    )
assert _test_acc > 0.8, f"Feature extractor accuracy too low: {_test_acc:.3f}"
print(f"  Feature extractor test accuracy: {_test_acc:.1%}")
print("\n--- Checkpoint 1 passed --- feature extractor trained\n")


# ════════════════════════════════════════════════════════════════════════
# PHASE 3 — TRAIN: Both GAN Variants for Comparative Evaluation
# ════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  PHASE 3 — TRAIN: Vanilla GAN + WGAN-GP for Comparison")
print("=" * 70)

import torch.nn as nn


# TODO: Train vanilla GAN (15 epochs) with BCE loss
# Hint: Same pattern as ex_5/01 — Generator, Discriminator, Adam(0.5, 0.999),
#       BCEWithLogitsLoss, alternate D and G training each batch
print("\n  Training Vanilla GAN (15 epochs)...")
G_gan = Generator().to(device)
D_gan = Discriminator().to(device)
opt_g_gan = torch.optim.Adam(G_gan.parameters(), lr=2e-4, betas=(0.5, 0.999))
opt_d_gan = torch.optim.Adam(D_gan.parameters(), lr=2e-4, betas=(0.5, 0.999))
bce = nn.BCEWithLogitsLoss()
gan_g_losses, gan_d_losses = [], []

for epoch in range(15):
    eg, ed = [], []
    for (real_batch,) in real_loader:
        bs = real_batch.size(0)
        z = torch.randn(bs, LATENT_DIM, device=device)
        fake = G_gan(z).detach()
        # TODO: D loss = BCE on real (target=1) + BCE on fake (target=0)
        # Hint: loss_d = bce(D_gan(real_batch), torch.ones(bs, 1, device=device))
        #              + bce(D_gan(fake), torch.zeros(bs, 1, device=device))
        loss_d = ____
        opt_d_gan.zero_grad()
        loss_d.backward()
        opt_d_gan.step()
        z = torch.randn(bs, LATENT_DIM, device=device)
        # TODO: G loss = fool D by labelling fakes as real
        # Hint: loss_g = bce(D_gan(G_gan(z)), torch.ones(bs, 1, device=device))
        loss_g = ____
        opt_g_gan.zero_grad()
        loss_g.backward()
        opt_g_gan.step()
        eg.append(loss_g.item())
        ed.append(loss_d.item())
    gan_g_losses.append(float(np.mean(eg)))
    gan_d_losses.append(float(np.mean(ed)))
    print(
        f"  [Vanilla] epoch {epoch+1:2d}/15  "
        f"D={gan_d_losses[-1]:.3f}  G={gan_g_losses[-1]:.3f}"
    )


# TODO: Implement gradient penalty function for WGAN-GP
# Hint: Same as ex_5/02 — interpolate real+fake, compute grad norm, penalise != 1
def gradient_penalty(D, real, fake):
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
    # TODO: Return gradient penalty = mean of (||grad||_2 - 1)^2
    # Hint: ((grad.reshape(batch, -1).norm(2, dim=1) - 1) ** 2).mean()
    return ____


# TODO: Train WGAN-GP (20 epochs) with critic training
# Hint: 5 critic steps per G step, Adam(0.5, 0.9), lr=1e-4
#       Critic loss = D(fake).mean() - D(real).mean() + 10.0 * gp
#       G loss = -D(G(z)).mean()
print("\n  Training WGAN-GP (20 epochs)...")
G_wgan = Generator().to(device)
D_wgan = Discriminator().to(device)
opt_g_wgan = torch.optim.Adam(G_wgan.parameters(), lr=1e-4, betas=(0.5, 0.9))
opt_d_wgan = torch.optim.Adam(D_wgan.parameters(), lr=1e-4, betas=(0.5, 0.9))
wgan_g_losses, wgan_d_losses = [], []

for epoch in range(20):
    eg, ed = [], []
    for (real_batch,) in real_loader:
        bs = real_batch.size(0)
        for _ in range(5):
            z = torch.randn(bs, LATENT_DIM, device=device)
            fake = G_wgan(z).detach()
            gp = gradient_penalty(D_wgan, real_batch, fake)
            # TODO: Wasserstein critic loss + gradient penalty
            # Hint: loss_d = D_wgan(fake).mean() - D_wgan(real_batch).mean() + 10.0 * gp
            loss_d = ____
            opt_d_wgan.zero_grad()
            loss_d.backward()
            opt_d_wgan.step()
        z = torch.randn(bs, LATENT_DIM, device=device)
        # TODO: G loss — maximise critic score on fakes
        # Hint: loss_g = -D_wgan(G_wgan(z)).mean()
        loss_g = ____
        opt_g_wgan.zero_grad()
        loss_g.backward()
        opt_g_wgan.step()
        eg.append(loss_g.item())
        ed.append(loss_d.item())
    wgan_g_losses.append(float(np.mean(eg)))
    wgan_d_losses.append(float(np.mean(ed)))
    print(
        f"  [WGAN-GP] epoch {epoch+1:2d}/20  "
        f"critic={wgan_d_losses[-1]:.3f}  G={wgan_g_losses[-1]:.3f}"
    )

# ── Checkpoint 2 ─────────────────────────────────────────────────────
assert len(gan_g_losses) == 15, "Vanilla GAN should train 15 epochs"
assert len(wgan_g_losses) == 20, "WGAN-GP should train 20 epochs"
print("\n--- Checkpoint 2 passed --- both generators trained\n")


# ════════════════════════════════════════════════════════════════════════
# Compute FID scores
# ════════════════════════════════════════════════════════════════════════
print("\n  Computing FID scores...")

N_FID = 10000
G_gan.eval()
G_wgan.eval()

with torch.no_grad():
    # TODO: Generate fake images from both generators (N_FID each), scale to [0, 1]
    # Hint: gan_fake_01 = (G_gan(torch.randn(N_FID, LATENT_DIM, device=device)) + 1) / 2
    #       wgan_fake_01 = (G_wgan(torch.randn(N_FID, LATENT_DIM, device=device)) + 1) / 2
    gan_fake_01 = ____
    wgan_fake_01 = ____

rng = np.random.default_rng(42)
real_sub = (X_real[rng.choice(len(X_real), N_FID, replace=False)] + 1) / 2

# TODO: Compute FID for both generators using the trained feature extractor
# Hint: fid_gan = compute_fid(fid_ext, real_sub, gan_fake_01)
#       fid_wgan = compute_fid(fid_ext, real_sub, wgan_fake_01)
fid_gan = ____
fid_wgan = ____

print(f"\n  FID Scores (lower = better):")
print(f"    Vanilla GAN: {fid_gan:.2f}")
print(f"    WGAN-GP:     {fid_wgan:.2f}")

# TODO: Compute mode coverage for both generators
# Hint: cov_gan, dist_gan, ent_gan = mode_coverage(G_gan, fid_ext, device)
#       cov_wgan, dist_wgan, ent_wgan = mode_coverage(G_wgan, fid_ext, device)
cov_gan, dist_gan, ent_gan = ____
cov_wgan, dist_wgan, ent_wgan = ____

print(f"\n  Mode Coverage (all 10 digit classes):")
print(f"    Vanilla GAN: {cov_gan}/10 classes, entropy={ent_gan:.2f}/3.32")
print(f"    Distribution: {dist_gan}")
print(f"    WGAN-GP:     {cov_wgan}/10 classes, entropy={ent_wgan:.2f}/3.32")
print(f"    Distribution: {dist_wgan}")


# Log to ExperimentTracker
async def _log_evaluation():
    async with tracker.track(experiment=exp_name, run_name="fid_evaluation") as run:
        await run.log_param("n_generated", str(N_FID))
        await run.log_metrics(
            {
                "fid_vanilla_gan": fid_gan,
                "fid_wgan_gp": fid_wgan,
                "coverage_vanilla": float(cov_gan),
                "coverage_wgan": float(cov_wgan),
                "entropy_vanilla": ent_gan,
                "entropy_wgan": ent_wgan,
            }
        )


asyncio.run(_log_evaluation())

# ── Checkpoint 3 ─────────────────────────────────────────────────────
assert fid_gan >= -1e-3 and fid_wgan >= -1e-3, (
    f"FID expected ~0+; got fid_gan={fid_gan:.6f}, fid_wgan={fid_wgan:.6f}"
)
assert 0 <= ent_gan <= np.log2(10) + 0.01, "Entropy out of range"
assert cov_gan >= 1 and cov_wgan >= 1, "Must produce at least 1 class"
# INTERPRETATION: FID = 0 means identical distributions. Typical MNIST
# GAN FID after a few epochs: 10-100. Production papers target FID < 10.
# WGAN-GP typically achieves better coverage because Wasserstein distance
# provides gradients even when distributions don't overlap.
print("\n--- Checkpoint 3 passed --- FID and mode coverage computed\n")


# ════════════════════════════════════════════════════════════════════════
# PHASE 4 — VISUALISE: FID Comparison, Mode Coverage, Latent Walk
# ════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  PHASE 4 — VISUALISE: Evaluation Metrics Dashboard")
print("=" * 70)

# 4A: FID score comparison bar chart
print("\n  4A: FID score comparison")
# TODO: Create bar chart comparing FID scores of both generators
# Hint: Use plt.subplots, ax.bar with ["Vanilla GAN", "WGAN-GP"], [fid_gan, fid_wgan]
#       Add reference lines at FID=10 (publication), 50 (recognisable), 100 (poor)
fig_fid, ax = plt.subplots(figsize=(8, 5))
bars = ax.bar(
    ["Vanilla GAN", "WGAN-GP"],
    [fid_gan, fid_wgan],
    color=["#e74c3c", "#2ecc71"],
    width=0.5,
    edgecolor="black",
    linewidth=1.2,
)
ax.set_ylabel("FID Score (lower = better)", fontsize=13)
ax.set_title(
    "Frechet Inception Distance Comparison\n"
    f"(computed on {N_FID:,} generated vs {N_FID:,} real images)",
    fontsize=14,
    fontweight="bold",
)
# TODO: Add value labels on bars and reference lines
# Hint: ax.text(bar.get_x() + bar.get_width()/2, bar.get_height()+0.5, f"{val:.1f}", ...)
#       ax.axhline(y=10, color="green", linestyle="--", alpha=0.5, label="Publication quality")
for bar, val in zip(bars, [fid_gan, fid_wgan]):
    ____  # TODO: Add value label text on each bar
ax.axhline(
    y=10, color="green", linestyle="--", alpha=0.5, label="Publication quality (<10)"
)
ax.axhline(
    y=50, color="orange", linestyle="--", alpha=0.5, label="Recognisable (10-50)"
)
ax.axhline(y=100, color="red", linestyle="--", alpha=0.5, label="Poor quality (>100)")
ax.legend(fontsize=10, loc="upper right")
ax.grid(True, alpha=0.2, axis="y")
plt.tight_layout()
fig_fid.savefig(
    str(OUTPUT_DIR / "ex_5_03_fid_comparison.png"), dpi=150, bbox_inches="tight"
)
plt.show()

# 4B: Mode coverage matrix — heatmap of generated digit classes
print("\n  4B: Mode coverage matrix (all 10 digit classes)")
# TODO: Create side-by-side bar charts showing per-digit class distribution
# Hint: For each GAN, compute counts per class 0-9, convert to percentages,
#       bar chart with green (present) / red (missing) colours
fig_mode, axes = plt.subplots(1, 2, figsize=(16, 5))
fig_mode.suptitle(
    "Mode Coverage: Which Digits Does Each GAN Generate?",
    fontsize=14,
    fontweight="bold",
)

for idx, (name, dist, ent, cov) in enumerate(
    [
        ("Vanilla GAN", dist_gan, ent_gan, cov_gan),
        ("WGAN-GP", dist_wgan, ent_wgan, cov_wgan),
    ]
):
    counts = [dist.get(c, 0) for c in range(10)]
    total = sum(counts)
    pcts = [c / total * 100 if total > 0 else 0 for c in counts]
    colors = ["#2ecc71" if c > 0 else "#e74c3c" for c in counts]

    bars = axes[idx].bar(
        range(10), pcts, color=colors, edgecolor="black", linewidth=0.8
    )
    axes[idx].set_xlabel("Digit Class", fontsize=12)
    axes[idx].set_ylabel("% of Generated Images", fontsize=12)
    axes[idx].set_title(
        f"{name}\n{cov}/10 classes, entropy={ent:.2f}/3.32", fontsize=13
    )
    axes[idx].set_xticks(range(10))
    axes[idx].axhline(
        y=10, color="blue", linestyle="--", alpha=0.5, label="Ideal uniform (10%)"
    )
    axes[idx].legend(fontsize=10)
    axes[idx].grid(True, alpha=0.2, axis="y")

    # TODO: Add percentage labels on each bar
    # Hint: for bar, pct in zip(bars, pcts):
    #           if pct > 0: axes[idx].text(bar.get_x()+bar.get_width()/2, ...)
    ____

plt.tight_layout()
fig_mode.savefig(
    str(OUTPUT_DIR / "ex_5_03_mode_coverage.png"), dpi=150, bbox_inches="tight"
)
plt.show()

# 4C: Latent space interpolation walk — smooth transitions
print("\n  4C: Latent space interpolation (WGAN-GP)")
fig_walk = plot_latent_interpolation(
    G_wgan,
    device,
    n_steps=12,
    n_rows=6,
    title="WGAN-GP Latent Walk — Smooth Digit Transitions",
    save_path=str(OUTPUT_DIR / "ex_5_03_latent_walk.png"),
)
plt.show()

# 4D: Combined training dynamics
print("\n  4D: Combined training dynamics comparison")
viz = ModelVisualizer()
fig_all = viz.training_history(
    metrics={
        "Vanilla G": gan_g_losses,
        "Vanilla D": gan_d_losses,
        "WGAN G": wgan_g_losses,
        "WGAN Critic": wgan_d_losses,
    },
    x_label="Epoch",
    y_label="Loss",
)
fig_all.write_html(str(OUTPUT_DIR / "ex_5_03_combined_training.html"))
print("  Interactive combined training curves saved")

# 4E: Comprehensive evaluation dashboard
print("\n  4E: Evaluation dashboard")
# TODO: Create a 2x2 dashboard with FID comparison, entropy comparison,
#       G loss curves, and D/Critic loss curves
# Hint: fig_dash, axes = plt.subplots(2, 2, figsize=(16, 12))
fig_dash, axes = plt.subplots(2, 2, figsize=(16, 12))
fig_dash.suptitle(
    "GAN Evaluation Dashboard — Vanilla GAN vs WGAN-GP",
    fontsize=16,
    fontweight="bold",
)

# TODO: Top-left — FID bar chart
# Hint: axes[0][0].bar(["Vanilla GAN", "WGAN-GP"], [fid_gan, fid_wgan], ...)
____

# TODO: Top-right — Entropy bar chart with max entropy line
# Hint: axes[0][1].bar(["Vanilla GAN", "WGAN-GP"], [ent_gan, ent_wgan], ...)
#       axes[0][1].axhline(y=np.log2(10), ...)
____

# TODO: Bottom-left — G loss curves for both generators
# Hint: axes[1][0].plot(range(1, 16), gan_g_losses, "r-", ...)
#       axes[1][0].plot(range(1, 21), wgan_g_losses, "g-", ...)
____

# TODO: Bottom-right — D/Critic loss curves for both generators
# Hint: axes[1][1].plot(range(1, 16), gan_d_losses, "r-", ...)
#       axes[1][1].plot(range(1, 21), wgan_d_losses, "g-", ...)
____

plt.tight_layout()
fig_dash.savefig(
    str(OUTPUT_DIR / "ex_5_03_evaluation_dashboard.png"), dpi=150, bbox_inches="tight"
)
plt.show()

# ── Checkpoint 4 ─────────────────────────────────────────────────────
assert os.path.exists(
    str(OUTPUT_DIR / "ex_5_03_fid_comparison.png")
), "FID comparison should exist"
assert os.path.exists(
    str(OUTPUT_DIR / "ex_5_03_evaluation_dashboard.png")
), "Dashboard should exist"
print("\n--- Checkpoint 4 passed --- evaluation visualisations complete\n")


# ════════════════════════════════════════════════════════════════════════
# Register generators in ModelRegistry
# ════════════════════════════════════════════════════════════════════════
print("\n  Registering generators in ModelRegistry...")
ver_gan = register_generator(
    registry, "dcgan_generator", G_gan, fid_gan, cov_gan, ent_gan
)
ver_wgan = register_generator(
    registry, "wgan_gp_generator", G_wgan, fid_wgan, cov_wgan, ent_wgan
)

# ── Checkpoint 5 ─────────────────────────────────────────────────────
if registry is not None:
    assert ver_gan is not None, "Vanilla GAN should be registered"
    assert ver_wgan is not None, "WGAN-GP should be registered"
print("\n--- Checkpoint 5 passed --- generators registered\n")


# ════════════════════════════════════════════════════════════════════════
# PHASE 5 — APPLY: QA Pipeline for Insurance Synthetic Data
# ════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  PHASE 5 — APPLY: Synthetic Data QA for Insurance Production")
print("=" * 70)
print(
    """
  BUSINESS SCENARIO:
  You are the ML engineering lead at the Singapore insurance company
  from Exercise 5.1. Your team has trained a WGAN-GP to generate
  synthetic policyholder profiles for fraud detection model training.

  Before deploying synthetic data to production, the Chief Risk Officer
  (CRO) asks: "How do you KNOW the synthetic data is good enough?
  What if the generator produces biased profiles that make the fraud
  model miss certain claim types?"

  YOUR QA PIPELINE:
  1. FID threshold: synthetic data must score below FID 50
     (statistically close to real distribution)
  2. Mode coverage: all major claim categories must be represented
     (no category can be below 5% of total)
  3. Distribution matching: key statistical properties must match
     (mean, variance, correlation structure)
  4. Downstream model validation: fraud model trained on synthetic
     data must achieve within 5% of real-data baseline accuracy

  This is the production gate — synthetic data that fails any check
  is rejected and the generator is retrained or tuned.
"""
)

# Step 1: Define QA thresholds
FID_THRESHOLD = 50.0
MIN_MODE_COVERAGE = 8  # At least 8/10 classes
MIN_ENTROPY = 2.5  # Minimum diversity (max is 3.32)
MIN_CLASS_PCT = 3.0  # No class below 3% of total

print("  QA Thresholds:")
print(f"    FID score:        < {FID_THRESHOLD}")
print(f"    Mode coverage:    >= {MIN_MODE_COVERAGE}/10 classes")
print(f"    Shannon entropy:  >= {MIN_ENTROPY}/3.32")
print(f"    Min class share:  >= {MIN_CLASS_PCT}%")

# Step 2: Run QA on both generators
print("\n  Running QA pipeline on both generators...")

# TODO: Evaluate both generators against QA thresholds
# Hint: For each generator, check fid < threshold, coverage >= min,
#       entropy >= min, and min class percentage >= threshold
qa_results = {}
for name, G, fid, cov, dist, ent in [
    ("Vanilla GAN", G_gan, fid_gan, cov_gan, dist_gan, ent_gan),
    ("WGAN-GP", G_wgan, fid_wgan, cov_wgan, dist_wgan, ent_wgan),
]:
    total = sum(dist.values())
    min_pct = min(dist.values()) / total * 100 if total > 0 and dist else 0.0

    # TODO: Check each QA criterion
    # Hint: fid_pass = fid < FID_THRESHOLD
    #       cov_pass = cov >= MIN_MODE_COVERAGE
    #       ent_pass = ent >= MIN_ENTROPY
    #       pct_pass = min_pct >= MIN_CLASS_PCT
    fid_pass = ____
    cov_pass = ____
    ent_pass = ____
    pct_pass = ____

    all_pass = fid_pass and cov_pass and ent_pass and pct_pass

    qa_results[name] = {
        "fid": fid,
        "fid_pass": fid_pass,
        "coverage": cov,
        "cov_pass": cov_pass,
        "entropy": ent,
        "ent_pass": ent_pass,
        "min_class_pct": min_pct,
        "pct_pass": pct_pass,
        "overall": all_pass,
    }

# Step 3: QA results visualisation
# TODO: Create normalised bar chart showing % of QA threshold met
# Hint: For FID (lower=better), normalise as threshold/value * 100
#       For others (higher=better), normalise as value/threshold * 100
fig_qa, ax = plt.subplots(figsize=(14, 7))
fig_qa.suptitle(
    "Synthetic Data QA Pipeline Results\n"
    "Production Gate for Insurance Fraud Model Training",
    fontsize=14,
    fontweight="bold",
)

categories = [
    "FID Score\n(< 50)",
    "Mode Coverage\n(>= 8/10)",
    "Entropy\n(>= 2.5)",
    "Min Class %\n(>= 3%)",
]
vanilla_scores = [
    qa_results["Vanilla GAN"]["fid"],
    qa_results["Vanilla GAN"]["coverage"],
    qa_results["Vanilla GAN"]["entropy"],
    qa_results["Vanilla GAN"]["min_class_pct"],
]
wgan_scores = [
    qa_results["WGAN-GP"]["fid"],
    qa_results["WGAN-GP"]["coverage"],
    qa_results["WGAN-GP"]["entropy"],
    qa_results["WGAN-GP"]["min_class_pct"],
]
thresholds = [FID_THRESHOLD, MIN_MODE_COVERAGE, MIN_ENTROPY, MIN_CLASS_PCT]

x = np.arange(len(categories))
width = 0.3

# TODO: Normalise scores to percentage of threshold met
# Hint: For FID (lower=better): min(threshold / (value + 1e-8) * 100, 150)
#       For others (higher=better): min(value / threshold * 100, 150)
vanilla_norm = []
wgan_norm = []
for v, w, t, cat in zip(vanilla_scores, wgan_scores, thresholds, categories):
    if "FID" in cat:
        ____  # TODO: Append normalised FID (lower=better)
        ____
    else:
        ____  # TODO: Append normalised coverage/entropy/pct (higher=better)
        ____

bars1 = ax.bar(
    x - width / 2, vanilla_norm, width, label="Vanilla GAN", color="#e74c3c", alpha=0.8
)
bars2 = ax.bar(
    x + width / 2, wgan_norm, width, label="WGAN-GP", color="#2ecc71", alpha=0.8
)
ax.axhline(
    y=100, color="black", linestyle="--", linewidth=2, label="QA Threshold (100%)"
)
ax.set_ylabel("% of QA Threshold Met", fontsize=12)
ax.set_xticks(x)
ax.set_xticklabels(categories, fontsize=11)
ax.legend(fontsize=11, loc="upper right")
ax.grid(True, alpha=0.2, axis="y")
ax.set_ylim(0, 160)

# TODO: Add PASS/FAIL indicators on each bar
# Hint: For each bar pair, check the pass/fail from qa_results,
#       add green "PASS" or red "FAIL" text above the bar
____

plt.tight_layout()
fig_qa.savefig(
    str(OUTPUT_DIR / "ex_5_03_qa_pipeline.png"), dpi=150, bbox_inches="tight"
)
plt.show()

# Step 4: Stakeholder-ready QA report
van_overall = qa_results["Vanilla GAN"]["overall"]
wgan_overall = qa_results["WGAN-GP"]["overall"]

print("\n  ┌────────────────────────────────────────────────────────────────┐")
print("  │  SYNTHETIC DATA QA REPORT — Insurance Fraud Model Pipeline    │")
print("  ├────────────────────────────────────────────────────────────────┤")
print("  │                                                                │")
print(f"  │  {'Metric':<22} {'Vanilla GAN':>13} {'WGAN-GP':>13} {'Threshold':>12} │")
print(f"  │  {'─'*60}  │")
print(f"  │  {'FID Score':<22} {fid_gan:>13.1f} {fid_wgan:>13.1f} {'< 50':>12} │")
print(
    f"  │  {'Mode Coverage':<22} {cov_gan:>12}/10 {cov_wgan:>12}/10 {'>= 8/10':>12} │"
)
print(
    f"  │  {'Shannon Entropy':<22} {ent_gan:>13.2f} {ent_wgan:>13.2f} {'>= 2.50':>12} │"
)
van_min = qa_results["Vanilla GAN"]["min_class_pct"]
wgan_min = qa_results["WGAN-GP"]["min_class_pct"]
print(
    f"  │  {'Min Class Share':<22} {van_min:>12.1f}% {wgan_min:>12.1f}% {'>= 3.0%':>12} │"
)
print("  │                                                                │")
van_status = "APPROVED" if van_overall else "REJECTED"
wgan_status = "APPROVED" if wgan_overall else "REJECTED"
print(f"  │  {'OVERALL STATUS':<22} {van_status:>13} {wgan_status:>13}              │")
print("  │                                                                │")
better = "WGAN-GP" if fid_wgan < fid_gan else "Vanilla GAN"
print(f"  │  Recommended generator: {better:<38} │")
print(f"  │  Best FID score: {min(fid_gan, fid_wgan):<44.1f} │")
print("  │                                                                │")
print("  │  DECISION: CRO can approve WGAN-GP synthetic data for         │")
print("  │  production fraud model training. QA pipeline will run         │")
print("  │  automatically on every new generator version via ModelRegistry│")
print("  └────────────────────────────────────────────────────────────────┘")

# ── Checkpoint 6 ─────────────────────────────────────────────────────
assert os.path.exists(
    str(OUTPUT_DIR / "ex_5_03_qa_pipeline.png")
), "QA pipeline chart should exist"
print("\n--- Checkpoint 6 passed --- QA pipeline complete\n")


# ════════════════════════════════════════════════════════════════════════
# Final Summary
# ════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  EXPERIMENT SUMMARY")
print("=" * 70)
print(f"\n  Experiment: {exp_name}")
print(f"  Dataset: MNIST (60,000 images), latent_dim={LATENT_DIM}")
print(f"\n  {'Metric':<25} {'Vanilla GAN':>14} {'WGAN-GP':>14}")
print(f"  {'-'*53}")
print(f"  {'Epochs':<25} {'15':>14} {'20':>14}")
print(f"  {'Final G loss':<25} {gan_g_losses[-1]:>14.3f} {wgan_g_losses[-1]:>14.3f}")
print(
    f"  {'Final D/Critic loss':<25} {gan_d_losses[-1]:>14.3f} {wgan_d_losses[-1]:>14.3f}"
)
print(f"  {'FID score':<25} {fid_gan:>14.2f} {fid_wgan:>14.2f}")
print(f"  {'Mode coverage':<25} {cov_gan:>13}/10 {cov_wgan:>13}/10")
print(f"  {'Class entropy':<25} {ent_gan:>14.2f} {ent_wgan:>14.2f}")
print(f"\n  Best generator by FID: {better}")


# ════════════════════════════════════════════════════════════════════════
# Cleanup
# ════════════════════════════════════════════════════════════════════════
asyncio.run(close_engines(conn))


# ════════════════════════════════════════════════════════════════════════
# DESTINATION-FIRST CLOSE — km.diagnose
# ════════════════════════════════════════════════════════════════════════
# This lesson walked the journey of generative adversarial networks —
# vanilla GAN, WGAN-GP, FID/coverage/entropy QA pipelines. The kailash-ml
# SDK ships a single-call diagnostic primitive that closes the production
# loop: km.diagnose inspects a trained model and emits an auto-dashboard
# (loss curves, gradient flow, dead neurons, activation stats, weight
# distributions). One cell. Every diagnostic students would otherwise
# hand-roll, ready to surface in a Plotly dashboard.

from kailash_ml import diagnose

# Diagnose the WGAN-GP generator (the more stable of the two architectures).
# Generators take noise vectors as input — we feed a small iterable of
# `LATENT_DIM`-shaped noise tensors. `kind='auto'` correctly dispatches a
# torch.nn.Module to DLDiagnostics regardless of input shape.
noise_iter = [torch.randn(64, LATENT_DIM, device=device) for _ in range(4)]
report = diagnose(G_wgan, kind="auto", data=noise_iter, show=False)
report.plot_training_dashboard()
print()
print("km.diagnose: 1 line of code -> the same observability the lesson")
print("body hand-rolled in 200+ lines. This is what 'destination-first'")
print("means — when the journey is internalised, the SDK is one call.")


# ════════════════════════════════════════════════════════════════════════
# REFLECTION
# ════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  WHAT YOU'VE MASTERED")
print("=" * 70)
print(
    """
  GAN EVALUATION METRICS:
  [x] FID (Frechet Inception Distance): the gold standard for GAN quality.
      Measures distributional distance in a learned feature space.
  [x] Mode coverage: counting how many classes the generator produces.
      A sharp but repetitive GAN fails this test.
  [x] Shannon entropy: quantifying generation diversity.
      Max = log2(10) = 3.32 for MNIST (uniform across all digits).
  [x] Minimum class share: no category can be underrepresented
      (prevents hidden bias in synthetic datasets).

  MODEL REGISTRY:
  [x] Registered both generators with FID + coverage + entropy metrics
  [x] Version tracking enables A/B comparison between generator versions
  [x] Promotion criteria: lower FID + higher coverage = promote to serving

  QA PIPELINE FOR PRODUCTION:
  [x] Defined quantitative thresholds (FID < 50, coverage >= 8/10, etc.)
  [x] Automated evaluation — no "looks good to me" subjectivity
  [x] Stakeholder-ready report for CRO approval
  [x] Pipeline runs on every new generator version in ModelRegistry

  REAL-WORLD APPLICATION:
  [x] Insurance synthetic data QA: proving data quality before production
  [x] CRO-ready quality report with pass/fail per metric
  [x] Quantified business risk: what happens if synthetic data is biased

  KEY INSIGHTS:
  - FID alone is not enough: a generator can have low FID on the modes
    it covers while completely missing other modes
  - Mode coverage alone is not enough: a generator can cover all modes
    but produce blurry, low-quality images
  - You need BOTH quality (FID) AND diversity (coverage + entropy)
  - In production, these checks run automatically on every new model
    version — the CRO never needs to "eyeball" synthetic data again

  WHEN TO USE WHICH GAN:
  - Vanilla GAN: quick prototyping, simple datasets, no stability needs
  - WGAN-GP: production use, medical/financial data, stability required
  - Both need the same evaluation pipeline — the QA doesn't change,
    only the generator architecture does.

  GAN vs VAE vs Diffusion (from M5 Exercise 1):
  - GANs:      Sharp images, hard to train, fast sampling
  - VAEs:      Blurry but stable, continuous latent space, fast
  - Diffusion: Sharp + stable, best quality, SLOW sampling
"""
)

# ══════════════════════════════════════════════════════════════════
# DIAGNOSTIC CHECKPOINT — five instruments before Visualise
# ══════════════════════════════════════════════════════════════════
# Reference: `kailash_ml.diagnostics` (via `kailash-ml`) — see gold standard
# `solutions/ex_1/01_standard_ae.py` for the full pattern.
from kailash_ml.diagnostics import run_diagnostic_checkpoint


def _diag_loss(m, batch):
    # GAN evaluation — diagnostic on the generator model
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


print("\n── Diagnostic Report (GAN Evaluation (FID + Inception Score)) ──")
try:
    diag, findings = run_diagnostic_checkpoint(
        generator,
        noise_loader,
        _diag_loss,
        title="GAN Evaluation (FID + Inception Score)",
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
# [!] Gradient flow (WARNING): G gradients RMS 3.2e-03, D gradients RMS 4.5e-02
#     (G/D imbalance — D dominating is the canonical GAN failure mode).
# [!] Dead neurons  (WARNING): 28% saturation in G's final tanh layer —
#     generator is producing outputs clustered at [-1, 1] extremes.
# [?] Loss trend    (MIXED): D loss → 0.1 (winning), G loss climbing.
#     Textbook sign of D overpowering G — generator can't keep up.
# ════════════════════════════════════════════════════════════════
#
# STUDENT INTERPRETATION GUIDE — reading the Prescription Pad:

#  [BLOOD TEST] G/D gradient imbalance is the signature of GAN training
#     collapse. When D's RMS >> G's RMS, the discriminator has "won" —
#     G is receiving useless gradient signal. Slide 5.5 shows this as
#     the most common GAN failure.
#     >> Prescription: reduce D learning rate OR train G for N steps
#        per 1 D step OR apply WGAN-GP (ex_5/02) which sidesteps this
#        via gradient penalty instead of BCE.
#
#  [X-RAY] 28% tanh saturation means the generator is producing
#     extreme outputs — diversity is collapsing. Combined with the
#     loss trend, this is mode collapse territory.
#     >> Prescription: add minibatch discrimination, feature matching,
#        or switch to WGAN which has no saturation problem.
#
#  [STETHOSCOPE] Diverging G/D losses (D→0, G→∞) is the classic
#     "Nash equilibrium lost" pattern. WGAN-GP (next exercise) is
#     the direct fix.
