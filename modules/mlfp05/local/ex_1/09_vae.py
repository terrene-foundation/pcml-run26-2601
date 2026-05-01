# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# MLFP05 — Exercise 1.9: Variational Autoencoder (Probabilistic Latent)
# ════════════════════════════════════════════════════════════════════════
#
# WHAT YOU'LL LEARN:
#   - Build a VAE with reparameterisation trick (z = mu + sigma * epsilon)
#   - Understand the ELBO loss: reconstruction + KL divergence
#   - Generate BRAND NEW images by sampling z ~ N(0, I)
#   - Visualise latent traversal to see what each dimension controls
#   - Apply to privacy-preserving synthetic patient data at NUH
#   - Verify synthetic data quality with statistical tests + privacy checks
#
# PREREQUISITES: 08_recurrent_ae.py
# ESTIMATED TIME: ~25 min
#
# TASKS:
#   1. Build VAE with mu/logvar heads and reparameterisation
#   2. Train with ELBO loss (reconstruction + KL divergence)
#   3. Visualise: reconstruction, generation, latent traversal
#   4. Apply: synthetic patient data for NUH PDPA compliance
#
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from shared.mlfp05.ex_1 import (
    INPUT_DIM,
    LATENT_DIM,
    EPOCHS,
    OUTPUT_DIR,
    device,
    load_fashion_mnist,
    setup_engines,
    train_variant,
    show_reconstruction,
    show_generated_samples,
    show_latent_traversal,
    register_model,
)


# ════════════════════════════════════════════════════════════════════════
# THEORY — Probabilistic Latent Space
# ════════════════════════════════════════════════════════════════════════
# The VAE replaces the deterministic latent code with a PROBABILITY
# DISTRIBUTION. The encoder outputs mean (mu) and log-variance (log_var).
# We sample z ~ N(mu, sigma^2) using the reparameterisation trick:
#   z = mu + sigma * epsilon,  where epsilon ~ N(0, I)
# This keeps gradients flowing through mu and sigma.
#
# Loss = reconstruction + KL(q(z|x) || N(0,I))
# The KL term pushes the latent distribution toward a standard Gaussian,
# which is what allows GENERATION: sample z ~ N(0,I), decode to images.
#
# Analogy: A recipe book vs a cooking style guide. A regular AE stores
# exact recipes (deterministic codes). A VAE stores a STYLE GUIDE
# (probability distribution) — "use about this much salt, roughly this
# temperature." From the style guide, you can generate infinite new
# recipes that taste like the original chef's cooking.


# ════════════════════════════════════════════════════════════════════════
# TASK 1 — Load data and engines
# ════════════════════════════════════════════════════════════════════════

X_flat, X_test_flat, X_img, X_test_img, flat_loader, img_loader = load_fashion_mnist()
conn, tracker, exp_name, registry, has_registry = setup_engines()


# ════════════════════════════════════════════════════════════════════════
# TASK 2 — Build and Train VAE
# ════════════════════════════════════════════════════════════════════════


class VAE(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int):
        super().__init__()
        # TODO: Build shared encoder body — nn.Sequential:
        #       Linear(input_dim, 256), ReLU, Linear(256, 128), ReLU
        self.encoder = ____

        # TODO: Two separate heads from the shared encoder output:
        #       fc_mu: Linear(128, latent_dim) — mean of q(z|x)
        #       fc_logvar: Linear(128, latent_dim) — log-variance of q(z|x)
        self.fc_mu = ____
        self.fc_logvar = ____

        # TODO: Build decoder — nn.Sequential:
        #       Linear(latent_dim, 128), ReLU, Linear(128, 256), ReLU,
        #       Linear(256, input_dim), Sigmoid
        self.decoder = ____

    def encode(self, x):
        # TODO: Pass x through encoder, return (mu, logvar)
        ____

    def reparameterise(self, mu, logvar):
        """z = mu + sigma * epsilon. Gradients flow through mu and sigma."""
        # TODO: std = exp(0.5 * logvar)
        #       eps = torch.randn_like(std)
        #       return mu + eps * std
        ____

    def forward(self, x):
        # TODO: encode -> reparameterise -> decode
        # Return (reconstruction, mu, logvar)
        ____

    def sample(self, n):
        """Sample from the prior N(0, I) and decode to images."""
        # TODO: z = torch.randn(n, latent_dim, device=...)
        #       return self.decoder(z)
        ____


KL_WEIGHT = 0.1


def vae_loss_fn(model, xb):
    """VAE loss: reconstruction (MSE) + KL divergence."""
    # TODO: Forward pass to get x_hat, mu, logvar
    # recon = F.mse_loss(x_hat, xb, reduction="sum") / xb.size(0)
    # kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / xb.size(0)
    # Return (recon + KL_WEIGHT * kl, {"recon": recon.item(), "kl": kl.item()})
    ____


print("\n" + "=" * 70)
print("  Variational AE — Probabilistic Latent Space")
print("=" * 70)
print("  Reparameterisation trick: z = mu + sigma * epsilon")
print(f"  KL weight: {KL_WEIGHT} (balance reconstruction vs regularity)")

# TODO: Create VAE(INPUT_DIM, LATENT_DIM) and train
vae_model = ____
vae_losses = ____


# ════════════════════════════════════════════════════════════════════════
# TASK 3 — Visualise: Reconstruction, Generation, Latent Traversal
# ════════════════════════════════════════════════════════════════════════

# TODO: Three visualisations:
# show_reconstruction(vae_model, X_test_flat, "VAE Reconstruction")
# show_generated_samples(vae_model, "VAE — Generated Samples from N(0,I)", grid_size=8)
# show_latent_traversal(vae_model, X_test_flat, "VAE — Latent Traversal", n_dims=5)
____
____
____

# ── Checkpoint ──────────────────────────────────────────────────────
assert len(vae_losses) == EPOCHS
assert vae_losses[-1] < vae_losses[0]
vae_model.eval()
with torch.no_grad():
    samples = vae_model.sample(n=16).cpu()
assert samples.shape == (16, INPUT_DIM), f"Expected (16, 784), got {samples.shape}"
assert samples.min() >= 0.0 and samples.max() <= 1.0
print("\n--- Checkpoint passed --- VAE trained + generation verified\n")

if has_registry:
    register_model(registry, "vae", vae_model, vae_losses[-1])


# ════════════════════════════════════════════════════════════════════════
# APPLY — Privacy-Preserving Synthetic Patient Data (NUH)
# ════════════════════════════════════════════════════════════════════════
# BUSINESS SCENARIO: You are a data scientist at National University
# Hospital (NUH). Researchers need patient data to study diabetes risk
# factors, but Singapore's PDPA prohibits sharing identifiable records.
# Your director asks: "Can we give researchers statistically useful
# data without exposing any real patient?"

print("\n" + "=" * 70)
print("  APPLICATION: PDPA-Compliant Synthetic Patient Data (NUH)")
print("=" * 70)

# --- Generate realistic patient data ---
N_PATIENTS = 10_000
patient_rng = np.random.default_rng(42)

# TODO: Generate correlated patient features:
# age ~ Normal(52, 15), clipped [21, 90]
# gender ~ Binomial(1, 0.48)
# bmi depends on age + gender + noise
# systolic_bp depends on age + bmi + noise
# diastolic_bp ~ 0.6 * systolic + noise
# cholesterol depends on age + bmi + noise
# hba1c depends on age + bmi + exponential noise
# glucose ~ 18 * hba1c + noise
# diagnosis ~ logistic(age, bmi, cholesterol, hba1c)
age = ____
gender = ____
bmi = ____
systolic_bp = ____
diastolic_bp = ____
cholesterol = ____
hba1c = ____
glucose = ____
diagnosis = ____

df = pl.DataFrame(
    {
        "age": age,
        "gender": gender,
        "bmi": bmi,
        "systolic_bp": systolic_bp,
        "diastolic_bp": diastolic_bp,
        "cholesterol": cholesterol,
        "hba1c": hba1c,
        "fasting_glucose": glucose,
        "diabetes_diagnosis": diagnosis,
    }
)
FEATURE_NAMES = df.columns
N_FEATURES = len(FEATURE_NAMES)
print(f"Patient dataset: {df.shape[0]:,} records, {N_FEATURES} features")
print(f"Diabetes prevalence: {diagnosis.mean()*100:.1f}%")

# --- Normalise and train VAE ---
data_np = df.to_numpy().astype(np.float32)
data_min = data_np.min(axis=0)
data_max = data_np.max(axis=0)
data_range = data_max - data_min
data_range[data_range == 0] = 1.0
data_norm = (data_np - data_min) / data_range

data_tensor = torch.tensor(data_norm, device=device)
patient_loader = DataLoader(TensorDataset(data_tensor), batch_size=256, shuffle=True)


class PatientVAE(nn.Module):
    def __init__(self, input_dim, latent_dim=8):
        super().__init__()
        # TODO: Build encoder, mu/logvar heads, decoder
        # Encoder: Linear(input_dim, 64), ReLU, Linear(64, 32), ReLU
        # fc_mu: Linear(32, latent_dim)
        # fc_logvar: Linear(32, latent_dim)
        # Decoder: Linear(latent_dim, 32), ReLU, Linear(32, 64), ReLU,
        #          Linear(64, input_dim), Sigmoid
        self.encoder = ____
        self.fc_mu = ____
        self.fc_logvar = ____
        self.decoder = ____

    def encode(self, x):
        # TODO: Return (mu, logvar)
        ____

    def reparameterise(self, mu, logvar):
        # TODO: Reparameterisation trick
        ____

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        # TODO: encode -> reparameterise -> decode
        # Return (reconstruction, mu, logvar)
        ____


PATIENT_LATENT = 8
# TODO: Create PatientVAE, optimizer. Train 100 epochs with ELBO loss.
patient_model = ____
patient_opt = ____

print("\nTraining VAE on patient records...")
for epoch in range(100):
    patient_model.train()
    epoch_loss, n_samples = 0.0, 0
    for (batch,) in patient_loader:
        # TODO: Forward, compute recon_loss + kl_loss, backprop
        ____
    if (epoch + 1) % 25 == 0:
        print(f"  Epoch {epoch+1:3d}/100: loss = {epoch_loss/n_samples:.4f}")

# --- Generate synthetic patients ---
patient_model.eval()
with torch.no_grad():
    z = torch.randn(N_PATIENTS, PATIENT_LATENT, device=device)
    synthetic_norm = patient_model.decode(z).cpu().numpy()

synthetic_raw = synthetic_norm * data_range + data_min
synthetic_raw[:, 1] = np.round(synthetic_raw[:, 1]).clip(0, 1)
synthetic_raw[:, -1] = np.round(synthetic_raw[:, -1]).clip(0, 1)

print(f"\nGenerated {N_PATIENTS:,} synthetic patients")

# --- Visualisation 1: Distribution comparison ---
# TODO: Grid of histograms comparing real vs synthetic distributions
# Save to OUTPUT_DIR / "ex1_generation_distributions.png"
n_cols = 3
n_rows = (N_FEATURES + n_cols - 1) // n_cols
fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, n_rows * 3.5))
____
plt.tight_layout()
plt.savefig(
    OUTPUT_DIR / "ex1_generation_distributions.png", dpi=150, bbox_inches="tight"
)
plt.show()

# --- Visualisation 2: Correlation comparison ---
# TODO: 1x3 figure: real correlation, synthetic correlation, absolute difference
# Save to OUTPUT_DIR / "ex1_generation_correlations.png"
real_corr = np.corrcoef(data_np.T)
synth_corr = np.corrcoef(synthetic_raw.T)
corr_mae = np.abs(real_corr - synth_corr).mean()

fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
____
plt.tight_layout()
plt.savefig(
    OUTPUT_DIR / "ex1_generation_correlations.png", dpi=150, bbox_inches="tight"
)
plt.show()

# --- Privacy test ---
from scipy.spatial.distance import cdist

sample_synth = patient_rng.choice(N_PATIENTS, size=1000, replace=False)
sample_real = patient_rng.choice(N_PATIENTS, size=1000, replace=False)
dist_matrix = cdist(synthetic_norm[sample_synth], data_norm[sample_real])
nn_distances = dist_matrix.min(axis=1)
self_dist = cdist(data_norm[sample_real[:500]], data_norm[sample_real[500:]])
self_nn = self_dist.min(axis=1)
privacy_safe = nn_distances.mean() >= self_nn.mean() * 0.8

# --- Statistical utility ---
tests_passed = 0
for i in range(N_FEATURES):
    real_mean = data_np[:, i].mean()
    synth_mean = synthetic_raw[:, i].mean()
    real_std = data_np[:, i].std()
    if real_std > 0:
        passed = abs(synth_mean - real_mean) / real_std < 0.3
    else:
        passed = abs(synth_mean - real_mean) < 0.1
    if passed:
        tests_passed += 1

# --- Business Impact ---
print("\n" + "=" * 64)
print("BUSINESS IMPACT SUMMARY — NUH PDPA-Compliant Synthetic Data")
print("=" * 64)
print(f"\nDataset: {N_PATIENTS:,} real -> {N_PATIENTS:,} synthetic records")
print(
    f"Statistical utility: {tests_passed}/{N_FEATURES} features pass ({tests_passed/N_FEATURES*100:.0f}%)"
)
print(f"Correlation MAE: {corr_mae:.4f}")
print(f"\nPrivacy assessment:")
print(f"  Mean synth-to-real NN distance: {nn_distances.mean():.4f}")
print(f"  Mean real-to-real NN distance:  {self_nn.mean():.4f}")
print(f"  Ratio (>1.0 = good):            {nn_distances.mean()/self_nn.mean():.3f}")
print(f"  Privacy safe: {'YES' if privacy_safe else 'NO'}")
print(f"\nResearch impact:")
print(f"  Before: 6-12 month ethics approval per data request")
print(f"  After: Instant access to synthetic data, ethics-exempt")
print(f"  Estimated: 3-5 research projects/year unblocked")
print(f"  Value: ~S$500K in grant revenue (S$100K avg per project)")
print("=" * 64)


# ════════════════════════════════════════════════════════════════════════
# REFLECTION
# ════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  WHAT YOU'VE MASTERED")
print("=" * 70)
print(
    """
  [x] Built a VAE with reparameterisation trick (z = mu + sigma * eps)
  [x] Trained with ELBO loss (reconstruction + KL divergence)
  [x] Generated BRAND NEW images from the learned prior N(0,I)
  [x] Explored latent traversal — each dimension controls one aspect
  [x] Applied to synthetic patient data generation for NUH
  [x] Verified statistical utility AND privacy preservation

  KEY INSIGHT: The VAE trades reconstruction sharpness for a regular
  latent space. The KL term pushes q(z|x) toward N(0,I), which means
  you can sample z ~ N(0,I) and get plausible outputs. For synthetic
  data, this is the key: the generated records are statistically
  similar to real records but are NOT copies of any real patient.
  Singapore's PDPA is satisfied; researchers get useful data.

  Next: 10_contractive_vae.py combines smooth + probabilistic latent spaces...
"""
)

# ══════════════════════════════════════════════════════════════════
# DIAGNOSTIC CHECKPOINT — five instruments (VAE with KL term)
# ══════════════════════════════════════════════════════════════════
# VAE-specific failure: posterior collapse (the encoder ignores x
# and outputs a fixed prior). It shows up in the Blood Test as
# gradients vanishing specifically on `fc_mu`/`fc_logvar` — the
# model stops routing information through the latent.
from kailash_ml.diagnostics import run_diagnostic_checkpoint


def _diag_loss(m, batch):
    xb = batch[0] if isinstance(batch, (tuple, list)) else batch
    loss, _ = vae_loss_fn(m, xb)
    return loss


print("\n── Diagnostic Report (VAE) ──")
diag, findings = run_diagnostic_checkpoint(
    vae_model,
    flat_loader,
    _diag_loss,
    title=f"VAE (KL={KL_WEIGHT})",
    n_batches=8,
    train_losses=vae_losses,
    show=False,
)

# ══════ EXPECTED OUTPUT (synthesized reference — full run produces similar pattern) ══════
# ════════════════════════════════════════════════════════════════
#   DL Diagnostics Report — Prescription Pad
# ════════════════════════════════════════════════════════════════
#   [!] Gradient flow (WARNING): Low gradients at
#       'fc_mu.weight' — RMS = 5.2e-05. KL penalty is
#       dampening mu-head updates (early sign of posterior
#       collapse risk). fc_logvar.weight RMS = 4.8e-05
#       (similar dampening).
#   [!] Dead neurons  (WARNING): 'decoder.2' (relu): 22%
#       dead during early epochs — sampled z lands far from
#       trained region.
#   [✓] Loss trend    (HEALTHY): total loss slope
#       -1.9e-03/epoch. Reconstruction term: -1.7e-03,
#       KL term: -2.1e-04. Both converging — no term
#       dominating.
# ════════════════════════════════════════════════════════════════
# Final train loss: ~0.039 (recon 0.031 + KL 0.008), 10 epochs, beta=1.
#
# STUDENT INTERPRETATION GUIDE — reading the Prescription Pad:
#
#  [BLOOD TEST — POSTERIOR-COLLAPSE DETECTOR] RMS 5.2e-05 on
#     fc_mu.weight is the key VAE health metric. The KL term
#     pulls q(z|x) toward N(0,I), which REDUCES gradient on
#     the mean-encoder. If this drops below 1e-5, the encoder
#     has given up and emits constant z regardless of input
#     — POSTERIOR COLLAPSE. Slide 5M covers the Bowman 2016
#     analysis: "the decoder ignores z and the encoder
#     surrenders."
#     >> Prescription: (a) KL annealing: start beta=0, linearly
#        ramp to 1 over first 5 epochs. (b) Free-bits: cap KL
#        loss from below at a small positive value (~0.5 per
#        latent dim). (c) Reduce KL_WEIGHT below 1.0 if above.
#
#  [X-RAY] 22% dead in decoder is an EARLY-EPOCH signature
#     specific to VAEs: the sampled z ~ N(mu, sigma) lands in
#     regions of latent space the decoder hasn't seen yet,
#     triggering ReLU gates that never activated before.
#     Usually recovers by epoch 5-6 as the decoder learns to
#     cover the [-3, 3] sigma region of N(0,I).
#     >> Prescription: If dead% STAYS above 15% past epoch 8,
#        the reparameterisation sampling is too aggressive —
#        clamp logvar to [-5, 5] to prevent sigma exploding.
#
#  [STETHOSCOPE — TWO-TERM READ] VAE loss = reconstruction
#     (pixelwise MSE / BCE) + KL divergence. Both SHOULD
#     decrease. If total loss drops but KL rises: you have
#     a regular AE pretending to be VAE (encoder ignoring KL).
#     If KL drops to near zero and total stalls: posterior
#     collapse (encoder ignoring input). The diag.epochs_df()
#     exposes both terms — always plot them separately.
#     >> Prescription: Watch both curves. Healthy VAE has
#        reconstruction decreasing 5-10x faster than KL — the
#        decoder learns first, then KL tightens the latent.
#
#  FIVE-INSTRUMENT TAKEAWAY: VAE needs a two-part Stethoscope
#  reading (recon + KL separately) because the sum can be
#  healthy while each term is pathological. This is the first
#  exercise where "total loss" lies — you will encounter the
#  same pattern in ex_5 GANs (generator vs discriminator loss
#  balance) and in ex_8 RL (policy vs value loss).
# ════════════════════════════════════════════════════════════════════


# ════════════════════════════════════════════════════════════════════════
# TASK 3 — Visualise: Reconstruction, Generation, Latent Traversal
# ════════════════════════════════════════════════════════════════════════

show_reconstruction(vae_model, X_test_flat, "VAE Reconstruction")
show_generated_samples(vae_model, "VAE — Generated Samples from N(0,I)", grid_size=8)
show_latent_traversal(vae_model, X_test_flat, "VAE — Latent Traversal", n_dims=5)

# ── Checkpoint ──────────────────────────────────────────────────────
assert len(vae_losses) == EPOCHS
assert vae_losses[-1] < vae_losses[0]
vae_model.eval()
with torch.no_grad():
    samples = vae_model.sample(n=16).cpu()
assert samples.shape == (16, INPUT_DIM), f"Expected (16, 784), got {samples.shape}"
assert samples.min() >= 0.0 and samples.max() <= 1.0
print("\n--- Checkpoint passed --- VAE trained + generation verified\n")

if has_registry:
    register_model(registry, "vae", vae_model, vae_losses[-1])


# ════════════════════════════════════════════════════════════════════════
# APPLY — Privacy-Preserving Synthetic Patient Data (NUH)
# ════════════════════════════════════════════════════════════════════════
# BUSINESS SCENARIO: You are a data scientist at National University
# Hospital (NUH). Researchers need patient data to study diabetes risk
# factors, but Singapore's PDPA prohibits sharing identifiable records.
# Your director asks: "Can we give researchers statistically useful
# data without exposing any real patient?"

print("\n" + "=" * 70)
print("  APPLICATION: PDPA-Compliant Synthetic Patient Data (NUH)")
print("=" * 70)

# --- Generate realistic patient data ---
N_PATIENTS = 10_000
patient_rng = np.random.default_rng(42)

age = (
    patient_rng.normal(loc=52, scale=15, size=N_PATIENTS)
    .clip(21, 90)
    .astype(np.float32)
)
gender = patient_rng.binomial(1, 0.48, size=N_PATIENTS).astype(np.float32)
bmi = (
    (22 + 0.05 * (age - 40) + 1.2 * gender + patient_rng.normal(0, 3.5, N_PATIENTS))
    .clip(16, 45)
    .astype(np.float32)
)
systolic_bp = (
    (100 + 0.4 * (age - 40) + 0.8 * (bmi - 24) + patient_rng.normal(0, 12, N_PATIENTS))
    .clip(85, 200)
    .astype(np.float32)
)
diastolic_bp = (
    (systolic_bp * 0.6 + patient_rng.normal(0, 8, N_PATIENTS))
    .clip(55, 120)
    .astype(np.float32)
)
cholesterol = (
    (150 + 0.5 * (age - 40) + 1.5 * (bmi - 24) + patient_rng.normal(0, 30, N_PATIENTS))
    .clip(100, 350)
    .astype(np.float32)
)
hba1c = (
    (
        5.0
        + 0.01 * (age - 40)
        + 0.05 * (bmi - 24)
        + patient_rng.exponential(0.4, N_PATIENTS)
    )
    .clip(4.0, 14.0)
    .astype(np.float32)
)
glucose = (
    (hba1c * 18 + patient_rng.normal(0, 15, N_PATIENTS))
    .clip(60, 300)
    .astype(np.float32)
)
diagnosis_logit = (
    -6.0
    + 0.03 * (age - 40)
    + 0.08 * (bmi - 24)
    + 0.005 * (cholesterol - 200)
    + 0.5 * (hba1c - 5.5)
)
diagnosis = patient_rng.binomial(1, 1 / (1 + np.exp(-diagnosis_logit))).astype(
    np.float32
)

df = pl.DataFrame(
    {
        "age": age,
        "gender": gender,
        "bmi": bmi,
        "systolic_bp": systolic_bp,
        "diastolic_bp": diastolic_bp,
        "cholesterol": cholesterol,
        "hba1c": hba1c,
        "fasting_glucose": glucose,
        "diabetes_diagnosis": diagnosis,
    }
)
FEATURE_NAMES = df.columns
N_FEATURES = len(FEATURE_NAMES)
print(f"Patient dataset: {df.shape[0]:,} records, {N_FEATURES} features")
print(f"Diabetes prevalence: {diagnosis.mean()*100:.1f}%")

# --- Normalise and train VAE ---
data_np = df.to_numpy().astype(np.float32)
data_min = data_np.min(axis=0)
data_max = data_np.max(axis=0)
data_range = data_max - data_min
data_range[data_range == 0] = 1.0
data_norm = (data_np - data_min) / data_range

data_tensor = torch.tensor(data_norm, device=device)
patient_loader = DataLoader(TensorDataset(data_tensor), batch_size=256, shuffle=True)


class PatientVAE(nn.Module):
    def __init__(self, input_dim, latent_dim=8):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64), nn.ReLU(), nn.Linear(64, 32), nn.ReLU()
        )
        self.fc_mu = nn.Linear(32, latent_dim)
        self.fc_logvar = nn.Linear(32, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, input_dim),
            nn.Sigmoid(),
        )

    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterise(self, mu, logvar):
        return mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterise(mu, logvar)
        return self.decode(z), mu, logvar


PATIENT_LATENT = 8
patient_model = PatientVAE(N_FEATURES, PATIENT_LATENT).to(device)
patient_opt = torch.optim.Adam(patient_model.parameters(), lr=1e-3)

print("\nTraining VAE on patient records...")
for epoch in range(100):
    patient_model.train()
    epoch_loss, n_samples = 0.0, 0
    for (batch,) in patient_loader:
        recon, mu, logvar = patient_model(batch)
        recon_loss = F.mse_loss(recon, batch, reduction="sum")
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        loss = recon_loss + kl_loss
        patient_opt.zero_grad()
        loss.backward()
        patient_opt.step()
        epoch_loss += loss.item()
        n_samples += len(batch)
    if (epoch + 1) % 25 == 0:
        print(f"  Epoch {epoch+1:3d}/100: loss = {epoch_loss/n_samples:.4f}")

# --- Generate synthetic patients ---
patient_model.eval()
with torch.no_grad():
    z = torch.randn(N_PATIENTS, PATIENT_LATENT, device=device)
    synthetic_norm = patient_model.decode(z).cpu().numpy()

synthetic_raw = synthetic_norm * data_range + data_min
synthetic_raw[:, 1] = np.round(synthetic_raw[:, 1]).clip(0, 1)
synthetic_raw[:, -1] = np.round(synthetic_raw[:, -1]).clip(0, 1)

print(f"\nGenerated {N_PATIENTS:,} synthetic patients")

# --- Visualisation 1: Distribution comparison ---
n_cols = 3
n_rows = (N_FEATURES + n_cols - 1) // n_cols
fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, n_rows * 3.5))
axes_flat = axes.flatten()

for i, name in enumerate(FEATURE_NAMES):
    real_vals = data_np[:, i]
    synth_vals = synthetic_raw[:, i]
    if name in ("gender", "diabetes_diagnosis"):
        x = np.arange(2)
        real_counts = [np.mean(real_vals == c) for c in [0, 1]]
        synth_counts = [np.mean(synth_vals == c) for c in [0, 1]]
        axes_flat[i].bar(
            x - 0.15, real_counts, 0.3, label="Real", color="#2196F3", alpha=0.8
        )
        axes_flat[i].bar(
            x + 0.15, synth_counts, 0.3, label="Synthetic", color="#FF9800", alpha=0.8
        )
        axes_flat[i].set_xticks(x)
        axes_flat[i].set_xticklabels(
            ["Female", "Male"] if name == "gender" else ["No", "Yes"]
        )
    else:
        axes_flat[i].hist(
            real_vals, bins=40, alpha=0.6, density=True, label="Real", color="#2196F3"
        )
        axes_flat[i].hist(
            synth_vals,
            bins=40,
            alpha=0.6,
            density=True,
            label="Synthetic",
            color="#FF9800",
        )
    axes_flat[i].set_title(name.replace("_", " ").title(), fontsize=11)
    axes_flat[i].legend(fontsize=8)
for j in range(N_FEATURES, len(axes_flat)):
    axes_flat[j].set_visible(False)
fig.suptitle("Real vs VAE-Generated Patient Distributions", fontsize=13, y=1.02)
plt.tight_layout()
plt.savefig(
    OUTPUT_DIR / "ex1_generation_distributions.png", dpi=150, bbox_inches="tight"
)
plt.show()

# --- Visualisation 2: Correlation comparison ---
real_corr = np.corrcoef(data_np.T)
synth_corr = np.corrcoef(synthetic_raw.T)
corr_mae = np.abs(real_corr - synth_corr).mean()

fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
for ax, corr_mat, title in [
    (axes[0], real_corr, "Real Data"),
    (axes[1], synth_corr, "Synthetic Data"),
    (axes[2], np.abs(real_corr - synth_corr), "Absolute Difference"),
]:
    vmin = -1 if title != "Absolute Difference" else 0
    vmax = 1 if title != "Absolute Difference" else 0.3
    cmap = "RdBu_r" if title != "Absolute Difference" else "Reds"
    im = ax.imshow(corr_mat, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_xticks(range(N_FEATURES))
    ax.set_yticks(range(N_FEATURES))
    short_names = [n[:8] for n in FEATURE_NAMES]
    ax.set_xticklabels(short_names, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(short_names, fontsize=8)
    ax.set_title(title, fontsize=12)
    plt.colorbar(im, ax=ax, shrink=0.8)
fig.suptitle(f"Correlation Preservation: MAE = {corr_mae:.4f}", fontsize=13)
plt.tight_layout()
plt.savefig(
    OUTPUT_DIR / "ex1_generation_correlations.png", dpi=150, bbox_inches="tight"
)
plt.show()

# --- Privacy test ---
from scipy.spatial.distance import cdist

sample_synth = patient_rng.choice(N_PATIENTS, size=1000, replace=False)
sample_real = patient_rng.choice(N_PATIENTS, size=1000, replace=False)
dist_matrix = cdist(synthetic_norm[sample_synth], data_norm[sample_real])
nn_distances = dist_matrix.min(axis=1)
self_dist = cdist(data_norm[sample_real[:500]], data_norm[sample_real[500:]])
self_nn = self_dist.min(axis=1)
privacy_safe = nn_distances.mean() >= self_nn.mean() * 0.8

# --- Statistical utility ---
tests_passed = 0
for i in range(N_FEATURES):
    real_mean = data_np[:, i].mean()
    synth_mean = synthetic_raw[:, i].mean()
    real_std = data_np[:, i].std()
    if real_std > 0:
        passed = abs(synth_mean - real_mean) / real_std < 0.3
    else:
        passed = abs(synth_mean - real_mean) < 0.1
    if passed:
        tests_passed += 1

# --- Business Impact ---
print("\n" + "=" * 64)
print("BUSINESS IMPACT SUMMARY — NUH PDPA-Compliant Synthetic Data")
print("=" * 64)
print(f"\nDataset: {N_PATIENTS:,} real -> {N_PATIENTS:,} synthetic records")
print(
    f"Statistical utility: {tests_passed}/{N_FEATURES} features pass ({tests_passed/N_FEATURES*100:.0f}%)"
)
print(f"Correlation MAE: {corr_mae:.4f}")
print(f"\nPrivacy assessment:")
print(f"  Mean synth-to-real NN distance: {nn_distances.mean():.4f}")
print(f"  Mean real-to-real NN distance:  {self_nn.mean():.4f}")
print(f"  Ratio (>1.0 = good):            {nn_distances.mean()/self_nn.mean():.3f}")
print(f"  Privacy safe: {'YES' if privacy_safe else 'NO'}")
print(f"\nResearch impact:")
print(f"  Before: 6-12 month ethics approval per data request")
print(f"  After: Instant access to synthetic data, ethics-exempt")
print(f"  Estimated: 3-5 research projects/year unblocked")
print(f"  Value: ~S$500K in grant revenue (S$100K avg per project)")
print("=" * 64)


# ════════════════════════════════════════════════════════════════════════
# REFLECTION
# ════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  WHAT YOU'VE MASTERED")
print("=" * 70)
print(
    """
  [x] Built a VAE with reparameterisation trick (z = mu + sigma * eps)
  [x] Trained with ELBO loss (reconstruction + KL divergence)
  [x] Generated BRAND NEW images from the learned prior N(0,I)
  [x] Explored latent traversal — each dimension controls one aspect
  [x] Applied to synthetic patient data generation for NUH
  [x] Verified statistical utility AND privacy preservation

  KEY INSIGHT: The VAE trades reconstruction sharpness for a regular
  latent space. The KL term pushes q(z|x) toward N(0,I), which means
  you can sample z ~ N(0,I) and get plausible outputs. For synthetic
  data, this is the key: the generated records are statistically
  similar to real records but are NOT copies of any real patient.
  Singapore's PDPA is satisfied; researchers get useful data.

  Next: 10_contractive_vae.py combines smooth + probabilistic latent spaces...
"""
)

