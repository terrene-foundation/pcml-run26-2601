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
