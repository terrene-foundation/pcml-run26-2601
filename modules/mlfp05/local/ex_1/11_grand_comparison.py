# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# MLFP05 — Exercise 1.11: Grand Comparison — All 10 Variants Side by Side
# ════════════════════════════════════════════════════════════════════════
#
# WHAT YOU'LL LEARN:
#   - Train ALL 10 autoencoder variants on the same data
#   - Compare reconstruction quality in a single mega-grid
#   - Compare training curves using ModelVisualizer
#   - Build a summary table with loss, parameters, and variant type
#   - Register all 10 models in ModelRegistry
#   - Choose the right variant for a given business problem
#
# PREREQUISITES: Files 01-10 (understanding each variant)
# ESTIMATED TIME: ~30 min (all 10 train sequentially)
#
# TASKS:
#   1. Train all 10 variants (reusing architectures from prior files)
#   2. Grand reconstruction comparison grid (9 image variants + original)
#   3. Training curves overlay
#   4. Summary table + model registration
#
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import asyncio
import pickle

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from kailash_ml import ModelVisualizer
from kailash_ml import ModelRegistry
from kailash_ml.types import MetricSpec

from shared.mlfp05.ex_1 import (
    INPUT_DIM,
    LATENT_DIM,
    EPOCHS,
    OUTPUT_DIR,
    device,
    load_fashion_mnist,
    setup_engines,
    train_variant,
    all_losses,
    all_models,
)


# ════════════════════════════════════════════════════════════════════════
# TASK 1 — Load data and engines
# ════════════════════════════════════════════════════════════════════════

X_flat, X_test_flat, X_img, X_test_img, flat_loader, img_loader = load_fashion_mnist()
conn, tracker, exp_name, registry, has_registry = setup_engines()

assert X_flat.shape[0] == 60000
assert X_test_flat.shape[0] == 10000
print("\n--- Data loaded ---\n")


# ════════════════════════════════════════════════════════════════════════
# MODEL DEFINITIONS — All 10 Variants (compact)
# ════════════════════════════════════════════════════════════════════════
# These are the same architectures from files 01-10. You must implement
# each one here to train all 10 in a single comparison run.


class StandardAE(nn.Module):
    def __init__(self, input_dim, hidden_dim=1024):
        super().__init__()
        # TODO: Overcomplete encoder/decoder (same as 01_standard_ae.py)
        self.encoder = ____
        self.decoder = ____

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z), z


class UndercompleteAE(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        # TODO: Bottleneck encoder/decoder (same as 02_undercomplete_ae.py)
        self.encoder = ____
        self.decoder = ____

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z), z


class DenoisingAE(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        # TODO: Same architecture as undercomplete (different training)
        self.encoder = ____
        self.decoder = ____

    def add_noise(self, x, sigma=0.3):
        return torch.clamp(x + sigma * torch.randn_like(x), 0.0, 1.0)

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z), z


class SparseAE(nn.Module):
    def __init__(self, input_dim, hidden_dim=256):
        super().__init__()
        # TODO: Sparse encoder/decoder (same as 04_sparse_ae.py)
        self.encoder = ____
        self.decoder = ____

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z), z


class ContractiveAE(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        # TODO: Explicit encoder layers for weight access
        self.enc1 = ____
        self.enc2 = ____
        self.enc3 = ____
        self.decoder = ____

    def encoder(self, x):
        return self.enc3(F.relu(self.enc2(F.relu(self.enc1(x)))))

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z), z


class ConvAE(nn.Module):
    def __init__(self, latent_dim=16):
        super().__init__()
        # TODO: Conv encoder + deconv decoder (same as 06_convolutional_ae.py)
        self.encoder = ____
        self.decoder = ____

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z), z


class StackedAE(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        # TODO: 5-layer deep encoder/decoder (same as 07_stacked_ae.py)
        self.encoder = ____
        self.decoder = ____

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z), z


class RecurrentAE(nn.Module):
    def __init__(self, seq_len, hidden_dim=64, latent_dim=16):
        super().__init__()
        self.seq_len = seq_len
        # TODO: LSTM encoder-decoder (same as 08_recurrent_ae.py)
        self.encoder_lstm = ____
        self.enc_to_latent = ____
        self.latent_to_dec = ____
        self.decoder_lstm = ____
        self.output_layer = ____

    def forward(self, x):
        # TODO: LSTM encode -> latent -> LSTM decode
        ____


class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        # TODO: Encoder body + mu/logvar heads + decoder (same as 09_vae.py)
        self.encoder = ____
        self.fc_mu = ____
        self.fc_logvar = ____
        self.decoder = ____

    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)
        return self.decoder(z), mu, logvar

    def sample(self, n):
        z = torch.randn(
            n, self.fc_mu.out_features, device=next(self.parameters()).device
        )
        return self.decoder(z)


class ContractiveVAE(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        # TODO: Explicit encoder layers + mu/logvar + decoder
        self.enc1 = ____
        self.enc2 = ____
        self.fc_mu = ____
        self.fc_logvar = ____
        self.decoder = ____

    def encoder(self, x):
        return F.relu(self.enc2(F.relu(self.enc1(x))))

    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)
        return self.decoder(z), mu, logvar

    def sample(self, n):
        z = torch.randn(
            n, self.fc_mu.out_features, device=next(self.parameters()).device
        )
        return self.decoder(z)


# ════════════════════════════════════════════════════════════════════════
# TASK 2 — Train All 10 Variants
# ════════════════════════════════════════════════════════════════════════

NOISE_SIGMA = 0.3
SPARSITY_WEIGHT = 1e-4
CONTRACTIVE_WEIGHT = 1e-4
KL_WEIGHT = 0.1
CVAE_CONTRACTIVE_WEIGHT = 1e-4


# TODO: Define loss functions for each variant
# std_loss: MSE only
# dae_loss: add noise, MSE against clean
# sparse_loss: MSE + L1 penalty
# cae_loss: MSE + Frobenius norm of encoder weights
# conv_loss: MSE only
# rec_loss: MSE only
# vae_loss: MSE + KL divergence
# cvae_loss: MSE + KL + Jacobian
def std_loss(m, xb):
    ____


def dae_loss(m, xb):
    ____


def sparse_loss(m, xb):
    ____


def cae_loss(m, xb):
    ____


def conv_loss(m, xb):
    ____


def rec_loss(m, xb):
    ____


def vae_loss(m, xb):
    ____


def cvae_loss(m, xb):
    ____


# Sensor data for recurrent AE
SEQ_LEN = 100
rng_s = np.random.RandomState(42)
sensor_data = []
for _ in range(5000):
    t = np.linspace(0, 4 * np.pi, SEQ_LEN)
    sig = rng_s.uniform(0.5, 1.0) * np.sin(
        rng_s.uniform(0.8, 1.2) * t + rng_s.uniform(0, 2 * np.pi)
    )
    sig += rng_s.uniform(0.2, 0.5) * np.sin(
        rng_s.uniform(1.8, 2.2) * t
    ) + 0.1 * rng_s.randn(SEQ_LEN)
    sig = (sig - sig.min()) / (sig.max() - sig.min() + 1e-8)
    sensor_data.append(sig)
sensor_t = torch.tensor(np.array(sensor_data, dtype=np.float32)).to(device)
sensor_loader = DataLoader(TensorDataset(sensor_t), batch_size=128, shuffle=True)

print("=" * 70)
print("  GRAND COMPARISON — Training All 10 Variants")
print("=" * 70)

# TODO: Create list of (name, model, loader, loss_fn) tuples for all 10 variants
# Then train each with train_variant
variants = [
    ("standard_ae", StandardAE(INPUT_DIM, 1024), flat_loader, std_loss),
    ("undercomplete_ae", UndercompleteAE(INPUT_DIM, LATENT_DIM), flat_loader, std_loss),
    ("denoising_ae", DenoisingAE(INPUT_DIM, LATENT_DIM), flat_loader, dae_loss),
    ("sparse_ae", SparseAE(INPUT_DIM), flat_loader, sparse_loss),
    ("contractive_ae", ContractiveAE(INPUT_DIM, LATENT_DIM), flat_loader, cae_loss),
    ("conv_ae", ConvAE(LATENT_DIM), img_loader, conv_loss),
    ("stacked_ae", StackedAE(INPUT_DIM, LATENT_DIM), flat_loader, std_loss),
    ("recurrent_ae", RecurrentAE(SEQ_LEN, 64, LATENT_DIM), sensor_loader, rec_loss),
    ("vae", VAE(INPUT_DIM, LATENT_DIM), flat_loader, vae_loss),
    ("contractive_vae", ContractiveVAE(INPUT_DIM, LATENT_DIM), flat_loader, cvae_loss),
]

for name, model, loader, loss_fn in variants:
    print(f"\n--- Training {name} ---")
    train_variant(tracker, exp_name, model, name, loader, loss_fn)

assert len(all_losses) == 10, f"Expected 10 variants, got {len(all_losses)}"
print(f"\n  All 10 variants trained successfully.")


# ════════════════════════════════════════════════════════════════════════
# TASK 3 — Grand Reconstruction Comparison Grid
# ════════════════════════════════════════════════════════════════════════

image_variants = [
    ("Standard AE", all_models["standard_ae"], False),
    ("Undercomplete AE", all_models["undercomplete_ae"], False),
    ("Denoising AE", all_models["denoising_ae"], False),
    ("Sparse AE", all_models["sparse_ae"], False),
    ("Contractive AE", all_models["contractive_ae"], False),
    ("Convolutional AE", all_models["conv_ae"], True),
    ("Stacked AE", all_models["stacked_ae"], False),
    ("VAE", all_models["vae"], False),
    ("Contractive VAE", all_models["contractive_vae"], False),
]

N_COMPARE = 8
n_variants = len(image_variants)

# TODO: Create (n_variants + 1) x N_COMPARE grid figure
# Row 0: original images
# Rows 1-9: each variant's reconstruction
# Save to OUTPUT_DIR / "ex1_grand_comparison.png"
fig, axes = plt.subplots(
    n_variants + 1, N_COMPARE, figsize=(16, (n_variants + 1) * 1.5)
)
____
plt.tight_layout()
fname = OUTPUT_DIR / "ex1_grand_comparison.png"
plt.savefig(fname, dpi=150, bbox_inches="tight")
plt.show()
print(f"  Saved: {fname}")


# ════════════════════════════════════════════════════════════════════════
# TASK 4 — Summary Table
# ════════════════════════════════════════════════════════════════════════

name_to_key = {
    "Standard AE": "standard_ae",
    "Undercomplete AE": "undercomplete_ae",
    "Denoising AE": "denoising_ae",
    "Sparse AE": "sparse_ae",
    "Contractive AE": "contractive_ae",
    "Convolutional AE": "conv_ae",
    "Stacked AE": "stacked_ae",
    "VAE": "vae",
    "Contractive VAE": "contractive_vae",
}

# TODO: Print comparison table: Variant, Final Loss, Params, Type
print("\n=== Autoencoder Variant Comparison ===")
print(f"{'Variant':<22} {'Final Loss':>12} {'Params':>10} {'Type':>12}")
print("-" * 60)
for name, model, _ in image_variants:
    key = name_to_key[name]
    final_loss = all_losses[key][-1]
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  {name:<20} {final_loss:>12.4f} {n_params:>10,} {'image':>12}")

rec_loss_val = all_losses["recurrent_ae"][-1]
rec_params = sum(p.numel() for p in all_models["recurrent_ae"].parameters())
print(
    f"  {'Recurrent AE':<20} {rec_loss_val:>12.4f} {rec_params:>10,} {'sequence':>12}"
)

# --- Training curves ---
# TODO: Use ModelVisualizer to plot all training curves
# Save to OUTPUT_DIR / "ex1_all_variants_training_curves.html"
viz = ModelVisualizer()
____


# ════════════════════════════════════════════════════════════════════════
# MODEL REGISTRATION
# ════════════════════════════════════════════════════════════════════════


async def register_all():
    if not has_registry:
        print("  ModelRegistry not available — skipping")
        return
    for name, model in all_models.items():
        model_bytes = pickle.dumps(model.state_dict())
        version = await registry.register_model(
            name=f"m5_{name}",
            artifact=model_bytes,
            metrics=[
                MetricSpec(name="final_loss", value=all_losses[name][-1]),
                MetricSpec(name="latent_dim", value=float(LATENT_DIM)),
                MetricSpec(name="epochs", value=float(EPOCHS)),
            ],
        )
        print(
            f"  Registered {name}: v{version.version}, loss={all_losses[name][-1]:.4f}"
        )


asyncio.run(register_all())

# ── Final Checkpoint ────────────────────────────────────────────────
assert len(all_losses) == 10
grand_path = OUTPUT_DIR / "ex1_grand_comparison.png"
assert grand_path.exists(), "Grand comparison image should be saved"
print("\n--- Final checkpoint passed --- grand comparison complete\n")

asyncio.run(conn.close())


# ════════════════════════════════════════════════════════════════════════
# DESTINATION-FIRST CLOSE — km.diagnose
# ════════════════════════════════════════════════════════════════════════
# This lesson walked the journey of the autoencoder family — 10 variants,
# 10 hand-rolled training loops, 10 reconstruction grids. The kailash-ml
# SDK ships a single-call diagnostic primitive that closes the production
# loop: km.diagnose inspects a trained model and emits an auto-dashboard
# (loss curves, gradient flow, dead neurons, activation stats, weight
# distributions). One cell. Every diagnostic students would otherwise
# hand-roll, ready to surface in a Plotly dashboard.

from kailash_ml import diagnose

# Pick the VAE (the lesson's most expressive variant) for the close.
# `kind='auto'` dispatches by model type — DLDiagnostics for torch.nn.Module.
# `data=` accepts any iterable yielding tensors; we reuse the flat_loader
# the lesson already constructed.
report = diagnose(all_models["vae"], kind="auto", data=flat_loader, show=False)
report.plot_training_dashboard()
print()
print("km.diagnose: 1 line of code -> the same observability the lesson")
print("body hand-rolled in 200+ lines. This is what 'destination-first'")
print("means — when the journey is internalised, the SDK is one call.")


# ════════════════════════════════════════════════════════════════════════
# REFLECTION
# ════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  WHAT YOU'VE MASTERED — THE COMPLETE AUTOENCODER FAMILY")
print("=" * 70)
print(
    """
  PROBLEM SOLVERS:
  [x] Standard AE: demonstrated the identity-function risk
  [x] Undercomplete AE: bottleneck forces 49:1 compression
  [x] Denoising AE: noise injection -> robust features
  [x] Sparse AE: L1 penalty -> specialist neurons

  ARCHITECTURE VARIANTS:
  [x] Contractive AE: Jacobian penalty -> smooth latent space
  [x] Convolutional AE: Conv2d -> sharpest image reconstructions
  [x] Stacked AE: depth -> hierarchical feature learning

  DIFFERENT MODALITIES:
  [x] Recurrent AE: LSTM -> temporal pattern capture

  GENERATIVE MODELS:
  [x] VAE: reparameterisation trick -> sample new data from N(0,I)
  [x] Contractive VAE: smooth + generative = best of both worlds

  ML ENGINEERING:
  [x] Tracked all 10 variants with ExperimentTracker
  [x] Registered all 10 models in ModelRegistry
  [x] Compared training curves with ModelVisualizer

  KEY INSIGHT: Every variant solves a specific failure mode:
  - Identity risk -> undercomplete bottleneck or regularisation
  - Memorisation -> noise injection (denoising) or sparsity
  - Sensitivity -> contractive penalty (Jacobian smoothness)
  - Generation need -> probabilistic latent space (VAE)
  - Spatial data -> convolutional architecture
  - Sequential data -> recurrent architecture

  Choosing the right variant is an engineering decision driven by
  your data type, use case, and failure tolerance.

  Next: Exercise 2 — CNNs with ResNet skip connections and SE
  attention for image classification...
"""
)
