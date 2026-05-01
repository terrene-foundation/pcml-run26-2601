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


class StandardAE(nn.Module):
    def __init__(self, input_dim, hidden_dim=1024):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid(),
        )

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z), z


class UndercompleteAE(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 256),
            nn.ReLU(),
            nn.Linear(256, input_dim),
            nn.Sigmoid(),
        )

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z), z


class DenoisingAE(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 256),
            nn.ReLU(),
            nn.Linear(256, input_dim),
            nn.Sigmoid(),
        )

    def add_noise(self, x, sigma=0.3):
        return torch.clamp(x + sigma * torch.randn_like(x), 0.0, 1.0)

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z), z


class SparseAE(nn.Module):
    def __init__(self, input_dim, hidden_dim=256):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
        )
        self.decoder = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid(),
        )

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z), z


class ContractiveAE(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        self.enc1 = nn.Linear(input_dim, 256)
        self.enc2 = nn.Linear(256, 64)
        self.enc3 = nn.Linear(64, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 256),
            nn.ReLU(),
            nn.Linear(256, input_dim),
            nn.Sigmoid(),
        )

    def encoder(self, x):
        return self.enc3(F.relu(self.enc2(F.relu(self.enc1(x)))))

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z), z


class ConvAE(nn.Module):
    def __init__(self, latent_dim=16):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 32 * 7 * 7),
            nn.ReLU(),
            nn.Unflatten(1, (32, 7, 7)),
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, 3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z), z


class StackedAE(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, input_dim),
            nn.Sigmoid(),
        )

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z), z


class RecurrentAE(nn.Module):
    def __init__(self, seq_len, hidden_dim=64, latent_dim=16):
        super().__init__()
        self.seq_len = seq_len
        self.encoder_lstm = nn.LSTM(
            input_size=1, hidden_size=hidden_dim, batch_first=True
        )
        self.enc_to_latent = nn.Linear(hidden_dim, latent_dim)
        self.latent_to_dec = nn.Linear(latent_dim, hidden_dim)
        self.decoder_lstm = nn.LSTM(
            input_size=hidden_dim, hidden_size=hidden_dim, batch_first=True
        )
        self.output_layer = nn.Sequential(nn.Linear(hidden_dim, 1), nn.Sigmoid())

    def forward(self, x):
        x_seq = x.unsqueeze(-1)
        _, (h_n, _) = self.encoder_lstm(x_seq)
        z = self.enc_to_latent(h_n.squeeze(0))
        dec_input = self.latent_to_dec(z).unsqueeze(1).repeat(1, self.seq_len, 1)
        dec_output, _ = self.decoder_lstm(dec_input)
        return self.output_layer(dec_output).squeeze(-1), z


class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256), nn.ReLU(), nn.Linear(256, 128), nn.ReLU()
        )
        self.fc_mu = nn.Linear(128, latent_dim)
        self.fc_logvar = nn.Linear(128, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, input_dim),
            nn.Sigmoid(),
        )

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
        self.enc1 = nn.Linear(input_dim, 256)
        self.enc2 = nn.Linear(256, 128)
        self.fc_mu = nn.Linear(128, latent_dim)
        self.fc_logvar = nn.Linear(128, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, input_dim),
            nn.Sigmoid(),
        )

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


def std_loss(m, xb):
    x_hat, _ = m(xb)
    return F.mse_loss(x_hat, xb), {}


def dae_loss(m, xb):
    noisy = m.add_noise(xb, NOISE_SIGMA)
    x_hat, _ = m(noisy)
    return F.mse_loss(x_hat, xb), {}


def sparse_loss(m, xb):
    x_hat, z = m(xb)
    return F.mse_loss(x_hat, xb) + SPARSITY_WEIGHT * torch.mean(torch.abs(z)), {}


def cae_loss(m, xb):
    x_hat, z = m(xb)
    return (
        F.mse_loss(x_hat, xb)
        + CONTRACTIVE_WEIGHT
        * sum(torch.sum(p**2) for p in [m.enc1.weight, m.enc2.weight, m.enc3.weight]),
        {},
    )


def conv_loss(m, xb):
    x_hat, _ = m(xb)
    return F.mse_loss(x_hat, xb), {}


def rec_loss(m, xb):
    x_hat, _ = m(xb)
    return F.mse_loss(x_hat, xb), {}


def vae_loss(m, xb):
    x_hat, mu, lv = m(xb)
    r = F.mse_loss(x_hat, xb, reduction="sum") / xb.size(0)
    kl = -0.5 * torch.sum(1 + lv - mu.pow(2) - lv.exp()) / xb.size(0)
    return r + KL_WEIGHT * kl, {}


def cvae_loss(m, xb):
    x_hat, mu, lv = m(xb)
    r = F.mse_loss(x_hat, xb, reduction="sum") / xb.size(0)
    kl = -0.5 * torch.sum(1 + lv - mu.pow(2) - lv.exp()) / xb.size(0)
    j = sum(torch.sum(p**2) for p in [m.enc1.weight, m.enc2.weight])
    return r + KL_WEIGHT * kl + CVAE_CONTRACTIVE_WEIGHT * j, {}


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

fig, axes = plt.subplots(
    n_variants + 1, N_COMPARE, figsize=(16, (n_variants + 1) * 1.5)
)
fig.suptitle(
    "Grand Comparison: Original + 9 Image AE Variants",
    fontsize=15,
    fontweight="bold",
    y=1.01,
)

for j in range(N_COMPARE):
    axes[0, j].imshow(X_test_img[j].cpu().squeeze(), cmap="gray")
    axes[0, j].axis("off")
axes[0, 0].set_ylabel("Original", fontsize=9, rotation=0, labelpad=55)

for row, (name, model, is_conv) in enumerate(image_variants, start=1):
    model.eval()
    with torch.no_grad():
        x = (
            X_test_img[:N_COMPARE].to(device)
            if is_conv
            else X_test_flat[:N_COMPARE].to(device)
        )
        result = model(x)
        x_hat = result[0]
    for j in range(N_COMPARE):
        img = x_hat[j].cpu().squeeze() if is_conv else x_hat[j].cpu().reshape(28, 28)
        axes[row, j].imshow(img.clamp(0, 1), cmap="gray")
        axes[row, j].axis("off")
    axes[row, 0].set_ylabel(name, fontsize=8, rotation=0, labelpad=60)

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
viz = ModelVisualizer()
fig_all = viz.training_history(
    metrics={name: all_losses[name_to_key[name]] for name, _, _ in image_variants},
    x_label="Epoch",
    y_label="Loss (MSE)",
)
fig_all.write_html(str(OUTPUT_DIR / "ex1_all_variants_training_curves.html"))
print(
    f"\n  Training curves saved to {OUTPUT_DIR / 'ex1_all_variants_training_curves.html'}"
)


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
