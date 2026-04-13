# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# MLFP05 — Exercise 1.10: Contractive VAE (Smooth + Probabilistic)
# ════════════════════════════════════════════════════════════════════════
#
# WHAT YOU'LL LEARN:
#   - Combine VAE's probabilistic latent space with Contractive AE's smoothness
#   - Compare latent interpolation: vanilla VAE vs Contractive VAE
#   - Understand WHY smooth + generative is the best of both worlds
#   - Apply to drug molecule similarity search at a Singapore biotech
#   - Build a latent-space nearest-neighbour system for molecular similarity
#
# PREREQUISITES: 09_vae.py
# ESTIMATED TIME: ~20 min
#
# TASKS:
#   1. Build Contractive VAE (VAE + Jacobian weight penalty)
#   2. Train and compare interpolation smoothness vs vanilla VAE
#   3. Visualise side-by-side interpolation comparison
#   4. Apply: molecular feature similarity search
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
from scipy.spatial.distance import cdist

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
    register_model,
)


# ════════════════════════════════════════════════════════════════════════
# THEORY — Smooth + Generative = Best of Both Worlds
# ════════════════════════════════════════════════════════════════════════
# The Contractive VAE combines:
#   - VAE's probabilistic latent space (can sample new data)
#   - Contractive AE's Jacobian penalty (smooth transitions)
#
# The result: a latent space where you can BOTH generate new samples
# AND navigate smoothly between them. Small steps in latent space
# produce small, meaningful changes in output.
#
# WHY THIS MATTERS: In drug discovery, you want to explore the space
# of possible molecules smoothly. A drug that works for condition A
# might have a close neighbour in latent space that works for condition
# B. The smooth latent space means "close in latent space" reliably
# means "structurally similar as molecules" — no sharp discontinuities
# where a tiny latent change produces a completely different molecule.


# ════════════════════════════════════════════════════════════════════════
# TASK 1 — Load data, engines, and train vanilla VAE for comparison
# ════════════════════════════════════════════════════════════════════════

X_flat, X_test_flat, X_img, X_test_img, flat_loader, img_loader = load_fashion_mnist()
conn, tracker, exp_name, registry, has_registry = setup_engines()


# --- Vanilla VAE (for comparison) ---
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

    def reparameterise(self, mu, logvar):
        return mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterise(mu, logvar)
        return self.decoder(z), mu, logvar

    def sample(self, n):
        z = torch.randn(
            n, self.fc_mu.out_features, device=next(self.parameters()).device
        )
        return self.decoder(z)


KL_WEIGHT = 0.1


def vae_loss_fn(model, xb):
    x_hat, mu, logvar = model(xb)
    recon = F.mse_loss(x_hat, xb, reduction="sum") / xb.size(0)
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / xb.size(0)
    return recon + KL_WEIGHT * kl, {}


print("Training vanilla VAE for comparison...")
vae_model = VAE(INPUT_DIM, LATENT_DIM)
vae_losses = train_variant(
    tracker,
    exp_name,
    vae_model,
    "vae_compare",
    flat_loader,
    vae_loss_fn,
    extra_params={"kl_weight": str(KL_WEIGHT)},
)


# ════════════════════════════════════════════════════════════════════════
# TASK 2 — Build and Train Contractive VAE
# ════════════════════════════════════════════════════════════════════════


class ContractiveVAE(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        # TODO: Define explicit encoder layers (for weight access):
        #       enc1: Linear(input_dim, 256)
        #       enc2: Linear(256, 128)
        #       fc_mu: Linear(128, latent_dim)
        #       fc_logvar: Linear(128, latent_dim)
        self.enc1 = ____
        self.enc2 = ____
        self.fc_mu = ____
        self.fc_logvar = ____

        # TODO: Build decoder — same as vanilla VAE
        self.decoder = ____

    def encoder(self, x):
        # TODO: Forward through enc1->ReLU->enc2->ReLU
        ____

    def encode(self, x):
        # TODO: encoder(x) -> (mu, logvar)
        ____

    def reparameterise(self, mu, logvar):
        # TODO: Reparameterisation trick
        ____

    def forward(self, x):
        # TODO: encode -> reparameterise -> decode
        # Return (reconstruction, mu, logvar)
        ____

    def sample(self, n):
        # TODO: Sample z ~ N(0,I), decode
        ____


CVAE_CONTRACTIVE_WEIGHT = 1e-4


def cvae_loss_fn(model, xb):
    # TODO: ELBO loss + Jacobian penalty on enc1.weight and enc2.weight
    # recon + KL_WEIGHT * kl + CVAE_CONTRACTIVE_WEIGHT * jacobian_penalty
    ____


print("\n" + "=" * 70)
print("  Contractive VAE — Smooth + Probabilistic")
print("=" * 70)

# TODO: Create ContractiveVAE(INPUT_DIM, LATENT_DIM) and train
cvae_model = ____
cvae_losses = ____

# TODO: show_reconstruction
____


# ════════════════════════════════════════════════════════════════════════
# TASK 3 — Compare Interpolation Smoothness: VAE vs CVAE
# ════════════════════════════════════════════════════════════════════════


def compare_interpolation(model_a, model_b, test_data, label_a, label_b, n_steps=10):
    # TODO: Create 2-row interpolation comparison figure
    # Row 0: model_a interpolation from image 0 to image 5
    # Row 1: model_b interpolation from image 0 to image 5
    # For each model: encode start/end -> interpolate in latent space -> decode
    # Save to OUTPUT_DIR / "ex1_vae_vs_cvae_interpolation.png"
    fig, axes = plt.subplots(2, n_steps + 2, figsize=(16, 3.5))
    ____
    plt.tight_layout()
    fname = OUTPUT_DIR / "ex1_vae_vs_cvae_interpolation.png"
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"  Saved: {fname}")


compare_interpolation(
    vae_model, cvae_model, X_test_flat, "Vanilla VAE", "Contractive VAE"
)

# ── Checkpoint ──────────────────────────────────────────────────────
assert len(cvae_losses) == EPOCHS
assert cvae_losses[-1] < cvae_losses[0]
print("\n--- Checkpoint passed --- contractive VAE trained\n")

if has_registry:
    register_model(registry, "contractive_vae", cvae_model, cvae_losses[-1])


# ════════════════════════════════════════════════════════════════════════
# APPLY — Drug Molecule Similarity Search
# ════════════════════════════════════════════════════════════════════════
# BUSINESS SCENARIO: You are an ML engineer at a Singapore biotech
# company (A*STAR spinoff). Drug discovery involves exploring vast
# molecular spaces. Given a lead compound that shows promise, you
# want to find structurally similar molecules that might have improved
# properties. The CVAE's smooth latent space means "nearby in latent
# space" = "structurally similar as molecules."

print("\n" + "=" * 70)
print("  APPLICATION: Molecular Similarity Search")
print("=" * 70)

# --- Generate synthetic molecular descriptor data ---
N_MOLECULES = 5000
N_DESCRIPTORS = 50
N_FAMILIES = 5
FAMILY_NAMES = ["Analgesics", "Antibiotics", "Antivirals", "Oncology", "Cardiovascular"]

mol_rng = np.random.default_rng(42)

# TODO: Generate clustered molecular data
# family_labels = random assignment to N_FAMILIES
# family_centers = random centers in N_DESCRIPTORS-dim space
# mol_data[i] = family_centers[label[i]] + noise
# Normalise to [0, 1]
family_labels = ____
family_centers = ____
mol_data = ____

mol_min, mol_max = mol_data.min(axis=0), mol_data.max(axis=0)
mol_range = mol_max - mol_min
mol_range[mol_range == 0] = 1.0
mol_norm = (mol_data - mol_min) / mol_range

mol_tensor = torch.tensor(mol_norm, device=device)
mol_loader = DataLoader(TensorDataset(mol_tensor), batch_size=128, shuffle=True)


class MolecularCVAE(nn.Module):
    def __init__(self, input_dim, latent_dim=10):
        super().__init__()
        # TODO: enc1: Linear(input_dim, 64)
        #       enc2: Linear(64, 32)
        #       fc_mu: Linear(32, latent_dim)
        #       fc_logvar: Linear(32, latent_dim)
        #       decoder: Linear(latent_dim, 32), ReLU, Linear(32, 64), ReLU,
        #                Linear(64, input_dim), Sigmoid
        self.enc1 = ____
        self.enc2 = ____
        self.fc_mu = ____
        self.fc_logvar = ____
        self.decoder = ____

    def encode(self, x):
        # TODO: enc1->ReLU->enc2->ReLU->(mu, logvar)
        ____

    def forward(self, x):
        # TODO: encode -> reparameterise -> decode
        ____


MOL_LATENT = 10
# TODO: Create MolecularCVAE, optimizer. Train 80 epochs with ELBO + Jacobian loss.
mol_model = ____
mol_opt = ____

print(f"Training CVAE on {N_MOLECULES} molecules ({N_DESCRIPTORS} descriptors)...")
for epoch in range(80):
    mol_model.train()
    for (batch,) in mol_loader:
        # TODO: Forward, compute ELBO + contractive loss, backprop
        ____

# --- Encode all molecules ---
mol_model.eval()
with torch.no_grad():
    all_mu, _ = mol_model.encode(mol_tensor)
    mol_latents = all_mu.cpu().numpy()

# --- Similarity search ---
query_idx = 42
query_latent = mol_latents[query_idx : query_idx + 1]
dists = cdist(query_latent, mol_latents, metric="euclidean")[0]
dists[query_idx] = float("inf")
top_k = 10
nearest = np.argsort(dists)[:top_k]

query_family = family_labels[query_idx]
retrieved_families = family_labels[nearest]
same_family_pct = (retrieved_families == query_family).mean() * 100

print(f"\nQuery molecule #{query_idx} (family: {FAMILY_NAMES[query_family]})")
print(f"Top-{top_k} nearest neighbours:")
for rank, ni in enumerate(nearest):
    fam = FAMILY_NAMES[family_labels[ni]]
    match = "SAME" if family_labels[ni] == query_family else "diff"
    print(f"  #{rank+1}: molecule {ni}, family={fam}, dist={dists[ni]:.3f} [{match}]")
print(f"Same-family retrieval: {same_family_pct:.0f}%")

# --- Visualise latent space (PCA) ---
from numpy.linalg import svd

# TODO: PCA to 2D, scatter plot coloured by drug family
# Save to OUTPUT_DIR / "ex1_molecular_latent_space.png"
centered = mol_latents - mol_latents.mean(axis=0)
_, _, Vt = svd(centered, full_matrices=False)
proj = centered @ Vt[:2].T

fig, ax = plt.subplots(figsize=(10, 8))
____
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "ex1_molecular_latent_space.png", dpi=150, bbox_inches="tight")
plt.show()

# --- Business Impact ---
print("\n" + "=" * 64)
print("BUSINESS IMPACT SUMMARY — Drug Molecule Similarity Search")
print("=" * 64)
print(f"\nMolecular library: {N_MOLECULES:,} compounds, {N_DESCRIPTORS} descriptors")
print(f"CVAE latent dimension: {MOL_LATENT}")
print(f"Same-family retrieval rate: {same_family_pct:.0f}% (top-{top_k})")
print(f"\nDrug discovery impact:")
print(f"  Traditional screening: test 10,000 compounds at S$100 each = S$1M")
print(f"  CVAE-guided search: prioritise top-100 neighbours first")
print(f"  Expected hit rate improvement: ~3-5x (from random screening)")
print(f"  Cost savings per drug programme: S$200K-400K in early screening")
print("=" * 64)


# ════════════════════════════════════════════════════════════════════════
# REFLECTION
# ════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  WHAT YOU'VE MASTERED")
print("=" * 70)
print(
    """
  [x] Built a Contractive VAE (VAE + Jacobian weight penalty)
  [x] Compared interpolation smoothness: vanilla VAE vs CVAE
  [x] Observed smoother transitions in CVAE's latent space
  [x] Applied to molecular similarity search in drug discovery
  [x] Built latent-space nearest-neighbour retrieval for molecules
  [x] Quantified drug screening cost savings

  KEY INSIGHT: The CVAE is the best of both worlds. The VAE component
  gives you a generative model (sample new molecules). The contractive
  component gives you a smooth latent space (nearby molecules are truly
  similar). Together, they enable both GENERATION (propose new drug
  candidates) and NAVIGATION (find similar molecules, interpolate
  between hits and misses).

  Next: 11_grand_comparison.py puts all 10 variants side by side...
"""
)
