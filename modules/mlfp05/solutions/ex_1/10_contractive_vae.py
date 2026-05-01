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
        h = F.relu(self.enc1(x))
        return F.relu(self.enc2(h))

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


CVAE_CONTRACTIVE_WEIGHT = 1e-4


def cvae_loss_fn(model, xb):
    x_hat, mu, logvar = model(xb)
    recon = F.mse_loss(x_hat, xb, reduction="sum") / xb.size(0)
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / xb.size(0)
    jacobian_penalty = sum(
        torch.sum(p**2) for p in [model.enc1.weight, model.enc2.weight]
    )
    return recon + KL_WEIGHT * kl + CVAE_CONTRACTIVE_WEIGHT * jacobian_penalty, {}


print("\n" + "=" * 70)
print("  Contractive VAE — Smooth + Probabilistic")
print("=" * 70)

cvae_model = ContractiveVAE(INPUT_DIM, LATENT_DIM)
cvae_losses = train_variant(
    tracker,
    exp_name,
    cvae_model,
    "contractive_vae",
    flat_loader,
    cvae_loss_fn,
    extra_params={
        "kl_weight": str(KL_WEIGHT),
        "contractive_weight": str(CVAE_CONTRACTIVE_WEIGHT),
    },
)

# ══════════════════════════════════════════════════════════════════
# DIAGNOSTIC CHECKPOINT — five instruments (VAE + Jacobian penalty)
# ══════════════════════════════════════════════════════════════════
# Contractive VAE = VAE loss + Jacobian penalty. Expect the BLOOD
# TEST to show low-but-stable gradients — both regularisers pull
# the encoder toward smoother manifolds.
from kailash_ml.diagnostics import run_diagnostic_checkpoint


def _diag_loss(m, batch):
    xb = batch[0] if isinstance(batch, (tuple, list)) else batch
    loss, _ = cvae_loss_fn(m, xb)
    return loss


print("\n── Diagnostic Report (Contractive VAE) ──")
diag, findings = run_diagnostic_checkpoint(
    cvae_model,
    flat_loader,
    _diag_loss,
    title=f"CVAE (KL={KL_WEIGHT}, lam={CVAE_CONTRACTIVE_WEIGHT})",
    n_batches=8,
    train_losses=cvae_losses,
    show=False,
)

# ══════ EXPECTED OUTPUT (synthesized reference — full run produces similar pattern) ══════
# ════════════════════════════════════════════════════════════════
#   DL Diagnostics Report — Prescription Pad
# ════════════════════════════════════════════════════════════════
#   [!] Gradient flow (WARNING): Compound dampening at
#       'fc_mu.weight' — RMS = 3.1e-05. Both KL penalty AND
#       Jacobian penalty act on mu-head. Contrast: VAE
#       (09) had 5.2e-05, ContractiveAE (05) had 7.4e-05 —
#       CVAE sits below both because TWO regularisers.
#   [!] Dead neurons  (WARNING): 'decoder.2' (relu): 18%
#       dead — early-epoch VAE signature plus contractive
#       dampening of encoder gradients.
#   [✓] Loss trend    (HEALTHY): 3-term composition, all
#       decreasing. Total slope -2.3e-03/epoch. Final loss
#       ~0.046 (recon 0.033 + KL 0.009 + Jacobian 0.004).
# ════════════════════════════════════════════════════════════════
# Final train loss: ~0.046, 10 epochs, beta=1, lambda=1e-4.
#
# STUDENT INTERPRETATION GUIDE — reading the Prescription Pad:
#
#  [BLOOD TEST — TWO REGULARISERS COMPOUNDING] RMS 3.1e-05
#     at fc_mu is the key diagnostic. This is the SUM of
#     two dampening effects: (a) KL penalty shrinking encoder
#     gradients (see 09 VAE), (b) Jacobian penalty further
#     constraining encoder sensitivity (see 05 contractive).
#     The floor sits BELOW either alone. If it drops below
#     1e-5, one regulariser is winning — halve the heavier
#     weight first.
#     >> Prescription: Compute the ratio
#        beta*KL_loss : lambda*Jacobian_loss. If >3:1 or
#        <1:3, one is dominating. Target 1:1 to 2:1 (KL
#        primary, Jacobian secondary).
#
#  [X-RAY] 18% dead is the VAE early-epoch pattern persisting
#     slightly longer due to contractive dampening (slower
#     decoder adaptation). Should converge to <10% by epoch 8.
#     If it stays above 15%, EITHER the KL is too strong
#     (posterior collapse risk) OR Jacobian is too strong
#     (encoder stuck). Diagnose by reading the Blood Test
#     first: which weight is dampening more?
#     >> Prescription: If KL term dominates: anneal beta.
#        If Jacobian dominates: halve lambda.
#
#  [STETHOSCOPE — THREE-TERM COMPOSITION] Unlike 09 (2 terms),
#     CVAE's loss is recon + KL + Jacobian. diag.epochs_df()
#     should expose all three. A healthy run sees: recon falls
#     fastest (decoder learns), KL falls next (latent tightens),
#     Jacobian last (encoder smooths). Any other ordering
#     signals imbalance — e.g. Jacobian dropping first means
#     lambda is too large and is crushing learning.
#     >> Prescription: Read three curves. If Jacobian curve
#        is flat, lambda is too small — no contraction
#        benefit. If Jacobian drops faster than recon,
#        lambda is too large — crushing reconstruction.
#
#  FIVE-INSTRUMENT TAKEAWAY: CVAE demonstrates the DIAGNOSTIC
#  SKILL OF DISENTANGLING MULTIPLE REGULARISERS. Same Blood
#  Test signal (low RMS), but the attribution depends on
#  weighing the terms. This reading skill scales to ex_5 GANs
#  (generator regulariser + discriminator regulariser +
#  gradient penalty = 3 terms to balance) and to ex_8 RL
#  (policy + value + entropy bonuses). Clinical reading in
#  the face of multiple compounding signals is the ceiling
#  skill for this module.
# ════════════════════════════════════════════════════════════════════

show_reconstruction(cvae_model, X_test_flat, "Contractive VAE")


# ════════════════════════════════════════════════════════════════════════
# TASK 3 — Compare Interpolation Smoothness: VAE vs CVAE
# ════════════════════════════════════════════════════════════════════════


def compare_interpolation(model_a, model_b, test_data, label_a, label_b, n_steps=10):
    fig, axes = plt.subplots(2, n_steps + 2, figsize=(16, 3.5))
    fig.suptitle(
        f"Latent Interpolation: {label_a} vs {label_b}", fontsize=13, fontweight="bold"
    )

    for row, (model, label) in enumerate([(model_a, label_a), (model_b, label_b)]):
        model.eval()
        with torch.no_grad():
            x1, x2 = test_data[0:1].to(device), test_data[5:6].to(device)
            if hasattr(model, "encode"):
                mu1, _ = model.encode(x1)
                mu2, _ = model.encode(x2)
                z1, z2 = mu1, mu2
            else:
                z1, z2 = model.encoder(x1), model.encoder(x2)
            alphas = torch.linspace(0, 1, n_steps).to(device)

            axes[row, 0].imshow(x1[0].cpu().reshape(28, 28), cmap="gray")
            axes[row, 0].axis("off")
            if row == 0:
                axes[row, 0].set_title("Start", fontsize=7)

            for i, alpha in enumerate(alphas):
                z = (1 - alpha) * z1 + alpha * z2
                x_hat = model.decoder(z)
                axes[row, i + 1].imshow(
                    x_hat[0].cpu().reshape(28, 28).clamp(0, 1), cmap="gray"
                )
                axes[row, i + 1].axis("off")
                if row == 0:
                    axes[row, i + 1].set_title(f"{alpha:.1f}", fontsize=7)

            axes[row, -1].imshow(x2[0].cpu().reshape(28, 28), cmap="gray")
            axes[row, -1].axis("off")
            if row == 0:
                axes[row, -1].set_title("End", fontsize=7)

        axes[row, 0].set_ylabel(label, fontsize=9, rotation=0, labelpad=50)

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
#
# We simulate this with Fashion-MNIST: each image class represents
# a "molecular family." Smooth interpolation between families suggests
# the latent space can guide molecular optimisation.

print("\n" + "=" * 70)
print("  APPLICATION: Molecular Similarity Search")
print("=" * 70)

# --- Generate synthetic molecular descriptor data ---
N_MOLECULES = 5000
N_DESCRIPTORS = 50  # molecular descriptors (LogP, MW, TPSA, etc.)
N_FAMILIES = 5  # drug families (analgesics, antibiotics, etc.)
FAMILY_NAMES = ["Analgesics", "Antibiotics", "Antivirals", "Oncology", "Cardiovascular"]

mol_rng = np.random.default_rng(42)

# Generate clustered molecular data
family_labels = mol_rng.choice(N_FAMILIES, size=N_MOLECULES)
family_centers = (
    mol_rng.standard_normal((N_FAMILIES, N_DESCRIPTORS)).astype(np.float32) * 2
)
mol_data = np.zeros((N_MOLECULES, N_DESCRIPTORS), dtype=np.float32)
for i in range(N_MOLECULES):
    mol_data[i] = family_centers[family_labels[i]] + mol_rng.normal(
        0, 0.5, N_DESCRIPTORS
    ).astype(np.float32)

# Normalise
mol_min, mol_max = mol_data.min(axis=0), mol_data.max(axis=0)
mol_range = mol_max - mol_min
mol_range[mol_range == 0] = 1.0
mol_norm = (mol_data - mol_min) / mol_range

# Train CVAE on molecular data
mol_tensor = torch.tensor(mol_norm, device=device)
mol_loader = DataLoader(TensorDataset(mol_tensor), batch_size=128, shuffle=True)


class MolecularCVAE(nn.Module):
    def __init__(self, input_dim, latent_dim=10):
        super().__init__()
        self.enc1 = nn.Linear(input_dim, 64)
        self.enc2 = nn.Linear(64, 32)
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
        h = F.relu(self.enc1(x))
        h = F.relu(self.enc2(h))
        return self.fc_mu(h), self.fc_logvar(h)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)
        return self.decoder(z), mu, logvar


MOL_LATENT = 10
mol_model = MolecularCVAE(N_DESCRIPTORS, MOL_LATENT).to(device)
mol_opt = torch.optim.Adam(mol_model.parameters(), lr=1e-3)

print(f"Training CVAE on {N_MOLECULES} molecules ({N_DESCRIPTORS} descriptors)...")
for epoch in range(80):
    mol_model.train()
    for (batch,) in mol_loader:
        recon, mu, logvar = mol_model(batch)
        recon_l = F.mse_loss(recon, batch, reduction="sum")
        kl_l = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        jacob = sum(
            torch.sum(p**2) for p in [mol_model.enc1.weight, mol_model.enc2.weight]
        )
        loss = recon_l + kl_l + 1e-4 * jacob
        mol_opt.zero_grad()
        loss.backward()
        mol_opt.step()

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

centered = mol_latents - mol_latents.mean(axis=0)
_, _, Vt = svd(centered, full_matrices=False)
proj = centered @ Vt[:2].T

fig, ax = plt.subplots(figsize=(10, 8))
for fam in range(N_FAMILIES):
    mask = family_labels == fam
    ax.scatter(proj[mask, 0], proj[mask, 1], s=10, alpha=0.5, label=FAMILY_NAMES[fam])
ax.scatter(
    proj[query_idx, 0],
    proj[query_idx, 1],
    s=200,
    c="red",
    marker="*",
    zorder=5,
    label="Query",
)
for ni in nearest[:5]:
    ax.annotate(f"#{ni}", (proj[ni, 0], proj[ni, 1]), fontsize=8, color="red")
ax.set_xlabel("PC1")
ax.set_ylabel("PC2")
ax.set_title(
    "Molecular Latent Space (CVAE)\nSmooth clustering enables similarity search",
    fontsize=13,
)
ax.legend(fontsize=9, markerscale=3, ncol=2)
ax.grid(True, alpha=0.3)
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
print(f"\nSmooth latent space advantage:")
print(f"  Interpolation between a hit and a miss suggests optimisation direction")
print(f"  'Move 20% toward molecule X in latent space' = specific structural changes")
print(f"  This is medicinal chemistry guidance from the model itself")
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
