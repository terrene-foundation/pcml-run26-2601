# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# MLFP05 — Exercise 1.7: Stacked Autoencoder (Deep Feature Hierarchy)
# ════════════════════════════════════════════════════════════════════════
#
# WHAT YOU'LL LEARN:
#   - Build a deep stacked AE (5 encoder layers, 5 decoder layers)
#   - Understand WHY depth enables hierarchical feature learning
#   - Observe that depth alone is not magic — skip connections help
#   - Apply to image search/retrieval using learned latent features
#   - Build a nearest-neighbour retrieval system from Fashion-MNIST
#
# PREREQUISITES: 06_convolutional_ae.py
# ESTIMATED TIME: ~20 min
#
# TASKS:
#   1. Build Stacked AE: 784 -> 512 -> 256 -> 128 -> 64 -> 16
#   2. Train and compare to shallower undercomplete AE
#   3. Visualise feature hierarchy via layer activations
#   4. Apply: image retrieval using latent space nearest neighbours
#
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

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
    get_fashion_mnist_labels,
)


# ════════════════════════════════════════════════════════════════════════
# THEORY — Depth Enables Feature Hierarchy
# ════════════════════════════════════════════════════════════════════════
# Stack multiple autoencoder layers to learn a hierarchy of features:
# low-level (edges) -> mid-level (textures) -> high-level (shapes).
# Each layer learns to encode the previous layer's output.
#
# Analogy: In document processing, features exist at multiple levels.
# Character-level features detect handwriting quality; word-level
# features detect names and addresses; document-level features detect
# document type (passport vs utility bill). A stacked AE learns this
# hierarchy automatically.
#
# CAVEAT: Depth without skip connections can cause vanishing gradients.
# The value of stacking is in FEATURE HIERARCHY — the intermediate
# representations at each layer capture increasingly abstract features.


# ════════════════════════════════════════════════════════════════════════
# TASK 1 — Load data and engines
# ════════════════════════════════════════════════════════════════════════

X_flat, X_test_flat, X_img, X_test_img, flat_loader, img_loader = load_fashion_mnist()
conn, tracker, exp_name, registry, has_registry = setup_engines()


# ════════════════════════════════════════════════════════════════════════
# TASK 2 — Build and Train Stacked AE
# ════════════════════════════════════════════════════════════════════════


class StackedAE(nn.Module):
    """Deep encoder with 5 layers of progressive compression."""

    def __init__(self, input_dim: int, latent_dim: int):
        super().__init__()
        # TODO: Build encoder — nn.Sequential with 5 linear layers:
        #       784 -> 512 -> 256 -> 128 -> 64 -> latent_dim
        #       Use ReLU between each layer (not after last)
        self.encoder = ____

        # TODO: Build decoder — mirror of encoder:
        #       latent_dim -> 64 -> 128 -> 256 -> 512 -> input_dim
        #       ReLU between layers, Sigmoid at end
        self.decoder = ____

    def forward(self, x):
        # TODO: Encode then decode. Return (reconstruction, latent_code)
        ____


def stacked_ae_loss(model, xb):
    # TODO: Forward, MSE loss. Return (loss, {})
    ____


print("\n" + "=" * 70)
print("  Stacked AE — Deep Feature Hierarchy")
print("=" * 70)
print("  5 encoder layers: 784->512->256->128->64->16")

# TODO: Create StackedAE(INPUT_DIM, LATENT_DIM) and train
stacked_model = ____
stacked_losses = ____


# ════════════════════════════════════════════════════════════════════════
# TASK 3 — Visualise
# ════════════════════════════════════════════════════════════════════════

# TODO: show_reconstruction
____

# ── Checkpoint ──────────────────────────────────────────────────────
assert len(stacked_losses) == EPOCHS
assert stacked_losses[-1] < stacked_losses[0]
# INTERPRETATION: More layers does NOT automatically mean better
# reconstruction. Depth without skip connections can cause vanishing
# gradients. The value of stacking is in FEATURE HIERARCHY.
print("\n--- Checkpoint passed --- stacked AE trained\n")

if has_registry:
    register_model(registry, "stacked_ae", stacked_model, stacked_losses[-1])


# ════════════════════════════════════════════════════════════════════════
# APPLY — Image Search / Feature Extraction
# ════════════════════════════════════════════════════════════════════════
# BUSINESS SCENARIO: You are building a visual search engine for a
# Singapore fashion marketplace. Users upload a photo of clothing
# and want to find "similar items" from the catalogue. The stacked
# AE's latent space provides a 16-dimensional feature vector per
# image — use nearest-neighbour search to find similar items.
#
# The depth of the stacked AE means the 16-dim vector captures
# HIERARCHICAL features: Layer 1 encodes edges, Layer 3 encodes
# shapes, Layer 5 encodes category-level semantics. This multi-level
# encoding makes the similarity search more meaningful than pixel
# comparison or shallow features.

print("\n" + "=" * 70)
print("  APPLICATION: Image Search via Latent Space Retrieval")
print("=" * 70)

# --- Encode all test images to latent space ---
stacked_model.eval()
with torch.no_grad():
    test_latents = stacked_model.encoder(X_test_flat.to(device)).cpu().numpy()

print(f"Encoded {len(test_latents)} images to {test_latents.shape[1]}-dim latent space")

# --- Get labels for evaluation ---
_, test_labels = get_fashion_mnist_labels()
test_labels_np = test_labels.numpy()

CLASS_NAMES = [
    "T-shirt",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle Boot",
]

# --- Nearest-neighbour retrieval ---
from scipy.spatial.distance import cdist

query_indices = [0, 100, 200, 500, 1000, 2000, 3000, 5000, 7000, 9000]
n_retrieve = 8

# TODO: For each query, compute distances to all test images,
# find top-8 nearest neighbours, compute same-class precision.
# Create retrieval grid: query image + 8 retrieved images per row.
# Green border = same class, red border = different class.
# Save to OUTPUT_DIR / "ex1_image_retrieval.png"
fig, axes = plt.subplots(
    len(query_indices), n_retrieve + 1, figsize=(18, 2 * len(query_indices))
)

retrieval_precisions = []
for row, qi in enumerate(query_indices):
    # TODO: Compute Euclidean distances from query to all test latents
    # Exclude self (set dist[qi] = inf)
    # Find nearest n_retrieve indices
    # Compute precision (fraction of retrieved with same label as query)
    ____

mean_precision = np.mean(retrieval_precisions)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "ex1_image_retrieval.png", dpi=150, bbox_inches="tight")
plt.show()
print(f"\nRetrieval precision@{n_retrieve}: {mean_precision:.1%}")

# --- Latent space clustering visualisation (PCA to 2D) ---
from numpy.linalg import svd

# TODO: PCA projection of 3000 sample latents to 2D
# Scatter plot coloured by class label
# Save to OUTPUT_DIR / "ex1_stacked_ae_latent_clusters.png"
sample_size = 3000
sample_idx = np.random.choice(len(test_latents), sample_size, replace=False)
sample_latents = test_latents[sample_idx]
sample_labels = test_labels_np[sample_idx]

centered = sample_latents - sample_latents.mean(axis=0)
_, _, Vt = svd(centered, full_matrices=False)
projected = centered @ Vt[:2].T

fig, ax = plt.subplots(figsize=(10, 8))
____
plt.tight_layout()
plt.savefig(
    OUTPUT_DIR / "ex1_stacked_ae_latent_clusters.png", dpi=150, bbox_inches="tight"
)
plt.show()

# --- Business Impact ---
print("\n" + "=" * 64)
print("BUSINESS IMPACT SUMMARY — Fashion Image Search Engine")
print("=" * 64)
print(f"\nRetrieval precision@{n_retrieve}: {mean_precision:.1%}")
print(
    f"Latent dimension: {LATENT_DIM} (vs 784 raw pixels = {784/LATENT_DIM:.0f}x smaller)"
)
print(f"\nFor a catalogue of 1M products:")
print(f"  Raw pixel search:     784 dims x 1M = 3.0 GB index")
print(
    f"  Latent space search:  {LATENT_DIM} dims x 1M = {LATENT_DIM * 4 / 1e6:.1f} MB index"
)
print(f"  Index size reduction: {784 / LATENT_DIM:.0f}x smaller")
print(
    f"  Search speed:         ~{784 / LATENT_DIM:.0f}x faster (fewer distance computations)"
)
print("=" * 64)


# ════════════════════════════════════════════════════════════════════════
# REFLECTION
# ════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  WHAT YOU'VE MASTERED")
print("=" * 70)
print(
    """
  [x] Built a 5-layer stacked autoencoder for deep feature hierarchy
  [x] Understood that depth enables multi-level abstraction
  [x] Applied latent features to image search/retrieval
  [x] Built nearest-neighbour retrieval with precision evaluation
  [x] Visualised latent space clustering (PCA 2D projection)
  [x] Quantified index size and search speed improvements

  KEY INSIGHT: Depth enables HIERARCHY. Layer 1 learns edges, Layer 3
  learns shapes, Layer 5 learns category semantics. The 16-dim latent
  vector is a compact summary that preserves this hierarchy. For image
  search, this means retrieving items that are semantically similar
  (same type of clothing) rather than just visually similar (same
  brightness distribution).

  Next: 08_recurrent_ae.py handles sequential/time-series data...
"""
)

# ══════════════════════════════════════════════════════════════════
# DIAGNOSTIC CHECKPOINT — five instruments (depth stress-test)
# ══════════════════════════════════════════════════════════════════
# 5-layer encoder = the DEEPEST dense AE in this exercise. Vanishing
# gradients are the primary risk at this depth without residuals.
# This is where the Blood Test earns its keep.
from kailash_ml.diagnostics import run_diagnostic_checkpoint


def _diag_loss(m, batch):
    xb = batch[0] if isinstance(batch, (tuple, list)) else batch
    loss, _ = stacked_ae_loss(m, xb)
    return loss


print("\n── Diagnostic Report (Stacked AE) ──")
diag, findings = run_diagnostic_checkpoint(
    stacked_model,
    flat_loader,
    _diag_loss,
    title="Stacked AE (5-layer)",
    n_batches=8,
    train_losses=stacked_losses,
    show=False,
)

# ══════ EXPECTED OUTPUT (synthesized reference — full run produces similar pattern) ══════
# ════════════════════════════════════════════════════════════════
#   DL Diagnostics Report — Prescription Pad
# ════════════════════════════════════════════════════════════════
#   [X] Gradient flow (CRITICAL): Vanishing gradients at
#       'encoder.3.weight' — min RMS = 4.2e-06, min
#       update_ratio = 8.9e-05. (Shallow layer encoder.0 RMS
#       ~3.1e-03 — ~750x spread across 5 layers.)
#       Fix: Kaiming init, GELU, or add skip/residual
#            connections between encoder blocks.
#   [!] Dead neurons  (WARNING): 'encoder.3' (relu): 44% dead.
#       Bottleneck ReLU saturation compounds vanishing
#       gradients — the two failure modes feed each other.
#   [✓] Loss trend    (HEALTHY): Loss converging
#       (train slope -1.1e-03/epoch) — slower than shallow
#       baseline (02) because deep layers are barely updating.
# ════════════════════════════════════════════════════════════════
# Final train loss: ~0.018 after 10 epochs, 5-layer encoder.
#
# STUDENT INTERPRETATION GUIDE — reading the Prescription Pad:
#
#  [BLOOD TEST — VANISHING GRADIENT TEXTBOOK CASE] 750x spread
#     in RMS across 5 layers (encoder.0: 3.1e-03 → encoder.3:
#     4.2e-06) is the canonical deep-dense-network failure.
#     Slide 5K walks through the math: each ReLU + Linear
#     layer multiplies the gradient by an expectation < 1,
#     and 5 layers gets you ~0.5^5 = ~3% of the shallow-layer
#     magnitude. Depth without skip connections is how AlexNet
#     barely worked and ResNet solved it.
#     >> Prescription: Three fixes of increasing impact:
#        (a) Kaiming He init (partial — slows the collapse)
#        (b) GELU or SiLU (fully non-saturating on negatives)
#        (c) Residual connections (FIXES it — gradient
#            bypasses depth via additive skip). Applied in
#            ex_2/02_resnet_se.py — forward reference.
#
#  [X-RAY] 44% dead at encoder.3 is the compounding failure:
#     if gradient is vanishing, dead ReLUs never recover
#     (the tiny gradient cannot re-activate them), and the
#     remaining live ReLUs overload to compensate, pushing
#     more of them into saturation next batch. Positive
#     feedback loop into deeper dysfunction.
#     >> Prescription: LeakyReLU (negative-slope leak lets
#        small gradients revive dead channels) OR a residual
#        block (additive skip bypasses the ReLU gate).
#
#  [STETHOSCOPE] Loss still DECREASES despite dysfunction —
#     this is the trap. Final loss 0.018 looks "fine" but
#     compare to 06_convolutional (~0.0048) at similar latent
#     size. The loss curve lies when the architecture is
#     pathological; only the Blood Test + X-Ray reveal that
#     most of the model isn't learning.
#     >> Prescription: ALWAYS read 3 instruments. The
#        Stethoscope alone would have shipped this model.
#
#  FIVE-INSTRUMENT TAKEAWAY: stacked AE is the "how NOT to go
#  deep" cautionary tale. The fix isn't more training or more
#  regularisation — it's ARCHITECTURAL (skip connections).
#  This motivates ex_2 ResNet-SE and foreshadows the same
#  pattern in ex_3 RNNs (where "skip" becomes LSTM/GRU gating).
# ════════════════════════════════════════════════════════════════════


# ════════════════════════════════════════════════════════════════════════
# TASK 3 — Visualise
# ════════════════════════════════════════════════════════════════════════

show_reconstruction(stacked_model, X_test_flat, "Stacked AE (5 Layers)")

# ── Checkpoint ──────────────────────────────────────────────────────
assert len(stacked_losses) == EPOCHS
assert stacked_losses[-1] < stacked_losses[0]
# INTERPRETATION: More layers does NOT automatically mean better
# reconstruction. Depth without skip connections can cause vanishing
# gradients. The value of stacking is in FEATURE HIERARCHY.
print("\n--- Checkpoint passed --- stacked AE trained\n")

if has_registry:
    register_model(registry, "stacked_ae", stacked_model, stacked_losses[-1])


# ════════════════════════════════════════════════════════════════════════
# APPLY — Image Search / Feature Extraction
# ════════════════════════════════════════════════════════════════════════
# BUSINESS SCENARIO: You are building a visual search engine for a
# Singapore fashion marketplace. Users upload a photo of clothing
# and want to find "similar items" from the catalogue. The stacked
# AE's latent space provides a 16-dimensional feature vector per
# image — use nearest-neighbour search to find similar items.
#
# The depth of the stacked AE means the 16-dim vector captures
# HIERARCHICAL features: Layer 1 encodes edges, Layer 3 encodes
# shapes, Layer 5 encodes category-level semantics. This multi-level
# encoding makes the similarity search more meaningful than pixel
# comparison or shallow features.

print("\n" + "=" * 70)
print("  APPLICATION: Image Search via Latent Space Retrieval")
print("=" * 70)

# --- Encode all test images to latent space ---
stacked_model.eval()
with torch.no_grad():
    test_latents = stacked_model.encoder(X_test_flat.to(device)).cpu().numpy()

print(f"Encoded {len(test_latents)} images to {test_latents.shape[1]}-dim latent space")

# --- Get labels for evaluation ---
_, test_labels = get_fashion_mnist_labels()
test_labels_np = test_labels.numpy()

CLASS_NAMES = [
    "T-shirt",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle Boot",
]

# --- Nearest-neighbour retrieval ---
from scipy.spatial.distance import cdist

# Sample 10 query images
query_indices = [0, 100, 200, 500, 1000, 2000, 3000, 5000, 7000, 9000]
n_retrieve = 8  # top-8 similar

fig, axes = plt.subplots(
    len(query_indices), n_retrieve + 1, figsize=(18, 2 * len(query_indices))
)
fig.suptitle(
    "Image Retrieval: Query (left) and Top-8 Nearest Neighbours (right)\n"
    "Green border = same class, Red border = different class",
    fontsize=13,
    y=1.02,
)

retrieval_precisions = []

for row, qi in enumerate(query_indices):
    query_latent = test_latents[qi : qi + 1]
    dists = cdist(query_latent, test_latents, metric="euclidean")[0]
    # Exclude self
    dists[qi] = float("inf")
    nearest_idx = np.argsort(dists)[:n_retrieve]

    query_label = test_labels_np[qi]
    retrieved_labels = test_labels_np[nearest_idx]
    precision = (retrieved_labels == query_label).mean()
    retrieval_precisions.append(precision)

    # Plot query
    axes[row, 0].imshow(X_test_flat[qi].cpu().reshape(28, 28), cmap="gray")
    axes[row, 0].set_title(f"Query\n{CLASS_NAMES[query_label]}", fontsize=8)
    axes[row, 0].axis("off")

    # Plot retrieved
    for col, ni in enumerate(nearest_idx):
        axes[row, col + 1].imshow(X_test_flat[ni].cpu().reshape(28, 28), cmap="gray")
        match = test_labels_np[ni] == query_label
        border_color = "#4CAF50" if match else "#F44336"
        for spine in axes[row, col + 1].spines.values():
            spine.set_edgecolor(border_color)
            spine.set_linewidth(3)
        axes[row, col + 1].set_title(
            f"{CLASS_NAMES[test_labels_np[ni]]}\nd={dists[ni]:.2f}", fontsize=7
        )
        axes[row, col + 1].axis("off")

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "ex1_image_retrieval.png", dpi=150, bbox_inches="tight")
plt.show()
print(f"Saved: {OUTPUT_DIR / 'ex1_image_retrieval.png'}")

mean_precision = np.mean(retrieval_precisions)
print(f"\nRetrieval precision@{n_retrieve}: {mean_precision:.1%}")

# --- Latent space clustering visualisation (PCA to 2D) ---
from numpy.linalg import svd

sample_size = 3000
sample_idx = np.random.choice(len(test_latents), sample_size, replace=False)
sample_latents = test_latents[sample_idx]
sample_labels = test_labels_np[sample_idx]

centered = sample_latents - sample_latents.mean(axis=0)
_, _, Vt = svd(centered, full_matrices=False)
projected = centered @ Vt[:2].T

fig, ax = plt.subplots(figsize=(10, 8))
for cls in range(10):
    mask = sample_labels == cls
    ax.scatter(
        projected[mask, 0], projected[mask, 1], s=10, alpha=0.5, label=CLASS_NAMES[cls]
    )
ax.set_xlabel("PC1", fontsize=12)
ax.set_ylabel("PC2", fontsize=12)
ax.set_title(
    "Stacked AE Latent Space (PCA 2D Projection)\nClusters show semantic grouping",
    fontsize=13,
)
ax.legend(fontsize=8, markerscale=3, ncol=2, loc="upper right")
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(
    OUTPUT_DIR / "ex1_stacked_ae_latent_clusters.png", dpi=150, bbox_inches="tight"
)
plt.show()

# --- Business Impact ---
print("\n" + "=" * 64)
print("BUSINESS IMPACT SUMMARY — Fashion Image Search Engine")
print("=" * 64)
print(f"\nRetrieval precision@{n_retrieve}: {mean_precision:.1%}")
print(
    f"Latent dimension: {LATENT_DIM} (vs 784 raw pixels = {784/LATENT_DIM:.0f}x smaller)"
)
print(f"\nFor a catalogue of 1M products:")
print(f"  Raw pixel search:     784 dims x 1M = 3.0 GB index")
print(
    f"  Latent space search:  {LATENT_DIM} dims x 1M = {LATENT_DIM * 4 / 1e6:.1f} MB index"
)
print(f"  Index size reduction: {784 / LATENT_DIM:.0f}x smaller")
print(
    f"  Search speed:         ~{784 / LATENT_DIM:.0f}x faster (fewer distance computations)"
)
print(f"\nSearch quality:")
print(
    f"  Same-class retrieval: {mean_precision:.0%} of top-{n_retrieve} results match query class"
)
print(f"  The latent space groups semantically similar items together")
print(f"  (see PCA cluster plot: shoes cluster, shirts cluster, etc.)")
print(f"\nKey insight: The stacked AE's depth means its 16-dim features")
print(f"capture HIERARCHICAL semantics — not just pixel similarity but")
print(f"category-level similarity. A coat retrieves other coats, not")
print(f"just images with similar pixel distributions.")
print("=" * 64)


# ════════════════════════════════════════════════════════════════════════
# REFLECTION
# ════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  WHAT YOU'VE MASTERED")
print("=" * 70)
print(
    """
  [x] Built a 5-layer stacked autoencoder for deep feature hierarchy
  [x] Understood that depth enables multi-level abstraction
  [x] Applied latent features to image search/retrieval
  [x] Built nearest-neighbour retrieval with precision evaluation
  [x] Visualised latent space clustering (PCA 2D projection)
  [x] Quantified index size and search speed improvements

  KEY INSIGHT: Depth enables HIERARCHY. Layer 1 learns edges, Layer 3
  learns shapes, Layer 5 learns category semantics. The 16-dim latent
  vector is a compact summary that preserves this hierarchy. For image
  search, this means retrieving items that are semantically similar
  (same type of clothing) rather than just visually similar (same
  brightness distribution).

  Next: 08_recurrent_ae.py handles sequential/time-series data...
"""
)

