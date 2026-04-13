# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# MLFP05 — Exercise 1.1: Standard Autoencoder (Identity Risk Demo)
# ════════════════════════════════════════════════════════════════════════
#
# WHAT YOU'LL LEARN:
#   - Build an overcomplete autoencoder (hidden dim > input dim)
#   - Demonstrate the identity-function risk — the model copies input
#   - Understand WHY this is dangerous in production (fraud, anomaly detection)
#   - Visualise near-perfect but deceptive reconstructions
#   - Track training with kailash-ml ExperimentTracker
#
# PREREQUISITES: M4.8 (neural network basics, loss functions, optimisers)
# ESTIMATED TIME: ~15 min
#
# TASKS:
#   1. Load Fashion-MNIST and set up engines
#   2. Build overcomplete Standard AE (hidden=1024 > input=784)
#   3. Train and visualise reconstructions
#   4. Interpret WHY near-perfect reconstruction is a problem
#
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import torch.nn as nn
import torch.nn.functional as F

from shared.mlfp05.ex_1 import (
    INPUT_DIM,
    EPOCHS,
    device,
    load_fashion_mnist,
    setup_engines,
    train_variant,
    show_reconstruction,
    register_model,
)

# ════════════════════════════════════════════════════════════════════════
# TASK 1 — Load data and set up engines
# ════════════════════════════════════════════════════════════════════════

X_flat, X_test_flat, X_img, X_test_img, flat_loader, img_loader = load_fashion_mnist()
conn, tracker, exp_name, registry, has_registry = setup_engines()

assert (
    X_flat.shape[0] == 60000
), f"Expected full 60K Fashion-MNIST, got {X_flat.shape[0]}"
assert X_test_flat.shape[0] == 10000, "Test set should have 10K images"
print("\n--- Data loaded and engines initialised ---\n")


# ════════════════════════════════════════════════════════════════════════
# THEORY — The Identity Risk
# ════════════════════════════════════════════════════════════════════════
# A standard autoencoder with hidden dimensions >= input dimension can
# learn the trivial identity function f(x) = x. It achieves near-zero
# loss by simply copying the input — no useful compression learned.
#
# Analogy: Imagine a filing clerk asked to summarise every document.
# If the clerk has unlimited filing cabinet space, they just photocopy
# the originals. The "summary" is the same as the document — useless.
# Only when you force the clerk into a SMALLER cabinet do they have to
# extract the key points.
#
# WHY THIS MATTERS: In fraud detection, a model that memorises every
# transaction pattern (including fraudulent ones) fails to flag anomalies.
# The identity risk is the autoencoder equivalent of overfitting. This
# file IS the application — it is a cautionary tale showing what goes
# wrong when you skip the bottleneck.


# ════════════════════════════════════════════════════════════════════════
# TASK 2 — Build the Standard (Overcomplete) Autoencoder
# ════════════════════════════════════════════════════════════════════════


class StandardAE(nn.Module):
    """Overcomplete autoencoder — hidden dim > input dim.

    This is intentionally "too powerful". With 1024-dim hidden layers for
    784-dim input, it CAN learn the identity function. We demonstrate
    this risk, then show how each subsequent variant solves it.
    """

    def __init__(self, input_dim: int, hidden_dim: int = 1024):
        super().__init__()
        # TODO: Build encoder — nn.Sequential with:
        #       Linear(input_dim, hidden_dim), ReLU,
        #       Linear(hidden_dim, hidden_dim), ReLU
        #       Note: hidden_dim=1024 > input_dim=784 — this is overcomplete
        self.encoder = ____

        # TODO: Build decoder — nn.Sequential with:
        #       Linear(hidden_dim, hidden_dim), ReLU,
        #       Linear(hidden_dim, input_dim), Sigmoid
        self.decoder = ____

    def forward(self, x):
        # TODO: Encode x, then decode. Return (reconstruction, latent_code)
        ____


def standard_ae_loss(model, xb):
    # TODO: Forward pass, compute MSE loss between reconstruction and input
    # Return (loss, empty_dict)
    ____


# ════════════════════════════════════════════════════════════════════════
# TASK 3 — Train and Visualise
# ════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("  Standard Autoencoder — Identity Risk Demo")
print("=" * 70)
print("  Hidden dim=1024 > input dim=784. Can the model just copy?")

# TODO: Create StandardAE instance with INPUT_DIM, hidden_dim=1024
standard_model = ____

# TODO: Train using train_variant with:
#       tracker, exp_name, standard_model, "standard_ae", flat_loader, standard_ae_loss
standard_losses = ____

# TODO: Visualise reconstructions using show_reconstruction
#       Pass: standard_model, X_test_flat, title="Standard AE (Overcomplete)"
____

# ── Checkpoint ──────────────────────────────────────────────────────
assert len(standard_losses) == EPOCHS, f"Expected {EPOCHS} losses"
assert standard_losses[-1] < standard_losses[0], "Loss should decrease"
print("\n--- Checkpoint passed --- standard AE trained\n")

if has_registry:
    register_model(registry, "standard_ae", standard_model, standard_losses[-1])


# ════════════════════════════════════════════════════════════════════════
# TASK 4 — Apply: The Cautionary Tale
# ════════════════════════════════════════════════════════════════════════
# The application of the Standard AE IS the risk demonstration itself.
# This is not a technique you deploy — it is a mistake you learn from.
#
# SCENARIO: A junior data scientist at a Singapore bank builds an
# anomaly detector using this overcomplete architecture. The model
# achieves 0.001 MSE on validation data — "incredible performance!"
# The model goes to production. Fraud losses INCREASE because the
# model reconstructs fraudulent transactions just as well as normal
# ones. Every transaction looks "normal" to the model because it
# learned to copy, not to understand.
#
# Visual proof: Look at the reconstruction grid above. The outputs
# are nearly pixel-perfect copies of the inputs. A model that can
# perfectly reconstruct ANYTHING has learned nothing about the
# structure of the data.
#
# The FIX: Every subsequent variant in this exercise addresses the
# identity risk through a different mechanism:
#   - Undercomplete AE: smaller bottleneck (forced compression)
#   - Denoising AE: noise injection (can't memorise noisy pixels)
#   - Sparse AE: L1 penalty (most neurons forced to zero)
#   - Contractive AE: Jacobian penalty (smooth latent space)
#   - VAE: KL divergence (regularised latent distribution)

# INTERPRETATION: The near-perfect reconstruction is DECEPTIVE. This
# model learned to copy, not to compress. In production, it would fail
# to detect anomalies because it reconstructs EVERYTHING well — even
# fraudulent transactions it should flag as unusual.
#
# BUSINESS IMPACT: At a bank processing 500K transactions/day, an
# identity-risk model in production means ZERO additional fraud
# detection versus the baseline — months of development wasted, and
# fraud losses continue unchecked. The cost is not just the lost
# engineering time; it is the false confidence that "we have an ML
# fraud detector" when in fact we have an expensive photocopier.

print("\n" + "=" * 70)
print("  KEY TAKEAWAY: Near-Perfect Reconstruction = Warning Sign")
print("=" * 70)
print(f"  Final loss: {standard_losses[-1]:.6f}")
print("  This loss is suspiciously low. The model has enough capacity")
print("  to memorise rather than generalise.")
print()
print("  In production, this model would:")
print("  - Reconstruct fraudulent transactions perfectly (no anomaly signal)")
print("  - Reconstruct novel patterns perfectly (no novelty detection)")
print("  - Waste compute on copying instead of learning structure")
print()
print("  SOLUTION: Read the next 9 variants to see how each one")
print("  solves this fundamental problem.")


# ════════════════════════════════════════════════════════════════════════
# REFLECTION
# ════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  WHAT YOU'VE MASTERED")
print("=" * 70)
print(
    """
  [x] Built an overcomplete autoencoder (hidden=1024 > input=784)
  [x] Observed the identity-function risk: near-zero loss, no learning
  [x] Understood why perfect reconstruction is a RED FLAG, not success
  [x] Identified the production failure mode: anomaly detection that
      detects nothing because the model copies everything
  [x] Tracked training with ExperimentTracker

  KEY INSIGHT: Loss alone does not prove a model is useful. A model
  that achieves 0.001 MSE by memorising is worse than one that
  achieves 0.05 MSE by learning structure. The reconstruction grid
  is your proof — look at the images, not just the numbers.

  Next: 02_undercomplete_ae.py fixes this with a bottleneck...
"""
)
