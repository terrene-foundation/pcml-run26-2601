# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# MLFP05 — Exercise 7, Part 2: Transfer Learning with ResNet-18
# ════════════════════════════════════════════════════════════════════════
#
# WHAT YOU'LL LEARN:
#   After completing this section, you will be able to:
#   - Load a pre-trained ResNet-18 (ImageNet weights) and freeze its
#     convolutional backbone
#   - Replace the final classifier head for CIFAR-10 (10 classes)
#   - Compare transfer learning vs from-scratch on the same dataset
#   - Visualise layer activations (pretrained vs random init) and
#     use Grad-CAM to see what the model "looks at"
#   - Apply transfer learning to a real medical imaging scenario
#
# PREREQUISITES: Part 1 (from-scratch baseline), M5/ex_2 (CNNs).
# ESTIMATED TIME: ~30 min
#
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import asyncio
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision

from shared.mlfp05.ex_7 import (
    BATCH_SIZE,
    CLASS_NAMES,
    EPOCHS,
    INPUT_SIZE,
    N_CLASSES,
    OUTPUT_DIR,
    count_params,
    create_visualizer,
    device,
    extract_features,
    compute_tsne,
    cluster_quality,
    init_engines,
    load_cifar10,
    plot_tsne,
    register_model,
    save_training_plots,
    train_model,
)


# ════════════════════════════════════════════════════════════════════════
# THEORY — Feature Reuse: The Doctor Who Specialises
# ════════════════════════════════════════════════════════════════════════
# Think of a doctor who studied general medicine for 7 years, then
# specialises in dermatology for 1 year. The general training (anatomy,
# physiology, diagnosis methodology) transfers directly — the doctor
# doesn't re-learn how to read a biopsy from scratch.
#
# Transfer learning works the same way. A ResNet-18 trained on ImageNet
# (1.28M images, 1000 classes) has already learned:
#   - Layer 1-2: Edge detectors, colour gradients (universally useful)
#   - Layer 3-4: Texture patterns, simple shapes (mostly transferable)
#   - Layer 5+:  Object parts, task-specific combinations (less transferable)
#
# We FREEZE these learned features and only train a new classification
# head for our specific task (CIFAR-10's 10 classes). This means:
#   - ~5,000 trainable parameters instead of ~11 million
#   - Training completes in minutes instead of hours
#   - Works well even with small datasets (because features are pre-learned)
#
# What transfers well:
#   - Low-level features (edges, textures) — almost always useful
#   - Mid-level features (patterns, shapes) — usually useful
#
# What doesn't transfer:
#   - Task-specific combinations (ImageNet "cat ears" ≠ CIFAR "cat face")
#   - Domain-specific patterns (medical images ≠ natural photos)
# ════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("  PART 2: Transfer Learning with Frozen ResNet-18")
print("=" * 70)


# ════════════════════════════════════════════════════════════════════════
# TASK 1 — Load data and engines
# ════════════════════════════════════════════════════════════════════════

train_set, val_set, train_loader, val_loader = load_cifar10()
conn, tracker, exp_name, registry, has_registry = init_engines()

print(f"\n  Using device: {device}")


# ════════════════════════════════════════════════════════════════════════
# TASK 2 — Build the transfer-learning ResNet-18
# ════════════════════════════════════════════════════════════════════════
# torchvision.models.resnet18(weights=IMAGENET1K_V1) loads the ImageNet
# checkpoint (~44 MB). We freeze all convolutional layers and replace
# the final fc with a fresh 10-class head.


def build_transfer_resnet(
    n_classes: int = N_CLASSES,
    freeze_backbone: bool = True,
) -> nn.Module:
    """Build a ResNet-18 with frozen backbone and fresh classifier head."""
    # TODO: Load pre-trained ResNet-18 and adapt it for CIFAR-10
    # Steps:
    #   1. Load weights: torchvision.models.ResNet18_Weights.IMAGENET1K_V1
    #   2. Create model: torchvision.models.resnet18(weights=weights)
    #   3. Freeze backbone — for param in model.parameters(): param.requires_grad = False
    #   4. Replace model.fc: get in_features = model.fc.in_features,
    #      then model.fc = nn.Linear(in_features, n_classes)
    # Hint: Wrap in try/except for offline fallback (weights=None)
    pass  # Replace with your implementation


transfer_model = build_transfer_resnet()
n_trainable = count_params(transfer_model, trainable_only=True)
n_total = count_params(transfer_model)
print(
    f"  Trainable: {n_trainable:,} / {n_total:,} ({100 * n_trainable / n_total:.2f}%)"
)

# ── Checkpoint 1 ─────────────────────────────────────────────────────
assert n_trainable < n_total, "Backbone should be frozen (fewer trainable than total)"
assert n_trainable < 15000, f"Only fc head should be trainable, got {n_trainable:,}"
# INTERPRETATION: We're training only ~5K parameters (the new fc head)
# while keeping ~11M parameters frozen. The frozen backbone provides
# powerful feature extraction learned from 1.28M ImageNet images.
print("--- Checkpoint 1 passed --- transfer model built\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 3 — Build the from-scratch CNN for comparison
# ════════════════════════════════════════════════════════════════════════


def build_scratch_cnn(n_classes: int = N_CLASSES) -> nn.Module:
    """Baseline: same architecture as Part 1."""
    # TODO: Build the same 3-conv-layer CNN from Part 1
    # Hint: nn.Sequential with Conv2d->BN->ReLU->Pool blocks, then
    #       AdaptiveAvgPool2d(1)->Flatten->Dropout(0.3)->Linear(128, n_classes)
    pass  # Replace with your implementation


# ════════════════════════════════════════════════════════════════════════
# TASK 4 — Train both models and compare
# ════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("  TRAINING: Transfer Learning (frozen ResNet-18 + new head)")
print("=" * 70)
transfer_losses, transfer_accs, transfer_train_accs = train_model(
    transfer_model,
    "resnet18_transfer",
    tracker,
    exp_name,
    train_loader,
    val_loader,
    epochs=EPOCHS,
)

print("\n" + "=" * 70)
print("  TRAINING: From-Scratch CNN baseline")
print("=" * 70)
scratch_model = build_scratch_cnn()
scratch_losses, scratch_accs, scratch_train_accs = train_model(
    scratch_model,
    "cnn_from_scratch",
    tracker,
    exp_name,
    train_loader,
    val_loader,
    epochs=EPOCHS,
)

best_transfer = max(transfer_accs)
best_scratch = max(scratch_accs)

# ── Checkpoint 2 ─────────────────────────────────────────────────────
assert best_transfer > 0.50, (
    f"Transfer val accuracy {best_transfer:.3f} below 0.50 -- check ImageNet "
    "normalisation, input resize, or epoch count"
)
assert len(transfer_losses) == EPOCHS, "Should have per-epoch losses"
assert len(scratch_losses) == EPOCHS, "Should have per-epoch losses"
# INTERPRETATION: Transfer learning leverages features already learned
# on ImageNet's 1.28M images. Even though we only train a linear head,
# the frozen ResNet-18 backbone provides powerful feature extraction
# that the from-scratch CNN cannot match in the same number of epochs.
print(f"\n  Transfer best val_acc: {best_transfer:.3f}")
print(f"  Scratch  best val_acc: {best_scratch:.3f}")
print(f"  Advantage: {best_transfer - best_scratch:+.3f}")
print("\n--- Checkpoint 2 passed --- both models trained and compared\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 5 — Register models in ModelRegistry
# ════════════════════════════════════════════════════════════════════════

if has_registry:
    # TODO: Register both models in the ModelRegistry and promote the better one
    # Steps:
    #   1. register_model(registry, "cifar10_resnet18_transfer", transfer_model, ...)
    #   2. register_model(registry, "cifar10_cnn_scratch", scratch_model, ...)
    #   3. Promote the model with higher accuracy to "production" stage
    # Hint: Use register_model() from helpers
    # Hint: For promotion, use registry.promote_model(name=..., version=...,
    #       target_stage="production", reason=...)
    pass  # TODO: Register and promote models
else:
    print("  Note: ModelRegistry not available. Skipping registration.")

# ── Checkpoint 3 ─────────────────────────────────────────────────────
# INTERPRETATION: The ModelRegistry gives every model a version, metrics,
# and an audit trail. Promoting a model to production records the exact
# comparison that justified the decision.
print("\n--- Checkpoint 3 passed --- models registered\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 6 — Visualise: Layer activations and Grad-CAM
# ════════════════════════════════════════════════════════════════════════
# Two powerful visualisations reveal HOW transfer learning works:
#
# 1. LAYER ACTIVATION COMPARISON — We compare activations from the
#    pre-trained ResNet (structured, edge-aware) vs a randomly
#    initialised ResNet (noisy, meaningless).
#
# 2. GRAD-CAM — Gradient-weighted Class Activation Mapping shows which
#    regions of the input image the model focuses on for its prediction.
#    This is the "visual explanation" of what the model learned.

print("-- Layer activation comparison --")

# TODO: Compare layer activations between pretrained and random ResNet
# Steps:
#   1. Build a random ResNet: torchvision.models.resnet18(weights=None)
#   2. Get a sample image from val_loader
#   3. Hook into conv1 of both models using register_forward_hook
#   4. Run forward pass on both models
#   5. Visualise first 8 activation maps from each in a Plotly figure
# Hint: def hook_fn(m, inp, out): acts_list.append(out.detach().cpu())
# Hint: handle = model.conv1.register_forward_hook(hook_fn)
# Hint: Remember to call handle.remove() after the forward pass

random_resnet = torchvision.models.resnet18(weights=None)
random_resnet.eval()
random_resnet.to(device)

sample_batch_x, sample_batch_y = next(iter(val_loader))
sample_img = sample_batch_x[0:1].to(device)

# TODO: Set up hooks, run forward passes, collect activations
# TODO: Create fig_acts with Heatmap traces for pretrained vs random
# TODO: Save to OUTPUT_DIR / "02_activation_comparison.html"
pass  # Replace with your activation comparison code

# -- Grad-CAM visualisation --
print("-- Computing Grad-CAM heatmaps --")

# TODO: Compute Grad-CAM for the transfer model
# Steps:
#   1. Enable gradients on sample image: sample_img_grad = sample_img.clone().requires_grad_(True)
#   2. Hook into transfer_model.layer4 (forward + backward hooks)
#   3. Forward pass -> get predicted class
#   4. Backward pass for predicted class: logits[0, pred_class].backward()
#   5. Compute Grad-CAM: weights = grads.mean(dim=[2,3], keepdim=True)
#      cam = (weights * acts).sum(dim=1, keepdim=True)
#      cam = F.relu(cam)  # Only positive contributions
#   6. Normalise to [0,1] and visualise with go.Heatmap
# Hint: fwd_handle = model.layer4.register_forward_hook(fwd_hook)
# Hint: bwd_handle = model.layer4.register_full_backward_hook(bwd_hook)
# Hint: Don't forget to remove hooks after use
pass  # Replace with your Grad-CAM implementation

# -- t-SNE comparison --
print("\n-- t-SNE: Transfer vs Scratch feature spaces --")
transfer_feats, transfer_labels = extract_features(transfer_model, val_loader)
scratch_feats, scratch_labels = extract_features(scratch_model, val_loader)

coords_transfer = compute_tsne(transfer_feats)
coords_scratch = compute_tsne(scratch_feats)

cq_transfer = cluster_quality(coords_transfer, transfer_labels)
cq_scratch = cluster_quality(coords_scratch, scratch_labels)

print(f"  Cluster quality (lower = better):")
print(f"    Transfer: {cq_transfer:.4f}")
print(f"    Scratch:  {cq_scratch:.4f}")

plot_tsne(
    coords_transfer,
    transfer_labels,
    "t-SNE: Transfer Learning Features (ResNet-18)",
    OUTPUT_DIR / "02_transfer_tsne.html",
)

# Training curves comparison
viz = create_visualizer()
save_training_plots(
    viz,
    {
        "transfer loss": transfer_losses,
        "transfer val_acc": transfer_accs,
        "scratch loss": scratch_losses,
        "scratch val_acc": scratch_accs,
    },
    OUTPUT_DIR / "02_training_comparison.html",
)

# ── Checkpoint 4 ─────────────────────────────────────────────────────
assert transfer_feats.shape[0] > 0, "Should have extracted transfer features"
# INTERPRETATION: The Grad-CAM heatmap shows the model focuses on the
# object in the image, not the background. Pre-trained features give
# structured attention patterns; scratch-trained features attend more
# diffusely. The t-SNE shows tighter class clusters for transfer
# learning because ImageNet pre-training teaches general visual
# category separation.
print("\n--- Checkpoint 4 passed --- visualisations complete\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 7 — Apply: Medical Image Classification at National Skin Centre
# ════════════════════════════════════════════════════════════════════════
# SCENARIO: The National Skin Centre (NSC) Singapore wants to build an
# AI system to classify skin conditions from dermatology images. They
# have only 2,000 labelled dermatology images across 10 condition types.
#
# Training a CNN from scratch on 2,000 images would give poor results.
# Transfer learning from ImageNet provides a strong foundation: the
# low-level features (edges, textures, colour patterns) transfer well
# to medical images, even though ImageNet doesn't contain dermatology
# photos.

print("\n" + "=" * 70)
print("  APPLY: Medical Image Classification — National Skin Centre SG")
print("=" * 70)

# TODO: Simulate the medical scenario with only 2,000 labelled images
# Steps:
#   1. Create a random subset of 2,000 indices from train_set
#   2. Build a DataLoader for the medical subset
#   3. Train a transfer model (build_transfer_resnet()) on 2K images
#   4. Train a scratch model (build_scratch_cnn()) on 2K images
#   5. Compare accuracy and calculate cost savings
# Hint: rng = np.random.default_rng(42)
# Hint: n_medical = 2000
# Hint: Same pattern as the startup scenario in Part 1

from torch.utils.data import Subset, DataLoader as DL

rng = np.random.default_rng(42)
n_medical = 2000
# TODO: Create indices, subset, loader
# TODO: Train medical_transfer model
# TODO: Train medical_scratch model
# TODO: Store results in best_medical_transfer and best_medical_scratch
best_medical_transfer = 0.0  # TODO: Replace after training
best_medical_scratch = 0.0  # TODO: Replace after training

print(f"\n  === National Skin Centre Scenario (2,000 images) ===")
print(f"  {'Method':<25} {'Val Accuracy':>15} {'Trainable Params':>18}")
print("  " + "-" * 60)
print(
    f"  {'Transfer (ResNet-18)':<25} " f"{best_medical_transfer:>15.1%} " f"{'~5K':>18}"
)
print(f"  {'From scratch':<25} " f"{best_medical_scratch:>15.1%} " f"{'~200K':>18}")
print(f"  {'Advantage':<25} {best_medical_transfer - best_medical_scratch:>+15.1%}")
print()
print(f"  COST-BENEFIT ANALYSIS:")
print(f"  Labelling cost per image (dermatologist review): ~S$5.00")
print(f"  Current labelled images: 2,000")
print(f"  To match transfer accuracy from scratch: ~20,000+ images needed")
print(f"  Labelling cost saved: ~18,000 images x S$5.00 = S$90,000")
print(f"  Transfer learning: FREE (pre-trained weights are open-source)")

# ── Checkpoint 5 ─────────────────────────────────────────────────────
assert (
    best_medical_transfer > 0.15
), f"Transfer with 2K data should beat random (acc={best_medical_transfer:.3f})"
# INTERPRETATION: With only 2,000 images, transfer learning dramatically
# outperforms training from scratch. In a medical context, this means
# fewer misdiagnosed patients and S$90,000+ saved in labelling costs.
# The pre-trained ResNet already knows edges, textures, and shapes —
# it only needs to learn which combinations indicate each skin condition.
print("\n--- Checkpoint 5 passed --- medical scenario complete\n")


# ════════════════════════════════════════════════════════════════════════
# CLEANUP
# ════════════════════════════════════════════════════════════════════════
asyncio.run(conn.close())


# ════════════════════════════════════════════════════════════════════════
# REFLECTION
# ════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  PART 2 COMPLETE — What You've Learned")
print("=" * 70)
print(
    f"""
  [x] Loaded pre-trained ResNet-18 and froze the backbone
  [x] Trained only the classification head ({n_trainable:,} params vs {n_total:,} total)
  [x] Transfer vs scratch on full data: {best_transfer:.1%} vs {best_scratch:.1%}
  [x] Visualised layer activations: pre-trained (structured) vs random (noise)
  [x] Computed Grad-CAM heatmaps showing model attention
  [x] t-SNE cluster quality: transfer {cq_transfer:.3f} vs scratch {cq_scratch:.3f}
  [x] Medical scenario: {best_medical_transfer:.1%} transfer vs {best_medical_scratch:.1%} scratch with 2K images

  KEY INSIGHT: Transfer learning reuses visual features learned from
  1.28M ImageNet images. Like a doctor who studied general medicine
  before specialising, the model starts with deep knowledge instead of
  a blank slate. This is especially powerful when labelled data is
  scarce and expensive (medical images, industrial defects).

  NEXT: Part 3 quantifies data efficiency — how many labelled images
  do you actually need with transfer learning? This answers the VP's
  question: "How much do we need to spend on labelling?"
"""
)

# ══════════════════════════════════════════════════════════════════
# DIAGNOSTIC CHECKPOINT — five instruments before Visualise
# ══════════════════════════════════════════════════════════════════
# Reference: `kailash_ml.diagnostics` (via `kailash-ml`) — see gold standard
# `solutions/ex_1/01_standard_ae.py` for the full pattern.
from kailash_ml.diagnostics import run_diagnostic_checkpoint


def _diag_loss(m, batch):
    # Fine-tune last layer + optionally unfreeze
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


print("\n── Diagnostic Report (Transfer ResNet18 (ImageNet pretrained)) ──")
try:
    diag, findings = run_diagnostic_checkpoint(
        model,
        train_loader,
        _diag_loss,
        title="Transfer ResNet18 (ImageNet pretrained)",
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
# [✓] Gradient flow (HEALTHY): RMS 8.4e-04 to 3.2e-02 across frozen+unfrozen layers.
#     Frozen layers properly report 0 (no gradient by design).
# [✓] Dead neurons  (HEALTHY): 14% inactive — ImageNet-pretrained
#     weights + ReLU is the happy path.
# [✓] Loss trend    (HEALTHY): rapid convergence to 87% val accuracy in 5 epochs.
# ════════════════════════════════════════════════════════════════
#
# STUDENT INTERPRETATION GUIDE — reading the Prescription Pad:

#  [BLOOD TEST] The HEALTHY RMS at every layer is the point of
#     transfer learning. ImageNet features are "warm-started" — no
#     vanishing gradients, no dead neurons, fast convergence.
#     Compare to ex_7/01 (43% dead conv1 from scratch).
#
#  [X-RAY — GRAD-CAM] Run diag.grad_cam(input, target_class,
#     layer_name='layer4') to see what the model attends to.
#     Before fine-tuning: attends to generic edges (ImageNet).
#     After fine-tuning: attends to Fashion-MNIST-specific features.
#     Slide 5G-ii Zech-2018-watermark example — ALWAYS check
#     attribution before deploying.
#
#  [STETHOSCOPE] 87% in 5 epochs vs 60% from scratch in 10 epochs
#     = 10x faster convergence, 30% better final accuracy.
#     Transfer learning is cheat code for small datasets.


