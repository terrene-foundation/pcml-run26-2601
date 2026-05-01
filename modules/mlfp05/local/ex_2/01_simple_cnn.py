# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# MLFP05 Exercise 2.1 — Simple CNN: Convolutions for Image Classification
# ════════════════════════════════════════════════════════════════════════
#
# WHAT YOU'LL LEARN:
#   After completing this file, you will be able to:
#   - Explain WHY convolutions work for images (spatial hierarchy, weight
#     sharing, translation invariance) in terms a non-technical manager
#     can understand
#   - Build a CNN with Conv2d, BatchNorm2d, ReLU, and MaxPool2d layers
#   - Train on the FULL CIFAR-10 dataset (50K images) with PyTorch Lightning
#   - Track the training run with kailash-ml ExperimentTracker
#   - Visualise learned convolutional filters and feature maps to SEE what
#     the network actually learned (not just loss curves)
#   - Apply this to automated product categorisation for a Singapore
#     e-commerce platform
#
# PREREQUISITES: M5/ex_1 (autoencoders, PyTorch training basics,
#   ExperimentTracker and ModelRegistry setup)
# ESTIMATED TIME: ~35 min
#
# DATASET: CIFAR-10 — 50,000 real 32x32 colour photos across 10 classes
#   (airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck).
#
# PHASES:
#   1. THEORY  — Why convolutions work for images
#   2. BUILD   — SimpleCNN architecture
#   3. TRAIN   — Train with Lightning + ExperimentTracker
#   4. VISUALISE — Learned filters + feature maps (model BEHAVIOUR)
#   5. APPLY   — Singapore e-commerce product categorisation
#
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from shared.mlfp05.ex_2 import (
    BATCH_SIZE,
    CLASS_NAMES,
    DEVICE,
    EPOCHS,
    N_CLASSES,
    count_parameters,
    create_visualizer,
    denormalise_cifar,
    init_engines,
    load_cifar10,
    save_training_plots,
    train_model,
)


# ════════════════════════════════════════════════════════════════════════
# PHASE 1 — THEORY: Why Convolutions Work for Images
# ════════════════════════════════════════════════════════════════════════
# Imagine you're a quality inspector at a factory. You don't stare at
# the entire product at once — you scan small regions looking for
# specific patterns: scratches, dents, misalignments. Then you step
# back and combine those local observations into a verdict.
#
# A convolutional neural network works the same way:
#
# 1. LOCAL RECEPTIVE FIELDS (the "scanning window"):
#    A 3x3 filter slides across the image, examining one small patch at
#    a time. A 32x32 image with a 3x3 filter produces 30x30 = 900
#    local observations per filter. This is dramatically more efficient
#    than connecting every pixel to every neuron (32x32x3 = 3,072 inputs
#    per neuron in a fully-connected layer).
#
# 2. WEIGHT SHARING (the "same inspector everywhere"):
#    The SAME 3x3 filter is applied at every spatial position. If an
#    edge detector works in the top-left corner, it works in the
#    bottom-right too. This means a 3x3 filter has only 9 learnable
#    parameters regardless of image size — a 1000x1000 image still
#    uses the same 9 weights.
#
# 3. SPATIAL HIERARCHY (the "zoom out" principle):
#    Layer 1 detects edges (horizontal, vertical, diagonal lines).
#    Layer 2 combines edges into textures (fur, fabric, metal).
#    Layer 3 combines textures into parts (wheel, wing, eye).
#    Each layer "zooms out" by one level of abstraction.
#
# 4. TRANSLATION INVARIANCE (the "it doesn't matter where"):
#    A cat in the top-left of the image activates the same filters as
#    a cat in the bottom-right. The network recognises WHAT, not WHERE.
#    MaxPool2d reinforces this by summarising each 2x2 region with its
#    maximum value.
#
# For a non-technical stakeholder: "Convolutions let the network learn
# its own inspection checklist — first small details like edges, then
# bigger patterns like shapes, then whole objects. The same checklist
# works regardless of where the object appears in the image."
#
# BatchNorm2d: After each convolution, normalise the activations so
# that training remains stable. Without it, deeper layers receive
# inputs with wildly different scales, making gradient updates erratic.
# Think of it as "recalibrating the instruments between measurements."

print("=" * 70)
print("  PHASE 1 — THEORY: Why Convolutions Work for Images")
print("=" * 70)
print(
    """
  KEY CONCEPTS:
  - Local receptive fields: scan small patches, not the whole image
  - Weight sharing: same 3x3 filter applied at every position
  - Spatial hierarchy: edges -> textures -> parts -> objects
  - Translation invariance: recognise WHAT, not WHERE
  - BatchNorm: stabilise training by normalising activations
  - MaxPool: summarise spatial regions, reduce dimensions by 2x
"""
)


# ════════════════════════════════════════════════════════════════════════
# PHASE 2 — BUILD: SimpleCNN Architecture
# ════════════════════════════════════════════════════════════════════════
# Architecture:
#   Conv2d(3->32, 3x3) -> BatchNorm -> ReLU -> MaxPool (32->16)
#   Conv2d(32->64, 3x3) -> BatchNorm -> ReLU -> MaxPool (16->8)
#   Flatten -> Linear(64*8*8 -> 128) -> ReLU -> Linear(128 -> 10)
#
# Why these choices:
#   - 3x3 kernels: the smallest filter that captures spatial structure.
#     Stacking two 3x3 layers gives a 5x5 receptive field with fewer
#     parameters than a single 5x5 filter (2*9=18 vs 25).
#   - Channel doubling (32->64): deeper layers need more feature maps
#     because they represent more complex combinations of lower features.
#   - Global architecture: features (convolutions) -> head (classification).
#     This separation is standard practice — features extract, head decides.

print("\n" + "=" * 70)
print("  PHASE 2 — BUILD: SimpleCNN Architecture")
print("=" * 70)


class SimpleCNN(nn.Module):
    """Plain CNN with BatchNorm and MaxPool for CIFAR-10 classification.

    Architecture:
        Conv(3->32) -> BN -> ReLU -> MaxPool(2)   [32x32 -> 16x16]
        Conv(32->64) -> BN -> ReLU -> MaxPool(2)   [16x16 -> 8x8]
        Flatten -> Linear(4096->128) -> ReLU -> Linear(128->10)
    """

    def __init__(self, n_classes: int = N_CLASSES):
        super().__init__()
        # TODO: Build the feature extraction backbone — nn.Sequential with:
        #   Block 1: Conv2d(3, 32, kernel_size=3, padding=1), BatchNorm2d(32),
        #            ReLU(), MaxPool2d(2)   [spatial: 32 -> 16]
        #   Block 2: Conv2d(32, 64, kernel_size=3, padding=1), BatchNorm2d(64),
        #            ReLU(), MaxPool2d(2)   [spatial: 16 -> 8]
        self.features = nn.Sequential(
            ____,
        )
        # TODO: Build the classification head — nn.Sequential with:
        #   Flatten(), Linear(64 * 8 * 8, 128), ReLU(), Linear(128, n_classes)
        self.head = nn.Sequential(
            ____,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO: Pass through features then head
        return ____


# ── Checkpoint 1: Architecture builds correctly ───────────────────────
model = SimpleCNN()
dummy_input = torch.randn(2, 3, 32, 32)
dummy_output = model(dummy_input)
assert dummy_output.shape == (
    2,
    N_CLASSES,
), f"SimpleCNN output should be (batch, {N_CLASSES}), got {dummy_output.shape}"

param_count = count_parameters(model)
print(f"\nSimpleCNN built successfully")
print(f"  Parameters: {param_count:,}")
print(f"  Input shape: (batch, 3, 32, 32)")
print(f"  Output shape: (batch, {N_CLASSES})")

# INTERPRETATION: ~560K parameters is modest by modern standards (ResNet-50
# has 25M). The two conv layers learn 32+64=96 distinct 3x3 filters.
# Layer 1 filters detect basic edges; layer 2 filters detect textures
# composed from those edges. The classifier head maps the 64 feature maps
# (each 8x8) to 10 class probabilities.
print("\n--- Checkpoint 1 passed --- SimpleCNN architecture verified\n")
del model, dummy_input, dummy_output


# ════════════════════════════════════════════════════════════════════════
# PHASE 3 — TRAIN: Train with Lightning + ExperimentTracker
# ════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("  PHASE 3 — TRAIN: SimpleCNN on full CIFAR-10")
print("=" * 70)

# Load data
X_train, y_train, X_val, y_val, train_loader, val_loader = load_cifar10()

# ── Checkpoint 2: Data loaded correctly ───────────────────────────────
assert X_train.shape[0] == 50000, (
    f"Expected full 50K CIFAR-10 training set, got {X_train.shape[0]}. "
    "CNNs need the full dataset to learn general spatial features."
)
assert X_val.shape[0] == 10000, f"Expected 10K validation, got {X_val.shape[0]}"
assert X_train.shape[1:] == (3, 32, 32), "CIFAR-10 images should be 3x32x32"
print("--- Checkpoint 2 passed --- CIFAR-10 loaded (50K train, 10K val)\n")

# Set up kailash-ml engines
conn, tracker, exp_name, registry, has_registry = init_engines()

# TODO: Train SimpleCNN
#   1. Instantiate SimpleCNN()
#   2. Call train_model(model, name, tracker, exp_name, train_loader, val_loader, epochs=EPOCHS)
#   3. train_model returns (losses_list, accs_list)
print(f"\nTraining SimpleCNN for {EPOCHS} epochs on {X_train.shape[0]:,} images...")
simple_cnn = ____
simple_losses, simple_accs = ____

# ── Checkpoint 3: Training converged ─────────────────────────────────
assert len(simple_losses) == EPOCHS, f"Expected {EPOCHS} epoch losses"
assert simple_losses[-1] < simple_losses[0], "Loss should decrease during training"
assert simple_accs[-1] > 0.4, (
    f"SimpleCNN val accuracy {simple_accs[-1]:.3f} too low -- "
    "expected > 0.4 on full CIFAR-10 after 8 epochs"
)
# INTERPRETATION: With 50K images the SimpleCNN learns real convolutional
# filters: edge detectors in layer 1, texture detectors in layer 2. The
# accuracy above ~55% shows the model is learning meaningful spatial
# features, not just memorising. BatchNorm is crucial here -- without it,
# training would be much slower and less stable.
print(f"\nSimpleCNN final: loss={simple_losses[-1]:.4f}, val_acc={simple_accs[-1]:.3f}")
print("--- Checkpoint 3 passed --- SimpleCNN trained successfully\n")

# Register in ModelRegistry
if has_registry:
    from shared.mlfp05.ex_2 import register_model

    # TODO: Register the trained model
    #   register_model(registry, "simple_cnn_cifar10", simple_cnn,
    #                  simple_losses[-1], simple_accs[-1])
    simple_version = ____

# Save training curves
viz = create_visualizer()
# TODO: Save loss and accuracy plots using save_training_plots
#   Args: (viz, metrics_dict, output_filename, y_label=...)
#   metrics_dict format: {"label": list_of_values}
____
____


# ════════════════════════════════════════════════════════════════════════
# PHASE 4 — VISUALISE: Learned Filters and Feature Maps
# ════════════════════════════════════════════════════════════════════════
# Loss curves tell you WHETHER the model is learning. Filter and feature
# map visualisations tell you WHAT the model is learning.
#
# Layer 1 filters should look like oriented edge detectors (Gabor-like
# patterns). Feature maps show which parts of the input image activate
# each filter — bright regions = strong activation = "this filter found
# something here."

print("=" * 70)
print("  PHASE 4 — VISUALISE: What Did the CNN Learn?")
print("=" * 70)

# 4a. Visualise Layer 1 learned filters (3x3 RGB filters)
# TODO: Extract layer 1 weights — simple_cnn.features[0].weight.data.cpu()
#   Shape will be (32, 3, 3, 3) — 32 filters, each 3x3 RGB
layer1_weights = ____

fig_filters, axes = plt.subplots(4, 8, figsize=(16, 8))
fig_filters.suptitle(
    "SimpleCNN Layer 1 Learned Filters (32 filters, 3x3 RGB)", fontsize=14
)
for i, ax in enumerate(axes.flat):
    if i < layer1_weights.shape[0]:
        # TODO: Normalise filter to [0, 1] for display
        #   Get filter i from layer1_weights, normalise: (f - f.min()) / (f.max() - f.min() + 1e-8)
        #   Then ax.imshow(filt.permute(1, 2, 0).numpy())
        filt = ____
        filt = ____
        ax.imshow(filt.permute(1, 2, 0).numpy())
    ax.axis("off")
plt.tight_layout()
plt.savefig("ex_2_01_learned_filters.png", dpi=150, bbox_inches="tight")
plt.close(fig_filters)
print("  Saved: ex_2_01_learned_filters.png")
print("  Look for: edge detectors (horizontal, vertical, diagonal lines)")
print("  Some filters will be colour-sensitive (responding to red/green/blue edges)")

# 4b. Visualise feature maps for a sample image
simple_cnn.eval()
sample_idx = 7  # A horse in CIFAR-10
sample_img = X_val[sample_idx : sample_idx + 1]  # (1, 3, 32, 32)
true_label = CLASS_NAMES[y_val[sample_idx].item()]

# TODO: Extract intermediate feature maps using forward hooks
#   1. Create a dict to store feature maps
#   2. Define a hook_fn(name) that returns a hook capturing output.detach().cpu()
#   3. Register hooks on: features[0] (conv1_raw), features[2] (conv1_relu),
#      features[4] (conv2_raw), features[6] (conv2_relu)
#   4. Run forward pass with torch.no_grad()
#   5. Remove all hooks
feature_maps = {}


def hook_fn(name):
    def hook(module, input, output):
        feature_maps[name] = output.detach().cpu()

    return hook


handles = []
# TODO: Register hooks on the 4 layers listed above
#   handles.append(simple_cnn.features[0].register_forward_hook(hook_fn("conv1_raw")))
#   ... (repeat for indices 2, 4, 6 with names conv1_relu, conv2_raw, conv2_relu)
____
____
____
____

with torch.no_grad():
    pred_logits = simple_cnn(sample_img)
    pred_class = CLASS_NAMES[pred_logits.argmax(dim=-1).item()]

for h in handles:
    h.remove()

# TODO: Plot original image + layer 1 feature maps + layer 2 feature maps
#   Row 0: original image (col 0) + first 8 conv1_relu maps (cols 1-8)
#   Row 1: next 8 conv1 maps
#   Row 2: 8 conv2_relu maps (8x8 resolution after pooling, use cmap="magma")
fig_fmaps, axes = plt.subplots(3, 9, figsize=(18, 7))
fig_fmaps.suptitle(
    f"Feature Maps for '{true_label}' (predicted: '{pred_class}')",
    fontsize=14,
)

# Row 0: original image + first 8 conv1 ReLU feature maps
orig_display = denormalise_cifar(sample_img.squeeze(0))
axes[0, 0].imshow(orig_display.permute(1, 2, 0).numpy())
axes[0, 0].set_title("Original", fontsize=9)
axes[0, 0].axis("off")
conv1_maps = feature_maps["conv1_relu"].squeeze(0)
for i in range(8):
    # TODO: imshow conv1_maps[i] with cmap="viridis"
    ____
    axes[0, i + 1].set_title(f"L1 F{i}", fontsize=8)
    axes[0, i + 1].axis("off")

# Row 1: next 8 conv1 feature maps
for i in range(8):
    idx = i + 8
    if idx < conv1_maps.shape[0]:
        axes[1, i].imshow(conv1_maps[idx].numpy(), cmap="viridis")
        axes[1, i].set_title(f"L1 F{idx}", fontsize=8)
    axes[1, i].axis("off")
axes[1, 8].axis("off")

# Row 2: 8 conv2 ReLU feature maps (8x8 after MaxPool)
conv2_maps = feature_maps["conv2_relu"].squeeze(0)
for i in range(8):
    axes[2, i].imshow(conv2_maps[i].numpy(), cmap="magma")
    axes[2, i].set_title(f"L2 F{i}", fontsize=8)
    axes[2, i].axis("off")
axes[2, 8].axis("off")

plt.tight_layout()
plt.savefig("ex_2_01_feature_maps.png", dpi=150, bbox_inches="tight")
plt.close(fig_fmaps)
print(f"\n  Saved: ex_2_01_feature_maps.png")
print(f"  Sample: true='{true_label}', predicted='{pred_class}'")
print("  Layer 1 maps: edges and colour boundaries (high-res, 32x32)")
print("  Layer 2 maps: textures and shapes (lower-res, 8x8 after pooling)")

# 4c. Visualise prediction confidence on a batch of images
with torch.no_grad():
    batch_imgs = X_val[:16]
    batch_logits = simple_cnn(batch_imgs)
    batch_probs = F.softmax(batch_logits, dim=-1)
    batch_preds = batch_logits.argmax(dim=-1)

fig_preds, axes = plt.subplots(2, 8, figsize=(20, 6))
fig_preds.suptitle("SimpleCNN Predictions on Validation Images", fontsize=14)
for i in range(16):
    row, col = i // 8, i % 8
    img_display = denormalise_cifar(batch_imgs[i])
    axes[row, col].imshow(img_display.permute(1, 2, 0).numpy())
    pred_name = CLASS_NAMES[batch_preds[i].item()]
    true_name = CLASS_NAMES[y_val[i].item()]
    conf = batch_probs[i][batch_preds[i]].item()
    correct = batch_preds[i].item() == y_val[i].item()
    colour = "green" if correct else "red"
    axes[row, col].set_title(
        f"{pred_name}\n({conf:.0%})",
        fontsize=8,
        color=colour,
    )
    axes[row, col].axis("off")
plt.tight_layout()
plt.savefig("ex_2_01_predictions.png", dpi=150, bbox_inches="tight")
plt.close(fig_preds)
print("  Saved: ex_2_01_predictions.png")
print("  Green = correct, Red = incorrect. Confidence shown in parentheses.")

# ── Checkpoint 4: Visualisations generated ───────────────────────────
import os

assert os.path.exists("ex_2_01_learned_filters.png"), "Filter visualisation missing"
assert os.path.exists("ex_2_01_feature_maps.png"), "Feature map visualisation missing"
assert os.path.exists("ex_2_01_predictions.png"), "Prediction visualisation missing"
print("\n--- Checkpoint 4 passed --- visual proof of model behaviour saved\n")


# ════════════════════════════════════════════════════════════════════════
# PHASE 5 — APPLY: Singapore E-Commerce Product Categorisation
# ════════════════════════════════════════════════════════════════════════
# SCENARIO: You are an ML engineer at a Singapore e-commerce platform
# (think Shopee, Lazada, or Carousell). The platform receives 500,000+
# new product listings per day. Sellers often mis-categorise products
# (a "Nike Air Max" listed under "Electronics" instead of "Shoes"),
# leading to:
#   - Poor search results (customers can't find what they want)
#   - Incorrect commission rates (different categories have different fees)
#   - Regulatory compliance issues (restricted items in wrong categories)
#
# BUSINESS CASE:
#   - Manual review: 500K listings/day * $0.02/listing = $10,000/day
#   - CNN auto-categorisation: $500/day compute + $2,000/day for edge cases
#   - Net savings: ~$7,500/day = ~$2.7M/year
#   - Additional revenue: improved search -> +3-5% conversion rate
#
# HOW THE CNN APPLIES:
#   Our SimpleCNN learned to classify 32x32 images into 10 categories.
#   A production system would:
#   1. Resize product images to standard dimensions
#   2. Run through a CNN trained on product categories (not CIFAR-10)
#   3. High-confidence predictions (>0.9) auto-categorise
#   4. Low-confidence predictions route to human reviewers
#   5. Track accuracy with ExperimentTracker, retrain monthly
#
# CIFAR-10 AS PROXY:
#   CIFAR-10's categories (airplane, automobile, truck, ship, etc.) map
#   to real e-commerce categories. The architecture patterns are
#   identical — only the training data changes.

print("=" * 70)
print("  PHASE 5 — APPLY: Singapore E-Commerce Product Categorisation")
print("=" * 70)

# TODO: Build the production triage system
#   1. Set simple_cnn.eval()
#   2. Define thresholds: HIGH_CONFIDENCE_THRESHOLD = 0.85, LOW_CONFIDENCE_THRESHOLD = 0.50
#   3. Create ECOMMERCE_MAPPING dict mapping CIFAR-10 classes to e-commerce categories:
#      airplane -> "Travel & Luggage", automobile -> "Automotive",
#      bird/cat/dog -> "Pet Supplies", deer -> "Home & Garden (Decor)",
#      frog -> "Toys & Collectibles", horse -> "Sports & Outdoors",
#      ship -> "Travel & Luggage", truck -> "Automotive"
simple_cnn.eval()
HIGH_CONFIDENCE_THRESHOLD = 0.85
LOW_CONFIDENCE_THRESHOLD = 0.50

ECOMMERCE_MAPPING = {____}

# TODO: Run inference on all validation images
#   with torch.no_grad():
#     val_logits = simple_cnn(X_val)
#     val_probs = F.softmax(val_logits, dim=-1)
#     val_preds = val_logits.argmax(dim=-1)
#     val_confidences = val_probs.gather(1, val_preds.unsqueeze(1)).squeeze()
#     val_correct = (val_preds == y_val).float()
with torch.no_grad():
    val_logits = ____
    val_probs = ____
    val_preds = ____
    val_confidences = ____
    val_correct = ____

# TODO: Create triage masks based on confidence thresholds
#   auto_approve_mask: confidences >= HIGH threshold
#   review_mask: confidences >= LOW threshold AND < HIGH threshold
#   reject_mask: confidences < LOW threshold
auto_approve_mask = ____
review_mask = ____
reject_mask = ____

n_total = len(y_val)
n_auto = auto_approve_mask.sum().item()
n_review = review_mask.sum().item()
n_reject = reject_mask.sum().item()

auto_acc = val_correct[auto_approve_mask].mean().item() if n_auto > 0 else 0
review_acc = val_correct[review_mask].mean().item() if n_review > 0 else 0

print(
    f"""
  PRODUCTION TRIAGE SIMULATION (10,000 product listings):

  Category         | Count  | % Total | Accuracy
  -----------------+--------+---------+---------
  Auto-approved    | {n_auto:>5,} | {n_auto/n_total:>6.1%}  | {auto_acc:.1%}
  Needs review     | {n_review:>5,} | {n_review/n_total:>6.1%}  | {review_acc:.1%}
  Low confidence   | {n_reject:>5,} | {n_reject/n_total:>6.1%}  | (routed to human)

  BUSINESS IMPACT (daily projection for 500K listings):
    Auto-approved listings:   {int(500000 * n_auto/n_total):>7,} (no human cost)
    Human review needed:      {int(500000 * n_review/n_total):>7,} (@ $0.02/review)
    Daily review cost:        ${int(500000 * (n_review + n_reject)/n_total * 0.02):>7,}
    vs. full manual review:   $  10,000
    Daily savings:            ${10000 - int(500000 * (n_review + n_reject)/n_total * 0.02):>7,}
    Projected annual savings: ${(10000 - int(500000 * (n_review + n_reject)/n_total * 0.02)) * 365:>10,}

  STAKEHOLDER INSIGHT:
    Auto-approved accuracy of {auto_acc:.1%} means {100 - auto_acc*100:.1f}% error rate
    on automated decisions. For an e-commerce platform, this means
    roughly {int(500000 * n_auto/n_total * (1 - auto_acc)):,} mis-categorised products
    per day slip through without human review.

    RECOMMENDATION: Accept this if mis-categorisation cost < $0.50/item.
    For high-value categories (electronics, luxury), lower the auto-approve
    threshold to 0.95 and accept higher review volume.
"""
)

# Show example auto-approved and review-needed listings
print("  EXAMPLE AUTO-APPROVED LISTINGS:")
auto_indices = torch.where(auto_approve_mask)[0][:5]
for idx in auto_indices:
    pred_name = CLASS_NAMES[val_preds[idx].item()]
    true_name = CLASS_NAMES[y_val[idx].item()]
    conf = val_confidences[idx].item()
    ecom_cat = ECOMMERCE_MAPPING[pred_name]
    status = "CORRECT" if pred_name == true_name else "MIS-CATEGORISED"
    print(
        f"    Listing #{idx.item()}: {pred_name} -> {ecom_cat} "
        f"(conf={conf:.2f}) [{status}]"
    )

print("\n  EXAMPLE REVIEW-NEEDED LISTINGS:")
review_indices = torch.where(review_mask)[0][:5]
for idx in review_indices:
    pred_name = CLASS_NAMES[val_preds[idx].item()]
    true_name = CLASS_NAMES[y_val[idx].item()]
    conf = val_confidences[idx].item()
    print(
        f"    Listing #{idx.item()}: predicted={pred_name} (conf={conf:.2f}), "
        f"true={true_name} -> NEEDS HUMAN REVIEW"
    )

# ── Checkpoint 5: Apply section complete ─────────────────────────────
assert n_auto + n_review + n_reject == n_total, "Triage should cover all samples"
assert auto_acc > 0.7, (
    f"Auto-approved accuracy {auto_acc:.3f} too low -- "
    "high-confidence predictions should be mostly correct"
)
print("\n--- Checkpoint 5 passed --- e-commerce application demonstrated\n")


# ════════════════════════════════════════════════════════════════════════
# Clean up
# ════════════════════════════════════════════════════════════════════════
import asyncio

asyncio.run(conn.close())


# ════════════════════════════════════════════════════════════════════════
# REFLECTION
# ════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  WHAT YOU'VE MASTERED")
print("=" * 70)
print(
    f"""
  THEORY:
  [x] Convolutions scan local patches with shared weights -- 3x3 filters
      detect edges at every position with only 9 parameters
  [x] Spatial hierarchy: edges -> textures -> parts -> objects, each
      layer building on the previous
  [x] BatchNorm stabilises training; MaxPool adds translation invariance

  BUILD + TRAIN:
  [x] SimpleCNN: Conv2d + BatchNorm + ReLU + MaxPool, two-block design
  [x] Trained on FULL CIFAR-10 (50K images) with ExperimentTracker
  [x] ~{{param_count:,}} parameters, achieves >{{simple_accs[-1]:.0%}} validation accuracy

  VISUALISE (the proof):
  [x] Layer 1 filters: oriented edge detectors (Gabor-like patterns)
  [x] Feature maps: which spatial regions activate each filter
  [x] Prediction grid: where the model succeeds and fails, with confidence

  APPLY:
  [x] Singapore e-commerce product auto-categorisation
  [x] Confidence-based triage: auto-approve vs human review
  [x] Business impact: projected annual savings from automation
  [x] Stakeholder-ready accuracy and error rate analysis

  KEY INSIGHT: A CNN is not a black box. The filters it learns are
  interpretable -- they are the visual features the network considers
  important. Filter and feature map visualisation is how you debug and
  explain CNN decisions to non-technical stakeholders.

  Next: In 02_resnet_se.py, you'll see why deeper networks fail and how
  residual connections and attention mechanisms fix the problem...
"""
)

# ══════════════════════════════════════════════════════════════════
# DIAGNOSTIC CHECKPOINT — five instruments + Grad-CAM for CNNs
# ══════════════════════════════════════════════════════════════════
# First classifier in M5: we use `diagnose_classifier` which wraps
# `run_diagnostic_checkpoint` with a cross-entropy loss function.
# For CNNs, Grad-CAM is the sixth instrument — it answers "which
# pixels drove the prediction?" and surfaces spurious shortcuts
# (Zech et al. 2018: hospitals' chest-X-ray models latched onto
# watermarks instead of pathology).
from kailash_ml import diagnose

print("\n── Diagnostic Report (SimpleCNN) ──")
report = diagnose(simple_cnn, kind="dl", data=val_loader)

# Grad-CAM on the last conv layer: verify the model looks at objects,
# not backgrounds. Pick a validation batch and a target class.
try:
    _vx, _vy = next(iter(val_loader))
    # Find the last conv layer in the model
    _last_conv = None
    for _name, _mod in simple_cnn.named_modules():
        if isinstance(_mod, nn.Conv2d):
            _last_conv = _name
    if _last_conv is not None:
        cam = diag.grad_cam(
            _vx[:4].to(DEVICE),
            target_class=int(_vy[0].item()),
            layer_name=_last_conv,
        )
        print(
            f"  Grad-CAM computed on layer '{_last_conv}', "
            f"heatmap shape={tuple(cam.shape)}"
        )
        # Students: overlay `cam[i]` onto `_vx[i]` (resize CAM to 32x32)
        # and inspect — if the hot region is off the object, the model
        # learned a shortcut (see Zech 2018 hospital-watermark study).
except Exception as _exc:  # pragma: no cover — visualisation optional
    print(f"  Grad-CAM skipped ({_exc})")

# ══════ EXPECTED OUTPUT (reference shape) ══════
# ══════════════════════════════════════════════════════════════════
#   DL Diagnostics Report — Prescription Pad
# ══════════════════════════════════════════════════════════════════
#   [✓] Gradient flow (HEALTHY): RMS range ~1e-4 – ~1e-2, uniform
#       across Conv2d and Linear layers (BatchNorm is keeping flow
#       healthy — this is WHY BN was invented).
#   [✓] Dead neurons  (HEALTHY): No ReLU layer above 30% inactive;
#       weight sharing in Conv2d naturally keeps channels alive.
#   [✓] Loss trend    (HEALTHY): Training converging, val accuracy
#       rising monotonically — no overfitting after 8 epochs.
#   + Grad-CAM: heatmap concentrates on the object, not the corners.
# ══════════════════════════════════════════════════════════════════
#
# STUDENT INTERPRETATION GUIDE:
#   - Healthy CNN signature: uniform gradient RMS + low dead %% +
#     Grad-CAM on the object. If any of the three fails, investigate
#     BEFORE trusting val accuracy.
#   - Zech 2018 lesson: a pneumonia classifier achieved SOTA accuracy
#     but Grad-CAM revealed it was attending to hospital watermarks
#     — a dataset-shortcut that would FAIL on any other hospital's
#     scans. Always visualise attribution; never ship on accuracy
#     alone.
#   - If Grad-CAM highlights background/corners, the model is
#     using spurious features. Fix: data augmentation, balanced
#     sampling, or a different loss.
# ══════════════════════════════════════════════════════════════════

# ── Checkpoint 3: Training converged ─────────────────────────────────
assert len(simple_losses) == EPOCHS, f"Expected {EPOCHS} epoch losses"
assert simple_losses[-1] < simple_losses[0], "Loss should decrease during training"
assert simple_accs[-1] > 0.4, (
    f"SimpleCNN val accuracy {simple_accs[-1]:.3f} too low -- "
    "expected > 0.4 on full CIFAR-10 after 8 epochs"
)
# INTERPRETATION: With 50K images the SimpleCNN learns real convolutional
# filters: edge detectors in layer 1, texture detectors in layer 2. The
# accuracy above ~55% shows the model is learning meaningful spatial
# features, not just memorising. BatchNorm is crucial here -- without it,
# training would be much slower and less stable.
print(f"\nSimpleCNN final: loss={simple_losses[-1]:.4f}, val_acc={simple_accs[-1]:.3f}")
print("--- Checkpoint 3 passed --- SimpleCNN trained successfully\n")

# Register in ModelRegistry
if has_registry:
    from shared.mlfp05.ex_2 import register_model

    simple_version = register_model(
        registry,
        "simple_cnn_cifar10",
        simple_cnn,
        simple_losses[-1],
        simple_accs[-1],
    )

# Save training curves
viz = create_visualizer()
save_training_plots(
    viz,
    {"SimpleCNN loss": simple_losses},
    "ex_2_01_simple_cnn_loss.html",
    y_label="Training Loss",
)
save_training_plots(
    viz,
    {"SimpleCNN val accuracy": simple_accs},
    "ex_2_01_simple_cnn_acc.html",
    y_label="Validation Accuracy",
)


# ════════════════════════════════════════════════════════════════════════
# PHASE 4 — VISUALISE: Learned Filters and Feature Maps
# ════════════════════════════════════════════════════════════════════════
# Loss curves tell you WHETHER the model is learning. Filter and feature
# map visualisations tell you WHAT the model is learning.
#
# Layer 1 filters should look like oriented edge detectors (Gabor-like
# patterns). Feature maps show which parts of the input image activate
# each filter — bright regions = strong activation = "this filter found
# something here."

print("=" * 70)
print("  PHASE 4 — VISUALISE: What Did the CNN Learn?")
print("=" * 70)

# 4a. Visualise Layer 1 learned filters (3x3 RGB filters)
layer1_weights = simple_cnn.features[0].weight.data.cpu()  # (32, 3, 3, 3)

fig_filters, axes = plt.subplots(4, 8, figsize=(16, 8))
fig_filters.suptitle(
    "SimpleCNN Layer 1 Learned Filters (32 filters, 3x3 RGB)", fontsize=14
)
for i, ax in enumerate(axes.flat):
    if i < layer1_weights.shape[0]:
        # Normalise each filter to [0, 1] for display
        filt = layer1_weights[i]  # (3, 3, 3) — RGB
        filt = (filt - filt.min()) / (filt.max() - filt.min() + 1e-8)
        ax.imshow(filt.permute(1, 2, 0).numpy())  # (3,3,3) -> (H,W,C)
    ax.axis("off")
plt.tight_layout()
plt.savefig("ex_2_01_learned_filters.png", dpi=150, bbox_inches="tight")
plt.close(fig_filters)
print("  Saved: ex_2_01_learned_filters.png")
print("  Look for: edge detectors (horizontal, vertical, diagonal lines)")
print("  Some filters will be colour-sensitive (responding to red/green/blue edges)")

# 4b. Visualise feature maps for a sample image
simple_cnn.eval()
sample_idx = 7  # A horse in CIFAR-10
sample_img = X_val[sample_idx : sample_idx + 1]  # (1, 3, 32, 32)
true_label = CLASS_NAMES[y_val[sample_idx].item()]

# Extract intermediate feature maps using hooks
feature_maps = {}


def hook_fn(name):
    def hook(module, input, output):
        feature_maps[name] = output.detach().cpu()

    return hook


handles = []
handles.append(simple_cnn.features[0].register_forward_hook(hook_fn("conv1_raw")))
handles.append(simple_cnn.features[2].register_forward_hook(hook_fn("conv1_relu")))
handles.append(simple_cnn.features[4].register_forward_hook(hook_fn("conv2_raw")))
handles.append(simple_cnn.features[6].register_forward_hook(hook_fn("conv2_relu")))

with torch.no_grad():
    pred_logits = simple_cnn(sample_img)
    pred_class = CLASS_NAMES[pred_logits.argmax(dim=-1).item()]

for h in handles:
    h.remove()

# Plot: original image + layer 1 feature maps + layer 2 feature maps
fig_fmaps, axes = plt.subplots(3, 9, figsize=(18, 7))
fig_fmaps.suptitle(
    f"Feature Maps for '{true_label}' (predicted: '{pred_class}')",
    fontsize=14,
)

# Row 0: original image (centre) + first 8 conv1 ReLU feature maps
orig_display = denormalise_cifar(sample_img.squeeze(0))
axes[0, 0].imshow(orig_display.permute(1, 2, 0).numpy())
axes[0, 0].set_title("Original", fontsize=9)
axes[0, 0].axis("off")
conv1_maps = feature_maps["conv1_relu"].squeeze(0)  # (32, 32, 32)
for i in range(8):
    axes[0, i + 1].imshow(conv1_maps[i].numpy(), cmap="viridis")
    axes[0, i + 1].set_title(f"L1 F{i}", fontsize=8)
    axes[0, i + 1].axis("off")

# Row 1: next 8 conv1 feature maps
for i in range(8):
    idx = i + 8
    if idx < conv1_maps.shape[0]:
        axes[1, i].imshow(conv1_maps[idx].numpy(), cmap="viridis")
        axes[1, i].set_title(f"L1 F{idx}", fontsize=8)
    axes[1, i].axis("off")
axes[1, 8].axis("off")

# Row 2: 8 conv2 ReLU feature maps (16x16 after MaxPool)
conv2_maps = feature_maps["conv2_relu"].squeeze(0)  # (64, 8, 8)
for i in range(8):
    axes[2, i].imshow(conv2_maps[i].numpy(), cmap="magma")
    axes[2, i].set_title(f"L2 F{i}", fontsize=8)
    axes[2, i].axis("off")
axes[2, 8].axis("off")

plt.tight_layout()
plt.savefig("ex_2_01_feature_maps.png", dpi=150, bbox_inches="tight")
plt.close(fig_fmaps)
print(f"\n  Saved: ex_2_01_feature_maps.png")
print(f"  Sample: true='{true_label}', predicted='{pred_class}'")
print("  Layer 1 maps: edges and colour boundaries (high-res, 32x32)")
print("  Layer 2 maps: textures and shapes (lower-res, 8x8 after pooling)")

# 4c. Visualise prediction confidence on a batch of images
with torch.no_grad():
    batch_imgs = X_val[:16]
    batch_logits = simple_cnn(batch_imgs)
    batch_probs = F.softmax(batch_logits, dim=-1)
    batch_preds = batch_logits.argmax(dim=-1)

fig_preds, axes = plt.subplots(2, 8, figsize=(20, 6))
fig_preds.suptitle("SimpleCNN Predictions on Validation Images", fontsize=14)
for i in range(16):
    row, col = i // 8, i % 8
    img_display = denormalise_cifar(batch_imgs[i])
    axes[row, col].imshow(img_display.permute(1, 2, 0).numpy())
    pred_name = CLASS_NAMES[batch_preds[i].item()]
    true_name = CLASS_NAMES[y_val[i].item()]
    conf = batch_probs[i][batch_preds[i]].item()
    correct = batch_preds[i].item() == y_val[i].item()
    colour = "green" if correct else "red"
    axes[row, col].set_title(
        f"{pred_name}\n({conf:.0%})",
        fontsize=8,
        color=colour,
    )
    axes[row, col].axis("off")
plt.tight_layout()
plt.savefig("ex_2_01_predictions.png", dpi=150, bbox_inches="tight")
plt.close(fig_preds)
print("  Saved: ex_2_01_predictions.png")
print("  Green = correct, Red = incorrect. Confidence shown in parentheses.")

# ── Checkpoint 4: Visualisations generated ───────────────────────────
import os

assert os.path.exists("ex_2_01_learned_filters.png"), "Filter visualisation missing"
assert os.path.exists("ex_2_01_feature_maps.png"), "Feature map visualisation missing"
assert os.path.exists("ex_2_01_predictions.png"), "Prediction visualisation missing"
print("\n--- Checkpoint 4 passed --- visual proof of model behaviour saved\n")


# ════════════════════════════════════════════════════════════════════════
# PHASE 5 — APPLY: Singapore E-Commerce Product Categorisation
# ════════════════════════════════════════════════════════════════════════
# SCENARIO: You are an ML engineer at a Singapore e-commerce platform
# (think Shopee, Lazada, or Carousell). The platform receives 500,000+
# new product listings per day. Sellers often mis-categorise products
# (a "Nike Air Max" listed under "Electronics" instead of "Shoes"),
# leading to:
#   - Poor search results (customers can't find what they want)
#   - Incorrect commission rates (different categories have different fees)
#   - Regulatory compliance issues (restricted items in wrong categories)
#
# BUSINESS CASE:
#   - Manual review: 500K listings/day * $0.02/listing = $10,000/day
#   - CNN auto-categorisation: $500/day compute + $2,000/day for edge cases
#   - Net savings: ~$7,500/day = ~$2.7M/year
#   - Additional revenue: improved search -> +3-5% conversion rate
#
# HOW THE CNN APPLIES:
#   Our SimpleCNN learned to classify 32x32 images into 10 categories.
#   A production system would:
#   1. Resize product images to standard dimensions
#   2. Run through a CNN trained on product categories (not CIFAR-10)
#   3. High-confidence predictions (>0.9) auto-categorise
#   4. Low-confidence predictions route to human reviewers
#   5. Track accuracy with ExperimentTracker, retrain monthly
#
# CIFAR-10 AS PROXY:
#   CIFAR-10's categories (airplane, automobile, truck, ship, etc.) map
#   to real e-commerce categories. The architecture patterns are
#   identical — only the training data changes.

print("=" * 70)
print("  PHASE 5 — APPLY: Singapore E-Commerce Product Categorisation")
print("=" * 70)

# Simulate the production triage system using our trained SimpleCNN
simple_cnn.eval()
HIGH_CONFIDENCE_THRESHOLD = 0.85
LOW_CONFIDENCE_THRESHOLD = 0.50

# Map CIFAR-10 classes to e-commerce product categories
ECOMMERCE_MAPPING = {
    "airplane": "Travel & Luggage",
    "automobile": "Automotive",
    "bird": "Pet Supplies",
    "cat": "Pet Supplies",
    "deer": "Home & Garden (Decor)",
    "dog": "Pet Supplies",
    "frog": "Toys & Collectibles",
    "horse": "Sports & Outdoors",
    "ship": "Travel & Luggage",
    "truck": "Automotive",
}

with torch.no_grad():
    val_logits = simple_cnn(X_val)
    val_probs = F.softmax(val_logits, dim=-1)
    val_preds = val_logits.argmax(dim=-1)
    val_confidences = val_probs.gather(1, val_preds.unsqueeze(1)).squeeze()
    val_correct = (val_preds == y_val).float()

# Triage: auto-approve, review, reject
auto_approve_mask = val_confidences >= HIGH_CONFIDENCE_THRESHOLD
review_mask = (val_confidences >= LOW_CONFIDENCE_THRESHOLD) & ~auto_approve_mask
reject_mask = val_confidences < LOW_CONFIDENCE_THRESHOLD

n_total = len(y_val)
n_auto = auto_approve_mask.sum().item()
n_review = review_mask.sum().item()
n_reject = reject_mask.sum().item()

auto_acc = val_correct[auto_approve_mask].mean().item() if n_auto > 0 else 0
review_acc = val_correct[review_mask].mean().item() if n_review > 0 else 0

print(
    f"""
  PRODUCTION TRIAGE SIMULATION (10,000 product listings):

  Category         | Count  | % Total | Accuracy
  -----------------+--------+---------+---------
  Auto-approved    | {n_auto:>5,} | {n_auto/n_total:>6.1%}  | {auto_acc:.1%}
  Needs review     | {n_review:>5,} | {n_review/n_total:>6.1%}  | {review_acc:.1%}
  Low confidence   | {n_reject:>5,} | {n_reject/n_total:>6.1%}  | (routed to human)

  BUSINESS IMPACT (daily projection for 500K listings):
    Auto-approved listings:   {int(500000 * n_auto/n_total):>7,} (no human cost)
    Human review needed:      {int(500000 * n_review/n_total):>7,} (@ $0.02/review)
    Daily review cost:        ${int(500000 * (n_review + n_reject)/n_total * 0.02):>7,}
    vs. full manual review:   $  10,000
    Daily savings:            ${10000 - int(500000 * (n_review + n_reject)/n_total * 0.02):>7,}
    Projected annual savings: ${(10000 - int(500000 * (n_review + n_reject)/n_total * 0.02)) * 365:>10,}

  STAKEHOLDER INSIGHT:
    Auto-approved accuracy of {auto_acc:.1%} means {100 - auto_acc*100:.1f}% error rate
    on automated decisions. For an e-commerce platform, this means
    roughly {int(500000 * n_auto/n_total * (1 - auto_acc)):,} mis-categorised products
    per day slip through without human review.

    RECOMMENDATION: Accept this if mis-categorisation cost < $0.50/item.
    For high-value categories (electronics, luxury), lower the auto-approve
    threshold to 0.95 and accept higher review volume.
"""
)

# Show example auto-approved and review-needed listings
print("  EXAMPLE AUTO-APPROVED LISTINGS:")
auto_indices = torch.where(auto_approve_mask)[0][:5]
for idx in auto_indices:
    pred_name = CLASS_NAMES[val_preds[idx].item()]
    true_name = CLASS_NAMES[y_val[idx].item()]
    conf = val_confidences[idx].item()
    ecom_cat = ECOMMERCE_MAPPING[pred_name]
    status = "CORRECT" if pred_name == true_name else "MIS-CATEGORISED"
    print(
        f"    Listing #{idx.item()}: {pred_name} -> {ecom_cat} "
        f"(conf={conf:.2f}) [{status}]"
    )

print("\n  EXAMPLE REVIEW-NEEDED LISTINGS:")
review_indices = torch.where(review_mask)[0][:5]
for idx in review_indices:
    pred_name = CLASS_NAMES[val_preds[idx].item()]
    true_name = CLASS_NAMES[y_val[idx].item()]
    conf = val_confidences[idx].item()
    print(
        f"    Listing #{idx.item()}: predicted={pred_name} (conf={conf:.2f}), "
        f"true={true_name} -> NEEDS HUMAN REVIEW"
    )

# ── Checkpoint 5: Apply section complete ─────────────────────────────
assert n_auto + n_review + n_reject == n_total, "Triage should cover all samples"
assert auto_acc > 0.7, (
    f"Auto-approved accuracy {auto_acc:.3f} too low -- "
    "high-confidence predictions should be mostly correct"
)
print("\n--- Checkpoint 5 passed --- e-commerce application demonstrated\n")


# ════════════════════════════════════════════════════════════════════════
# Clean up
# ════════════════════════════════════════════════════════════════════════
import asyncio

asyncio.run(conn.close())


# ════════════════════════════════════════════════════════════════════════
# REFLECTION
# ════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  WHAT YOU'VE MASTERED")
print("=" * 70)
print(
    """
  THEORY:
  [x] Convolutions scan local patches with shared weights -- 3x3 filters
      detect edges at every position with only 9 parameters
  [x] Spatial hierarchy: edges -> textures -> parts -> objects, each
      layer building on the previous
  [x] BatchNorm stabilises training; MaxPool adds translation invariance

  BUILD + TRAIN:
  [x] SimpleCNN: Conv2d + BatchNorm + ReLU + MaxPool, two-block design
  [x] Trained on FULL CIFAR-10 (50K images) with ExperimentTracker
  [x] ~{param_count:,} parameters, achieves >{simple_accs[-1]:.0%} validation accuracy

  VISUALISE (the proof):
  [x] Layer 1 filters: oriented edge detectors (Gabor-like patterns)
  [x] Feature maps: which spatial regions activate each filter
  [x] Prediction grid: where the model succeeds and fails, with confidence

  APPLY:
  [x] Singapore e-commerce product auto-categorisation
  [x] Confidence-based triage: auto-approve vs human review
  [x] Business impact: projected annual savings from automation
  [x] Stakeholder-ready accuracy and error rate analysis

  KEY INSIGHT: A CNN is not a black box. The filters it learns are
  interpretable -- they are the visual features the network considers
  important. Filter and feature map visualisation is how you debug and
  explain CNN decisions to non-technical stakeholders.

  Next: In 02_resnet_se.py, you'll see why deeper networks fail and how
  residual connections and attention mechanisms fix the problem...
"""
)

