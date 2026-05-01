# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# MLFP04 — Exercise 8.5: Regularisation, Clipping, Early Stop, ONNX Export
# ════════════════════════════════════════════════════════════════════════
#
# WHAT YOU'LL LEARN:
#   - Contrast baseline vs dropout vs batch-norm vs both
#   - Apply gradient clipping to prevent exploding gradients
#   - Implement early stopping on validation loss
#   - Export the trained CNN to ONNX via kailash-ml OnnxBridge
#
# PREREQUISITES: 04_optimisers_schedulers.py
#
# ESTIMATED TIME: ~45 min
#
# TASKS:
#   1. Theory — regularisation tricks compound
#   2. Build — four regularisation variants + EarlyStopping class
#   3. Train — full loop with clipping, scheduler, early stop
#   4. Visualise — val loss across variants + final gradient norms
#   5. Apply — DSO Labs Singapore: ONNX export cut inference cost by 7x
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.optim as optim

from kailash_ml import OnnxBridge

from shared.mlfp04.ex_8 import (
    N_CLASSES,
    OUTPUT_DIR,
    TriageCNN,
    build_sg_loaders,
    device,
    eval_cnn,
    train_cnn_one_epoch,
    viz,
)

print("\n" + "=" * 70)
print("  Regularisation, Clipping, Early Stopping, ONNX Export")
print("=" * 70)

# ════════════════════════════════════════════════════════════════════════
# THEORY — Compounding regularisers
# ════════════════════════════════════════════════════════════════════════
# Dropout breaks co-adaptation. BatchNorm stabilises the loss landscape.
# Weight decay shrinks weights. Gradient clipping caps updates. Early
# stopping uses training-time itself as a regulariser. Combined, they
# make training robust without heavy per-task tuning.

train_loader, test_loader, X_test_np, y_test_np = build_sg_loaders()
criterion = nn.BCEWithLogitsLoss()


# ════════════════════════════════════════════════════════════════════════
# TASK 2 — BUILD: regularisation variants and EarlyStopping
# ════════════════════════════════════════════════════════════════════════


def build_variant(use_dropout: bool, use_bn: bool) -> nn.Module:
    """Small CNN variant with configurable regularisation."""
    layers: list[nn.Module] = [nn.Conv2d(1, 32, 3, padding=1)]
    if use_bn:
        layers.append(nn.BatchNorm2d(32))
    layers += [nn.ReLU(), nn.MaxPool2d(2), nn.Conv2d(32, 64, 3, padding=1)]
    if use_bn:
        layers.append(nn.BatchNorm2d(64))
    layers += [
        nn.ReLU(),
        nn.AdaptiveAvgPool2d(4),
        nn.Flatten(),
        nn.Linear(64 * 4 * 4, 64),
        nn.ReLU(),
    ]
    if use_dropout:
        layers.append(nn.Dropout(0.3))
    layers.append(nn.Linear(64, N_CLASSES))
    return nn.Sequential(*layers).to(device)


class EarlyStopping:
    """Monitor validation loss and stop when it plateaus."""

    def __init__(self, patience: int = 4, min_delta: float = 1e-3) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float("inf")
        self.should_stop = False

    def step(self, val_loss: float) -> bool:
        # TODO: If val_loss is more than min_delta better than best_loss,
        # update best_loss and reset counter. Otherwise, increment counter
        # and set should_stop=True when counter >= patience.
        if val_loss < self.best_loss - self.min_delta:
            ____
        else:
            ____
            if ____:
                self.should_stop = True
        return self.should_stop


# ════════════════════════════════════════════════════════════════════════
# TASK 3 — TRAIN: variant grid + full pipeline
# ════════════════════════════════════════════════════════════════════════
print("\n--- Regularisation grid (6 epochs each) ---")
variant_labels = [
    (False, False, "Baseline"),
    (True, False, "Dropout"),
    (False, True, "BatchNorm"),
    (True, True, "Dropout+BN"),
]

variant_val_histories: dict[str, list[float]] = {}
for use_do, use_bn, label in variant_labels:
    net = build_variant(use_do, use_bn)
    opt_ = optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
    val_curve: list[float] = []
    for _ in range(6):
        train_cnn_one_epoch(net, train_loader, opt_, criterion)
        val_curve.append(eval_cnn(net, test_loader, criterion))
    variant_val_histories[label] = val_curve
    print(f"  {label:<12}: val {val_curve[0]:.4f} -> {val_curve[-1]:.4f}")

assert (
    variant_val_histories["Dropout+BN"][-1]
    <= variant_val_histories["Baseline"][-1] + 0.2
), "Task 3: Dropout+BN should not be dramatically worse than baseline"

# Full pipeline: TriageCNN + AdamW + cosine + clip + early stop
print("\n--- Full pipeline: TriageCNN + AdamW + cosine + clip + early stop ---")
model = TriageCNN(n_classes=N_CLASSES, dropout_rate=0.3).to(device)
optimiser = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

n_epochs = 12
# TODO: Create a cosine annealing scheduler with T_max=n_epochs, eta_min=1e-6.
scheduler = ____
# TODO: Create an EarlyStopping instance with patience=4, min_delta=1e-3.
stopper = ____

history = {"train_loss": [], "val_loss": [], "lr": [], "grad_norm": []}
for epoch in range(n_epochs):
    # TODO: Call train_cnn_one_epoch with clip_value=1.0 so the shared helper
    # applies gradient-norm clipping at 1.0.
    train_loss, grad = ____
    val_loss = eval_cnn(model, test_loader, criterion)
    history["train_loss"].append(train_loss)
    history["val_loss"].append(val_loss)
    history["lr"].append(scheduler.get_last_lr()[0])
    history["grad_norm"].append(grad)
    scheduler.step()
    print(
        f"  Epoch {epoch + 1:>2}/{n_epochs}: "
        f"train={train_loss:.4f} val={val_loss:.4f} "
        f"lr={history['lr'][-1]:.6f} grad={grad:.3f}"
    )
    if stopper.step(val_loss):
        print(f"  -> early stop triggered at epoch {epoch + 1}")
        break

assert stopper.best_loss < math.inf, "Task 3: early stopping must record a best loss"
assert all(
    g > 0 for g in history["grad_norm"]
), "Task 3: gradient norms must be positive"
print("\n[ok] Checkpoint passed — full training pipeline complete\n")

# ════════════════════════════════════════════════════════════════════════
# TASK 4 — VISUALISE
# ════════════════════════════════════════════════════════════════════════
fig_variants = viz.training_history(variant_val_histories, x_label="Epoch")
fig_variants.update_layout(title="Regularisation Variants — Validation BCE")
fig_variants.write_html(OUTPUT_DIR / "05_variants_val.html")

fig_full = viz.training_history(
    {
        "Train BCE": history["train_loss"],
        "Val BCE": history["val_loss"],
        "Grad Norm": history["grad_norm"],
    },
    x_label="Epoch",
)
fig_full.update_layout(title="Full Pipeline — Loss and Gradient Norm")
fig_full.write_html(OUTPUT_DIR / "05_full_pipeline.html")
print("[viz] 05_variants_val.html + 05_full_pipeline.html saved")

# ════════════════════════════════════════════════════════════════════════
# TASK 5 — APPLY: DSO National Laboratories Singapore — ONNX Export
# ════════════════════════════════════════════════════════════════════════
# SCENARIO: DSO runs imagery triage on edge devices with 512MB RAM and
# no Python runtime. PyTorch's 1.8GB footprint doesn't fit; ONNX Runtime
# (48MB, C++) does. Kailash-ml's OnnxBridge wraps the conversion,
# validates the exported model, and records the export for governance.

print("--- ONNX export via kailash-ml OnnxBridge + torch.onnx ---")

# Step 1: Compatibility check with the kailash-ml engine layer.
# TODO: Instantiate an OnnxBridge().
bridge = ____

compat = bridge.check_compatibility(model, framework="pytorch")
print(f"Compatible:  {compat.compatible}")
print(f"Confidence:  {compat.confidence}")

# Step 2: Export via torch.onnx.export (OnnxBridge.export natively handles
# sklearn/lightgbm; PyTorch models go through torch.onnx). Kailash-ml's
# InferenceServer (Module 5) consumes the resulting .onnx file.
onnx_output = OUTPUT_DIR / "triage_cnn.onnx"
model.eval()
dummy_input = torch.from_numpy(X_test_np[:1]).to(device)

# TODO: Call torch.onnx.export(model, dummy_input, str(onnx_output),
#   input_names=["image"], output_names=["logits"],
#   dynamic_axes={"image": {0: "batch"}, "logits": {0: "batch"}},
#   opset_version=17)
____

export_size_bytes = onnx_output.stat().st_size
print(f"Exported:    {onnx_output.name}")
print(f"Model size:  {export_size_bytes / 1024:.1f} KB")

# Step 3: Validate against the PyTorch original using ONNX Runtime.
import numpy as _np
import onnxruntime as _ort

session = _ort.InferenceSession(str(onnx_output))
sample = X_test_np[:10]
with torch.no_grad():
    torch_out = model(torch.from_numpy(sample).to(device)).cpu().numpy()

# TODO: Run the ONNX session on the numpy sample and measure the max
# absolute difference from torch_out.
# Hint: session.run(None, {"image": sample})[0]  -> numpy array
onnx_out = ____
max_diff = float(_np.max(_np.abs(torch_out - onnx_out)))
print(f"Max diff (torch vs onnx): {max_diff:.8f}")

assert compat.compatible, "Task 5: TriageCNN must be ONNX-compatible"
assert onnx_output.exists(), "Task 5: ONNX file must be written"
assert (
    max_diff < 1e-4
), f"Task 5: ONNX output must match PyTorch within 1e-4, got {max_diff}"
print("\n[ok] Checkpoint passed — model exported to ONNX and validated\n")

# BUSINESS IMPACT:
#   PyTorch runtime 1.8GB -> ONNX Runtime 48MB. 7x reduction in inference
#   cost per edge node, same numerical output within 1e-4 tolerance.

# ════════════════════════════════════════════════════════════════════════
# REFLECTION
# ════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("  WHAT YOU'VE MASTERED")
print("=" * 70)
print(
    """
  [x] Compared four regularisation variants on identical budgets
  [x] Built EarlyStopping with patience + min_delta
  [x] Wired cosine decay + gradient clipping + early stopping together
  [x] Exported the trained model to ONNX via kailash-ml OnnxBridge

  This completes Exercise 8 and Module 4. Module 5 takes these blocks
  to real Fashion-MNIST, CIFAR-10, and transfer-learning backbones.
"""
)
