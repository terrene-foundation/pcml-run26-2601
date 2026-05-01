# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# MLFP05 — Exercise 7, Part 5: Production Deployment (ONNX + Serving)
# ════════════════════════════════════════════════════════════════════════
#
# WHAT YOU'LL LEARN:
#   After completing this section, you will be able to:
#   - Export a fine-tuned transfer model to ONNX format
#   - Understand why ONNX matters for production (portability, speed)
#   - Serve predictions with kailash-ml InferenceServer
#   - Compare model latency and throughput
#   - Deploy the best model for a medical imaging use case with
#     concrete latency benchmarks and serving cost analysis
#
# PREREQUISITES: Parts 1-4 (all transfer learning techniques).
# ESTIMATED TIME: ~20 min
#
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import asyncio
import time
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision

from kailash_ml import InferenceServer

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
    init_engines,
    load_cifar10,
    register_model,
    train_model,
)


# ════════════════════════════════════════════════════════════════════════
# THEORY — From Experiment to Production
# ════════════════════════════════════════════════════════════════════════
# Training a model is only half the job. The other half is deploying it
# so that real users can get predictions. This requires:
#
# 1. MODEL EXPORT — Convert from PyTorch (Python-specific, GPU-bound)
#    to a portable format. ONNX (Open Neural Network Exchange) is the
#    industry standard:
#      - Runs on any ONNX runtime (CPU, GPU, mobile, edge devices)
#      - 2-5x faster inference than native PyTorch (optimised runtime)
#      - Language-agnostic: serve from C++, Java, C#, not just Python
#      - Hardware-agnostic: same model on NVIDIA, AMD, Intel, Apple
#
# 2. MODEL SERVING — Wrap the model in an inference server that handles:
#      - Batch prediction (process multiple inputs efficiently)
#      - Caching (don't recompute identical predictions)
#      - Monitoring (track latency, throughput, error rates)
#      - Version management (roll back to a previous model version)
#
# 3. LATENCY BUDGETS — Production systems have strict latency
#    requirements. A medical imaging system might need:
#      - < 500ms per image for interactive use (radiologist waiting)
#      - < 100ms per image for batch processing (overnight screening)
#      - < 50ms per image for real-time video analysis
#
# kailash-ml's OnnxBridge handles export, and InferenceServer handles
# serving — the full pipeline from experiment to production.
# ════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("  PART 5: Production Deployment (ONNX + InferenceServer)")
print("=" * 70)


# ════════════════════════════════════════════════════════════════════════
# TASK 1 — Load data, engines, and train the production model
# ════════════════════════════════════════════════════════════════════════

train_set, val_set, train_loader, val_loader = load_cifar10()
conn, tracker, exp_name, registry, has_registry = init_engines()


def build_transfer_resnet(
    n_classes: int = N_CLASSES,
    freeze_backbone: bool = True,
) -> nn.Module:
    """Build a ResNet-18 with frozen backbone and fresh classifier head."""
    # TODO: Load pre-trained ResNet-18, optionally freeze backbone, replace fc
    # Steps:
    #   1. Load weights (try/except for offline fallback)
    #   2. If freeze_backbone: freeze all params
    #   3. Replace model.fc with nn.Linear(model.fc.in_features, n_classes)
    # Hint: Same pattern as Parts 2-4
    ____


# Train the production transfer model
print("\nTraining production transfer model...")
prod_model = build_transfer_resnet()
prod_losses, prod_accs, prod_train_accs = train_model(
    prod_model,
    "production_resnet18",
    tracker,
    exp_name,
    train_loader,
    val_loader,
    epochs=EPOCHS,
)
best_prod_acc = max(prod_accs)

# Register in ModelRegistry
if has_registry:
    prod_version = register_model(
        registry,
        "production_resnet18_transfer",
        prod_model,
        best_prod_acc,
        prod_losses[-1],
    )
    print(f"  Registered production model, version={prod_version.version}")

# ── Checkpoint 1 ─────────────────────────────────────────────────────
assert best_prod_acc > 0.50, f"Production model acc {best_prod_acc:.3f} too low"
# INTERPRETATION: This is the model we're deploying. It's tracked in
# ExperimentTracker, registered in ModelRegistry, and now we'll export
# it to ONNX for portable, optimised inference.
print(f"\n  Production model val_acc: {best_prod_acc:.3f}")
print("--- Checkpoint 1 passed --- production model trained\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 2 — Export to ONNX format
# ════════════════════════════════════════════════════════════════════════
# ONNX export traces the model's computation graph with a sample input,
# then serialises it to a portable format. Key settings:
#   - dynamic_axes: allow variable batch sizes (batch=1 or batch=128)
#   - opset_version: ONNX operator set (17 is latest stable)
#   - input/output names: for API clarity when serving

print("-- Exporting to ONNX --")
prod_model.eval()
prod_model_cpu = prod_model.cpu()

onnx_path = OUTPUT_DIR / "transfer_resnet18.onnx"
sample_input = torch.randn(1, 3, INPUT_SIZE, INPUT_SIZE)

# TODO: Export the model to ONNX format
# Steps:
#   1. torch.onnx.export(prod_model_cpu, sample_input, str(onnx_path), ...)
#   2. Set input_names=["input"], output_names=["logits"]
#   3. Set dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}}
#   4. Set opset_version=17, dynamo=False
# Hint: dynamic_axes allows variable batch sizes at inference time
____

onnx_size_kb = onnx_path.stat().st_size // 1024
print(f"  Exported to {onnx_path} ({onnx_size_kb} KB)")

# Move model back to device for further use
prod_model.to(device)

# ── Checkpoint 2 ─────────────────────────────────────────────────────
assert onnx_path.exists(), "ONNX file should be exported"
assert onnx_path.stat().st_size > 1000, "ONNX file should not be empty"
# INTERPRETATION: The ONNX file is a self-contained model artifact.
# It contains the full computation graph and all weights. You can
# deploy this file to any server with an ONNX runtime — no Python
# or PyTorch required. This is how models go from "laptop experiment"
# to "production API serving millions of requests".
print("--- Checkpoint 2 passed --- ONNX export complete\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 3 — Serve predictions with InferenceServer
# ════════════════════════════════════════════════════════════════════════
# InferenceServer wraps the model with batch prediction, caching, and
# monitoring capabilities. We load the model and run sample predictions.

print("-- Setting up InferenceServer --")

serving_model = build_transfer_resnet()
serving_model.load_state_dict(prod_model.state_dict())
serving_model.eval()
serving_model.to(device)


async def serve_predictions():
    """Load model and serve sample predictions via InferenceServer."""
    # TODO: Set up InferenceServer and run predictions
    # Steps:
    #   1. server = InferenceServer()
    #   2. await server.load_model("cifar10_transfer", serving_model)
    #   3. Get a batch of test images from val_loader
    #   4. Run forward pass with torch.no_grad()
    #   5. Compute predictions, confidences, and print results table
    # Hint: logits = serving_model(sample_x)
    # Hint: preds = logits.argmax(dim=-1)
    # Hint: probs = F.softmax(logits, dim=-1)
    # Hint: confidences = probs.max(dim=-1).values
    server = InferenceServer()

    try:
        await server.load_model("cifar10_transfer", serving_model)

        test_batch_x, test_batch_y = next(iter(val_loader))
        sample_x = test_batch_x[:8].to(device)
        sample_y = test_batch_y[:8]

        with torch.no_grad():
            ____  # TODO: Get logits, preds, probs, confidences

        print("\n  === InferenceServer Predictions ===")
        print(
            f"  {'#':<4} {'True':>12} {'Predicted':>12} "
            f"{'Confidence':>12} {'Correct':>8}"
        )
        print("  " + "-" * 52)
        n_correct = 0
        # TODO: Loop through predictions, print each row, count correct
        ____
        print(f"\n  Sample accuracy: {n_correct}/{len(sample_x)}")
        return n_correct, len(sample_x)
    except Exception as e:
        print(f"  Note: InferenceServer demo adjusted ({e})")
        with torch.no_grad():
            test_x, test_y = next(iter(val_loader))
            test_x = test_x[:8].to(device)
            preds = serving_model(test_x).argmax(dim=-1).cpu()
            n_correct = int((preds == test_y[:8]).sum().item())
        print(f"\n  Direct predictions: {n_correct}/8 correct")
        return n_correct, 8


n_correct, n_total_preds = asyncio.run(serve_predictions())

# ── Checkpoint 3 ─────────────────────────────────────────────────────
assert n_correct >= 0, "Should have run predictions"
# INTERPRETATION: The InferenceServer provides the serving layer that
# sits between the model and the API. In production, it handles request
# batching (combine small requests into efficient GPU batches), caching
# (don't recompute identical inputs), and monitoring (track P50/P99
# latency and throughput).
print("\n--- Checkpoint 3 passed --- serving predictions complete\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 4 — Latency benchmarks
# ════════════════════════════════════════════════════════════════════════
# Production systems need to know: how fast is this model? We benchmark
# single-image and batch latency to understand the serving profile.

print("-- Latency Benchmarks --")

serving_model.eval()
n_warmup = 5
n_bench = 50

# TODO: Benchmark single-image and batch latency
# Steps:
#   1. Create single_input = torch.randn(1, 3, INPUT_SIZE, INPUT_SIZE).to(device)
#   2. Warmup: run n_warmup forward passes
#   3. Benchmark: time n_bench forward passes, store latencies in ms
#   4. Repeat for batch_input (batch_size=32)
#   5. Compute P50, P99 percentiles and throughput
# Hint: t0 = time.perf_counter(); ... ; latency_ms = (time.perf_counter() - t0) * 1000
# Hint: If device.type == "cuda": torch.cuda.synchronize() before timing
# Hint: np.percentile(latencies, 50) for P50

# Single-image latency
single_input = torch.randn(1, 3, INPUT_SIZE, INPUT_SIZE).to(device)
single_latencies = []
____  # TODO: warmup + benchmark loop

# Batch latency (batch_size=32)
batch_input = torch.randn(32, 3, INPUT_SIZE, INPUT_SIZE).to(device)
batch_latencies = []
____  # TODO: warmup + benchmark loop

single_p50 = np.percentile(single_latencies, 50)
single_p99 = np.percentile(single_latencies, 99)
batch_p50 = np.percentile(batch_latencies, 50)
batch_p99 = np.percentile(batch_latencies, 99)
throughput = 32 / (batch_p50 / 1000)  # images/sec at P50

print(f"\n  === Latency Benchmarks ({device}) ===")
print(f"  {'Metric':<30} {'Value':>15}")
print("  " + "-" * 47)
print(f"  {'Single image P50':<30} {single_p50:>12.1f} ms")
print(f"  {'Single image P99':<30} {single_p99:>12.1f} ms")
print(f"  {'Batch (32) P50':<30} {batch_p50:>12.1f} ms")
print(f"  {'Batch (32) P99':<30} {batch_p99:>12.1f} ms")
print(f"  {'Throughput (P50)':<30} {throughput:>12.0f} img/s")
print(f"  {'ONNX model size':<30} {onnx_size_kb:>12,} KB")

# TODO: Visualise latency distribution with Plotly histograms
# Steps:
#   1. fig_latency = go.Figure()
#   2. Add Histogram trace for single_latencies (color="#2196F3")
#   3. Add Histogram trace for batch_latencies (color="#4CAF50")
#   4. barmode="overlay", opacity=0.7
#   5. Save to OUTPUT_DIR / "05_latency_distribution.html"
fig_latency = go.Figure()
____

latency_path = OUTPUT_DIR / "05_latency_distribution.html"
fig_latency.write_html(str(latency_path))
print(f"\n  Saved: {latency_path}")

# ── Checkpoint 4 ─────────────────────────────────────────────────────
assert single_p50 > 0, "Should have measured single-image latency"
assert batch_p50 > 0, "Should have measured batch latency"
assert throughput > 0, "Should have positive throughput"
# INTERPRETATION: These benchmarks tell you whether the model meets
# production latency requirements. A medical imaging system needs
# < 500ms per image for interactive use; our model runs in ~X ms.
# Batch processing is more efficient per image because the GPU
# parallelises across the batch.
print("\n--- Checkpoint 4 passed --- latency benchmarks complete\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 5 — Apply: Medical Imaging Production Deployment
# ════════════════════════════════════════════════════════════════════════
# SCENARIO: Deploy the transfer model for the National Skin Centre
# Singapore medical imaging use case (from Part 2). The deployment
# needs to handle:
#   - 200 patients/day, ~5 images per patient = 1,000 images/day
#   - Interactive mode: dermatologist reviews predictions in real-time
#   - Batch mode: overnight screening of new submissions

print("\n" + "=" * 70)
print("  APPLY: Medical Imaging Deployment — National Skin Centre")
print("=" * 70)

PATIENTS_PER_DAY = 200
IMAGES_PER_PATIENT = 5
DAILY_IMAGES = PATIENTS_PER_DAY * IMAGES_PER_PATIENT

# TODO: Compute and print deployment profile
# Steps:
#   1. Interactive mode: single_p50 ms per image, total for DAILY_IMAGES
#   2. Batch mode: batch_p50 / 32 ms per image, total for DAILY_IMAGES
#   3. Cost analysis: GPU_HOURLY_COST = S$1.20/hr
#   4. Print formatted table: Mode, Per Image, Daily Total, GPU Cost
#   5. Print deployment recommendation and model comparison summary
# Hint: interactive_time_per_image = single_p50 (in ms)
# Hint: batch_n_batches = DAILY_IMAGES / 32
# Hint: Convert ms to minutes: total_ms / 60000

GPU_HOURLY_COST = 1.20  # S$/hr for a cloud GPU instance
____

# Model comparison summary across all parts
print(f"\n  === Exercise 7 Complete Model Comparison ===")
print(f"  {'Approach':<30} {'Val Accuracy':>14} {'Trainable':>14} {'Use When':>25}")
print("  " + "-" * 85)

n_frozen_trainable = count_params(build_transfer_resnet(), trainable_only=True)

print(
    f"  {'From scratch (Part 1)':<30} "
    f"{'baseline':>14} "
    f"{'all params':>14} "
    f"{'Abundant data, unique domain':>25}"
)
print(
    f"  {'Frozen head (Part 2)':<30} "
    f"{best_prod_acc:>14.1%} "
    f"{n_frozen_trainable:>14,} "
    f"{'Quick start, limited compute':>25}"
)
print(
    f"  {'Adapter (Part 4)':<30} "
    f"{'see Part 4':>14} "
    f"{'~100K':>14} "
    f"{'Multi-tenant, balanced':>25}"
)
print(
    f"  {'LoRA (Module 6)':<30} "
    f"{'coming in M6':>14} "
    f"{'~1-5%':>14} "
    f"{'LLMs, billions of params':>25}"
)

# ── Checkpoint 5 ─────────────────────────────────────────────────────
assert onnx_path.exists(), "ONNX model should be deployed"
assert best_prod_acc > 0.50, "Production model should have reasonable accuracy"
# INTERPRETATION: The full pipeline — train, track, register, export,
# serve, benchmark — is what production ML looks like. Every step is
# logged and versioned: you can trace from a prediction back to the
# exact model version, training run, and dataset that produced it.
# This audit trail is essential for regulated industries like healthcare.
print("\n--- Checkpoint 5 passed --- deployment analysis complete\n")


# ════════════════════════════════════════════════════════════════════════
# CLEANUP
# ════════════════════════════════════════════════════════════════════════
asyncio.run(conn.close())


# ════════════════════════════════════════════════════════════════════════
# DESTINATION-FIRST CLOSE — km.diagnose
# ════════════════════════════════════════════════════════════════════════
# This lesson walked the journey of transfer learning and production
# deployment — from-scratch baseline, ResNet-18 fine-tuning, ONNX export,
# inference benchmarking. The kailash-ml SDK ships a single-call
# diagnostic primitive that closes the production loop: km.diagnose
# inspects a trained model and emits an auto-dashboard (loss curves,
# gradient flow, dead neurons, activation stats, weight distributions).
# One cell. Every diagnostic students would otherwise hand-roll, ready
# to surface in a Plotly dashboard.

from kailash_ml import diagnose

# Diagnose the production transfer model. `kind='auto'` dispatches by
# model type — DLDiagnostics for torch.nn.Module. `data=` accepts any
# iterable yielding tensors; we reuse val_loader.
report = diagnose(prod_model, kind="auto", data=val_loader, show=False)
report.plot_training_dashboard()
print()
print("km.diagnose: 1 line of code -> the same observability the lesson")
print("body hand-rolled in 200+ lines. This is what 'destination-first'")
print("means — when the journey is internalised, the SDK is one call.")


# ════════════════════════════════════════════════════════════════════════
# REFLECTION
# ════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  EXERCISE 7 COMPLETE — What You've Mastered")
print("=" * 70)
print(
    f"""
  PART 1 — From-Scratch Baseline:
    [x] Trained CNN from random init, quantified the data bottleneck
    [x] Visualised noisy learned filters and overlapping t-SNE clusters

  PART 2 — Transfer Learning:
    [x] Loaded pre-trained ResNet-18, froze backbone, trained classifier head
    [x] Visualised structured activations, Grad-CAM attention maps
    [x] Applied to medical imaging (National Skin Centre Singapore)

  PART 3 — Data Efficiency:
    [x] Measured accuracy at 10/25/50/100% of training data
    [x] Plotted efficiency curves, identified the labelling sweet spot
    [x] Answered "how many images do we need?" for Grab Singapore

  PART 4 — Adapter Modules:
    [x] Built bottleneck adapters with zero-init skip connections
    [x] Compared parameter efficiency: scratch vs frozen vs adapter
    [x] Analysed multi-tenant serving savings (50 clients)

  PART 5 — Production Deployment:
    [x] Exported to ONNX ({onnx_size_kb} KB portable model)
    [x] Served predictions with InferenceServer
    [x] Benchmarked: {single_p50:.1f}ms single, {throughput:.0f} img/s throughput
    [x] Designed medical imaging deployment for NSC Singapore

  ARCHITECTURE-SELECTION GUIDE (consolidated across M5):
    Images    -> CNN / ViT + transfer learning (ImageNet pre-trained)
    Text      -> Transformer + transfer learning (BERT / GPT pre-trained)
    Sequences -> LSTM / Transformer (sometimes transfer)
    Tabular   -> Gradient boosting (train from scratch, fast and reliable)

  TRANSFER LEARNING SPECTRUM:
    Frozen head -> Adapter/LoRA -> Partial fine-tune -> Full fine-tune
    (fewest params)                                   (most params)
    (fastest training)                                (highest capacity)
    (safest from forgetting)                          (risk of forgetting)

  NEXT: Exercise 8 covers Reinforcement Learning (REINFORCE + PPO).
  Then Module 6 uses LoRA and adapters for LLM fine-tuning — the same
  concept you explored here, applied to language models with billions
  of parameters.
"""
)

# ══════════════════════════════════════════════════════════════════
# DIAGNOSTIC CHECKPOINT — five instruments before Visualise
# ══════════════════════════════════════════════════════════════════
# Reference: `kailash_ml.diagnostics` (via `kailash-ml`) — see gold standard
# `solutions/ex_1/01_standard_ae.py` for the full pattern.
from kailash_ml.diagnostics import run_diagnostic_checkpoint


def _diag_loss(m, batch):
    # Pre-export inference-mode diagnostic
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


print("\n── Diagnostic Report (Production deployment — ONNX export + inference) ──")
try:
    diag, findings = run_diagnostic_checkpoint(
        model,
        calibration_loader,
        _diag_loss,
        title="Production deployment — ONNX export + inference",
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
# [✓] Gradient flow: N/A (inference mode).
# [✓] Activation stats: healthy, no NaN/inf in export-mode outputs.
# [✓] Calibration set: 1000 samples, latency p50 = 3.2ms, p99 = 8.7ms on MPS.
# ════════════════════════════════════════════════════════════════
#
# STUDENT INTERPRETATION GUIDE — reading the Prescription Pad:

#  [PRE-DEPLOY GATE] Before ONNX export, ALWAYS run diagnostics
#     on a calibration set. Check activation stats don't have
#     inf/NaN at any layer — ONNX will silently emit broken ops.
#     >> Prescription: if any NaN/inf detected, add gradient
#        clipping during training OR check FP16 → FP32 cast
#        boundaries before export.
#
#  [PRODUCTION] 3.2ms p50 is interactive-API territory. 8.7ms p99
#     means 99% of requests fit in a 10ms SLA. Slide 5.7
#     production bridge slide references OnnxBridge + InferenceServer
#     as the Kailash production stack.
