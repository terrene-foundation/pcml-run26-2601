# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# MLFP05 — Lesson 5.0 (Prelude): The Destination — kailash-ml in 5 lines
# ════════════════════════════════════════════════════════════════════════
#
# WHY THIS LESSON COMES FIRST:
#   The next 8 lessons teach you how to build neural networks from the
#   ground up — autoencoders, CNNs, RNNs, transformers, GANs, GNNs,
#   transfer learning, RL. By the end of M5 you will know what every
#   line of a PyTorch training loop does and why.
#
#   Before that journey, see the destination. This is what production
#   ML training looks like in 2026 with the unified `kailash-ml` 1.1+
#   surface. The same training that takes 80 lines of hand-written
#   PyTorch fits in 3 — and runs on Apple Silicon's Metal Performance
#   Shaders backend automatically, with mixed-precision, with no flags.
#
#   Note on async: `km.train()` is async (kailash-ml 1.0+). In a CLI
#   script you wrap with `asyncio.run(km.train(...))`. In a notebook
#   with `nest_asyncio`, top-level `await km.train(...)` works.
#
# WHAT YOU'LL DO:
#   1. Print the auto-detected compute backend
#   2. Train a binary classifier in three lines
#   3. Construct an MLEngine and inspect its defaults
#
# PREREQUISITES: M4.8 (neural-network basics)
# ESTIMATED TIME: ~5 min (<30s of compute)
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import asyncio  # noqa: F401  — used in TASK 2

import polars as pl
from sklearn.datasets import make_classification

import kailash_ml as km


# ════════════════════════════════════════════════════════════════════════
# TASK 1 — Print the compute backend kailash-ml auto-selected
# ════════════════════════════════════════════════════════════════════════
# Hint: km.device() returns a BackendInfo with .backend, .accelerator,
# .precision, and .capabilities — print each so you can see what your
# machine resolved to (mps on Mac, cuda on NVIDIA, cpu otherwise).

print("=" * 72)
print("TASK 1 — Compute backend (auto-detected)")
print("=" * 72)
backend = ____
print(f"  backend       : {backend.____}")
print(f"  accelerator   : {backend.____}")
print(f"  precision     : {backend.____}")
print(f"  capabilities  : {sorted(backend.____)}")

# ── Checkpoint 1 ────────────────────────────────────────────────────────
assert backend.backend in {
    "mps",
    "cuda",
    "cpu",
    "rocm",
    "xpu",
}, "backend should be one of mps/cuda/cpu/rocm/xpu"
print("✓ Checkpoint 1 passed — backend detected\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 2 — Train a classifier in three lines
# ════════════════════════════════════════════════════════════════════════
# Hint: km.train(df, target='y') runs a default RandomForestClassifier
# pipeline end-to-end (split, fit, evaluate, package metrics). Pass the
# polars DataFrame and the name of the target column.
#
# km.train is async — call it via asyncio.run(km.train(df, target='y'))
# in this CLI script. In a notebook with nest_asyncio, top-level await
# works directly.

print("=" * 72)
print("TASK 2 — km.train() three-line zero-config training")
print("=" * 72)

X, y = make_classification(
    n_samples=800, n_features=10, n_informative=6, random_state=42
)
df = pl.DataFrame({**{f"f{i}": X[:, i] for i in range(10)}, "y": y})
print(f"  dataset shape : {df.shape}")

# TODO: call asyncio.run(km.train(df, target='y'))  — km.train is async
result = ____

print(f"  metrics       : {result.metrics}")

# ── Checkpoint 2 ────────────────────────────────────────────────────────
assert "accuracy" in result.metrics, "training should report an accuracy metric"
assert (
    result.metrics["accuracy"] > 0.8
), "make_classification with n_informative=6 should be easy (>0.8)"
print("✓ Checkpoint 2 passed — model trained\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 3 — Construct an MLEngine and inspect its defaults
# ════════════════════════════════════════════════════════════════════════
# Hint: km.MLEngine() with no arguments uses accelerator='auto' which
# resolves to the same backend km.device() picked. Inspect .accelerator
# and .backend_info to confirm.

print("=" * 72)
print("TASK 3 — MLEngine: the unified surface")
print("=" * 72)
engine = ____
print(f"  engine.accelerator  : {engine.accelerator}")
print(
    f"  engine.backend_info : {engine.backend_info.backend}/"
    f"{engine.backend_info.precision}"
)

# ── Checkpoint 3 ────────────────────────────────────────────────────────
assert (
    engine.accelerator == backend.accelerator
), "MLEngine should pick the same backend as km.device()"
print("✓ Checkpoint 3 passed — engine wired to detected backend\n")


# ════════════════════════════════════════════════════════════════════════
# REFLECTION
# ════════════════════════════════════════════════════════════════════════
print("=" * 72)
print("Reflection")
print("=" * 72)
print(
    "What you've mastered:\n"
    "  ✓ Compute backend selected automatically (no env vars, no flags)\n"
    "  ✓ Production training in 3 lines (km.train returns metrics)\n"
    "  ✓ The MLEngine surface every M5 lesson will use\n\n"
    "Next: In Lesson 5.1 you will build the autoencoder that lives\n"
    "underneath one of these km.train() calls — every layer, every loss\n"
    "function, every gradient update. The destination will make sense\n"
    "once you have walked the journey to it.\n"
)
