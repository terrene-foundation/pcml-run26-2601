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
#   Note on async: `km.train()` and `km.register()` are async (kailash-ml
#   1.0+ canonical pair). In a notebook with `nest_asyncio`, `await` works
#   at top level. In a CLI script (this file), wrap with `asyncio.run()`.
#
# WHAT YOU'LL SEE:
#   1. `km.device()` — auto-detects MPS / CUDA / ROCm / Intel XPU / CPU
#   2. `km.train(df, target=...)` — three-line zero-config training
#   3. `MLEngine().fit(...)` — the unified surface every framework family
#      goes through
#
# WHAT YOU WON'T DO YET:
#   Build the model. Compute the loss. Write a training loop. Backprop.
#   That comes next, in Lesson 5.1. You will return to this prelude in
#   the M5 reflection at the end and recognise every line.
#
# PREREQUISITES: M4.8 (neural-network basics)
# ESTIMATED TIME: ~5 min (<30s of compute)
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import asyncio

import polars as pl
from sklearn.datasets import make_classification

import kailash_ml as km


# ════════════════════════════════════════════════════════════════════════
# DEMO 1 — `km.device()`: the same backend the rest of the SDK picks
# ════════════════════════════════════════════════════════════════════════
#
# kailash-ml's BackendInfo answers four questions every M5 lesson asks:
#   * which compute backend?           (mps / cuda / rocm / xpu / cpu)
#   * which Lightning accelerator?     (mps / gpu / cpu)
#   * which precision?                 (16-mixed / bf16-mixed / 32)
#   * what capabilities?               (fp16, bf16, tensor cores, …)
#
# On an Apple Silicon Mac you will see `backend=mps precision=16-mixed`.
# On an NVIDIA RTX 4090 you will see `backend=cuda precision=16-mixed`.
# On a CPU-only laptop you will see `backend=cpu precision=32`.

print("=" * 72)
print("DEMO 1 — Compute backend (auto-detected)")
print("=" * 72)
backend = km.device()
print(f"  backend       : {backend.backend}")
print(f"  accelerator   : {backend.accelerator}")
print(f"  precision     : {backend.precision}")
print(f"  capabilities  : {sorted(backend.capabilities)}")
print(f"  device count  : {backend.device_count}")
print()


# ════════════════════════════════════════════════════════════════════════
# DEMO 2 — `km.train(df, target='y')`: production training in 3 lines
# ════════════════════════════════════════════════════════════════════════
#
# A real binary classification problem with 10 features, 800 rows. The
# whole training pipeline (data split, model selection, hyperparameter
# defaults, evaluation, metrics packaging) runs through a single call.
#
# The default family is sklearn (zero-config RandomForestClassifier) so
# this demo is fast on any machine. Lessons 5.1+ will swap the family
# to `lightning` and `torch` for the deep architectures.

print("=" * 72)
print("DEMO 2 — km.train() three-line zero-config training")
print("=" * 72)

X, y = make_classification(
    n_samples=800, n_features=10, n_informative=6, random_state=42
)
df = pl.DataFrame({**{f"f{i}": X[:, i] for i in range(10)}, "y": y})
print(f"  dataset shape : {df.shape}  (polars-native, no pandas)")

# km.train is async in kailash-ml 1.0+ — wrap with asyncio.run() in a CLI script.
# In a notebook (with nest_asyncio applied at the top), `await km.train(...)`
# works at the top of a cell.
result = asyncio.run(km.train(df, target="y"))

print(f"  result type   : {type(result).__name__}")
print(f"  metrics       : {result.metrics}")
print()


# ════════════════════════════════════════════════════════════════════════
# DEMO 3 — `MLEngine()`: the unified surface every M5 lesson sits on top of
# ════════════════════════════════════════════════════════════════════════
#
# `km.train()` is a convenience wrapper around `MLEngine`. Every M5 lesson
# from 5.1 onwards will compose its own `MLEngine()` (with custom feature
# stores, registries, and trackers) — but the surface is the same:
#
#     engine = MLEngine()                          # auto-detects MPS
#     trainable = SklearnTrainable(model, ...)    # or LightningTrainable
#     result = engine.fit(df, target='y', trainable=trainable)
#
# The engine knows about: feature stores, model registries, experiment
# trackers, hyperparameter search, ensemble training, drift monitors,
# inference servers, and the Trainable protocol. You will use most of
# these by Lesson 5.4.

print("=" * 72)
print("DEMO 3 — MLEngine: the unified surface")
print("=" * 72)
engine = km.MLEngine()
print(f"  engine.accelerator  : {engine.accelerator}")
print(
    f"  engine.backend_info : {engine.backend_info.backend}/"
    f"{engine.backend_info.precision}"
)
print(f"  engine.store_url    : {engine.store_url}")
print(f"  engine.tenant_id    : {engine.tenant_id}")
print()


# ════════════════════════════════════════════════════════════════════════
# REFLECTION
# ════════════════════════════════════════════════════════════════════════
print("=" * 72)
print("Reflection")
print("=" * 72)
print(
    "What you've seen:\n"
    "  ✓ Compute backend selected automatically (no env vars, no flags)\n"
    "  ✓ Production training in 3 lines (km.train returns metrics)\n"
    "  ✓ The MLEngine surface every M5 lesson will use\n\n"
    "Next: In Lesson 5.1 you will build the autoencoder that lives\n"
    "underneath one of these km.train() calls — every layer, every loss\n"
    "function, every gradient update. The destination will make sense\n"
    "once you have walked the journey to it.\n"
)
