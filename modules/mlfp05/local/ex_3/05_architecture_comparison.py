# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# MLFP05 — Exercise 3.5: Architecture Comparison — RNN vs LSTM vs GRU vs Attention
# ════════════════════════════════════════════════════════════════════════
#
# WHAT YOU'LL LEARN:
#   After completing this section, you will be able to:
#   - Train all four architectures under identical conditions for fair comparison
#   - Compare accuracy, parameter efficiency, gradient health, and latency
#   - Run multi-stock generalisation tests (does the best model generalise?)
#   - Generate a comprehensive comparison dashboard with visual evidence
#   - Make architecture selection decisions for different business scenarios
#   - Register the overall best model in ModelRegistry
#
# PREREQUISITES: 01-04 (all four architecture files).
# ESTIMATED TIME: ~25-30 min
#
# DATASET: STI + APAC/global stocks via yfinance (2010-2024).
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from shared.mlfp05.ex_3 import (
    BATCH_SIZE,
    CLIP,
    EPOCHS,
    FEATURES,
    FORECAST_HORIZON,
    HIDDEN_DIM,
    LR,
    OUTPUT_DIR,
    SEQ_LEN,
    TICKERS,
    build_dataset,
    init_environment,
    load_stock_data,
    prepare_dataloaders,
    setup_engines,
    train_model,
    register_best_model,
    get_visualizer,
)

# ════════════════════════════════════════════════════════════════════════
# THEORY — Choosing the Right Sequence Architecture
# ════════════════════════════════════════════════════════════════════════
#
# By now you have built four sequence architectures. Here is when to
# use each one in production:
#
# ┌──────────────────┬─────────────────────────────────────────────────┐
# │ Architecture     │ Best For                                       │
# ├──────────────────┼─────────────────────────────────────────────────┤
# │ Vanilla RNN      │ Short sequences (<10 steps), simple patterns,  │
# │                  │ tiny models for edge/mobile deployment          │
# ├──────────────────┼─────────────────────────────────────────────────┤
# │ LSTM             │ Long sequences (20-200 steps), complex          │
# │                  │ dependencies, when you need fine-grained        │
# │                  │ memory control via forget/input/output gates    │
# ├──────────────────┼─────────────────────────────────────────────────┤
# │ GRU              │ Same tasks as LSTM but when latency or model    │
# │                  │ size matters. ~75% of LSTM parameters,          │
# │                  │ measurably faster inference.                    │
# ├──────────────────┼─────────────────────────────────────────────────┤
# │ LSTM+Attention   │ When you need explainability (which timesteps   │
# │                  │ matter?) or when different parts of the         │
# │                  │ sequence are unequally important. Healthcare,   │
# │                  │ finance, any domain with "trigger events."      │
# └──────────────────┴─────────────────────────────────────────────────┘
#
# For sequences longer than ~200 steps, Transformers (Exercise 4)
# replace all of the above with parallel self-attention.
# ════════════════════════════════════════════════════════════════════════

device = init_environment()


# ════════════════════════════════════════════════════════════════════════
# TASK 1 — Load data and set up experiment tracking
# ════════════════════════════════════════════════════════════════════════
stock_data, PRIMARY, primary_df = load_stock_data()

(
    train_loader,
    val_loader,
    X_train_t,
    y_train_t,
    X_val_t,
    y_val_t,
    norm_mean,
    norm_std,
    n_train_w,
    N_FEATURES,
) = prepare_dataloaders(primary_df, device)

conn, tracker, exp_name, registry, has_registry = setup_engines(
    PRIMARY, experiment_suffix="comparison"
)

# ── Checkpoint 1 ─────────────────────────────────────────────────────
assert X_train_t.shape[1] == SEQ_LEN
assert tracker is not None
print("--- Checkpoint 1 passed --- data and tracking ready\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 2 — Define all four architectures
# ════════════════════════════════════════════════════════════════════════
class VanillaRNN(nn.Module):
    def __init__(
        self, input_dim: int, hidden_dim: int, horizon: int = FORECAST_HORIZON
    ):
        super().__init__()
        # TODO: Define nn.RNN layer (input_dim, hidden_dim, batch_first=True, nonlinearity="tanh")
        # TODO: Define nn.Linear prediction head (hidden_dim, horizon)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO: out, _ = self.rnn(x); return self.head(out[:, -1])
        pass


class LSTMRegressor(nn.Module):
    def __init__(
        self, input_dim: int, hidden_dim: int, horizon: int = FORECAST_HORIZON
    ):
        super().__init__()
        # TODO: Define nn.LSTM layer (input_dim, hidden_dim, batch_first=True)
        # TODO: Define nn.Linear prediction head (hidden_dim, horizon)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO: out, _ = self.lstm(x); return self.head(out[:, -1])
        pass


class GRURegressor(nn.Module):
    def __init__(
        self, input_dim: int, hidden_dim: int, horizon: int = FORECAST_HORIZON
    ):
        super().__init__()
        # TODO: Define nn.GRU layer (input_dim, hidden_dim, batch_first=True)
        # TODO: Define nn.Linear prediction head (hidden_dim, horizon)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO: out, _ = self.gru(x); return self.head(out[:, -1])
        pass


class TemporalAttention(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        # TODO: Define W — nn.Linear(hidden_dim, hidden_dim)
        # TODO: Define v — nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, lstm_outputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # TODO: energy = tanh(self.W(lstm_outputs))
        # TODO: scores = self.v(energy).squeeze(-1)
        # TODO: weights = softmax(scores, dim=-1)
        # TODO: context = bmm(weights.unsqueeze(1), lstm_outputs).squeeze(1)
        # TODO: return context, weights
        pass


class LSTMWithAttention(nn.Module):
    def __init__(
        self, input_dim: int, hidden_dim: int, horizon: int = FORECAST_HORIZON
    ):
        super().__init__()
        # TODO: Define nn.LSTM, TemporalAttention, nn.Linear head

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # TODO: lstm_out -> attention -> head
        # TODO: Return pred, attn_weights
        pass


# Instantiate all four
models = {
    "VanillaRNN": VanillaRNN(input_dim=N_FEATURES, hidden_dim=HIDDEN_DIM),
    "LSTM": LSTMRegressor(input_dim=N_FEATURES, hidden_dim=HIDDEN_DIM),
    "GRU": GRURegressor(input_dim=N_FEATURES, hidden_dim=HIDDEN_DIM),
    "LSTM+Attention": LSTMWithAttention(input_dim=N_FEATURES, hidden_dim=HIDDEN_DIM),
}
is_attn = {"VanillaRNN": False, "LSTM": False, "GRU": False, "LSTM+Attention": True}

param_counts = {}
for name, model in models.items():
    n_params = sum(p.numel() for p in model.parameters())
    param_counts[name] = n_params
    print(f"  {name}: {n_params:,} parameters")

# ── Checkpoint 2 ─────────────────────────────────────────────────────
assert len(models) == 4
assert param_counts["GRU"] < param_counts["LSTM"]
print("--- Checkpoint 2 passed --- all four architectures defined\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 3 — Train all four architectures under identical conditions
# ════════════════════════════════════════════════════════════════════════
print(f"\n== Training all four on {PRIMARY} (identical conditions) ==")
all_results = {}
for name, model in models.items():
    print(f"\n--- {name} ---")
    results = train_model(
        model,
        name,
        tracker,
        exp_name,
        train_loader,
        val_loader,
        device,
        attn=is_attn[name],
    )
    all_results[name] = results

# ── Checkpoint 3 ─────────────────────────────────────────────────────
for name, res in all_results.items():
    assert len(res["train_losses"]) == EPOCHS, f"{name} should have {EPOCHS} epochs"
    assert res["final_val_loss"] < 5.0, f"{name} val loss suspiciously high"
print("\n--- Checkpoint 3 passed --- all four architectures trained\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 4 — Comprehensive comparison table
# ════════════════════════════════════════════════════════════════════════
print("=" * 80)
print(f"  {'Model':<18s} {'Params':>8s} {'Train':>8s} {'Val':>8s} {'GradNorm':>10s}")
print("-" * 80)
for name, res in all_results.items():
    print(
        f"  {name:<18s} {param_counts[name]:>8,d} "
        f"{res['train_losses'][-1]:>8.4f} {res['final_val_loss']:>8.4f} "
        f"{np.mean(res['gradient_norms']):>10.4f}"
    )
print("=" * 80)

best_name = min(all_results, key=lambda k: all_results[k]["final_val_loss"])
best_val = all_results[best_name]["final_val_loss"]
print(f"\n  Best model: {best_name} (val_loss={best_val:.4f})")


# ════════════════════════════════════════════════════════════════════════
# TASK 5 — Inference latency benchmark
# ════════════════════════════════════════════════════════════════════════
def _predict_for_bench(model, x, attn=False):
    out = model(x)
    return out[0] if attn else out


def benchmark_inference(
    model: nn.Module, name: str, attn: bool, n_runs: int = 200
) -> float:
    model.eval()
    test_input = torch.randn(1, SEQ_LEN, N_FEATURES, device=device)
    with torch.no_grad():
        for _ in range(20):
            _predict_for_bench(model, test_input, attn)
    if device.type == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    with torch.no_grad():
        for _ in range(n_runs):
            _predict_for_bench(model, test_input, attn)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed_ms = (time.perf_counter() - start) / n_runs * 1000
    return elapsed_ms


print("\n== Inference Latency ==")
latencies = {}
for name, model in models.items():
    lat = benchmark_inference(model, name, is_attn[name])
    latencies[name] = lat
    print(f"  {name}: {lat:.3f} ms/inference")

fastest = min(latencies, key=latencies.get)
print(f"  Fastest: {fastest} ({latencies[fastest]:.3f} ms)")


# ════════════════════════════════════════════════════════════════════════
# TASK 6 — Multi-stock generalisation test
# ════════════════════════════════════════════════════════════════════════
print(f"\n== Multi-Stock Generalisation ({best_name}) ==")
multi_stock_results: dict[str, float] = {PRIMARY: best_val}

# TODO: For each stock in stock_data (except PRIMARY):
#   - Build dataset using build_dataset()
#   - Create train/val DataLoaders
#   - Instantiate the best architecture (check best_name)
#   - Train for 8 epochs with Adam, lr=LR, clip=CLIP
#   - Evaluate val loss and store in multi_stock_results
#   - Print per-stock val loss
for symbol, sdf in stock_data.items():
    if symbol == PRIMARY or len(sdf) < SEQ_LEN + FORECAST_HORIZON + 50:
        continue
    # TODO: Build dataset, train best model, evaluate, store result
    pass

avg_cross_stock = np.mean(list(multi_stock_results.values()))
print(f"\n  Average cross-stock val loss: {avg_cross_stock:.4f}")

# ── Checkpoint 4 ─────────────────────────────────────────────────────
assert len(multi_stock_results) >= 2, "Need multi-stock results"
print("--- Checkpoint 4 passed --- multi-stock generalisation complete\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 7 — Comprehensive comparison dashboard
# ════════════════════════════════════════════════════════════════════════
viz = get_visualizer()

# 7A: Training curves (all models overlaid)
train_metrics = {}
for label, res in all_results.items():
    train_metrics[f"{label} train"] = res["train_losses"]
    train_metrics[f"{label} val"] = res["val_losses"]
viz.training_history(
    metrics=train_metrics, x_label="Epoch", y_label="MSE Loss"
).write_html(str(OUTPUT_DIR / "05_comparison_training_curves.html"))

# 7B: Gradient norms (all models overlaid)
grad_metrics = {k: v["gradient_norms"] for k, v in all_results.items()}
viz.training_history(
    metrics=grad_metrics, x_label="Epoch", y_label="Gradient L2 Norm"
).write_html(str(OUTPUT_DIR / "05_comparison_gradient_norms.html"))

# 7C: Prediction vs actual for best model
best_model = models[best_name]
best_model.eval()
with torch.no_grad():
    if is_attn[best_name]:
        val_preds, val_attn_weights = best_model(X_val_t)
    else:
        val_preds = best_model(X_val_t)
        val_attn_weights = None

close_mean, close_std = norm_mean[0, 0], norm_std[0, 0]
preds_denorm = val_preds.cpu().numpy() * close_std + close_mean
actual_denorm = y_val_t.cpu().numpy() * close_std + close_mean

pred_df = pl.DataFrame(
    {"actual": actual_denorm[:, 0].tolist(), "predicted": preds_denorm[:, 0].tolist()}
)
viz.scatter(pred_df, x="actual", y="predicted").write_html(
    str(OUTPUT_DIR / "05_comparison_pred_vs_actual.html")
)

# 7D: Comprehensive comparison figure (matplotlib)
# TODO: Create 2x3 subplot figure (20, 12) with these panels:
#   (0,0): Val loss over epochs — all 4 models overlaid with legend
#   (0,1): Gradient norms over epochs — all 4 models overlaid
#   (0,2): Parameter count vs val loss scatter (efficiency frontier)
#     - Annotate each point with model name
#   (1,0): Latency bar chart — bars coloured per model
#     - Add ms values above bars
#   (1,1): Predicted vs actual time series (best model, 150 points)
#   (1,2): Multi-stock generalisation bar chart
#     - Primary stock in model colour, others in grey
#     - Horizontal line for average cross-stock loss
#   Suptitle: f"Architecture Comparison: ... on {PRIMARY}"
#   Save to OUTPUT_DIR / "05_comparison_dashboard.png"
model_names = list(all_results.keys())
colors = {
    "VanillaRNN": "#F44336",
    "LSTM": "#2196F3",
    "GRU": "#4CAF50",
    "LSTM+Attention": "#FF9800",
}
fig, axes = plt.subplots(2, 3, figsize=(20, 12))
# TODO: Fill in all 6 panels of the dashboard
fig.tight_layout()
fig.savefig(str(OUTPUT_DIR / "05_comparison_dashboard.png"), dpi=150)
plt.close(fig)
print("  Saved: 05_comparison_dashboard.png")

# 7E: Horizon error by day for all models
print("\n== Forecast Error by Horizon Day (all models) ==")
print(f"  {'Day':<6s}", end="")
for name in model_names:
    print(f"  {name:>16s}", end="")
print()

all_horizon_rmses = {}
for name in model_names:
    model = models[name]
    model.eval()
    with torch.no_grad():
        if is_attn[name]:
            vp, _ = model(X_val_t)
        else:
            vp = model(X_val_t)
    vp_denorm = vp.cpu().numpy() * close_std + close_mean
    rmses = []
    for day in range(FORECAST_HORIZON):
        rmse = float(np.mean((vp_denorm[:, day] - actual_denorm[:, day]) ** 2)) ** 0.5
        rmses.append(rmse)
    all_horizon_rmses[name] = rmses

for day in range(FORECAST_HORIZON):
    print(f"  Day {day+1:<3d}", end="")
    for name in model_names:
        print(f"  {all_horizon_rmses[name][day]:>16.2f}", end="")
    print()

# ── Checkpoint 5 ─────────────────────────────────────────────────────
assert (OUTPUT_DIR / "05_comparison_dashboard.png").exists()
assert (OUTPUT_DIR / "05_comparison_training_curves.html").exists()
assert (OUTPUT_DIR / "05_comparison_pred_vs_actual.html").exists()
print("\n--- Checkpoint 5 passed --- comparison dashboard generated\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 8 — Register overall best model
# ════════════════════════════════════════════════════════════════════════
register_best_model(
    models[best_name],
    best_name,
    best_val,
    PRIMARY,
    registry,
    has_registry,
)

# ── Checkpoint 6 ─────────────────────────────────────────────────────
assert best_val < 5.0, "Best model val loss should be reasonable"
print("--- Checkpoint 6 passed --- best model registered\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 9 — Architecture selection guide (decision framework)
# ════════════════════════════════════════════════════════════════════════
print("=" * 80)
print("  ARCHITECTURE SELECTION GUIDE")
print("=" * 80)
print(
    f"""
  Based on the experiments above, here is a decision framework:

  QUESTION 1: How long are your sequences?
    < 10 steps  -> VanillaRNN (simplest, fastest, adequate)
    10-200 steps -> LSTM or GRU (gates preserve long-range dependencies)
    > 200 steps  -> Transformer (Exercise 4 — parallel self-attention)

  QUESTION 2: Does latency matter?
    Yes (real-time, sensor, trading) -> GRU ({latencies['GRU']:.3f}ms vs LSTM {latencies['LSTM']:.3f}ms)
    No (batch, offline, training)    -> LSTM or LSTM+Attention

  QUESTION 3: Do you need explainability?
    Yes (healthcare, finance, regulated) -> LSTM+Attention (attention heatmaps)
    No (internal tool, non-regulated)    -> LSTM or GRU (simpler)

  QUESTION 4: How much data do you have?
    Small dataset (<1K sequences) -> GRU (fewer parameters, less overfitting)
    Large dataset (>10K sequences) -> LSTM+Attention (can leverage capacity)

  Results from THIS experiment on {PRIMARY}:
    Best accuracy:     {best_name} (val={best_val:.4f})
    Most efficient:    GRU ({param_counts['GRU']:,} params)
    Fastest inference: {fastest} ({latencies[fastest]:.3f}ms)
    Best gradient flow: LSTM or LSTM+Attention (gated architectures)
"""
)


# ══════════════════════════════════════════════════════════════════════
# REFLECTION
# ══════════════════════════════════════════════════════════════════════
print("=" * 70)
print("  WHAT YOU'VE MASTERED — EXERCISE 3 COMPLETE")
print("=" * 70)
print(
    f"""
  [x] Loaded data across {len(stock_data)} tickers
  [x] Built VanillaRNN, LSTM, GRU, and LSTM+Attention in torch.nn
  [x] Wrote LSTM gate equations as vectorised torch operations
  [x] Multi-step forecasting: {SEQ_LEN}-day window -> next {FORECAST_HORIZON} days
  [x] Tracked every variant with ExperimentTracker (per-epoch loss + grad norms)
  [x] Vanishing gradients: demonstrated and explained with visual evidence
  [x] Temporal attention: learnable focus over past timesteps (preview of M5.4)
  [x] Best model ({best_name}) registered in ModelRegistry
  [x] Multi-stock generalisation across {len(multi_stock_results)} tickers
  [x] Architecture selection guide for real-world decision making
  [x] Applied to Singapore business scenarios:
      - Ya Kun Kaya Toast: F&B demand forecasting (RNN)
      - SGX/DBS: Equity forecasting with prediction intervals (LSTM)
      - SMRT: Predictive maintenance for trains (GRU)
      - SGH: Clinical deterioration prediction with explainability (Attention)

  Key insight: There is no single "best" architecture. The right choice
  depends on sequence length, latency requirements, explainability needs,
  and data volume. RNNs fail on long sequences. LSTMs fix this with
  additive cell-state updates. GRUs match with fewer parameters. Attention
  lets the model choose which past steps matter. Error compounds across
  the forecast horizon.

  This exercise teaches architectures, not market timing.

  Next: Exercise 4 — Transformers replace recurrence with pure attention.
"""
)
