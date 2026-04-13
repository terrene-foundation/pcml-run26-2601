# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# MLFP05 — Exercise 3.3: GRU as a Lightweight Alternative
# ════════════════════════════════════════════════════════════════════════
#
# WHAT YOU'LL LEARN:
#   After completing this section, you will be able to:
#   - Explain the GRU vs LSTM tradeoff (fewer parameters, similar accuracy)
#   - Build a GRU regressor with torch.nn.GRU for multi-step forecasting
#   - Compare parameter counts: GRU uses ~75% of LSTM's parameters
#   - Measure inference latency differences between GRU and LSTM
#   - Visualise hidden state dynamics unique to GRU's architecture
#   - Track training with ExperimentTracker
#
# PREREQUISITES: 02_lstm.py (understand gating mechanisms).
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
import torch
import torch.nn as nn

from shared.mlfp05.ex_3 import (
    BATCH_SIZE,
    CLIP,
    EPOCHS,
    FORECAST_HORIZON,
    HIDDEN_DIM,
    LR,
    OUTPUT_DIR,
    SEQ_LEN,
    init_environment,
    load_stock_data,
    prepare_dataloaders,
    setup_engines,
    train_model,
    register_best_model,
    get_visualizer,
    plot_training_curves,
    plot_predictions,
    plot_time_series_overlay,
    plot_horizon_error,
)

# ════════════════════════════════════════════════════════════════════════
# THEORY — GRU: Simpler Gates, Similar Power
# ════════════════════════════════════════════════════════════════════════
#
# The GRU (Gated Recurrent Unit) was introduced in 2014 by Cho et al.
# as a simplification of LSTM. The key differences:
#
# LSTM has 3 gates + cell state:
#   forget, input, output gates + separate cell state C_t
#   Parameters: 4 * (input_dim + hidden_dim) * hidden_dim
#
# GRU has 2 gates, NO separate cell state:
#   z_t = sigma(W_z [h_{t-1}, x_t])          UPDATE gate
#       "How much of the old hidden state to keep?"
#       Combines LSTM's forget + input gates into one decision
#
#   r_t = sigma(W_r [h_{t-1}, x_t])          RESET gate
#       "How much of the old hidden state to use when computing the candidate?"
#       When r=0, the candidate ignores history — like starting fresh
#
#   h_tilde = tanh(W [r_t * h_{t-1}, x_t])   CANDIDATE hidden state
#       The reset gate filters what history is relevant for the candidate
#
#   h_t = (1 - z_t) * h_{t-1} + z_t * h_tilde   HIDDEN STATE UPDATE
#       Linear interpolation: z=1 means "fully update", z=0 means "keep old"
#       This is equivalent to LSTM's cell highway, but simpler
#
# INTUITION for non-technical professionals:
#   If LSTM is a notebook with separate pages for long-term and short-term
#   notes, GRU is a single page where you decide how much to erase before
#   writing new notes. Simpler, faster to use, and usually just as effective.
#
# WHEN TO CHOOSE GRU OVER LSTM:
#   - Real-time systems where latency matters (fewer computations per step)
#   - Smaller datasets where fewer parameters reduce overfitting risk
#   - Mobile/edge deployment where model size matters
#   - When empirical testing shows similar accuracy (which is often)
#
# WHEN LSTM IS BETTER:
#   - Very long sequences (>100 steps) where the separate cell state helps
#   - Tasks requiring fine-grained memory control (e.g., language modelling)
#   - When you have abundant data and compute to train the extra parameters
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
    PRIMARY, experiment_suffix="gru"
)

# ── Checkpoint 1 ─────────────────────────────────────────────────────
assert X_train_t.shape[1] == SEQ_LEN
assert tracker is not None
print("--- Checkpoint 1 passed --- data and tracking ready\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 2 — Build the GRU and LSTM for comparison
# ════════════════════════════════════════════════════════════════════════
class GRURegressor(nn.Module):
    def __init__(
        self, input_dim: int, hidden_dim: int, horizon: int = FORECAST_HORIZON
    ):
        super().__init__()
        # TODO: Define GRU layer — nn.GRU(input_dim, hidden_dim, batch_first=True)
        # TODO: Define prediction head — nn.Linear(hidden_dim, horizon)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO: Pass x through self.gru -> out, _
        # TODO: Return self.head(out[:, -1]) — last hidden state -> (batch, horizon)
        pass


class LSTMRegressor(nn.Module):
    """LSTM for direct comparison with GRU."""

    def __init__(
        self, input_dim: int, hidden_dim: int, horizon: int = FORECAST_HORIZON
    ):
        super().__init__()
        # TODO: Define LSTM layer — nn.LSTM(input_dim, hidden_dim, batch_first=True)
        # TODO: Define prediction head — nn.Linear(hidden_dim, horizon)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO: Pass x through self.lstm -> out, _
        # TODO: Return self.head(out[:, -1])
        pass


gru_model = GRURegressor(input_dim=N_FEATURES, hidden_dim=HIDDEN_DIM)
lstm_compare = LSTMRegressor(input_dim=N_FEATURES, hidden_dim=HIDDEN_DIM)

n_params_gru = sum(p.numel() for p in gru_model.parameters())
n_params_lstm = sum(p.numel() for p in lstm_compare.parameters())
param_ratio = n_params_gru / n_params_lstm * 100

print(f"GRU parameters:  {n_params_gru:,}")
print(f"LSTM parameters: {n_params_lstm:,}")
print(
    f"GRU is {param_ratio:.0f}% of LSTM parameter count ({100-param_ratio:.0f}% fewer)"
)

# ── Checkpoint 2 ─────────────────────────────────────────────────────
assert n_params_gru < n_params_lstm, "GRU should have fewer parameters than LSTM"
assert param_ratio < 85, f"GRU should be ~75% of LSTM params, got {param_ratio:.0f}%"
dummy_input = torch.randn(2, SEQ_LEN, N_FEATURES, device=device)
gru_model.to(device)
dummy_out = gru_model(dummy_input)
assert dummy_out.shape == (2, FORECAST_HORIZON)
print("--- Checkpoint 2 passed --- GRU architecture verified\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 3 — Train the GRU
# ════════════════════════════════════════════════════════════════════════
print(f"\n== Training GRU on {PRIMARY} ==")
gru_results = train_model(
    gru_model,
    "GRU",
    tracker,
    exp_name,
    train_loader,
    val_loader,
    device,
)

# ── Checkpoint 3 ─────────────────────────────────────────────────────
assert len(gru_results["train_losses"]) == EPOCHS
assert gru_results["final_val_loss"] < 5.0
print(f"\n  Final val loss: {gru_results['final_val_loss']:.4f}")
print("--- Checkpoint 3 passed --- GRU trained\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 4 — Train LSTM for head-to-head comparison
# ════════════════════════════════════════════════════════════════════════
print(f"\n== Training LSTM (comparison) on {PRIMARY} ==")
lstm_results = train_model(
    lstm_compare,
    "LSTM_compare",
    tracker,
    exp_name,
    train_loader,
    val_loader,
    device,
)

print(f"\n  == Head-to-Head Comparison ==")
print(f"  {'Metric':<25s} {'GRU':>10s} {'LSTM':>10s}")
print(f"  {'Parameters':.<25s} {n_params_gru:>10,d} {n_params_lstm:>10,d}")
print(
    f"  {'Final val loss':.<25s} {gru_results['final_val_loss']:>10.4f} {lstm_results['final_val_loss']:>10.4f}"
)
print(
    f"  {'Avg grad norm':.<25s} {np.mean(gru_results['gradient_norms']):>10.4f} {np.mean(lstm_results['gradient_norms']):>10.4f}"
)

# ── Checkpoint 4 ─────────────────────────────────────────────────────
assert lstm_results["final_val_loss"] < 5.0
print("--- Checkpoint 4 passed --- head-to-head comparison complete\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 5 — Visualise: inference latency comparison
# ════════════════════════════════════════════════════════════════════════
# In production systems (real-time sensor monitoring, trading), inference
# latency matters as much as accuracy. GRU's fewer operations per step
# translate to measurable speed gains.


def benchmark_inference(model: nn.Module, name: str, n_runs: int = 100) -> float:
    """Measure average inference latency in milliseconds."""
    model.eval()
    test_input = torch.randn(1, SEQ_LEN, N_FEATURES, device=device)

    # Warmup
    with torch.no_grad():
        for _ in range(10):
            model(test_input)

    # Timed runs
    if device.type == "cuda":
        torch.cuda.synchronize()

    start = time.perf_counter()
    with torch.no_grad():
        for _ in range(n_runs):
            model(test_input)

    if device.type == "cuda":
        torch.cuda.synchronize()

    elapsed_ms = (time.perf_counter() - start) / n_runs * 1000
    print(f"  {name}: {elapsed_ms:.3f} ms/inference (avg over {n_runs} runs)")
    return elapsed_ms


print("\n== Inference Latency Comparison ==")
gru_latency = benchmark_inference(gru_model, "GRU")
lstm_latency = benchmark_inference(lstm_compare, "LSTM")
speedup = lstm_latency / max(gru_latency, 1e-6)
print(f"  GRU speedup: {speedup:.2f}x faster than LSTM")

# TODO: Create side-by-side bar charts (2 subplots, figsize 14x5):
#   Left: Inference latency bars for GRU (green) and LSTM (blue)
#     - Add ms values as text above each bar
#   Right: Parameter count bars for GRU (green) and LSTM (blue)
#     - Add comma-formatted counts above each bar
#   Suptitle: f"GRU vs LSTM: {param_ratio:.0f}% parameters, {speedup:.2f}x faster"
#   Save to OUTPUT_DIR / "03_gru_latency_comparison.png"
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
# TODO: Fill in both bar chart subplots
fig.tight_layout()
fig.savefig(str(OUTPUT_DIR / "03_gru_latency_comparison.png"), dpi=150)
plt.close(fig)
print("  Saved: 03_gru_latency_comparison.png")

# ── Checkpoint 5 ─────────────────────────────────────────────────────
assert gru_latency > 0 and lstm_latency > 0
assert (OUTPUT_DIR / "03_gru_latency_comparison.png").exists()
print("--- Checkpoint 5 passed --- latency comparison complete\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 6 — Visualise: GRU hidden state dynamics
# ════════════════════════════════════════════════════════════════════════
# GRU's update and reset gates control how information flows.
# Visualise how the hidden state evolves differently from LSTM.


def visualise_gru_gates(model: nn.Module, sample: torch.Tensor) -> None:
    """Extract and visualise GRU gate activations using hooks."""
    model.eval()

    # TODO: Run step-by-step through the GRU to capture hidden states
    #   gru_layer = model.gru
    #   Initialise h = zeros(1, 1, HIDDEN_DIM) on device
    #   Loop through timesteps: out, h = gru_layer(sample[:, t:t+1, :], h)
    #   Append h.squeeze().cpu().numpy() to hidden_states list
    # TODO: Stack into hidden_matrix of shape (seq_len, hidden_dim)

    # TODO: Create 2-row subplot (14, 8):
    #   Top: heatmap of hidden_matrix.T with "RdBu_r" cmap
    #     Title: "GRU Hidden State Evolution (all dimensions)"
    #   Bottom: line plot of top-5 most active dimensions by variance
    #     Title includes "(sharper transitions = update gate)"
    #   Suptitle: "GRU Hidden State Dynamics"
    #   Save to OUTPUT_DIR / "03_gru_hidden_dynamics.png"
    pass


sample_input = X_val_t[:1]
visualise_gru_gates(gru_model, sample_input)


# ════════════════════════════════════════════════════════════════════════
# TASK 7 — Visualise: predicted vs actual + training curves
# ════════════════════════════════════════════════════════════════════════
viz = get_visualizer()
plot_training_curves(viz, gru_results, "GRU", "03_gru")

preds_denorm, actual_denorm, _ = plot_predictions(
    viz, gru_model, X_val_t, y_val_t, norm_mean, norm_std, "03_gru"
)

plot_time_series_overlay(
    preds_denorm,
    actual_denorm,
    "03_gru",
    title=f"GRU: Predicted vs Actual Close ({PRIMARY})",
)

rmses = plot_horizon_error(preds_denorm, actual_denorm, "GRU")

# ── Checkpoint 6 ─────────────────────────────────────────────────────
assert (OUTPUT_DIR / "03_gru_training_curves.html").exists()
assert (OUTPUT_DIR / "03_gru_time_series_overlay.png").exists()
assert (OUTPUT_DIR / "03_gru_hidden_dynamics.png").exists()
print("--- Checkpoint 6 passed --- GRU visualisations generated\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 8 — Register model
# ════════════════════════════════════════════════════════════════════════
register_best_model(
    gru_model,
    "GRU",
    gru_results["final_val_loss"],
    PRIMARY,
    registry,
    has_registry,
)


# ════════════════════════════════════════════════════════════════════════
# APPLY — SMRT Predictive Maintenance: Real-Time Sensor Monitoring
# ════════════════════════════════════════════════════════════════════════
#
# BUSINESS SCENARIO:
#   You are a data engineer at SMRT Corporation, which operates
#   Singapore's MRT (Mass Rapid Transit) network carrying ~3.4 million
#   trips per day. Train wheels, bearings, and axles generate vibration
#   data captured by accelerometers at 1-second intervals.
#
# WHY GRU (NOT LSTM)?
#   Predictive maintenance on rail infrastructure requires REAL-TIME
#   inference — the model must predict the next minute's vibration
#   reading before the sensor captures it. With 200+ sensors per train
#   and 200+ trains running simultaneously:
#     - At 200 sensors x 200 trains x 60 readings/min = 2.4M inferences/min
#     - GRU's speed advantage means more sensors served per second
#
# DELIVERABLES:
#   - Next-minute vibration prediction with anomaly threshold
#   - Latency comparison: can GRU serve all sensors in real-time?
#   - Maintenance alert: "bearing X shows increasing vibration trend"
print("\n" + "=" * 70)
print("  APPLY: SMRT Predictive Maintenance — Vibration Monitoring")
print("=" * 70)

# TODO: Generate realistic vibration sensor data
#   - n_readings = 60 * 24 * 30 (30 days at 1-minute intervals)
#   - Base vibration: 2.5 + 0.3 * sin(daily cycle)
#   - Add gradual bearing degradation in the last 5 days (0 -> 1.5 increase)
#   - Add Gaussian noise (std=0.2)
np.random.seed(42)
SENSOR_SEQ = 60  # 1-hour lookback (60 minutes)
SENSOR_HORIZON = 1  # predict next minute
ANOMALY_THRESHOLD = 3.5  # mm/s^2 — bearing replacement recommended above this

# TODO: Normalise vibration data and build windowed dataset
# TODO: Train both GRU (hidden=16) and LSTM (hidden=16) on sensor data
# TODO: Evaluate both models and compute MAE
# TODO: Benchmark inference latency for sensor-sized models (n_runs=500)
# TODO: Calculate real-time capacity: 60_000 / latency_ms = inferences per minute

# TODO: Anomaly detection: count predictions exceeding ANOMALY_THRESHOLD
# TODO: Print comparison table: MAE, latency, capacity, anomaly counts

# TODO: Visualise (2-row figure, 14x8):
#   Top: Time series of actual vs GRU predicted vibration (last 1000 min)
#     - Add horizontal anomaly threshold line (red dotted)
#     - Fill anomaly zone in red
#   Bottom: Daily average vibration bar chart (last 7 days)
#     - Color green if below threshold, red if above
#   Save to OUTPUT_DIR / "03_gru_smrt_vibration.png"

# ── Checkpoint 7 (Apply) ────────────────────────────────────────────
# assert gru_mae < 1.0, "GRU vibration MAE should be reasonable"
# assert abs(gru_mae - lstm_mae) < 0.5, "GRU and LSTM should have similar accuracy"
# assert (OUTPUT_DIR / "03_gru_smrt_vibration.png").exists()
# print("--- Checkpoint 7 passed --- SMRT predictive maintenance application complete\n")


# ══════════════════════════════════════════════════════════════════════
# REFLECTION
# ══════════════════════════════════════════════════════════════════════
print("=" * 70)
print("  WHAT YOU'VE MASTERED")
print("=" * 70)
print(
    f"""
  [x] Built GRU regressor ({n_params_gru:,} params, {param_ratio:.0f}% of LSTM)
  [x] Head-to-head: GRU val={gru_results['final_val_loss']:.4f} vs LSTM val={lstm_results['final_val_loss']:.4f}
  [x] Latency: GRU is {speedup:.2f}x faster than LSTM on stock data
  [x] Hidden state dynamics: GRU's update gate creates sharper transitions
  [x] Applied GRU to SMRT predictive maintenance (vibration monitoring)

  Key insight: GRU achieves similar accuracy to LSTM with 25% fewer
  parameters and measurably lower latency. The tradeoff matters most in
  real-time systems (sensor monitoring, trading) where you need to serve
  thousands of predictions per second. For offline batch processing where
  latency doesn't matter, LSTM and GRU are interchangeable — pick either
  and spend your time on feature engineering instead.

  Next: 04_temporal_attention.py — letting the model decide which past
  timesteps matter most.
"""
)
