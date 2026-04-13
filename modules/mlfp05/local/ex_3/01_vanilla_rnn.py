# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# MLFP05 — Exercise 3.1: Vanilla RNN for Sequence Prediction
# ════════════════════════════════════════════════════════════════════════
#
# WHAT YOU'LL LEARN:
#   After completing this section, you will be able to:
#   - Explain WHY sequences need memory (and why feedforward networks fail)
#   - Build a vanilla RNN with torch.nn.RNN for multi-step forecasting
#   - Measure gradient norms across timesteps to SEE vanishing gradients
#   - Track training with ExperimentTracker (per-epoch metrics)
#   - Visualise predicted vs actual time-series overlays
#
# PREREQUISITES: M5/ex_2 (CNNs, PyTorch training loops, batch norm).
# ESTIMATED TIME: ~25-30 min
#
# DATASET: STI + APAC/global stocks via yfinance (2010-2024).
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from shared.mlfp05.ex_3 import (
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
# THEORY — Why Sequences Need Memory
# ════════════════════════════════════════════════════════════════════════
#
# Imagine predicting tomorrow's stock price. A feedforward network sees
# today's numbers in isolation — it has no concept of "yesterday" or
# "last week." But markets have TRENDS: a stock rising for five straight
# days means something different from a stock that just jumped today.
#
# A Recurrent Neural Network (RNN) solves this by keeping a "hidden state"
# h_t that carries information from previous timesteps:
#
#   h_t = tanh(W_xh * x_t + W_hh * h_{t-1} + b)
#
# At each step, the RNN reads the new input x_t AND its own memory h_{t-1}.
# This is like reading a book: you understand each sentence because you
# remember what came before.
#
# THE CATCH — Vanishing Gradients:
# During backpropagation, gradients must flow backward through EVERY
# timestep. Each step multiplies the gradient by W_hh and squashes it
# through tanh. After 20-60 steps, the gradient either:
#   - Shrinks to near-zero (vanishing) — the network forgets early inputs
#   - Explodes to infinity (exploding) — training becomes unstable
#
# This is THE fundamental limitation of vanilla RNNs, and THE reason
# LSTMs and GRUs were invented (see 02_lstm.py and 03_gru.py).
# ════════════════════════════════════════════════════════════════════════

device = init_environment()


# ════════════════════════════════════════════════════════════════════════
# TASK 1 — Load multi-stock data and build windowed datasets
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

# ── Checkpoint 1 ─────────────────────────────────────────────────────
assert len(stock_data) >= 2, f"Need >= 2 tickers, got {len(stock_data)}"
assert X_train_t.shape[1] == SEQ_LEN
assert y_train_t.shape[1] == FORECAST_HORIZON, "Multi-step target shape mismatch"
print("--- Checkpoint 1 passed --- data loaded and windowed\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 2 — Set up ExperimentTracker
# ════════════════════════════════════════════════════════════════════════
conn, tracker, exp_name, registry, has_registry = setup_engines(
    PRIMARY, experiment_suffix="vanilla_rnn"
)

# ── Checkpoint 2 ─────────────────────────────────────────────────────
assert tracker is not None, "ExperimentTracker should be initialised"
assert exp_name is not None, "Experiment should be created"
print("--- Checkpoint 2 passed --- ExperimentTracker ready\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 3 — Build the Vanilla RNN
# ════════════════════════════════════════════════════════════════════════
# h_t = tanh(W_hh h_{t-1} + W_xh x_t + b)
# We use only the LAST hidden state h_T for prediction — all the
# information about the sequence must be compressed into one vector.
# This is a severe bottleneck for long sequences.
class VanillaRNN(nn.Module):
    def __init__(
        self, input_dim: int, hidden_dim: int, horizon: int = FORECAST_HORIZON
    ):
        super().__init__()
        # TODO: Define RNN layer — nn.RNN(input_dim, hidden_dim, batch_first=True, nonlinearity="tanh")
        # TODO: Define prediction head — nn.Linear(hidden_dim, horizon)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO: Pass x through self.rnn to get output sequence and hidden state
        #   out, _ = self.rnn(x)  # out shape: (batch, seq, hidden)
        # TODO: Take the LAST hidden state and pass through the prediction head
        #   return self.head(out[:, -1])  # shape: (batch, horizon)
        pass


rnn_model = VanillaRNN(input_dim=N_FEATURES, hidden_dim=HIDDEN_DIM)
n_params = sum(p.numel() for p in rnn_model.parameters())
print(f"VanillaRNN: {n_params:,} parameters")

# ── Checkpoint 3 ─────────────────────────────────────────────────────
dummy_input = torch.randn(2, SEQ_LEN, N_FEATURES, device=device)
rnn_model.to(device)
dummy_out = rnn_model(dummy_input)
assert dummy_out.shape == (
    2,
    FORECAST_HORIZON,
), f"Expected (2, {FORECAST_HORIZON}), got {dummy_out.shape}"
print("--- Checkpoint 3 passed --- VanillaRNN architecture verified\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 4 — Train the Vanilla RNN
# ════════════════════════════════════════════════════════════════════════
print(f"\n== Training VanillaRNN on {PRIMARY} ==")
rnn_results = train_model(
    rnn_model,
    "VanillaRNN",
    tracker,
    exp_name,
    train_loader,
    val_loader,
    device,
)

# ── Checkpoint 4 ─────────────────────────────────────────────────────
assert len(rnn_results["train_losses"]) == EPOCHS, f"Should have {EPOCHS} epochs"
assert rnn_results["final_val_loss"] < 5.0, "Val loss suspiciously high"
print(f"\n  Final val loss: {rnn_results['final_val_loss']:.4f}")
print(f"  Avg gradient norm: {np.mean(rnn_results['gradient_norms']):.4f}")
print("--- Checkpoint 4 passed --- VanillaRNN trained\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 5 — Visualise: vanishing gradients in action
# ════════════════════════════════════════════════════════════════════════
# We hand-roll an RNN step-by-step and measure the gradient norm at
# each timestep after backpropagating from the LAST step. If gradients
# vanish, the norm at step 0 will be orders of magnitude smaller than
# at step T — the network literally cannot learn from early inputs.


def _collect_grad_norms(hiddens: list[torch.Tensor]) -> list[float]:
    """Extract gradient norms from a list of hidden states after backward()."""
    return [float(h.grad.norm().item()) if h.grad is not None else 0.0 for h in hiddens]


def gradient_decay_rnn(seq_len: int = 60) -> list[float]:
    """Gradient norm at each timestep for a vanilla RNN."""
    torch.manual_seed(0)
    hd = 16
    # TODO: Create weight matrices W_xh (N_FEATURES, hd), W_hh (hd, hd), bias b (hd)
    #   All need requires_grad_(True). Scale W matrices by 0.5 with .mul_(0.5)
    # TODO: Create random input x of shape (1, seq_len, N_FEATURES) on device
    # TODO: Initialise hidden state h = zeros(1, hd) with requires_grad_(True)
    # TODO: Loop through each timestep t in range(seq_len):
    #   h = torch.tanh(x[:, t] @ W_xh + h @ W_hh + b)
    #   h.retain_grad()   # keep gradient for this intermediate step
    #   hiddens.append(h)
    # TODO: Backpropagate from the last hidden state:
    #   hiddens[-1].pow(2).sum().backward()
    # TODO: Return _collect_grad_norms(hiddens)
    pass


GRAD_SEQ_LEN = 60
rnn_decay = gradient_decay_rnn(seq_len=GRAD_SEQ_LEN)

rnn_ratio = rnn_decay[0] / max(rnn_decay[-1], 1e-12)
print(f"\n== Gradient Decay ({GRAD_SEQ_LEN} steps) ==")
print(
    f"  RNN:  first={rnn_decay[0]:.4e}  last={rnn_decay[-1]:.4e}  ratio={rnn_ratio:.4e}"
)
print("  Early timesteps receive near-zero gradient — the network forgets them.")

# TODO: Plot gradient decay on a log-scale y-axis (semilogy)
#   - x-axis: range(GRAD_SEQ_LEN) — timestep index
#   - y-axis: rnn_decay — gradient norm at each step
#   - Use ax.semilogy() for log scale
#   - Color: "#F44336", linewidth=2, label="Vanilla RNN"
#   - Add annotation at step 5 pointing to the low gradient region:
#     "Gradients vanish\nfor early timesteps"
#   - Save to OUTPUT_DIR / "01_rnn_gradient_decay.png"
fig, ax = plt.subplots(figsize=(12, 5))
# TODO: Plot and annotate the gradient decay curve
fig.tight_layout()
fig.savefig(str(OUTPUT_DIR / "01_rnn_gradient_decay.png"), dpi=150)
plt.close(fig)
print("  Saved: 01_rnn_gradient_decay.png")

# ── Checkpoint 5 ─────────────────────────────────────────────────────
assert (
    rnn_decay[0] < rnn_decay[-1]
), "RNN should show vanishing gradients (early < late)"
print("--- Checkpoint 5 passed --- vanishing gradient problem demonstrated\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 6 — Visualise: predicted vs actual time-series overlay
# ════════════════════════════════════════════════════════════════════════
viz = get_visualizer()

# Training curves
plot_training_curves(viz, rnn_results, "VanillaRNN", "01_rnn")

# Prediction scatter + denormalised arrays
preds_denorm, actual_denorm, _ = plot_predictions(
    viz, rnn_model, X_val_t, y_val_t, norm_mean, norm_std, "01_rnn"
)

# Time-series overlay — the key visual proof of model behaviour
plot_time_series_overlay(
    preds_denorm,
    actual_denorm,
    "01_rnn",
    title=f"VanillaRNN: Predicted vs Actual Close ({PRIMARY})",
)


# Hidden state evolution — show how the RNN's memory changes over time
def plot_hidden_state_evolution(model: nn.Module, sample: torch.Tensor) -> None:
    """Visualise how hidden states evolve across timesteps for a single sample."""
    model.eval()
    with torch.no_grad():
        # TODO: Get the rnn_layer from model.rnn
        # TODO: Initialise h = zeros(1, 1, HIDDEN_DIM) on device
        # TODO: Loop through each timestep t in range(sample.shape[1]):
        #   _, h = rnn_layer(sample[:, t:t+1, :], h)
        #   Append h.squeeze().cpu().numpy() to hidden_states list
        pass

    # TODO: Stack hidden_states into a matrix of shape (seq_len, hidden_dim)
    hidden_matrix = None  # np.stack(hidden_states)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8))

    # TODO: Plot heatmap of hidden_matrix.T using ax1.imshow()
    #   aspect="auto", cmap="RdBu_r", interpolation="nearest"
    #   Add colorbar, labels, title

    # TODO: Find top-5 most active dimensions by variance
    #   Plot each as a line on ax2 with labels

    fig.tight_layout()
    fig.savefig(str(OUTPUT_DIR / "01_rnn_hidden_state_evolution.png"), dpi=150)
    plt.close(fig)
    print("  Saved: 01_rnn_hidden_state_evolution.png")


sample_input = X_val_t[:1]  # single validation sample
plot_hidden_state_evolution(rnn_model, sample_input)

# Horizon error
rmses = plot_horizon_error(preds_denorm, actual_denorm, "VanillaRNN")

# ── Checkpoint 6 ─────────────────────────────────────────────────────
assert (OUTPUT_DIR / "01_rnn_training_curves.html").exists()
assert (OUTPUT_DIR / "01_rnn_pred_vs_actual.html").exists()
assert (OUTPUT_DIR / "01_rnn_time_series_overlay.png").exists()
assert (OUTPUT_DIR / "01_rnn_hidden_state_evolution.png").exists()
print("--- Checkpoint 6 passed --- visualisations generated\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 7 — Register model
# ════════════════════════════════════════════════════════════════════════
register_best_model(
    rnn_model,
    "VanillaRNN",
    rnn_results["final_val_loss"],
    PRIMARY,
    registry,
    has_registry,
)

# ── Checkpoint 7 ─────────────────────────────────────────────────────
assert rnn_results["final_val_loss"] < 5.0, "Val loss should be reasonable"
print("--- Checkpoint 7 passed --- model registered\n")


# ════════════════════════════════════════════════════════════════════════
# APPLY — Singapore F&B Demand Forecasting
# ════════════════════════════════════════════════════════════════════════
#
# BUSINESS SCENARIO:
#   You are a data analyst at Ya Kun Kaya Toast, one of Singapore's
#   most beloved F&B chains with 100+ outlets across the island.
#   Management wants to predict daily sales volume per outlet to
#   optimise ingredient ordering and staff scheduling.
#
# WHY A VANILLA RNN?
#   Daily F&B sales have SHORT-TERM patterns: weekday/weekend cycles,
#   lunch rushes, payday spikes. A 5-7 day lookback captures the weekly
#   cycle. Vanilla RNNs work well for short sequences — the vanishing
#   gradient problem only bites when you need to remember events from
#   30+ days ago.
#
# BUSINESS INTERPRETATION:
#   - Predicting 3 days ahead lets Ya Kun order kaya, bread, and eggs
#     with the right lead time for their supplier (Gardenia, local farms)
#   - Overprediction: wasted ingredients (kaya toast bread has 2-day shelf life)
#   - Underprediction: stockouts during peak hours, lost revenue + customer churn
#   - At ~$8 average transaction across 100 outlets, even a 5% improvement
#     in demand forecasting accuracy saves ~$146K/year in waste reduction
print("=" * 70)
print("  APPLY: Ya Kun Kaya Toast — Daily Demand Forecasting")
print("=" * 70)

# TODO: Generate realistic F&B sales data with weekly seasonality
#   - n_days = 365 * 3 (3 years)
#   - base_sales = 800 + 200 * sin(2*pi*arange(n_days)/7) for weekly cycle
#   - Add weekend_boost (150 for days 5,6), payday_boost (100 for first 3 days of month)
#   - Add linear trend and Gaussian noise
#   - Clip minimum to 200
np.random.seed(42)
n_days = 365 * 3
# TODO: Build sales array with seasonality, trend, and noise

# TODO: Build windowed dataset with CAFE_SEQ=7, CAFE_HORIZON=3
#   - Normalise using training set mean/std
#   - Create windows of shape (n_windows, 7, 1) -> targets (n_windows, 3)
CAFE_SEQ = 7
CAFE_HORIZON = 3

# TODO: Create and train a VanillaRNN(input_dim=1, hidden_dim=32, horizon=CAFE_HORIZON)
#   - Use Adam optimizer with lr=1e-3
#   - Train for 20 epochs with gradient clipping at 1.0
#   - Evaluate on validation set

# TODO: Denormalise predictions and compute business metrics
#   - mae_day1: Mean Absolute Error for day-1 prediction
#   - accuracy_improvement: 1 - mae_day1 / mean(actual)
#   - waste reduction and annual savings calculation

# TODO: Plot cafe demand forecast vs actual (60-day window)
#   - Actual as solid blue line, predicted as dashed orange
#   - Save to OUTPUT_DIR / "01_rnn_yakun_demand_forecast.png"

# ── Checkpoint 8 (Apply) ────────────────────────────────────────────
# assert mae_day1 < 200, "Day-1 MAE should be reasonable for F&B demand"
# assert (OUTPUT_DIR / "01_rnn_yakun_demand_forecast.png").exists()
# print("--- Checkpoint 8 passed --- Singapore F&B application complete\n")


# ══════════════════════════════════════════════════════════════════════
# REFLECTION
# ══════════════════════════════════════════════════════════════════════
print("=" * 70)
print("  WHAT YOU'VE MASTERED")
print("=" * 70)
print(
    f"""
  [x] Built a vanilla RNN for multi-step stock forecasting ({SEQ_LEN}-day -> {FORECAST_HORIZON}-day)
  [x] Demonstrated vanishing gradients: gradient ratio = {rnn_ratio:.4e}
  [x] Hidden state evolution: how the RNN's memory changes across timesteps
  [x] Predicted vs actual time-series overlay (visual proof of model behaviour)
  [x] Tracked training with ExperimentTracker (loss + gradient norms)
  [x] Applied RNN to Singapore F&B demand forecasting (Ya Kun Kaya Toast)

  Key insight: Vanilla RNNs work for SHORT sequences (5-10 steps) where
  recent patterns dominate. For longer dependencies — stock trends over
  months, seasonal patterns, patient histories — gradients vanish and
  the model forgets. That is exactly what LSTMs solve.

  Next: 02_lstm.py — how gates solve the vanishing gradient problem.
"""
)
