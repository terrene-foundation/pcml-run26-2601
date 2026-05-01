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
        self.rnn = nn.RNN(input_dim, hidden_dim, batch_first=True, nonlinearity="tanh")
        self.head = nn.Linear(hidden_dim, horizon)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.rnn(x)  # (batch, seq, hidden)
        return self.head(out[:, -1])  # use last hidden state -> (batch, horizon)


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

# ══════════════════════════════════════════════════════════════════
# DIAGNOSTIC CHECKPOINT — the vanishing-gradient TEXTBOOK case
# ══════════════════════════════════════════════════════════════════
# VanillaRNN is the vanishing-gradient poster child. This diagnostic
# run is EXPECTED to fire a CRITICAL Blood Test finding — that is
# the pedagogical point. LSTM (02) and GRU (03) fix it.
from kailash_ml import diagnose

print("\n── Diagnostic Report (VanillaRNN) ──")
report = diagnose(rnn_model, kind="dl", data=val_loader, show=False)

# ══════ EXPECTED OUTPUT (reference shape — BAD but INSTRUCTIVE) ══════
# ══════════════════════════════════════════════════════════════════
#   DL Diagnostics Report — Prescription Pad
# ══════════════════════════════════════════════════════════════════
#   [X] Gradient flow (CRITICAL): Vanishing gradients at
#       'rnn.weight_hh_l0' — min RMS ~1e-6. Fix: switch to LSTM/GRU,
#       shorten sequence, or use gradient clipping.
#   [!] Dead neurons  (depends on Tanh saturation): 'rnn' (tanh)
#       showing saturation (|x|>0.99) — the classic recurrent
#       vanishing-gradient fingerprint.
#   [?] Loss trend    (HEALTHY or plateaued early): loss stops
#       dropping because gradients through time are ~0.
# ══════════════════════════════════════════════════════════════════
#
# STUDENT INTERPRETATION GUIDE:
#   - The `rnn.weight_hh_l0` parameter is the hidden-to-hidden
#     matrix. Gradients propagating backward through time repeatedly
#     multiply by it; if its spectral radius < 1, gradients shrink
#     exponentially with sequence length. Bengio et al. 1994.
#   - Tanh saturation (|x|>0.99) is a second diagnostic — once
#     the recurrence saturates, derivatives collapse toward zero
#     and no learning can propagate backward through that step.
#   - THIS is the failure LSTM's gating mechanism solves — cells
#     can selectively preserve gradient flow across time.
#   - Do NOT "fix" this RNN; its pathology IS the lesson. Move to
#     02_lstm.py to see a healthier Prescription Pad on the same task.
# ══════════════════════════════════════════════════════════════════

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
    W_xh = torch.randn(N_FEATURES, hd, device=device).mul_(0.5).requires_grad_(True)
    W_hh = torch.randn(hd, hd, device=device).mul_(0.5).requires_grad_(True)
    b = torch.zeros(hd, device=device, requires_grad=True)
    x = torch.randn(1, seq_len, N_FEATURES, device=device)
    h = torch.zeros(1, hd, device=device, requires_grad=True)
    hiddens: list[torch.Tensor] = []
    for t in range(seq_len):
        h = torch.tanh(x[:, t] @ W_xh + h @ W_hh + b)
        h.retain_grad()
        hiddens.append(h)
    hiddens[-1].pow(2).sum().backward()
    return _collect_grad_norms(hiddens)


GRAD_SEQ_LEN = 60
rnn_decay = gradient_decay_rnn(seq_len=GRAD_SEQ_LEN)

rnn_ratio = rnn_decay[0] / max(rnn_decay[-1], 1e-12)
print(f"\n== Gradient Decay ({GRAD_SEQ_LEN} steps) ==")
print(
    f"  RNN:  first={rnn_decay[0]:.4e}  last={rnn_decay[-1]:.4e}  ratio={rnn_ratio:.4e}"
)
print("  Early timesteps receive near-zero gradient — the network forgets them.")

# Plot gradient decay
fig, ax = plt.subplots(figsize=(12, 5))
ax.semilogy(
    range(GRAD_SEQ_LEN), rnn_decay, color="#F44336", linewidth=2, label="Vanilla RNN"
)
ax.set_xlabel("Timestep (0 = earliest)")
ax.set_ylabel("Gradient Norm (log scale)")
ax.set_title("Vanishing Gradients: Gradient Norm vs Timestep")
ax.legend()
ax.grid(True, alpha=0.3)
ax.annotate(
    "Gradients vanish\nfor early timesteps",
    xy=(5, rnn_decay[5]),
    xytext=(15, rnn_decay[GRAD_SEQ_LEN // 2]),
    arrowprops=dict(arrowstyle="->", color="#333"),
    fontsize=10,
)
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
        # Run through RNN manually to capture all hidden states
        rnn_layer = model.rnn
        h = torch.zeros(1, 1, HIDDEN_DIM, device=device)
        hidden_states = []
        for t in range(sample.shape[1]):
            _, h = rnn_layer(sample[:, t : t + 1, :], h)
            hidden_states.append(h.squeeze().cpu().numpy())

    hidden_matrix = np.stack(hidden_states)  # (seq_len, hidden_dim)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8))

    # Heatmap of all hidden dimensions over time
    im = ax1.imshow(
        hidden_matrix.T, aspect="auto", cmap="RdBu_r", interpolation="nearest"
    )
    ax1.set_xlabel("Timestep")
    ax1.set_ylabel("Hidden Dimension")
    ax1.set_title("RNN Hidden State Evolution (all 64 dimensions)")
    plt.colorbar(im, ax=ax1, label="Activation")

    # Line plot of top-5 most active dimensions
    variances = np.var(hidden_matrix, axis=0)
    top5 = np.argsort(variances)[-5:][::-1]
    for i, dim in enumerate(top5):
        ax2.plot(hidden_matrix[:, dim], label=f"Dim {dim}", linewidth=1.5, alpha=0.8)
    ax2.set_xlabel("Timestep")
    ax2.set_ylabel("Activation")
    ax2.set_title("Top-5 Most Active Hidden Dimensions")
    ax2.legend(loc="upper right")
    ax2.grid(True, alpha=0.3)

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

# Generate realistic F&B sales data with weekly seasonality
np.random.seed(42)
n_days = 365 * 3  # 3 years
day_of_week = np.tile(np.arange(7), n_days // 7 + 1)[:n_days]

# Base sales: higher on weekends (Sat=5, Sun=6), lunch rush patterns
base_sales = 800 + 200 * np.sin(2 * np.pi * np.arange(n_days) / 7)  # weekly cycle
weekend_boost = np.where(day_of_week >= 5, 150, 0)
payday_boost = np.where(np.arange(n_days) % 30 < 3, 100, 0)  # payday spike
trend = np.linspace(0, 100, n_days)  # gradual growth
noise = np.random.normal(0, 50, n_days)
sales = base_sales + weekend_boost + payday_boost + trend + noise
sales = np.maximum(sales, 200).astype(np.float32)

# Build simple 1-feature windowed dataset
CAFE_SEQ = 7  # 1 week lookback
CAFE_HORIZON = 3  # predict next 3 days
sales_norm = (sales - sales[: int(0.8 * n_days)].mean()) / (
    sales[: int(0.8 * n_days)].std() + 1e-8
)
n_w = n_days - CAFE_SEQ - CAFE_HORIZON + 1
X_cafe = np.stack(
    [sales_norm[i : i + CAFE_SEQ].reshape(-1, 1) for i in range(n_w)]
).astype(np.float32)
y_cafe = np.stack(
    [sales_norm[i + CAFE_SEQ : i + CAFE_SEQ + CAFE_HORIZON] for i in range(n_w)]
).astype(np.float32)
split = int(0.8 * n_w)

cafe_model = VanillaRNN(input_dim=1, hidden_dim=32, horizon=CAFE_HORIZON).to(device)
opt = torch.optim.Adam(cafe_model.parameters(), lr=1e-3)
X_ct = torch.from_numpy(X_cafe[:split]).to(device)
y_ct = torch.from_numpy(y_cafe[:split]).to(device)
X_cv = torch.from_numpy(X_cafe[split:]).to(device)
y_cv = torch.from_numpy(y_cafe[split:]).to(device)

# Train
cafe_model.train()
for epoch in range(20):
    opt.zero_grad()
    loss = nn.functional.mse_loss(cafe_model(X_ct), y_ct)
    loss.backward()
    nn.utils.clip_grad_norm_(cafe_model.parameters(), max_norm=1.0)
    opt.step()

# Evaluate
cafe_model.eval()
with torch.no_grad():
    cafe_preds = cafe_model(X_cv).cpu().numpy()
    cafe_actual = y_cv.cpu().numpy()

# Denormalise
s_mean = sales[: int(0.8 * n_days)].mean()
s_std = sales[: int(0.8 * n_days)].std()
cafe_preds_d = cafe_preds * s_std + s_mean
cafe_actual_d = cafe_actual * s_std + s_mean

# Business metrics
mae_day1 = float(np.mean(np.abs(cafe_preds_d[:, 0] - cafe_actual_d[:, 0])))
avg_daily_revenue = 800 * 8  # 800 transactions * $8 average
waste_pct_before = 12.0  # industry average 12% food waste
accuracy_improvement = max(0, 1.0 - mae_day1 / float(np.mean(cafe_actual_d[:, 0])))
waste_pct_after = waste_pct_before * (1 - accuracy_improvement * 0.4)
annual_savings = (
    100 * avg_daily_revenue * 365 * (waste_pct_before - waste_pct_after) / 100
)

print(f"\n  Outlet-level demand forecasting (3-day ahead):")
print(f"    Day-1 MAE: {mae_day1:.0f} transactions")
print(f"    Forecast accuracy: {accuracy_improvement*100:.1f}%")
print(f"    Food waste reduction: {waste_pct_before:.1f}% -> {waste_pct_after:.1f}%")
print(f"    Estimated annual savings (100 outlets): S${annual_savings:,.0f}")
print(f"\n  Business decision: RNN works for short-term (7-day) F&B demand")
print(f"  because weekly cycles are the dominant pattern. For longer-range")
print(f"  planning (30+ days, festive seasons), LSTM would be needed —")
print(f"  vanilla RNN forgets events beyond ~10 timesteps.")

# Visualise the cafe scenario
fig, ax = plt.subplots(figsize=(14, 5))
n_show = 60
ax.plot(
    range(n_show),
    cafe_actual_d[:n_show, 0],
    label="Actual Daily Sales",
    color="#2196F3",
    linewidth=1.5,
)
ax.plot(
    range(n_show),
    cafe_preds_d[:n_show, 0],
    label="RNN Predicted (Day 1)",
    color="#FF5722",
    linewidth=1.5,
    linestyle="--",
    alpha=0.85,
)
ax.set_xlabel("Day")
ax.set_ylabel("Daily Transactions")
ax.set_title("Ya Kun Kaya Toast: RNN Demand Forecast vs Actual (60-day window)")
ax.legend()
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(str(OUTPUT_DIR / "01_rnn_yakun_demand_forecast.png"), dpi=150)
plt.close(fig)
print("  Saved: 01_rnn_yakun_demand_forecast.png")

# ── Checkpoint 8 (Apply) ────────────────────────────────────────────
assert mae_day1 < 200, "Day-1 MAE should be reasonable for F&B demand"
assert (OUTPUT_DIR / "01_rnn_yakun_demand_forecast.png").exists()
print("--- Checkpoint 8 passed --- Singapore F&B application complete\n")


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
  [x] Quantified business impact: S${annual_savings:,.0f}/year savings

  Key insight: Vanilla RNNs work for SHORT sequences (5-10 steps) where
  recent patterns dominate. For longer dependencies — stock trends over
  months, seasonal patterns, patient histories — gradients vanish and
  the model forgets. That is exactly what LSTMs solve.

  Next: 02_lstm.py — how gates solve the vanishing gradient problem.
"""
)
