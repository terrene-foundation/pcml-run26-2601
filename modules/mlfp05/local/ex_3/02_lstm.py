# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# MLFP05 — Exercise 3.2: LSTM for Sequence Prediction
# ════════════════════════════════════════════════════════════════════════
#
# WHAT YOU'LL LEARN:
#   After completing this section, you will be able to:
#   - Explain how LSTM gates solve the vanishing gradient problem
#   - Write the six LSTM gate equations as vectorised torch operations
#   - Build an LSTM regressor with torch.nn.LSTM for multi-step forecasting
#   - Compare LSTM vs RNN gradient preservation quantitatively
#   - Track training with ExperimentTracker and register in ModelRegistry
#   - Visualise gate activations and cell state evolution
#
# PREREQUISITES: 01_vanilla_rnn.py (understand vanishing gradients).
# ESTIMATED TIME: ~30-40 min
#
# DATASET: STI + APAC/global stocks via yfinance (2010-2024).
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import torch
import torch.nn as nn

from shared.mlfp05.ex_3 import (
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
    plot_training_curves,
    plot_predictions,
    plot_time_series_overlay,
    plot_horizon_error,
)

# ════════════════════════════════════════════════════════════════════════
# THEORY — How Gates Solve Vanishing Gradients
# ════════════════════════════════════════════════════════════════════════
#
# The vanilla RNN's problem: information must pass through tanh at EVERY
# timestep. After 20+ steps, gradients shrink to zero and the network
# forgets early inputs entirely.
#
# LSTM's solution: a SEPARATE "cell state" C_t that acts as a HIGHWAY.
# Information can flow through C_t with minimal transformation — like an
# express lane on the highway that bypasses all the local traffic.
#
# The six gate equations (this is the core of LSTM):
#
#   f_t = sigma(W_f [h_{t-1}, x_t] + b_f)     FORGET gate
#       "What fraction of the old memory should I keep?"
#       sigma outputs 0-1: 0 = forget everything, 1 = remember everything
#
#   i_t = sigma(W_i [h_{t-1}, x_t] + b_i)     INPUT gate
#       "How much of the new candidate should I write to memory?"
#
#   g_t = tanh(W_g [h_{t-1}, x_t] + b_g)      CANDIDATE cell
#       "What is the new information I could store?"
#
#   C_t = f_t * C_{t-1} + i_t * g_t            CELL UPDATE
#       The key equation: ADDITIVE update, not multiplicative!
#       This is why gradients survive — addition preserves them.
#
#   o_t = sigma(W_o [h_{t-1}, x_t] + b_o)      OUTPUT gate
#       "How much of the cell state should I expose as output?"
#
#   h_t = o_t * tanh(C_t)                       HIDDEN STATE
#       The output: filtered cell state, passed to the next layer.
#
# INTUITION for non-technical professionals:
#   Think of LSTM as a notebook with a pencil:
#   - The FORGET gate erases irrelevant old notes
#   - The INPUT gate decides what new notes to write
#   - The CELL STATE is the notebook itself (persistent memory)
#   - The OUTPUT gate decides which notes to share with others
#   - Vanilla RNN is like trying to remember everything in your head
#     without writing anything down — you forget quickly.
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
    PRIMARY, experiment_suffix="lstm"
)

# ── Checkpoint 1 ─────────────────────────────────────────────────────
assert X_train_t.shape[1] == SEQ_LEN
assert tracker is not None
print("--- Checkpoint 1 passed --- data and tracking ready\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 2 — Build LSTM architectures
# ════════════════════════════════════════════════════════════════════════


# 2A: Production LSTM — uses torch.nn.LSTM (optimised C++/CUDA)
class LSTMRegressor(nn.Module):
    def __init__(
        self, input_dim: int, hidden_dim: int, horizon: int = FORECAST_HORIZON
    ):
        super().__init__()
        # TODO: Define LSTM layer — nn.LSTM(input_dim, hidden_dim, batch_first=True)
        # TODO: Define prediction head — nn.Linear(hidden_dim, horizon)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO: Pass x through self.lstm -> out, (h_n, c_n)
        # TODO: Return self.head(out[:, -1]) — last hidden state -> (batch, horizon)
        pass


# 2B: Hand-rolled LSTM cell — makes the gate equations concrete
# Use nn.LSTM in production; this is for LEARNING the equations.
class LSTMCellFromScratch(nn.Module):
    """Implements the six LSTM gate equations as explicit torch operations."""

    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        # TODO: Single linear layer that computes all 4 gates in one matrix multiply
        #   self.gates = nn.Linear(input_dim + hidden_dim, 4 * hidden_dim)
        #   This concatenates [x_t, h_prev] and produces i, f, g, o in one pass
        self.hidden_dim = hidden_dim

    def forward(self, x_t: torch.Tensor, h_prev: torch.Tensor, c_prev: torch.Tensor):
        """One timestep of the LSTM.

        Args:
            x_t: input at this timestep (batch, input_dim)
            h_prev: previous hidden state (batch, hidden_dim)
            c_prev: previous cell state (batch, hidden_dim)

        Returns:
            h_next, c_next
        """
        # TODO: Concatenate x_t and h_prev along dim=-1
        # TODO: Pass through self.gates and chunk into 4 parts: i, f, g, o
        # TODO: Apply activations:
        #   i = torch.sigmoid(i)   # input gate
        #   f = torch.sigmoid(f)   # forget gate
        #   g = torch.tanh(g)      # candidate cell
        #   o = torch.sigmoid(o)   # output gate
        # TODO: Cell update (ADDITIVE — the key insight):
        #   c_next = f * c_prev + i * g
        # TODO: Hidden state output:
        #   h_next = o * torch.tanh(c_next)
        # TODO: Return h_next, c_next
        pass


lstm_model = LSTMRegressor(input_dim=N_FEATURES, hidden_dim=HIDDEN_DIM)
n_params_lstm = sum(p.numel() for p in lstm_model.parameters())
n_params_cell = sum(p.numel() for p in LSTMCellFromScratch(N_FEATURES, 16).parameters())
print(f"LSTMRegressor: {n_params_lstm:,} parameters")
print(f"LSTMCellFromScratch (hidden=16): {n_params_cell:,} parameters")

# ── Checkpoint 2 ─────────────────────────────────────────────────────
# Verify the hand-rolled cell produces correct shapes
cell = LSTMCellFromScratch(input_dim=N_FEATURES, hidden_dim=16).to(device)
h, c = torch.zeros(4, 16, device=device), torch.zeros(4, 16, device=device)
x_seq = torch.randn(4, SEQ_LEN, N_FEATURES, device=device)
for t in range(x_seq.size(1)):
    h, c = cell(x_seq[:, t], h, c)
assert h.shape == (4, 16), f"Expected (4, 16), got {h.shape}"
assert c.shape == (4, 16), f"Cell state shape mismatch"
print(f"Hand-rolled LSTMCell: h={tuple(h.shape)}, c={tuple(c.shape)} -- verified")
print("--- Checkpoint 2 passed --- LSTM architectures built\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 3 — Train the LSTM
# ════════════════════════════════════════════════════════════════════════
print(f"\n== Training LSTM on {PRIMARY} ==")
lstm_results = train_model(
    lstm_model,
    "LSTM",
    tracker,
    exp_name,
    train_loader,
    val_loader,
    device,
)

# ── Checkpoint 3 ─────────────────────────────────────────────────────
assert len(lstm_results["train_losses"]) == EPOCHS
assert lstm_results["final_val_loss"] < 5.0
print(f"\n  Final val loss: {lstm_results['final_val_loss']:.4f}")
print("--- Checkpoint 3 passed --- LSTM trained\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 4 — Visualise: gradient preservation (LSTM vs RNN)
# ════════════════════════════════════════════════════════════════════════
# Compare gradient flow through 60 timesteps: the LSTM's additive cell
# update preserves gradients far better than the RNN's tanh chain.


def _collect_grad_norms(hiddens: list[torch.Tensor]) -> list[float]:
    return [float(h.grad.norm().item()) if h.grad is not None else 0.0 for h in hiddens]


def gradient_decay_rnn(seq_len: int = 60) -> list[float]:
    """Gradient norm at each timestep for a vanilla RNN (for comparison)."""
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


def gradient_decay_lstm(seq_len: int = 60) -> list[float]:
    """Gradient norm at each timestep for an LSTM (hand-rolled)."""
    torch.manual_seed(0)
    hd = 16
    cell_gd = LSTMCellFromScratch(N_FEATURES, hd).to(device)
    # TODO: Create random input x of shape (1, seq_len, N_FEATURES) on device
    # TODO: Initialise h = zeros(1, hd) and c = zeros(1, hd) with requires_grad_(True)
    # TODO: Loop through each timestep, calling cell_gd(x[:, t], h, c)
    #   h.retain_grad() at each step, append h to hiddens list
    # TODO: Backpropagate: hiddens[-1].pow(2).sum().backward()
    # TODO: Return _collect_grad_norms(hiddens)
    pass


GRAD_SEQ_LEN = 60
rnn_decay = gradient_decay_rnn(GRAD_SEQ_LEN)
lstm_decay = gradient_decay_lstm(GRAD_SEQ_LEN)

rnn_ratio = rnn_decay[0] / max(rnn_decay[-1], 1e-12)
lstm_ratio = lstm_decay[0] / max(lstm_decay[-1], 1e-12)

print(f"\n== Gradient Decay ({GRAD_SEQ_LEN} steps) ==")
print(
    f"  RNN:  first={rnn_decay[0]:.4e}  last={rnn_decay[-1]:.4e}  ratio={rnn_ratio:.4e}"
)
print(
    f"  LSTM: first={lstm_decay[0]:.4e}  last={lstm_decay[-1]:.4e}  ratio={lstm_ratio:.4e}"
)
print(
    f"  LSTM preserves gradients {lstm_ratio/max(rnn_ratio, 1e-12):.0f}x better than RNN"
)

# TODO: Plot side-by-side gradient decay comparison (2 subplots)
#   Left: semilogy of RNN decay (red) and LSTM decay (green) vs timestep
#   Right: normalised gradient flow (each normalised to its last-step value)
#   Save to OUTPUT_DIR / "02_lstm_gradient_comparison.png"
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))
# TODO: Fill in both subplots
fig.tight_layout()
fig.savefig(str(OUTPUT_DIR / "02_lstm_gradient_comparison.png"), dpi=150)
plt.close(fig)
print("  Saved: 02_lstm_gradient_comparison.png")

# ── Checkpoint 4 ─────────────────────────────────────────────────────
assert lstm_ratio > rnn_ratio, "LSTM should preserve gradients better than RNN"
print("--- Checkpoint 4 passed --- gradient preservation demonstrated\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 5 — Visualise: gate activations and cell state
# ════════════════════════════════════════════════════════════════════════
# Show what the forget, input, and output gates actually DO on real data.
# This is the visual proof that LSTM "decides" what to remember.


def visualise_gate_activations(sample: torch.Tensor) -> None:
    """Run a sample through the hand-rolled LSTM cell and plot gate activations."""
    cell_viz = LSTMCellFromScratch(N_FEATURES, 16).to(device)
    cell_viz.eval()

    seq_len = sample.shape[1]
    h = torch.zeros(1, 16, device=device)
    c = torch.zeros(1, 16, device=device)

    forget_gates, input_gates, output_gates, cell_states = [], [], [], []

    with torch.no_grad():
        for t in range(seq_len):
            x_t = sample[:, t]
            # TODO: Concatenate x_t and h, pass through cell_viz.gates
            # TODO: Chunk into i_g, f_g, g_g, o_g (4 chunks)
            # TODO: Apply sigmoid to f_g, i_g, o_g and append .cpu().numpy().flatten()
            #   to forget_gates, input_gates, output_gates lists
            # TODO: Call cell_viz(x_t, h, c) to advance the state
            # TODO: Append c.cpu().numpy().flatten() to cell_states
            pass

    # TODO: Stack each list into numpy matrices of shape (seq_len, 16)
    # TODO: Create 2x2 subplot figure (16, 10):
    #   (0,0): forget_mat.T with "Reds" cmap — "Forget Gate (what to erase)"
    #   (0,1): input_mat.T with "Greens" cmap — "Input Gate (what to write)"
    #   (1,0): output_mat.T with "Blues" cmap — "Output Gate (what to expose)"
    #   (1,1): cell_mat.T with "RdBu_r" cmap — "Cell State (the memory)"
    # TODO: Save to OUTPUT_DIR / "02_lstm_gate_activations.png"
    pass


sample_input = X_val_t[:1]
visualise_gate_activations(sample_input)


# ════════════════════════════════════════════════════════════════════════
# TASK 6 — Visualise: predicted vs actual time-series overlay
# ════════════════════════════════════════════════════════════════════════
viz = get_visualizer()
plot_training_curves(viz, lstm_results, "LSTM", "02_lstm")

preds_denorm, actual_denorm, _ = plot_predictions(
    viz, lstm_model, X_val_t, y_val_t, norm_mean, norm_std, "02_lstm"
)

plot_time_series_overlay(
    preds_denorm,
    actual_denorm,
    "02_lstm",
    title=f"LSTM: Predicted vs Actual Close ({PRIMARY})",
)

rmses = plot_horizon_error(preds_denorm, actual_denorm, "LSTM")

# ── Checkpoint 5 ─────────────────────────────────────────────────────
assert (OUTPUT_DIR / "02_lstm_training_curves.html").exists()
assert (OUTPUT_DIR / "02_lstm_gate_activations.png").exists()
assert (OUTPUT_DIR / "02_lstm_time_series_overlay.png").exists()
print("--- Checkpoint 5 passed --- LSTM visualisations generated\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 7 — Register model
# ════════════════════════════════════════════════════════════════════════
register_best_model(
    lstm_model,
    "LSTM",
    lstm_results["final_val_loss"],
    PRIMARY,
    registry,
    has_registry,
)


# ════════════════════════════════════════════════════════════════════════
# APPLY — SGX Equity Forecasting for a Singapore Hedge Fund
# ════════════════════════════════════════════════════════════════════════
#
# BUSINESS SCENARIO:
#   You are a quantitative analyst at a Singapore hedge fund. Your PM
#   wants a model that predicts next-5-day returns for DBS Group
#   (Singapore's largest bank by market cap) to inform position sizing.
#
# WHY LSTM?
#   Equity returns have LONG-RANGE dependencies: earnings cycles (quarterly),
#   macro trends (interest rates, Fed decisions), sector rotation. A vanilla
#   RNN forgets these. LSTM's cell state preserves information across
#   20-60 day lookback windows — matching the fund's typical holding period.
#
# DELIVERABLES:
#   - Point prediction with prediction intervals (67% and 95%)
#   - Trading decision framework: BUY/HOLD/SELL based on predicted return
#   - Risk-adjusted return attribution
print("\n" + "=" * 70)
print("  APPLY: SGX Equity Forecasting — DBS Group (D05.SI)")
print("=" * 70)

# TODO: Use DBS data if available in stock_data, else fall back to primary
dbs_symbol = "DBS.SI"
# TODO: Build dataset for DBS using build_dataset()
# TODO: Create train/val tensors and DataLoader
# TODO: Train a dedicated LSTMRegressor on DBS data for EPOCHS

# TODO: Evaluate and denormalise predictions to real prices
# TODO: Compute prediction intervals using residual distribution:
#   residuals = preds[:, 0] - actual[:, 0]
#   res_std = np.std(residuals)
#   67% CI: +/- 1.0 * res_std
#   95% CI: +/- 1.96 * res_std

# TODO: Trading decision framework:
#   predicted_5d_return = (latest_pred[-1] - latest_pred[0]) / latest_pred[0] * 100
#   BUY if return > 1.5%, SELL if < -1.5%, else HOLD

# TODO: Plot prediction intervals (100-day window):
#   - Actual as solid blue, predicted as dashed green
#   - 95% CI as light green fill, 67% CI as darker green fill
#   - Save to OUTPUT_DIR / "02_lstm_dbs_prediction_intervals.png"

# ── Checkpoint 6 (Apply) ────────────────────────────────────────────
# assert decision in ("BUY", "HOLD", "SELL"), "Trading decision must be valid"
# assert (OUTPUT_DIR / "02_lstm_dbs_prediction_intervals.png").exists()
# print("--- Checkpoint 6 passed --- SGX equity application complete\n")


# ══════════════════════════════════════════════════════════════════════
# REFLECTION
# ══════════════════════════════════════════════════════════════════════
print("=" * 70)
print("  WHAT YOU'VE MASTERED")
print("=" * 70)
print(
    f"""
  [x] Built LSTM regressor with torch.nn.LSTM for multi-step forecasting
  [x] Wrote LSTM gate equations as vectorised torch operations (LSTMCellFromScratch)
  [x] Gradient preservation: LSTM vs RNN comparison
  [x] Visualised gate activations: forget, input, output gates + cell state
  [x] Predicted vs actual time-series overlay with prediction intervals
  [x] Applied LSTM to SGX equity forecasting with trading decision framework

  Key insight: LSTM's cell state is a HIGHWAY for information. The additive
  update (C_t = f*C + i*g) preserves gradients where RNN's multiplicative
  chain (h = tanh(Wh + Wx)) destroys them. The forget/input/output gates
  let the network LEARN what to remember, not just hope gradients survive.

  Next: 03_gru.py — a lighter alternative with fewer parameters.
"""
)
