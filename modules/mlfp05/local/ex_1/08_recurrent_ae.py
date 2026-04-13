# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# MLFP05 — Exercise 1.8: Recurrent Autoencoder (Sequential Data)
# ════════════════════════════════════════════════════════════════════════
#
# WHAT YOU'LL LEARN:
#   - Build an LSTM-based autoencoder for time-series data
#   - Understand WHY recurrent architecture preserves temporal order
#   - Visualise original vs reconstructed time-series overlays
#   - Apply to SGX financial anomaly / regime change detection
#   - Quantify portfolio drawdown reduction in S$ for a S$100M fund
#
# PREREQUISITES: 07_stacked_ae.py
# ESTIMATED TIME: ~20 min
#
# TASKS:
#   1. Generate synthetic sensor vibration data
#   2. Build LSTM encoder-decoder architecture
#   3. Train and visualise time-series reconstruction
#   4. Apply: SGX regime change detection with portfolio impact
#
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from shared.mlfp05.ex_1 import (
    LATENT_DIM,
    EPOCHS,
    OUTPUT_DIR,
    device,
    setup_engines,
    train_variant,
    show_timeseries_reconstruction,
    register_model,
)


# ════════════════════════════════════════════════════════════════════════
# THEORY — Temporal Order Matters
# ════════════════════════════════════════════════════════════════════════
# For sequential data, temporal order matters. An LSTM encoder reads
# the sequence step by step, building a summary (hidden state). The
# LSTM decoder unrolls the summary back into the original sequence.
#
# Analogy: Reading a book chapter vs reading the same words shuffled
# randomly. The LSTM encoder reads the chapter in order and produces
# a summary that captures the narrative arc. A flat MLP reads the
# shuffled words and has no sense of what comes before or after.
#
# WHY THIS MATTERS: Sensor data, financial returns, medical signals
# all have temporal structure. A vibration spike AFTER a speed change
# means something different than a spike BEFORE one.


# ════════════════════════════════════════════════════════════════════════
# TASK 1 — Generate Sensor Data and Set Up Engines
# ════════════════════════════════════════════════════════════════════════

conn, tracker, exp_name, registry, has_registry = setup_engines()

SEQ_LEN = 100
N_SERIES_TRAIN = 5000
N_SERIES_TEST = 500


def generate_sensor_data(n_samples, seq_len, seed=42):
    """Generate synthetic industrial vibration sensor data."""
    # TODO: Generate n_samples time series, each of length seq_len
    # Each series: sum of 2 sinusoids at random frequencies + noise
    # t = np.linspace(0, 4*pi, seq_len)
    # signal = amp1*sin(freq1*t+phase) + amp2*sin(freq2*t) + 0.1*noise
    # Normalise each series to [0, 1]
    ____


# TODO: Generate train and test sensor data
sensor_train = ____
sensor_test = ____

sensor_train_t = torch.tensor(sensor_train).to(device)
sensor_test_t = torch.tensor(sensor_test).to(device)
sensor_loader = DataLoader(TensorDataset(sensor_train_t), batch_size=128, shuffle=True)

print(
    f"Sensor data: {sensor_train.shape[0]} train, {sensor_test.shape[0]} test, seq_len={SEQ_LEN}"
)


# ════════════════════════════════════════════════════════════════════════
# TASK 2 — Build and Train Recurrent AE
# ════════════════════════════════════════════════════════════════════════


class RecurrentAE(nn.Module):
    """LSTM-based autoencoder for sequential data."""

    def __init__(self, seq_len: int, hidden_dim: int = 64, latent_dim: int = 16):
        super().__init__()
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim
        # TODO: Define layers:
        #   encoder_lstm: nn.LSTM(input_size=1, hidden_size=hidden_dim, batch_first=True)
        #   enc_to_latent: nn.Linear(hidden_dim, latent_dim)
        #   latent_to_dec: nn.Linear(latent_dim, hidden_dim)
        #   decoder_lstm: nn.LSTM(input_size=hidden_dim, hidden_size=hidden_dim, batch_first=True)
        #   output_layer: nn.Sequential(nn.Linear(hidden_dim, 1), nn.Sigmoid())
        self.encoder_lstm = ____
        self.enc_to_latent = ____
        self.latent_to_dec = ____
        self.decoder_lstm = ____
        self.output_layer = ____

    def forward(self, x):
        # TODO: Implement LSTM AE forward pass:
        # 1. x_seq = x.unsqueeze(-1)  — add feature dim
        # 2. Encode: _, (h_n, _) = self.encoder_lstm(x_seq)
        # 3. Compress: z = self.enc_to_latent(h_n.squeeze(0))
        # 4. Expand: dec_input = self.latent_to_dec(z).unsqueeze(1).repeat(1, seq_len, 1)
        # 5. Decode: dec_output, _ = self.decoder_lstm(dec_input)
        # 6. Output: x_hat = self.output_layer(dec_output).squeeze(-1)
        # Return (x_hat, z)
        ____


def recurrent_ae_loss(model, xb):
    # TODO: Forward, MSE loss. Return (loss, {})
    ____


print("\n" + "=" * 70)
print("  Recurrent AE — Time-Series (Sensor Data)")
print("=" * 70)
print("  LSTM encoder reads sequence -> latent -> LSTM decoder reconstructs.")

# TODO: Create RecurrentAE(SEQ_LEN, hidden_dim=64, latent_dim=LATENT_DIM) and train
recurrent_model = ____
recurrent_losses = ____


# ════════════════════════════════════════════════════════════════════════
# TASK 3 — Visualise Time-Series Reconstruction
# ════════════════════════════════════════════════════════════════════════

# TODO: show_timeseries_reconstruction
____

# ── Checkpoint ──────────────────────────────────────────────────────
assert len(recurrent_losses) == EPOCHS
assert recurrent_losses[-1] < recurrent_losses[0]
# INTERPRETATION: The time-series overlay shows reconstructed (red
# dashed) tracking original (blue solid). Where lines diverge =
# patterns harder to compress. High reconstruction error = anomaly.
print("\n--- Checkpoint passed --- recurrent AE trained\n")

if has_registry:
    register_model(registry, "recurrent_ae", recurrent_model, recurrent_losses[-1])


# ════════════════════════════════════════════════════════════════════════
# APPLY — SGX Financial Regime Change Detection
# ════════════════════════════════════════════════════════════════════════
# BUSINESS SCENARIO: You are a quantitative analyst at a Singapore
# hedge fund monitoring SGX equities for regime changes. Markets shift
# between calm and crisis states. Your PM asks: "Can we detect regime
# changes early enough to reduce portfolio drawdown?"

print("\n" + "=" * 70)
print("  APPLICATION: SGX Regime Change Detection (S$100M Fund)")
print("=" * 70)

# --- Generate SGX equity data ---
N_DAYS = 1500
N_STOCKS = 5
STOCK_NAMES = ["DBS", "OCBC", "Singtel", "CapitaLand", "Keppel"]
fin_rng = np.random.default_rng(42)

# TODO: Generate correlated daily returns for 5 SGX stocks
# base_returns, base_vols, correlation matrix, Cholesky decomposition
# Include 4 crisis periods with regime_labels
# Crisis: higher vol, negative drift
base_returns = np.array([0.08, 0.07, 0.04, 0.06, 0.05])
base_vols = np.array([0.18, 0.16, 0.22, 0.20, 0.24])
corr = np.array(
    [
        [1.00, 0.85, 0.45, 0.55, 0.50],
        [0.85, 1.00, 0.40, 0.50, 0.45],
        [0.45, 0.40, 1.00, 0.35, 0.30],
        [0.55, 0.50, 0.35, 1.00, 0.60],
        [0.50, 0.45, 0.30, 0.60, 1.00],
    ]
)
L = np.linalg.cholesky(corr)

daily_returns = np.zeros((N_DAYS, N_STOCKS), dtype=np.float32)
regime_labels = np.zeros(N_DAYS, dtype=np.int32)
crisis_periods = [
    (300, 360, "COVID Crash (Mar 2020)"),
    (600, 640, "Rate Hike Shock (2022)"),
    (900, 930, "Banking Stress (2023)"),
    (1200, 1230, "Geopolitical Crisis"),
]

# TODO: Fill daily_returns using correlated random draws
# Normal days: positive drift + normal vol
# Crisis days: negative drift + 3x vol
for day in range(N_DAYS):
    in_crisis = any(start <= day < end for start, end, _ in crisis_periods)
    z = fin_rng.standard_normal(N_STOCKS)
    corr_z = L @ z
    if in_crisis:
        regime_labels[day] = 1
        daily_returns[day] = ____
    else:
        daily_returns[day] = ____

prices = 100 * np.exp(np.cumsum(daily_returns, axis=0))
print(f"Generated {N_DAYS} trading days, {regime_labels.sum()} crisis days")

# --- Create sequences ---
FIN_SEQ_LEN = 20
# TODO: Create rolling features (returns + 5-day rolling vol)
# Normalise, create sequences of length FIN_SEQ_LEN
# Split into normal-only training set
rolling_vol = np.zeros_like(daily_returns)
for i in range(5, N_DAYS):
    rolling_vol[i] = daily_returns[i - 5 : i].std(axis=0)
features = np.concatenate([daily_returns, rolling_vol], axis=1)
N_FEATURES = features.shape[1]
feat_mean = features.mean(axis=0)
feat_std = features.std(axis=0)
feat_std[feat_std == 0] = 1.0
features_norm = ((features - feat_mean) / feat_std).astype(np.float32)

sequences, seq_labels, seq_days = [], [], []
for i in range(N_DAYS - FIN_SEQ_LEN):
    sequences.append(features_norm[i : i + FIN_SEQ_LEN])
    seq_labels.append(int(regime_labels[i : i + FIN_SEQ_LEN].any()))
    seq_days.append(i + FIN_SEQ_LEN)
sequences = np.array(sequences, dtype=np.float32)
seq_labels = np.array(seq_labels)
seq_days = np.array(seq_days)

normal_mask = seq_labels == 0
train_seqs = sequences[normal_mask][: int(normal_mask.sum() * 0.8)]
fin_train_tensor = torch.tensor(train_seqs, device=device)
fin_test_tensor = torch.tensor(sequences, device=device)
fin_train_loader = DataLoader(
    TensorDataset(fin_train_tensor), batch_size=128, shuffle=True
)

print(
    f"Training on {len(train_seqs)} normal-only sequences (shape: {FIN_SEQ_LEN}x{N_FEATURES})"
)


# --- LSTM Autoencoder ---
class LSTMEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, n_layers, batch_first=True)

    def forward(self, x):
        _, (h, c) = self.lstm(x)
        return h, c


class LSTMDecoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, seq_len, n_layers=1):
        super().__init__()
        self.seq_len = seq_len
        self.lstm = nn.LSTM(input_dim, hidden_dim, n_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, h, c):
        x_repeated = x[:, -1:, :].repeat(1, self.seq_len, 1)
        out, _ = self.lstm(x_repeated, (h, c))
        return self.fc(out)


class LSTMAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, seq_len):
        super().__init__()
        # TODO: Create encoder and decoder
        self.encoder = ____
        self.decoder = ____

    def forward(self, x):
        # TODO: Encode then decode
        ____


HIDDEN_DIM = 32
# TODO: Create LSTMAutoencoder, optimizer, criterion. Train 60 epochs.
fin_model = ____
fin_opt = ____
fin_criterion = nn.MSELoss()

print("\nTraining LSTM autoencoder on normal market periods...")
for epoch in range(60):
    fin_model.train()
    epoch_loss, n_batches = 0.0, 0
    for (batch,) in fin_train_loader:
        # TODO: Forward, loss, backprop
        ____
    if (epoch + 1) % 15 == 0:
        print(f"  Epoch {epoch+1:3d}/60: loss = {epoch_loss/n_batches:.6f}")

# --- Compute reconstruction errors ---
fin_model.eval()
with torch.no_grad():
    chunk_size = 512
    all_fin_errors = []
    for i in range(0, len(fin_test_tensor), chunk_size):
        chunk = fin_test_tensor[i : i + chunk_size]
        recon = fin_model(chunk)
        chunk_errors = ((chunk - recon) ** 2).mean(dim=(1, 2)).cpu().numpy()
        all_fin_errors.append(chunk_errors)
    recon_errors = np.concatenate(all_fin_errors)

normal_errors = recon_errors[seq_labels == 0]
crisis_errors = recon_errors[seq_labels == 1]
threshold = np.percentile(normal_errors, 95)
is_anomaly = recon_errors > threshold

print(f"\nNormal error: {normal_errors.mean():.4f}, Crisis: {crisis_errors.mean():.4f}")
print(f"Separation: {crisis_errors.mean() / normal_errors.mean():.1f}x")

# --- Visualisation 1: Price with anomaly overlay ---
# TODO: 2-row figure: DBS price with crisis shading, anomaly score with threshold
# Save to OUTPUT_DIR / "ex1_timeseries_anomaly_overlay.png"
fig, axes = plt.subplots(2, 1, figsize=(16, 10), gridspec_kw={"height_ratios": [2, 1]})
____
plt.tight_layout()
plt.savefig(
    OUTPUT_DIR / "ex1_timeseries_anomaly_overlay.png", dpi=150, bbox_inches="tight"
)
plt.show()

# --- Visualisation 2: Event detection ---
# TODO: For each crisis period, check if anomaly was detected.
# Compute lead time (days of early warning).
# Bar chart of lead times per event.
# Save to OUTPUT_DIR / "ex1_timeseries_event_detection.png"
event_detected, event_lead_days = [], []
for start, end, name in crisis_periods:
    window_mask = (seq_days >= start - 30) & (seq_days <= end)
    window_anomalies = is_anomaly[window_mask]
    detected = window_anomalies.any()
    event_detected.append(detected)
    if detected:
        first_idx = np.where(window_anomalies)[0][0]
        first_day = seq_days[window_mask][first_idx]
        event_lead_days.append(max(start - first_day, 0))
    else:
        event_lead_days.append(0)

fig, ax = plt.subplots(figsize=(14, 6))
____
plt.tight_layout()
plt.savefig(
    OUTPUT_DIR / "ex1_timeseries_event_detection.png", dpi=150, bbox_inches="tight"
)
plt.show()

# --- Portfolio drawdown analysis ---
# TODO: Compare passive vs anomaly-adjusted portfolio
# When anomaly detected, reduce exposure to 20% for 5 days
# Compute cumulative returns, max drawdown for both strategies
# Save to OUTPUT_DIR / "ex1_timeseries_portfolio_drawdown.png"
PORTFOLIO_VALUE = 100_000_000
portfolio_returns = daily_returns.mean(axis=1)
passive_cum = np.cumprod(1 + portfolio_returns) * PORTFOLIO_VALUE

anomaly_by_day = np.zeros(N_DAYS, dtype=bool)
for i, day in enumerate(seq_days):
    if is_anomaly[i] and day < N_DAYS:
        anomaly_by_day[day : min(day + 5, N_DAYS)] = True
adjusted_returns = np.where(anomaly_by_day, portfolio_returns * 0.2, portfolio_returns)
adjusted_cum = np.cumprod(1 + adjusted_returns) * PORTFOLIO_VALUE

passive_peak = np.maximum.accumulate(passive_cum)
passive_dd = (passive_cum - passive_peak) / passive_peak
adjusted_peak = np.maximum.accumulate(adjusted_cum)
adjusted_dd = (adjusted_cum - adjusted_peak) / adjusted_peak

fig, axes = plt.subplots(2, 1, figsize=(16, 10), gridspec_kw={"height_ratios": [2, 1]})
____
plt.tight_layout()
plt.savefig(
    OUTPUT_DIR / "ex1_timeseries_portfolio_drawdown.png", dpi=150, bbox_inches="tight"
)
plt.show()

# --- Business Impact ---
passive_worst_loss = PORTFOLIO_VALUE * abs(passive_dd.min())
adjusted_worst_loss = PORTFOLIO_VALUE * abs(adjusted_dd.min())
dollar_saved = passive_worst_loss - adjusted_worst_loss

print("\n" + "=" * 64)
print("BUSINESS IMPACT SUMMARY — SGX Regime Detection (S$100M Fund)")
print("=" * 64)
print(f"\nEvents detected: {sum(event_detected)}/{len(crisis_periods)}")
for i, (_, _, name) in enumerate(crisis_periods):
    status = f"Detected {event_lead_days[i]}d early" if event_detected[i] else "MISSED"
    print(f"  {name}: {status}")
print(f"\nPassive max drawdown:   {passive_dd.min()*100:.1f}%")
print(f"Adjusted max drawdown:  {adjusted_dd.min()*100:.1f}%")
print(f"Capital preserved:      S${dollar_saved:,.0f}")
print(f"\nFinal portfolio value:")
print(f"  Passive:              S${passive_cum[-1]:,.0f}")
print(f"  Anomaly-adjusted:     S${adjusted_cum[-1]:,.0f}")
print("=" * 64)


# ════════════════════════════════════════════════════════════════════════
# REFLECTION
# ════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  WHAT YOU'VE MASTERED")
print("=" * 70)
print(
    """
  [x] Built an LSTM encoder-decoder for time-series data
  [x] Understood temporal order preservation via recurrent architecture
  [x] Visualised original vs reconstructed vibration patterns
  [x] Applied to SGX regime change detection with early warning
  [x] Built portfolio anomaly-adjusted strategy
  [x] Quantified S$ capital preserved at worst drawdown

  KEY INSIGHT: The LSTM encoder reads a sequence step-by-step,
  building a compressed summary that captures temporal patterns.
  When the market enters a regime the model has never seen (crisis),
  reconstruction error spikes — the model is saying "this sequence
  does not match any pattern I learned from normal markets." That
  signal, days before the worst of the crash, is worth millions
  in avoided drawdown.

  Next: 09_vae.py introduces probabilistic latent spaces...
"""
)
