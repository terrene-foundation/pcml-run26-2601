# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# MLFP05 — Exercise 1.3: Denoising Autoencoder (Noise Robustness)
# ════════════════════════════════════════════════════════════════════════
#
# WHAT YOU'LL LEARN:
#   - Build a denoising autoencoder (DAE) with noise injection training
#   - Understand WHY noise acts as implicit regularisation
#   - Visualise the 3-row grid: original -> noisy -> cleaned
#   - Apply to SMRT MRT sensor signal cleaning
#   - Quantify business impact: reduced false alerts, fewer missed faults
#
# PREREQUISITES: 02_undercomplete_ae.py
# ESTIMATED TIME: ~20 min
#
# TASKS:
#   1. Build DAE architecture (same as undercomplete, different training)
#   2. Train with noise injection on Fashion-MNIST
#   3. Visualise denoising with 3-row comparison grid
#   4. Apply: SMRT sensor data cleaning with SNR improvement analysis
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
    INPUT_DIM,
    LATENT_DIM,
    EPOCHS,
    OUTPUT_DIR,
    device,
    load_fashion_mnist,
    setup_engines,
    train_variant,
    show_denoising_grid,
    show_reconstruction,
    register_model,
)


# ════════════════════════════════════════════════════════════════════════
# THEORY — Noise Injection as Regularisation
# ════════════════════════════════════════════════════════════════════════
# Add Gaussian noise to the input, then train the model to reconstruct
# the CLEAN original. The noise acts as implicit regularisation — the
# encoder cannot memorise pixel values because the pixels are corrupted
# differently every epoch.
#
# Analogy: A court stenographer who must produce accurate transcripts
# even when the courtroom is noisy. Over time, they learn to focus on
# the CONTENT of speech (the signal) and ignore ambient noise. The DAE
# does the same with images — it learns STRUCTURE, not noise.
#
# WHY THIS MATTERS: In medical imaging, scans often have noise (sensor
# artifacts, patient movement). In IoT, sensor readings have electrical
# interference. A DAE trained on clean + noisy pairs learns to clean
# up real signals, potentially revealing patterns hidden by noise.


# ════════════════════════════════════════════════════════════════════════
# TASK 1 — Load data and engines
# ════════════════════════════════════════════════════════════════════════

X_flat, X_test_flat, X_img, X_test_img, flat_loader, img_loader = load_fashion_mnist()
conn, tracker, exp_name, registry, has_registry = setup_engines()


# ════════════════════════════════════════════════════════════════════════
# TASK 2 — Build and Train Denoising AE
# ════════════════════════════════════════════════════════════════════════


class DenoisingAE(nn.Module):
    """Same architecture as undercomplete, but trained with noise injection."""

    def __init__(self, input_dim: int, latent_dim: int):
        super().__init__()
        # TODO: Build encoder — same as UndercompleteAE:
        #       Linear(input_dim, 256), ReLU, Linear(256, 64), ReLU,
        #       Linear(64, latent_dim)
        self.encoder = ____

        # TODO: Build decoder — mirror of encoder ending with Sigmoid
        self.decoder = ____

    def add_noise(self, x, sigma=0.3):
        # TODO: Add Gaussian noise: clamp(x + sigma * randn_like(x), 0, 1)
        ____

    def forward(self, x):
        # TODO: Encode then decode. Return (reconstruction, latent_code)
        ____


NOISE_SIGMA = 0.3


def dae_loss(model, xb):
    """Train on noisy input, reconstruct clean target."""
    # TODO: Add noise to xb using model.add_noise(xb, sigma=NOISE_SIGMA)
    # Forward pass on NOISY input, compute MSE against CLEAN xb
    # Return (loss, empty_dict)
    ____


print("\n" + "=" * 70)
print("  Denoising AE — Noise Injection (sigma=0.3)")
print("=" * 70)
print("  Input: corrupted image. Target: clean image. The model learns to denoise.")

# TODO: Create DenoisingAE(INPUT_DIM, LATENT_DIM) and train
dae_model = ____
dae_losses = ____


# ════════════════════════════════════════════════════════════════════════
# TASK 3 — Visualise: 3-Row Denoising Grid
# ════════════════════════════════════════════════════════════════════════

# TODO: show_denoising_grid(dae_model, X_test_flat, "Denoising AE (3-Row Comparison)")
____

# ── Checkpoint ──────────────────────────────────────────────────────
assert len(dae_losses) == EPOCHS
assert dae_losses[-1] < dae_losses[0]
# INTERPRETATION: The 3-row grid tells the story:
# Row 1 (Original): the clean clothing image
# Row 2 (Noisy): what the model receives — grainy, corrupted
# Row 3 (Cleaned): what the model outputs — noise removed, structure preserved
print("\n--- Checkpoint passed --- denoising AE trained\n")

if has_registry:
    register_model(registry, "denoising_ae", dae_model, dae_losses[-1])


# ════════════════════════════════════════════════════════════════════════
# APPLY — SMRT MRT Sensor Signal Cleaning
# ════════════════════════════════════════════════════════════════════════
# BUSINESS SCENARIO: You are an IoT engineer at SMRT (Singapore MRT).
# Vibration and temperature sensors on MRT trains generate readings
# every second. Sensor noise — electrical interference, sensor drift,
# dust on contacts — corrupts the signal. Noisy signals trigger false
# maintenance alerts (costly) or mask real faults (dangerous).

print("\n" + "=" * 70)
print("  APPLICATION: SMRT Sensor Data Cleaning")
print("=" * 70)

# --- Generate realistic MRT sensor time-series data ---
N_SENSORS = 10
WINDOW_SIZE = 100
N_WINDOWS = 5000
SENSOR_NOISE_SIGMA = 0.4

sensor_rng = np.random.default_rng(42)
sensor_names = [
    "Vibration_X",
    "Vibration_Y",
    "Vibration_Z",
    "Temperature_Bearing",
    "Temperature_Motor",
    "Current_Draw",
    "Voltage",
    "Brake_Pressure",
    "Door_Actuator",
    "HVAC_Flow",
]


def generate_clean_window(rng_local):
    # TODO: Generate a clean sensor window (WINDOW_SIZE x N_SENSORS)
    # For each sensor: sum of 3 sinusoids at different frequencies + slow drift
    # t = np.linspace(0, 2*pi, WINDOW_SIZE)
    # freq = rng_local.uniform(0.5, 4.0), amp = rng_local.uniform(0.3, 1.0)
    # signal = amp*sin(freq*t+phase) + harmonics + drift
    ____


# TODO: Generate clean_windows, add noise to create noisy_windows
# clean_windows = np.stack([generate_clean_window(sensor_rng) for _ in range(N_WINDOWS)])
# noise = sensor_rng.normal(0, SENSOR_NOISE_SIGMA, clean_windows.shape)
# noisy_windows = clean_windows + noise
clean_windows = ____
noise = ____
noisy_windows = ____

# TODO: Normalise using mean/std of clean windows
# Reshape to (N_WINDOWS, WINDOW_SIZE*N_SENSORS), create tensors and loader
feat_mean = clean_windows.reshape(-1, N_SENSORS).mean(axis=0)
feat_std = clean_windows.reshape(-1, N_SENSORS).std(axis=0)
feat_std[feat_std == 0] = 1.0

clean_norm = ((clean_windows - feat_mean) / feat_std).astype(np.float32)
noisy_norm = ((noisy_windows - feat_mean) / feat_std).astype(np.float32)

n_train = int(N_WINDOWS * 0.8)
SENSOR_INPUT_DIM = WINDOW_SIZE * N_SENSORS
train_noisy = torch.tensor(noisy_norm[:n_train].reshape(n_train, -1), device=device)
train_clean = torch.tensor(clean_norm[:n_train].reshape(n_train, -1), device=device)
test_noisy = torch.tensor(
    noisy_norm[n_train:].reshape(N_WINDOWS - n_train, -1), device=device
)
test_clean = torch.tensor(
    clean_norm[n_train:].reshape(N_WINDOWS - n_train, -1), device=device
)
sensor_train_loader = DataLoader(
    TensorDataset(train_noisy, train_clean), batch_size=128, shuffle=True
)

print(
    f"Generated {N_WINDOWS} sensor windows: {WINDOW_SIZE} timesteps x {N_SENSORS} sensors"
)


class SensorDenoisingAE(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int = 128):
        super().__init__()
        # TODO: Build encoder — Linear(input_dim, 512), ReLU,
        #       Linear(512, 256), ReLU, Linear(256, latent_dim), ReLU
        self.encoder = ____

        # TODO: Build decoder — mirror, NO Sigmoid (regression output)
        #       Linear(latent_dim, 256), ReLU, Linear(256, 512), ReLU,
        #       Linear(512, input_dim)
        self.decoder = ____

    def forward(self, x):
        # TODO: Return decoder(encoder(x))
        ____


# TODO: Create model, optimizer, criterion. Train 60 epochs on noisy->clean pairs.
sensor_model = ____
sensor_opt = ____
sensor_criterion = nn.MSELoss()

print("\nTraining sensor denoising autoencoder...")
for epoch in range(60):
    sensor_model.train()
    epoch_loss = 0.0
    n_batches = 0
    for noisy_batch, clean_batch in sensor_train_loader:
        # TODO: Forward noisy_batch through model, MSE vs clean_batch, backprop
        ____
    if (epoch + 1) % 15 == 0:
        print(f"  Epoch {epoch+1:3d}/60: loss = {epoch_loss/n_batches:.6f}")

# --- Evaluate ---
sensor_model.eval()
with torch.no_grad():
    cleaned_test = sensor_model(test_noisy).cpu().numpy()
test_noisy_np = test_noisy.cpu().numpy()
test_clean_np = test_clean.cpu().numpy()


def compute_snr(clean_sig, signal):
    noise_sig = signal - clean_sig
    signal_power = np.mean(clean_sig**2)
    noise_power = np.mean(noise_sig**2)
    if noise_power == 0:
        return float("inf")
    return 10 * np.log10(signal_power / noise_power)


# TODO: Compute SNR per sensor for noisy and cleaned signals
# Loop over N_SENSORS, extract per-sensor data, compute SNR
snr_noisy_per_sensor = []
snr_cleaned_per_sensor = []
for s in range(N_SENSORS):
    clean_s = test_clean_np.reshape(-1, WINDOW_SIZE, N_SENSORS)[:, :, s].ravel()
    noisy_s = test_noisy_np.reshape(-1, WINDOW_SIZE, N_SENSORS)[:, :, s].ravel()
    cleaned_s = cleaned_test.reshape(-1, WINDOW_SIZE, N_SENSORS)[:, :, s].ravel()
    snr_noisy_per_sensor.append(compute_snr(clean_s, noisy_s))
    snr_cleaned_per_sensor.append(compute_snr(clean_s, cleaned_s))

snr_improvement = np.mean(snr_cleaned_per_sensor) - np.mean(snr_noisy_per_sensor)
print(
    f"\nSignal quality: Noisy={np.mean(snr_noisy_per_sensor):.1f}dB, "
    f"Cleaned={np.mean(snr_cleaned_per_sensor):.1f}dB, Improvement=+{snr_improvement:.1f}dB"
)

# --- Visualisation 1: 3-panel time-series ---
# TODO: Create 3-row figure showing clean signal, noisy signal, DAE-cleaned signal
# for one window and one sensor. Include SNR values in titles.
# Save to OUTPUT_DIR / "ex1_sensor_denoising_timeseries.png"
fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
window_idx, sensor_idx = 5, 0
____
plt.tight_layout()
plt.savefig(
    OUTPUT_DIR / "ex1_sensor_denoising_timeseries.png", dpi=150, bbox_inches="tight"
)
plt.show()

# --- Visualisation 2: SNR improvement per sensor ---
# TODO: Create grouped bar chart — noisy vs cleaned SNR per sensor
# Save to OUTPUT_DIR / "ex1_sensor_snr_improvement.png"
fig, ax = plt.subplots(figsize=(12, 6))
____
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "ex1_sensor_snr_improvement.png", dpi=150, bbox_inches="tight")
plt.show()

# --- Visualisation 3: Frequency spectrum comparison ---
# TODO: 1x3 subplot showing FFT magnitude of clean, noisy, cleaned signals
# Save to OUTPUT_DIR / "ex1_sensor_frequency_spectrum.png"
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
____
plt.tight_layout()
plt.savefig(
    OUTPUT_DIR / "ex1_sensor_frequency_spectrum.png", dpi=150, bbox_inches="tight"
)
plt.show()

# --- Business Impact ---
SMRT_TRAINS = 150
SENSORS_PER_TRAIN = 40
FALSE_ALERT_RATE_NOISY = 0.05
FALSE_ALERT_RATE_CLEAN = 0.008
MISSED_FAULT_RATE_NOISY = 0.28
MISSED_FAULT_RATE_CLEAN = 0.11
REAL_FAULTS_PER_QUARTER = 12
COST_PER_MISSED_FAULT = 200_000
COST_PER_FALSE_ALERT = 5_000

# TODO: Compute quarterly savings
# false_alerts_noisy_q = SMRT_TRAINS * SENSORS_PER_TRAIN * 90 * FALSE_ALERT_RATE_NOISY
# savings_false_alerts = (noisy - clean) * COST_PER_FALSE_ALERT
# savings_missed_faults = (missed_noisy - missed_clean) * COST_PER_MISSED_FAULT
false_alerts_noisy_q = ____
false_alerts_clean_q = ____
missed_faults_noisy = ____
missed_faults_clean = ____
savings_false_alerts = ____
savings_missed_faults = ____
total_quarterly_savings = ____

print("\n" + "=" * 64)
print("BUSINESS IMPACT SUMMARY — SMRT Predictive Maintenance")
print("=" * 64)
print(f"\nSMRT fleet: {SMRT_TRAINS} trains x {SENSORS_PER_TRAIN} sensors")
print(f"DAE signal improvement: +{snr_improvement:.1f} dB average")
print(f"\nFalse maintenance alerts per quarter:")
print(f"  With noisy data:    {false_alerts_noisy_q:>10,}")
print(f"  With DAE-cleaned:   {false_alerts_clean_q:>10,}")
print(
    f"  Reduction:          {false_alerts_noisy_q - false_alerts_clean_q:>10,} ({(1 - false_alerts_clean_q/false_alerts_noisy_q):.0%})"
)
print(f"\nMissed real faults per quarter (of {REAL_FAULTS_PER_QUARTER}):")
print(
    f"  With noisy data:    {missed_faults_noisy:>10} ({MISSED_FAULT_RATE_NOISY:.0%})"
)
print(
    f"  With DAE-cleaned:   {missed_faults_clean:>10} ({MISSED_FAULT_RATE_CLEAN:.0%})"
)
print(f"\nQuarterly cost savings:")
print(f"  False alert reduction: {'S$' + f'{savings_false_alerts:,.0f}':>14}")
print(f"  Fewer missed faults:   {'S$' + f'{savings_missed_faults:,.0f}':>14}")
print(f"  Total per quarter:     {'S$' + f'{total_quarterly_savings:,.0f}':>14}")
print(f"  Total per year:        {'S$' + f'{total_quarterly_savings * 4:,.0f}':>14}")
print("=" * 64)


# ════════════════════════════════════════════════════════════════════════
# REFLECTION
# ════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  WHAT YOU'VE MASTERED")
print("=" * 70)
print(
    """
  [x] Built a denoising autoencoder with Gaussian noise injection
  [x] Understood noise as implicit regularisation (can't memorise pixels)
  [x] Visualised the 3-row proof: original -> noisy -> cleaned
  [x] Applied DAE to SMRT sensor data cleaning (10 sensor types)
  [x] Measured SNR improvement per sensor
  [x] Quantified business impact: false alert reduction + missed fault prevention

  KEY INSIGHT: The DAE learns ROBUST features that survive corruption.
  The same architecture that removes noise from Fashion-MNIST images
  removes electrical interference from MRT vibration sensors. The
  principle is universal: noise injection forces the model to find
  the signal underneath.

  Next: 04_sparse_ae.py forces specialist neurons with L1 penalty...
"""
)
