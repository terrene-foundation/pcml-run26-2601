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

# ══════════════════════════════════════════════════════════════════
# DIAGNOSTIC CHECKPOINT — five instruments before Visualise
# ══════════════════════════════════════════════════════════════════
# The DAE trains on noisy input with a clean target — the diagnostic
# loss mirrors that. Expect HEALTHIER activations than vanilla AE:
# noise acts as implicit regularisation, keeping ReLUs "alive"
# across the batch (dead-neuron % should drop vs 01/02).
from kailash_ml.diagnostics import run_diagnostic_checkpoint


def _diag_loss(m, batch):
    xb = batch[0] if isinstance(batch, (tuple, list)) else batch
    loss, _ = dae_loss(m, xb)
    return loss


print("\n── Diagnostic Report (Denoising AE) ──")
diag, findings = run_diagnostic_checkpoint(
    dae_model,
    flat_loader,
    _diag_loss,
    title=f"Denoising AE (sigma={NOISE_SIGMA})",
    n_batches=8,
    train_losses=dae_losses,
    show=False,
)

# ══════ EXPECTED OUTPUT (synthesized reference — full run produces similar pattern) ══════
# ════════════════════════════════════════════════════════════════
#   DL Diagnostics Report — Prescription Pad
# ════════════════════════════════════════════════════════════════
#   [!] Dead neurons  (WARNING): 'encoder.1' (relu): 18% dead
#       neurons — much lower than the 59% seen in 01 because
#       noise injection keeps gradients flowing across channels.
#   [✓] Gradient flow (HEALTHY): min RMS = 3.8e-04 at
#       'decoder.2.weight' (two orders of magnitude above the
#       vanishing threshold). Noise forces the encoder to reuse
#       every channel.
#   [✓] Loss trend    (HEALTHY): Loss converging to a HIGHER
#       floor than 01 (~0.024 vs ~0.007) — the desired signal.
# ════════════════════════════════════════════════════════════════
# Final train loss: ~0.024 after 10 epochs, sigma=0.3.
#
# STUDENT INTERPRETATION GUIDE — reading the Prescription Pad:
#
#  [STETHOSCOPE] The HIGHER loss floor is the SUCCESS signal
#     for a denoising AE, not a failure. You are measuring
#     reconstruction of CLEAN targets from NOISY inputs — the
#     irreducible noise (sigma^2 per pixel) sets a floor below
#     which no model can go. Slide 5G covers this: "the DAE's
#     loss floor is a measurement of its robustness budget."
#     >> Prescription: No fix needed. A loss that drops to ~0
#        would mean the model is memorising noise patterns —
#        worse than ours.
#
#  [X-RAY] 18% dead neurons at encoder.1 — far below the 59%
#     observed in 01_standard_ae.py. Every batch shows the
#     encoder a DIFFERENT noisy version of the same image, so
#     dead-ReLU channels get re-activated by the next batch's
#     noise pattern. Noise injection acts as an implicit
#     activation regulariser.
#     >> Prescription: No switch to GELU needed here — the
#        noise is already doing the job. You'll see this
#        contrast again in ex_2 CNN augmentation (same principle,
#        different domain).
#
#  [BLOOD TEST] Gradient RMS ~3.8e-04 is healthy. Contrast with
#     01's 9.46e-06 (three orders of magnitude worse). This is
#     why DAEs are the default in fraud/anomaly pipelines: they
#     self-regularise without needing architectural tricks.
#     >> Prescription: If RMS drops below 1e-5, sigma is too
#        large (SNR too low for the encoder to extract signal).
#        Halve NOISE_SIGMA and retrain.
#
#  FIVE-INSTRUMENT TAKEAWAY: noise IS the regulariser. A clean-
#  input baseline would show the identity-risk of 01; injecting
#  noise flips every red instrument to green while RAISING the
#  loss. This forward-references 04_sparse (different regulariser,
#  same clinical-reading skill: context determines pathology).
# ════════════════════════════════════════════════════════════════════
#
# STUDENT INTERPRETATION GUIDE — above block maps each finding to
# the 5-instrument rubric (Slide 5A-5F) so you can replicate the
# reading on any DAE you build.


# ════════════════════════════════════════════════════════════════════════
# TASK 3 — Visualise: 3-Row Denoising Grid
# ════════════════════════════════════════════════════════════════════════

show_denoising_grid(dae_model, X_test_flat, "Denoising AE (3-Row Comparison)")

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
    t = np.linspace(0, 2 * np.pi, WINDOW_SIZE)
    window = np.zeros((WINDOW_SIZE, N_SENSORS), dtype=np.float32)
    for s in range(N_SENSORS):
        freq = rng_local.uniform(0.5, 4.0)
        amp = rng_local.uniform(0.3, 1.0)
        phase = rng_local.uniform(0, 2 * np.pi)
        signal = amp * np.sin(freq * t + phase)
        signal += 0.3 * amp * np.sin(2 * freq * t + rng_local.uniform(0, np.pi))
        signal += 0.1 * amp * np.sin(3 * freq * t + rng_local.uniform(0, np.pi))
        signal += 0.2 * np.sin(0.1 * t + rng_local.uniform(0, np.pi))
        window[:, s] = signal
    return window


clean_windows = np.stack([generate_clean_window(sensor_rng) for _ in range(N_WINDOWS)])
noise = sensor_rng.normal(0, SENSOR_NOISE_SIGMA, clean_windows.shape).astype(np.float32)
noisy_windows = clean_windows + noise

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
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, latent_dim),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, input_dim),
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))


sensor_model = SensorDenoisingAE(SENSOR_INPUT_DIM).to(device)
sensor_opt = torch.optim.Adam(sensor_model.parameters(), lr=1e-3)
sensor_criterion = nn.MSELoss()

print("\nTraining sensor denoising autoencoder...")
for epoch in range(60):
    sensor_model.train()
    epoch_loss = 0.0
    n_batches = 0
    for noisy_batch, clean_batch in sensor_train_loader:
        recon = sensor_model(noisy_batch)
        loss = sensor_criterion(recon, clean_batch)
        sensor_opt.zero_grad()
        loss.backward()
        sensor_opt.step()
        epoch_loss += loss.item()
        n_batches += 1
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
fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
window_idx, sensor_idx = 5, 0
t = np.arange(WINDOW_SIZE)
clean_signal = test_clean_np.reshape(-1, WINDOW_SIZE, N_SENSORS)[
    window_idx, :, sensor_idx
]
noisy_signal = test_noisy_np.reshape(-1, WINDOW_SIZE, N_SENSORS)[
    window_idx, :, sensor_idx
]
cleaned_signal = cleaned_test.reshape(-1, WINDOW_SIZE, N_SENSORS)[
    window_idx, :, sensor_idx
]

axes[0].plot(t, clean_signal, color="#4CAF50", linewidth=1.5)
axes[0].set_title("Original Clean Signal (Ground Truth)", fontsize=12)
axes[0].set_ylabel(sensor_names[sensor_idx])
axes[0].grid(True, alpha=0.3)

axes[1].plot(t, noisy_signal, color="#F44336", linewidth=1, alpha=0.8)
axes[1].plot(
    t, clean_signal, color="#4CAF50", linewidth=1.5, alpha=0.4, label="Ground truth"
)
axes[1].set_title(
    f"Noisy Sensor Reading (SNR = {compute_snr(clean_signal, noisy_signal):.1f} dB)"
)
axes[1].set_ylabel(sensor_names[sensor_idx])
axes[1].legend(fontsize=9)
axes[1].grid(True, alpha=0.3)

axes[2].plot(t, cleaned_signal, color="#2196F3", linewidth=1.5)
axes[2].plot(
    t, clean_signal, color="#4CAF50", linewidth=1.5, alpha=0.4, label="Ground truth"
)
axes[2].set_title(
    f"DAE-Cleaned Signal (SNR = {compute_snr(clean_signal, cleaned_signal):.1f} dB)"
)
axes[2].set_ylabel(sensor_names[sensor_idx])
axes[2].set_xlabel("Timestep")
axes[2].legend(fontsize=9)
axes[2].grid(True, alpha=0.3)

fig.suptitle("Denoising Autoencoder: MRT Sensor Signal Cleaning", fontsize=13, y=1.01)
plt.tight_layout()
plt.savefig(
    OUTPUT_DIR / "ex1_sensor_denoising_timeseries.png", dpi=150, bbox_inches="tight"
)
plt.show()

# --- Visualisation 2: SNR improvement per sensor ---
fig, ax = plt.subplots(figsize=(12, 6))
x = np.arange(N_SENSORS)
width = 0.35
ax.bar(
    x - width / 2,
    snr_noisy_per_sensor,
    width,
    label="Noisy",
    color="#F44336",
    alpha=0.8,
)
ax.bar(
    x + width / 2,
    snr_cleaned_per_sensor,
    width,
    label="DAE-Cleaned",
    color="#2196F3",
    alpha=0.8,
)
ax.set_xlabel("Sensor")
ax.set_ylabel("Signal-to-Noise Ratio (dB)")
ax.set_title("SNR Improvement Per Sensor After DAE Cleaning", fontsize=13)
ax.set_xticks(x)
ax.set_xticklabels(sensor_names, rotation=45, ha="right", fontsize=9)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3, axis="y")
for i in range(N_SENSORS):
    improvement = snr_cleaned_per_sensor[i] - snr_noisy_per_sensor[i]
    ax.annotate(
        f"+{improvement:.1f}dB",
        xy=(x[i] + width / 2, snr_cleaned_per_sensor[i]),
        ha="center",
        va="bottom",
        fontsize=8,
        color="#1565C0",
        fontweight="bold",
    )
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "ex1_sensor_snr_improvement.png", dpi=150, bbox_inches="tight")
plt.show()

# --- Visualisation 3: Frequency spectrum comparison ---
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for ax, sig, title, color in [
    (axes[0], clean_signal, "Clean (Ground Truth)", "#4CAF50"),
    (axes[1], noisy_signal, "Noisy", "#F44336"),
    (axes[2], cleaned_signal, "DAE-Cleaned", "#2196F3"),
]:
    fft_vals = np.abs(np.fft.rfft(sig))
    freqs = np.fft.rfftfreq(len(sig))
    ax.plot(freqs[1:], fft_vals[1:], color=color, linewidth=1.5)
    ax.fill_between(freqs[1:], 0, fft_vals[1:], color=color, alpha=0.2)
    ax.set_title(title, fontsize=12)
    ax.set_xlabel("Frequency")
    ax.set_ylabel("Magnitude")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, max(np.abs(np.fft.rfft(clean_signal))[1:]) * 1.5)
fig.suptitle("Frequency Spectrum: DAE Removes High-Frequency Noise", fontsize=13)
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

false_alerts_noisy_q = int(
    SMRT_TRAINS * SENSORS_PER_TRAIN * 90 * FALSE_ALERT_RATE_NOISY
)
false_alerts_clean_q = int(
    SMRT_TRAINS * SENSORS_PER_TRAIN * 90 * FALSE_ALERT_RATE_CLEAN
)
missed_faults_noisy = int(REAL_FAULTS_PER_QUARTER * MISSED_FAULT_RATE_NOISY)
missed_faults_clean = int(REAL_FAULTS_PER_QUARTER * MISSED_FAULT_RATE_CLEAN)

savings_false_alerts = (
    false_alerts_noisy_q - false_alerts_clean_q
) * COST_PER_FALSE_ALERT
savings_missed_faults = (
    missed_faults_noisy - missed_faults_clean
) * COST_PER_MISSED_FAULT
total_quarterly_savings = savings_false_alerts + savings_missed_faults

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

