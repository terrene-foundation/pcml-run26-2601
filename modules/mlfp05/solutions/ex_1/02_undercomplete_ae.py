# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# MLFP05 — Exercise 1.2: Undercomplete Autoencoder (Bottleneck Compression)
# ════════════════════════════════════════════════════════════════════════
#
# WHAT YOU'LL LEARN:
#   - Build an undercomplete AE with bottleneck (784 -> 16 = 49:1 compression)
#   - Understand WHY forced compression solves the identity risk
#   - Visualise blurry but meaningful reconstructions
#   - Apply to credit card fraud detection at DBS Singapore
#   - Quantify business impact in S$ with precision-recall analysis
#
# PREREQUISITES: 01_standard_ae.py (identity risk understanding)
# ESTIMATED TIME: ~20 min
#
# TASKS:
#   1. Build undercomplete AE (784 -> 256 -> 64 -> 16)
#   2. Train on Fashion-MNIST and visualise reconstructions
#   3. Apply: fraud detection at DBS using anomaly reconstruction error
#   4. Business impact analysis with S$ projections
#
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
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
    show_reconstruction,
    register_model,
)


# ════════════════════════════════════════════════════════════════════════
# THEORY — Forced Compression via Bottleneck
# ════════════════════════════════════════════════════════════════════════
# The fix for the identity risk is simple: make the bottleneck SMALLER
# than the input. With latent_dim=16, the encoder must compress 784
# pixels into just 16 numbers — a 49:1 compression ratio.
#
# Analogy: A 50-page quarterly report compressed into a one-page
# executive summary. The summary MUST capture the key points because
# there is no room for everything. That forced compression is exactly
# what the undercomplete bottleneck does.
#
# The encoder must learn WHAT MATTERS in each image — the difference
# between a shirt and a shoe, not the exact pixel values. This is
# representation learning: extracting structure from data.


# ════════════════════════════════════════════════════════════════════════
# TASK 1 — Load data and engines
# ════════════════════════════════════════════════════════════════════════

X_flat, X_test_flat, X_img, X_test_img, flat_loader, img_loader = load_fashion_mnist()
conn, tracker, exp_name, registry, has_registry = setup_engines()


# ════════════════════════════════════════════════════════════════════════
# TASK 2 — Build and Train Undercomplete AE
# ════════════════════════════════════════════════════════════════════════


class UndercompleteAE(nn.Module):
    """Bottleneck forces compression: 784 -> 256 -> 64 -> 16 -> 64 -> 256 -> 784."""

    def __init__(self, input_dim: int, latent_dim: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 256),
            nn.ReLU(),
            nn.Linear(256, input_dim),
            nn.Sigmoid(),
        )

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z), z


def undercomplete_ae_loss(model, xb):
    x_hat, _ = model(xb)
    return F.mse_loss(x_hat, xb), {}


print("\n" + "=" * 70)
print("  Undercomplete AE — Forced Compression (latent=16)")
print("=" * 70)
print("  784 pixels -> 16 numbers. Compression ratio 49:1.")

undercomplete_model = UndercompleteAE(INPUT_DIM, LATENT_DIM)
undercomplete_losses = train_variant(
    tracker,
    exp_name,
    undercomplete_model,
    "undercomplete_ae",
    flat_loader,
    undercomplete_ae_loss,
)

# ══════════════════════════════════════════════════════════════════
# DIAGNOSTIC CHECKPOINT — five instruments before Visualise
# ══════════════════════════════════════════════════════════════════
# Same pattern as 01_standard_ae.py — see that file for the full
# Prescription Pad walkthrough. Here we expect a DIFFERENT picture:
# the undercomplete bottleneck (latent=16) blocks identity-copy, so
# gradients should be healthier across the encoder *while* the
# train loss stays noticeably higher than the overcomplete AE —
# that higher loss is the SIGNAL of genuine compression learning.
from kailash_ml.diagnostics import run_diagnostic_checkpoint


def _diag_loss(m, batch):
    xb = batch[0] if isinstance(batch, (tuple, list)) else batch
    loss, _ = undercomplete_ae_loss(m, xb)
    return loss


print("\n── Diagnostic Report (Undercomplete AE) ──")
diag, findings = run_diagnostic_checkpoint(
    undercomplete_model,
    flat_loader,
    _diag_loss,
    title="Undercomplete AE (latent=16)",
    n_batches=8,
    train_losses=undercomplete_losses,
    show=False,
)

# ══════ EXPECTED OUTPUT (synthesized reference — full run produces similar pattern) ══════
# ════════════════════════════════════════════════════════════════
#   DL Diagnostics Report — Prescription Pad
# ════════════════════════════════════════════════════════════════
#   [✓] Gradient flow (HEALTHY): min RMS = 4.3e-04 at
#       'decoder.2.weight'. Two orders of magnitude HIGHER
#       than 01's 9.46e-06. Bottleneck forces every channel
#       to carry signal.
#   [✓] Dead neurons  (HEALTHY): max 11% dead on decoder.1.
#       Tight bottleneck means no ReLU can afford to die.
#       Contrast 01's 59% dead.
#   [✓] Loss trend    (HEALTHY): train slope -1.6e-03/epoch.
#       Final loss ~0.028 — HIGHER than 01's 0.007. This
#       HIGHER loss IS the success signal.
# ════════════════════════════════════════════════════════════════
# Final train loss: ~0.028 after 10 epochs, latent=16 < input=784.
#
# STUDENT INTERPRETATION GUIDE — reading the Prescription Pad:
#
#  [STETHOSCOPE — INVERTED SUCCESS CRITERION] HIGHER final
#     loss than 01 is the WIN condition. The model is
#     forbidden from copying (latent=16 cannot encode 784
#     pixels pointwise), so reconstruction is necessarily
#     lossy. The loss quantifies how much information the
#     16-dim bottleneck retained. Slide 5C covers this: "an
#     autoencoder that achieves zero loss on an overcomplete
#     architecture has learned nothing; an undercomplete AE
#     with non-trivial loss has learned STRUCTURE."
#     >> Prescription: No fix. Final loss <0.01 on this
#        latent size would actually be suspicious — verify
#        the bottleneck isn't accidentally bypassed (check
#        for skip connections that shouldn't be there).
#
#  [BLOOD TEST — BOTTLENECK REVIVING GRADIENTS] RMS 4.3e-04
#     at decoder.2 is 45x healthier than 01's 9.46e-06. The
#     mechanism: every decoder layer MUST receive signal
#     from a 16-dim vector to reconstruct 784 pixels, so
#     backprop distributes gradient across more channels.
#     Compare this to the overcomplete case where most
#     channels get vanishing gradient because the identity
#     function doesn't need them.
#     >> Prescription: If RMS drops BELOW 1e-5 even with the
#        bottleneck, latent dim is TOO tight (information
#        destroyed faster than the model can learn). Try
#        latent=32 or 64.
#
#  [X-RAY — CAPACITY DISCIPLINE] 11% dead is within the
#     normal ReLU operating range. Contrast 01's 59% where
#     the overcomplete architecture let half the channels
#     shut down. Undercomplete architecture enforces
#     capacity discipline: every neuron must contribute or
#     the model cannot achieve even its lossy reconstruction.
#     >> Prescription: Dead% >25% here indicates either
#        LR too high (killing channels faster than they
#        learn) or latent still too large (capacity excess
#        allowing some neurons to be redundant).
#
#  FIVE-INSTRUMENT TAKEAWAY: undercomplete AE demonstrates
#  the ARCHITECTURAL fix for the identity-risk pathology of
#  01. Same 5 instruments, inverted readings: higher loss
#  is better, fewer dead neurons, stronger gradients.
#  This is the "normal" autoencoder baseline that variants
#  3-10 refine with explicit regularisers. You'll see this
#  flip repeatedly in the course: the same metric reads
#  differently based on the WHY of the model (design intent).
# ════════════════════════════════════════════════════════════════════


# ════════════════════════════════════════════════════════════════════════
# VISUALISE — Reconstruction grid
# ════════════════════════════════════════════════════════════════════════

show_reconstruction(
    undercomplete_model, X_test_flat, f"Undercomplete AE (latent={LATENT_DIM})"
)

# ── Checkpoint ──────────────────────────────────────────────────────
assert len(undercomplete_losses) == EPOCHS
assert undercomplete_losses[-1] < undercomplete_losses[0]
# INTERPRETATION: The reconstructions are blurry but recognisable.
# The model kept the SHAPE (is it a shirt? a shoe?) but lost DETAIL
# (exact button placement, stitching pattern). This is the information
# bottleneck principle: compress enough, and the model learns structure.
print("\n--- Checkpoint passed --- undercomplete AE trained\n")

if has_registry:
    register_model(
        registry, "undercomplete_ae", undercomplete_model, undercomplete_losses[-1]
    )


# ════════════════════════════════════════════════════════════════════════
# APPLY — Credit Card Fraud Detection at DBS Singapore
# ════════════════════════════════════════════════════════════════════════
# BUSINESS SCENARIO: You are a fraud analyst at DBS Bank. 99.8% of
# daily transactions are legitimate. You have NO labelled fraud
# examples — only a gut feeling that "unusual" transactions deserve
# investigation. Your manager asks: "Can we catch more fraud without
# drowning investigators in false alerts?"
#
# TECHNIQUE: Train on ONLY normal transactions so the AE learns what
# "normal" looks like. At inference, legitimate transactions reconstruct
# well (low error); fraudulent ones reconstruct poorly (high error)
# because the encoder never learned their patterns.

print("\n" + "=" * 70)
print("  APPLICATION: Credit Card Fraud Detection at DBS")
print("=" * 70)

# --- Generate realistic Singapore bank transaction data ---
N_TOTAL = 200_000
FRAUD_RATE = 0.002  # 0.2% fraud — realistic for Singapore card-present

n_fraud = int(N_TOTAL * FRAUD_RATE)
n_normal = N_TOTAL - n_fraud
rng = np.random.default_rng(42)

# Normal transaction features
normal_amounts = rng.lognormal(mean=3.5, sigma=1.2, size=n_normal).clip(0.5, 5000)
normal_hour = rng.normal(loc=14, scale=4, size=n_normal).clip(0, 23).astype(int)
normal_merchant_cat = rng.choice(
    range(15),
    size=n_normal,
    p=[
        0.18,
        0.15,
        0.12,
        0.10,
        0.08,
        0.07,
        0.06,
        0.05,
        0.04,
        0.04,
        0.03,
        0.03,
        0.02,
        0.02,
        0.01,
    ],
)
normal_is_online = rng.binomial(1, 0.35, size=n_normal)
normal_distance = rng.exponential(scale=5, size=n_normal).clip(0, 50)
normal_freq_24h = rng.poisson(lam=2, size=n_normal)
normal_amt_ratio = rng.normal(1.0, 0.3, size=n_normal).clip(0.1, 3.0)
normal_foreign = rng.binomial(1, 0.08, size=n_normal)

# Fraud transaction features — shifted distributions
fraud_amounts = rng.lognormal(mean=5.5, sigma=1.5, size=n_fraud).clip(10, 50000)
fraud_hour = rng.choice([0, 1, 2, 3, 4, 22, 23], size=n_fraud)
fraud_merchant_cat = rng.choice(
    range(15),
    size=n_fraud,
    p=[
        0.02,
        0.02,
        0.03,
        0.03,
        0.05,
        0.05,
        0.05,
        0.08,
        0.10,
        0.10,
        0.12,
        0.12,
        0.08,
        0.08,
        0.07,
    ],
)
fraud_is_online = rng.binomial(1, 0.75, size=n_fraud)
fraud_distance = rng.exponential(scale=40, size=n_fraud).clip(0, 200)
fraud_freq_24h = rng.poisson(lam=8, size=n_fraud)
fraud_amt_ratio = rng.normal(4.0, 1.5, size=n_fraud).clip(0.5, 15.0)
fraud_foreign = rng.binomial(1, 0.45, size=n_fraud)

# Combine into polars DataFrame
amounts = np.concatenate([normal_amounts, fraud_amounts])
hours = np.concatenate([normal_hour, fraud_hour])
merchant_cats = np.concatenate([normal_merchant_cat, fraud_merchant_cat])
is_online = np.concatenate([normal_is_online, fraud_is_online])
distances = np.concatenate([normal_distance, fraud_distance])
freq_24h = np.concatenate([normal_freq_24h, fraud_freq_24h])
amt_ratios = np.concatenate([normal_amt_ratio, fraud_amt_ratio])
foreign = np.concatenate([normal_foreign, fraud_foreign])
labels = np.concatenate([np.zeros(n_normal), np.ones(n_fraud)])

df = pl.DataFrame(
    {
        "amount": amounts,
        "hour": hours,
        "merchant_category": merchant_cats,
        "is_online": is_online,
        "distance_from_home_km": distances,
        "transactions_last_24h": freq_24h,
        "amount_vs_avg_ratio": amt_ratios,
        "is_foreign": foreign,
        "is_fraud": labels,
    }
).sample(fraction=1.0, seed=42, shuffle=True)

print(
    f"Dataset: {df.shape[0]:,} transactions, {df.filter(pl.col('is_fraud') == 1).shape[0]} fraud ({FRAUD_RATE*100:.1f}%)"
)

# --- Prepare training data (normal-only) ---
feature_cols = [c for c in df.columns if c != "is_fraud"]
all_features = df.select(feature_cols).to_numpy().astype(np.float32)
all_labels = df["is_fraud"].to_numpy()

feat_min = all_features.min(axis=0)
feat_max = all_features.max(axis=0)
feat_range = feat_max - feat_min
feat_range[feat_range == 0] = 1.0
all_features_norm = (all_features - feat_min) / feat_range

normal_mask = all_labels == 0
fraud_mask = all_labels == 1
normal_features = all_features_norm[normal_mask]
fraud_features = all_features_norm[fraud_mask]

n_train = int(len(normal_features) * 0.8)
train_features = normal_features[:n_train]
test_normal = normal_features[n_train:]
test_fraud = fraud_features
test_features = np.vstack([test_normal, test_fraud])
test_labels = np.concatenate([np.zeros(len(test_normal)), np.ones(len(test_fraud))])

train_tensor = torch.tensor(train_features, device=device)
test_tensor = torch.tensor(test_features, device=device)
fraud_train_loader = DataLoader(
    TensorDataset(train_tensor), batch_size=512, shuffle=True
)

print(f"Training on {len(train_features):,} normal-only transactions")
print(f"Test set: {len(test_normal):,} normal + {len(test_fraud):,} fraud")

# --- Build and train fraud detector ---
FRAUD_INPUT_DIM = len(feature_cols)


class FraudDetectorAE(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, input_dim),
            nn.Sigmoid(),
        )

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)


fraud_model = FraudDetectorAE(FRAUD_INPUT_DIM, 3).to(device)
fraud_opt = torch.optim.Adam(fraud_model.parameters(), lr=1e-3)

print("\nTraining fraud detection autoencoder...")
for epoch in range(50):
    fraud_model.train()
    epoch_loss = 0.0
    n_batches = 0
    for (batch,) in fraud_train_loader:
        recon = fraud_model(batch)
        loss = F.mse_loss(recon, batch)
        fraud_opt.zero_grad()
        loss.backward()
        fraud_opt.step()
        epoch_loss += loss.item()
        n_batches += 1
    if (epoch + 1) % 10 == 0:
        print(f"  Epoch {epoch+1:3d}/50: loss = {epoch_loss/n_batches:.6f}")

# --- Compute reconstruction errors ---
fraud_model.eval()
with torch.no_grad():
    recon_test = fraud_model(test_tensor)
    errors = ((test_tensor - recon_test) ** 2).mean(dim=1).cpu().numpy()

normal_errors = errors[test_labels == 0]
fraud_errors = errors[test_labels == 1]

print(f"\nReconstruction error statistics:")
print(
    f"  Normal: mean={normal_errors.mean():.6f}, p95={np.percentile(normal_errors, 95):.6f}"
)
print(
    f"  Fraud:  mean={fraud_errors.mean():.6f}, p95={np.percentile(fraud_errors, 95):.6f}"
)
print(f"  Separation ratio: {fraud_errors.mean() / normal_errors.mean():.1f}x")

# --- Visualisation 1: Error distributions ---
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].hist(
    normal_errors, bins=80, alpha=0.7, label="Normal", color="#2196F3", density=True
)
axes[0].hist(
    fraud_errors, bins=80, alpha=0.7, label="Fraud", color="#F44336", density=True
)
axes[0].axvline(
    np.percentile(normal_errors, 95),
    color="#FF9800",
    linestyle="--",
    label=f"95th pctl = {np.percentile(normal_errors, 95):.4f}",
)
axes[0].set_xlabel("Reconstruction Error (MSE)")
axes[0].set_ylabel("Density")
axes[0].set_title("Reconstruction Error Distribution\nNormal vs Fraud", fontsize=13)
axes[0].legend(fontsize=10)

bp = axes[1].boxplot(
    [normal_errors, fraud_errors],
    labels=["Normal", "Fraud"],
    patch_artist=True,
    widths=0.5,
)
bp["boxes"][0].set_facecolor("#2196F3")
bp["boxes"][1].set_facecolor("#F44336")
axes[1].set_ylabel("Reconstruction Error (MSE)")
axes[1].set_title("Error Comparison: Normal vs Fraud", fontsize=13)
plt.tight_layout()
plt.savefig(
    OUTPUT_DIR / "ex1_fraud_error_distribution.png", dpi=150, bbox_inches="tight"
)
plt.show()

# --- Visualisation 2: Precision-Recall ---
thresholds = np.linspace(errors.min(), np.percentile(errors, 99.5), 200)
precisions, recalls, f1_scores = [], [], []
for t in thresholds:
    predicted_fraud = errors > t
    true_fraud = test_labels == 1
    tp = np.sum(predicted_fraud & true_fraud)
    fp = np.sum(predicted_fraud & ~true_fraud)
    fn = np.sum(~predicted_fraud & true_fraud)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    precisions.append(precision)
    recalls.append(recall)
    f1_scores.append(f1)

precisions = np.array(precisions)
recalls = np.array(recalls)
f1_scores = np.array(f1_scores)
best_f1_idx = np.argmax(f1_scores)
best_threshold = thresholds[best_f1_idx]
best_precision = precisions[best_f1_idx]
best_recall = recalls[best_f1_idx]
best_f1 = f1_scores[best_f1_idx]

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].plot(recalls, precisions, color="#673AB7", linewidth=2)
axes[0].scatter(
    [best_recall],
    [best_precision],
    color="#F44336",
    s=100,
    zorder=5,
    label=f"Best F1={best_f1:.3f}\n(P={best_precision:.3f}, R={best_recall:.3f})",
)
axes[0].set_xlabel("Recall (Fraud Caught)")
axes[0].set_ylabel("Precision (True Among Flagged)")
axes[0].set_title("Precision-Recall Curve", fontsize=13)
axes[0].legend(fontsize=10)
axes[0].grid(True, alpha=0.3)

axes[1].plot(thresholds, f1_scores, color="#009688", linewidth=2, label="F1 Score")
axes[1].plot(
    thresholds, precisions, color="#2196F3", linewidth=1.5, alpha=0.7, label="Precision"
)
axes[1].plot(
    thresholds, recalls, color="#F44336", linewidth=1.5, alpha=0.7, label="Recall"
)
axes[1].axvline(
    best_threshold,
    color="#FF9800",
    linestyle="--",
    label=f"Optimal threshold = {best_threshold:.5f}",
)
axes[1].set_xlabel("Reconstruction Error Threshold")
axes[1].set_ylabel("Score")
axes[1].set_title("Threshold Selection", fontsize=13)
axes[1].legend(fontsize=9)
axes[1].grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "ex1_fraud_precision_recall.png", dpi=150, bbox_inches="tight")
plt.show()

# --- Visualisation 3: Top anomalies ---
top_k = 20
top_indices = np.argsort(errors)[-top_k:][::-1]
fig, ax = plt.subplots(figsize=(12, 6))
colors = ["#F44336" if test_labels[i] == 1 else "#2196F3" for i in top_indices]
ax.barh(range(top_k), errors[top_indices], color=colors)
ax.set_yticks(range(top_k))
ax.set_yticklabels(
    [f"Txn #{i} ({'FRAUD' if test_labels[i]==1 else 'Normal'})" for i in top_indices],
    fontsize=9,
)
ax.set_xlabel("Reconstruction Error (Anomaly Score)")
ax.set_title(
    f"Top {top_k} Most Anomalous Transactions\nRed = True Fraud, Blue = Normal",
    fontsize=13,
)
ax.axvline(
    best_threshold,
    color="#FF9800",
    linestyle="--",
    linewidth=2,
    label=f"Detection threshold = {best_threshold:.5f}",
)
ax.legend(fontsize=10)
ax.invert_yaxis()
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "ex1_fraud_top_anomalies.png", dpi=150, bbox_inches="tight")
plt.show()

# --- Business Impact Analysis ---
DBS_DAILY_TRANSACTIONS = 2_000_000
AVG_FRAUD_VALUE_SGD = 800
RULE_BASED_RECALL = 0.67
DAILY_FRAUD_COUNT = int(DBS_DAILY_TRANSACTIONS * FRAUD_RATE)
FPR_AT_BEST = np.sum((errors > best_threshold) & (test_labels == 0)) / np.sum(
    test_labels == 0
)

daily_fraud_caught_ae = int(DAILY_FRAUD_COUNT * best_recall)
daily_fraud_caught_rules = int(DAILY_FRAUD_COUNT * RULE_BASED_RECALL)
daily_additional_caught = daily_fraud_caught_ae - daily_fraud_caught_rules
daily_false_alerts = int(DBS_DAILY_TRANSACTIONS * (1 - FRAUD_RATE) * FPR_AT_BEST)
daily_value_saved = daily_additional_caught * AVG_FRAUD_VALUE_SGD
annual_value_saved = daily_value_saved * 365

print("\n" + "=" * 64)
print("BUSINESS IMPACT SUMMARY — DBS Singapore Card Fraud Detection")
print("=" * 64)
print(f"\nDBS daily card transactions:     {DBS_DAILY_TRANSACTIONS:>12,}")
print(f"Estimated daily fraud events:    {DAILY_FRAUD_COUNT:>12,}")
print(f"Average fraud value:             {'S$' + str(AVG_FRAUD_VALUE_SGD):>12}")
print(f"\nCurrent rule-based system:")
print(f"  Fraud recall:                  {RULE_BASED_RECALL:>11.0%}")
print(f"  Fraud caught/day:              {daily_fraud_caught_rules:>12,}")
print(f"\nAutoencoder-based system (optimal threshold = {best_threshold:.5f}):")
print(f"  Fraud recall:                  {best_recall:>11.1%}")
print(f"  Precision:                     {best_precision:>11.1%}")
print(f"  Fraud caught/day:              {daily_fraud_caught_ae:>12,}")
print(f"  False alerts/day:              {daily_false_alerts:>12,}")
print(f"\nIncremental impact:")
print(f"  Additional fraud caught/day:   {daily_additional_caught:>12,}")
print(f"  Value saved per day:           {'S$' + f'{daily_value_saved:,.0f}':>12}")
print(f"  Value saved per year:          {'S$' + f'{annual_value_saved:,.0f}':>12}")
print("=" * 64)


# ════════════════════════════════════════════════════════════════════════
# REFLECTION
# ════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  WHAT YOU'VE MASTERED")
print("=" * 70)
print(
    """
  [x] Built an undercomplete AE with 49:1 compression (784 -> 16)
  [x] Observed blurry but meaningful reconstructions — structure preserved
  [x] Applied bottleneck AE to credit card fraud detection at DBS
  [x] Computed precision-recall curves for threshold selection
  [x] Quantified business impact: S$ value of additional fraud prevented

  KEY INSIGHT: The bottleneck forces the encoder to learn what MATTERS.
  A shirt's overall shape is preserved; its button stitching is lost.
  In fraud detection, normal transaction PATTERNS are preserved;
  fraudulent patterns (unusual amount + time + merchant) cannot be
  reconstructed, producing the anomaly signal.

  Next: 03_denoising_ae.py adds noise robustness...
"""
)
