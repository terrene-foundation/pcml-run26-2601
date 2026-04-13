# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# MLFP05 — Exercise 3.4: Temporal Attention over LSTM Hidden States
# ════════════════════════════════════════════════════════════════════════
#
# WHAT YOU'LL LEARN:
#   After completing this section, you will be able to:
#   - Explain WHY fixed-length hidden states are a bottleneck
#   - Implement additive (Bahdanau) attention over LSTM hidden states
#   - Visualise attention weight heatmaps to see WHICH past timesteps matter
#   - Compare LSTM+Attention vs plain LSTM on multi-step forecasting
#   - Understand attention as the bridge to Transformers (M5.4)
#   - Track training with ExperimentTracker
#
# PREREQUISITES: 02_lstm.py (understand LSTM hidden states).
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
# THEORY — Why Fixed-Length Hidden States Are a Bottleneck
# ════════════════════════════════════════════════════════════════════════
#
# In a standard LSTM, prediction uses ONLY the final hidden state h_T:
#   y_hat = W @ h_T + b
#
# This means ALL information from 20 timesteps must be compressed into
# a single vector of size hidden_dim (64 in our case). This is like
# writing a 20-page report and being told to summarise it in one tweet.
#
# THE PROBLEM:
#   For stock prediction, day 3 might be highly relevant (earnings
#   announcement) while days 7-12 are noise (sideways trading). But
#   h_T treats all days equally — it is a RUNNING AVERAGE of information,
#   not a SELECTIVE summary.
#
# ATTENTION solves this by learning a WEIGHTED combination of ALL hidden
# states {h_1, h_2, ..., h_T}, where the weights reflect relevance:
#
#   energy_t = tanh(W @ h_t)       "How relevant is timestep t?"
#   a_t = softmax(v @ energy_t)     "Normalise to probability distribution"
#   context = sum(a_t * h_t)        "Weighted summary of all timesteps"
#
# INTUITION for non-technical professionals:
#   Imagine reading a financial report before making an investment decision.
#   Without attention: you read all 20 pages and try to remember everything.
#   With attention: you HIGHLIGHT the key paragraphs (earnings, guidance,
#   risk factors) and base your decision primarily on those highlights.
#   The attention weights ARE the highlighter marks.
#
# WHY THIS MATTERS:
#   Attention is THE foundational idea behind Transformers (GPT, BERT,
#   Claude) — but Transformers use SELF-attention (each timestep attends
#   to all others) instead of this simpler form. This exercise gives you
#   the intuition that makes Transformers click in Exercise 4.
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
    PRIMARY, experiment_suffix="temporal_attention"
)

# ── Checkpoint 1 ─────────────────────────────────────────────────────
assert X_train_t.shape[1] == SEQ_LEN
assert tracker is not None
print("--- Checkpoint 1 passed --- data and tracking ready\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 2 — Build the Temporal Attention mechanism
# ════════════════════════════════════════════════════════════════════════
class TemporalAttention(nn.Module):
    """Additive (Bahdanau) attention over LSTM hidden states.

    Given LSTM outputs H = {h_1, ..., h_T} of shape (batch, seq, hidden):
      1. Project each h_t through a learned matrix W: energy = tanh(H @ W)
      2. Score each projected state with a learned vector v: scores = energy @ v
      3. Normalise to attention weights: weights = softmax(scores)
      4. Compute weighted context: context = sum(weights * H)

    The weights tell us WHICH timesteps the model considers most relevant.
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        # TODO: Define W — nn.Linear(hidden_dim, hidden_dim)
        # TODO: Define v — nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, lstm_outputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            lstm_outputs: (batch, seq, hidden)

        Returns:
            context: (batch, hidden) — weighted summary
            weights: (batch, seq) — attention distribution over timesteps
        """
        # TODO: Compute energy = tanh(self.W(lstm_outputs))  — shape: (batch, seq, hidden)
        # TODO: Compute scores = self.v(energy).squeeze(-1)  — shape: (batch, seq)
        # TODO: Compute weights = softmax(scores, dim=-1)    — shape: (batch, seq)
        # TODO: Compute context using batch matrix multiply:
        #   context = torch.bmm(weights.unsqueeze(1), lstm_outputs).squeeze(1)
        #   This computes the weighted sum across all timesteps
        # TODO: Return context, weights
        pass


class LSTMWithAttention(nn.Module):
    """LSTM + Temporal Attention for sequence prediction.

    Instead of using only h_T, this model uses a learned weighted
    combination of ALL hidden states {h_1, ..., h_T}.
    """

    def __init__(
        self, input_dim: int, hidden_dim: int, horizon: int = FORECAST_HORIZON
    ):
        super().__init__()
        # TODO: Define LSTM layer — nn.LSTM(input_dim, hidden_dim, batch_first=True)
        # TODO: Define attention module — TemporalAttention(hidden_dim)
        # TODO: Define prediction head — nn.Linear(hidden_dim, horizon)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # TODO: Pass x through LSTM to get all hidden states: lstm_out, _ = self.lstm(x)
        # TODO: Apply attention: context, attn_weights = self.attention(lstm_out)
        # TODO: Predict: pred = self.head(context)
        # TODO: Return pred, attn_weights (both are needed — weights for visualisation)
        pass


# Plain LSTM for comparison
class LSTMRegressor(nn.Module):
    def __init__(
        self, input_dim: int, hidden_dim: int, horizon: int = FORECAST_HORIZON
    ):
        super().__init__()
        # TODO: Define LSTM layer — nn.LSTM(input_dim, hidden_dim, batch_first=True)
        # TODO: Define prediction head — nn.Linear(hidden_dim, horizon)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO: Pass through LSTM, return head(out[:, -1])
        pass


attn_model = LSTMWithAttention(input_dim=N_FEATURES, hidden_dim=HIDDEN_DIM)
lstm_model = LSTMRegressor(input_dim=N_FEATURES, hidden_dim=HIDDEN_DIM)

n_params_attn = sum(p.numel() for p in attn_model.parameters())
n_params_lstm = sum(p.numel() for p in lstm_model.parameters())
attn_overhead = n_params_attn - n_params_lstm

print(f"LSTM+Attention: {n_params_attn:,} parameters")
print(f"Plain LSTM:     {n_params_lstm:,} parameters")
print(
    f"Attention adds: {attn_overhead:,} parameters ({attn_overhead/n_params_lstm*100:.1f}% overhead)"
)

# ── Checkpoint 2 ─────────────────────────────────────────────────────
dummy_input = torch.randn(2, SEQ_LEN, N_FEATURES, device=device)
attn_model.to(device)
dummy_pred, dummy_weights = attn_model(dummy_input)
assert dummy_pred.shape == (2, FORECAST_HORIZON), f"Prediction shape mismatch"
assert dummy_weights.shape == (2, SEQ_LEN), f"Attention weights shape mismatch"
assert torch.allclose(
    dummy_weights.sum(dim=-1), torch.ones(2, device=device), atol=1e-5
), "Attention weights should sum to 1"
print("--- Checkpoint 2 passed --- LSTM+Attention architecture verified\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 3 — Train LSTM+Attention and plain LSTM
# ════════════════════════════════════════════════════════════════════════
print(f"\n== Training LSTM+Attention on {PRIMARY} ==")
attn_results = train_model(
    attn_model,
    "LSTM_Attention",
    tracker,
    exp_name,
    train_loader,
    val_loader,
    device,
    attn=True,
)

print(f"\n== Training plain LSTM (comparison) on {PRIMARY} ==")
lstm_results = train_model(
    lstm_model,
    "LSTM_plain",
    tracker,
    exp_name,
    train_loader,
    val_loader,
    device,
)

improvement = (
    (lstm_results["final_val_loss"] - attn_results["final_val_loss"])
    / lstm_results["final_val_loss"]
    * 100
)
print(f"\n  Comparison:")
print(f"    LSTM+Attention val loss: {attn_results['final_val_loss']:.4f}")
print(f"    Plain LSTM val loss:     {lstm_results['final_val_loss']:.4f}")
print(f"    Improvement: {improvement:+.1f}%")

# ── Checkpoint 3 ─────────────────────────────────────────────────────
assert len(attn_results["train_losses"]) == EPOCHS
assert attn_results["final_val_loss"] < 5.0
print("--- Checkpoint 3 passed --- both models trained\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 4 — Visualise: attention weight heatmaps
# ════════════════════════════════════════════════════════════════════════
# THIS is the key visual insight: which past timesteps does the model
# consider most relevant for its prediction?

attn_model.eval()
with torch.no_grad():
    val_preds, val_attn_weights = attn_model(X_val_t)

attn_np = val_attn_weights.cpu().numpy()  # (n_val, seq_len)

# TODO: Select 5 diverse sample indices: beginning, quarter, middle, three-quarter, end
sample_indices = [
    0,
    len(attn_np) // 4,
    len(attn_np) // 2,
    3 * len(attn_np) // 4,
    len(attn_np) - 1,
]

# TODO: Create 3x2 subplot figure (16, 14)
#   First 5 subplots: individual sample attention profiles
#     - Bar chart of attention weights for each sample
#     - Color bars using plt.cm.YlOrRd normalised by max weight
#     - Annotate top-3 timesteps with their weight values
#   6th subplot: average attention across ALL validation samples
#     - Compute avg_attn = attn_np.mean(axis=0)
#     - Calculate recency bias: ratio of avg weight for last 5 days vs first 5 days
#     - Add text annotation with recency ratio
#   Save to OUTPUT_DIR / "04_attention_weights_heatmap.png"
fig, axes = plt.subplots(3, 2, figsize=(16, 14))
avg_attn = attn_np.mean(axis=0)
recent_avg = avg_attn[-5:].mean()
early_avg = avg_attn[:5].mean()
recency_ratio = recent_avg / max(early_avg, 1e-8)
# TODO: Fill in all 6 subplots
fig.tight_layout()
fig.savefig(str(OUTPUT_DIR / "04_attention_weights_heatmap.png"), dpi=150)
plt.close(fig)
print("  Saved: 04_attention_weights_heatmap.png")

# TODO: Create attention heatmap matrix (14, 8)
#   - Show attn_np[:100] as an image with imshow, cmap="YlOrRd"
#   - x-axis: timestep, y-axis: validation sample index
#   - Save to OUTPUT_DIR / "04_attention_heatmap_matrix.png"
fig2, ax2 = plt.subplots(figsize=(14, 8))
# TODO: Plot heatmap matrix
fig2.tight_layout()
fig2.savefig(str(OUTPUT_DIR / "04_attention_heatmap_matrix.png"), dpi=150)
plt.close(fig2)
print("  Saved: 04_attention_heatmap_matrix.png")

# Print summary
top3_overall = np.argsort(avg_attn)[-3:][::-1]
print(f"\n  Top-3 most attended timesteps (across all validation):")
for rank, t in enumerate(top3_overall):
    print(
        f"    #{rank+1}: Day t-{SEQ_LEN-t} (step {t}), avg weight = {avg_attn[t]:.4f}"
    )
print(f"  Recency bias: {recency_ratio:.1f}x (recent 5 days vs early 5 days)")

# ── Checkpoint 4 ─────────────────────────────────────────────────────
assert (OUTPUT_DIR / "04_attention_weights_heatmap.png").exists()
assert (OUTPUT_DIR / "04_attention_heatmap_matrix.png").exists()
assert abs(attn_np.sum(axis=-1) - 1.0).max() < 1e-4, "Attention weights must sum to 1"
print("--- Checkpoint 4 passed --- attention heatmaps generated\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 5 — Visualise: predicted vs actual + training curves
# ════════════════════════════════════════════════════════════════════════
viz = get_visualizer()
plot_training_curves(viz, attn_results, "LSTM+Attention", "04_attention")

preds_denorm, actual_denorm, _ = plot_predictions(
    viz,
    attn_model,
    X_val_t,
    y_val_t,
    norm_mean,
    norm_std,
    "04_attention",
    attn=True,
)

plot_time_series_overlay(
    preds_denorm,
    actual_denorm,
    "04_attention",
    title=f"LSTM+Attention: Predicted vs Actual Close ({PRIMARY})",
)

rmses = plot_horizon_error(preds_denorm, actual_denorm, "LSTM+Attention")

# ── Checkpoint 5 ─────────────────────────────────────────────────────
assert (OUTPUT_DIR / "04_attention_training_curves.html").exists()
assert (OUTPUT_DIR / "04_attention_time_series_overlay.png").exists()
print("--- Checkpoint 5 passed --- LSTM+Attention visualisations generated\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 6 — Register model
# ════════════════════════════════════════════════════════════════════════
register_best_model(
    attn_model,
    "LSTM_Attention",
    attn_results["final_val_loss"],
    PRIMARY,
    registry,
    has_registry,
)


# ════════════════════════════════════════════════════════════════════════
# APPLY — Clinical Event Prediction at Singapore General Hospital (SGH)
# ════════════════════════════════════════════════════════════════════════
#
# BUSINESS SCENARIO:
#   You are a clinical data scientist at Singapore General Hospital
#   (SGH), the largest acute tertiary hospital in Singapore. ICU nurses
#   monitor vital signs (heart rate, blood pressure, SpO2, temperature,
#   respiratory rate) every 5 minutes. You need to predict patient
#   deterioration 30 minutes ahead.
#
# WHY ATTENTION (NOT PLAIN LSTM)?
#   Patient deterioration often has a SPECIFIC trigger moment — a
#   sudden BP drop, a temperature spike, an SpO2 dip. Plain LSTM
#   compresses 6 hours of vitals into one vector and loses the trigger.
#   Attention lets the model "look back" and focus on THE critical
#   reading that predicts deterioration.
#
# THE ATTENTION HEATMAP IS THE EXPLANATION:
#   When the model predicts "this patient will deteriorate," the
#   attention weights show WHICH past vital readings drove the prediction.
#   This is clinically actionable: "The model flagged this patient
#   because of the SpO2 dip at 2:35am and the rising trend since 3:00am."
#   Doctors need this explainability to trust the system.
#
# DELIVERABLES:
#   - Deterioration probability prediction with attention heatmap
#   - Per-patient attention analysis: which vital signs matter when
#   - Clinical alert with interpretable explanation
print("\n" + "=" * 70)
print("  APPLY: SGH Clinical Event Prediction — ICU Vital Signs")
print("=" * 70)

# TODO: Generate realistic ICU vital signs data
#   - n_patients = 200, readings_per_patient = 72 (6 hours at 5-min intervals)
#   - 5 vitals: HR (60-100), SBP (110-140), SpO2 (95-100), Temp (36.5-37.5), RR (12-20)
#   - 30% of patients deteriorate: introduce trigger event at random time
#     - SpO2 drops, HR rises, BP drops from trigger point onward
np.random.seed(42)
n_patients = 200
readings_per_patient = 72
n_vitals = 5

# TODO: Build ClinicalAttentionModel using TemporalAttention + LSTM + classifier
#   class ClinicalAttentionModel(nn.Module):
#     - LSTM layer: nn.LSTM(input_dim, hidden_dim=32, batch_first=True)
#     - Attention: TemporalAttention(hidden_dim=32)
#     - Classifier: nn.Sequential(Linear(32, 32), ReLU, Linear(32, 1), Sigmoid)
#     - forward returns (probability, attention_weights)

# TODO: Train for 30 epochs with binary cross-entropy loss
# TODO: Evaluate: compute precision, recall, F1 score

# TODO: For a deteriorating patient, extract attention weights and show
#   which readings drove the prediction (top-5 by attention weight)
#   Print the actual vital values at those timesteps

# TODO: Visualise (2-row figure, 16x10):
#   Top: Normalised vital signs with attention-weighted background shading
#   Bottom: Attention weight bars per timestep
#   Save to OUTPUT_DIR / "04_attention_clinical_patient.png"

# TODO: Calculate business impact:
#   ICU beds (~200), cost per day (S$5,000), early intervention saves 15% stay

# ── Checkpoint 6 (Apply) ────────────────────────────────────────────
# assert f1 > 0.3, f"F1 should be reasonable for synthetic clinical data"
# print("--- Checkpoint 6 passed --- SGH clinical application complete\n")


# ══════════════════════════════════════════════════════════════════════
# REFLECTION
# ══════════════════════════════════════════════════════════════════════
print("=" * 70)
print("  WHAT YOU'VE MASTERED")
print("=" * 70)
print(
    f"""
  [x] Built temporal attention mechanism (Bahdanau/additive attention)
  [x] LSTM+Attention vs plain LSTM: {improvement:+.1f}% val loss improvement
  [x] Attention heatmaps: visualised which timesteps the model focuses on
  [x] Recency bias: recent days get {recency_ratio:.1f}x more attention than early days
  [x] Applied to SGH clinical deterioration prediction
  [x] Attention provides EXPLAINABILITY: which past readings drove the alert

  Key insight: Attention is a LEARNED HIGHLIGHTING mechanism. Instead of
  compressing 20 timesteps into one vector, the model learns to weight
  each timestep by relevance. The weights are interpretable — they show
  you WHY the model made its prediction. This explainability is critical
  in healthcare, finance, and any domain where humans need to trust
  the model's decisions.

  Bridge to Transformers: This is "temporal attention" — the decoder
  attends to the encoder's hidden states. Transformers (M5.4) use
  "self-attention" — every position attends to every other position,
  WITHOUT the sequential bottleneck of LSTM. That is the key innovation.

  Next: 05_architecture_comparison.py — side-by-side comparison of all variants.
"""
)
