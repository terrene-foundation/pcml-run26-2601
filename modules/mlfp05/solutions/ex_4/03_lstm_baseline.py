# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# MLFP05 — Exercise 4.3: LSTM Baseline for Comparison
# ════════════════════════════════════════════════════════════════════════
#
# WHAT YOU'LL LEARN:
#   After completing this exercise, you will be able to:
#   - Explain why baselines are essential for evaluating new architectures
#   - Build a bidirectional LSTM text classifier for fair comparison
#   - Contrast sequential (LSTM) vs parallel (Transformer) processing
#   - Train the LSTM on the same data and compare training dynamics
#   - Apply LSTM classification to a Singapore airline customer feedback use case
#
# PREREQUISITES: ex_4/01_self_attention_from_scratch.py
# ESTIMATED TIME: ~20 min
# DATASET: AG News — 120,000 real news headlines, 4 classes.
#
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from shared.mlfp05.ex_4 import (
    CLASS_NAMES,
    DEVICE,
    EPOCHS_SCRATCH,
    MAX_LEN,
    build_vocab,
    load_ag_news,
    prepare_dataloaders,
    setup_engines,
    text_to_indices,
    train_model,
)

print(f"Using device: {DEVICE}")


# ════════════════════════════════════════════════════════════════════════
# THEORY — Why We Need a Baseline
# ════════════════════════════════════════════════════════════════════════
# When evaluating a new architecture (like the Transformer), you need a
# fair comparison point. Without a baseline, you cannot tell whether the
# Transformer's accuracy comes from:
#   (a) The attention mechanism itself, or
#   (b) Having more parameters, better hyperparameters, or more training.
#
# The LSTM is the strongest pre-Transformer baseline for text. It processes
# tokens sequentially, maintaining a hidden state that accumulates context.
# The key differences:
#
#   LSTM (sequential):
#     - Processes tokens one at a time: O(n) sequential steps
#     - Information flows through a hidden state "bottleneck"
#     - Bidirectional LSTM reads forward AND backward, doubling the context
#     - Cannot be parallelised across positions during training
#
#   Transformer (parallel):
#     - Processes all tokens simultaneously: O(1) depth (but O(n^2) attention)
#     - Every token directly attends to every other token
#     - Positional encoding provides word order information
#     - Fully parallelisable during training (much faster on GPU)
#
# By training both on the same data with similar parameter counts, we
# isolate the architectural difference and measure what attention buys.
# ════════════════════════════════════════════════════════════════════════


# ════════════════════════════════════════════════════════════════════════
# TASK 1 — Load data and set up engines
# ════════════════════════════════════════════════════════════════════════
train_df, test_df = load_ag_news()
vocab = build_vocab(train_df["text"].to_list())
train_loader, val_loader, train_t, train_y, test_t, test_y = prepare_dataloaders(
    train_df, test_df, vocab
)
conn, tracker, exp_name, registry, has_registry, bridge = setup_engines()
print(f"  vocab size: {len(vocab)}, seq_len: {MAX_LEN}")


# ════════════════════════════════════════════════════════════════════════
# TASK 2 — Build: Bidirectional LSTM Text Classifier
# ════════════════════════════════════════════════════════════════════════
class LSTMClassifier(nn.Module):
    """Bidirectional LSTM for text classification.

    Architecture: embedding -> bidirectional LSTM -> mean pool -> classifier.

    Bidirectional processing reads the sequence both forward and backward,
    giving each token context from both directions. This partially addresses
    the LSTM's sequential limitation, but information must still propagate
    through the hidden state chain -- unlike attention, which is direct.
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 128,
        hidden_dim: int = 128,
        n_layers: int = 2,
        n_classes: int = 4,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            embed_dim,
            hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0,
            bidirectional=True,
        )
        self.head_drop = nn.Dropout(dropout)
        # Bidirectional doubles the hidden dimension
        self.head = nn.Linear(hidden_dim * 2, n_classes)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        x = self.embed(tokens)
        lstm_out, _ = self.lstm(x)  # (B, L, 2*H)
        # Mean pool over non-pad positions
        pad_mask = tokens == 0
        lengths = (~pad_mask).sum(dim=1, keepdim=True).clamp(min=1).float()
        lstm_out = lstm_out.masked_fill(pad_mask.unsqueeze(-1), 0.0)
        pooled = lstm_out.sum(dim=1) / lengths
        return self.head(self.head_drop(pooled))


# ── Checkpoint 1 ─────────────────────────────────────────────────────
lstm_test = LSTMClassifier(vocab_size=100).to(DEVICE)
dummy_tokens = torch.randint(0, 100, (2, MAX_LEN), device=DEVICE)
lstm_out = lstm_test(dummy_tokens)
assert lstm_out.shape == (2, 4), "LSTM should output (batch, 4 classes)"
param_count = sum(p.numel() for p in lstm_test.parameters())
print(f"\n  LSTMClassifier output shape: {tuple(lstm_out.shape)} (expected (2, 4))")
print(f"  Parameter count: {param_count:,}")
print("\n--- Checkpoint 1 passed --- LSTM architecture ready\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 3 — Train: LSTM on full AG News with ExperimentTracker
# ════════════════════════════════════════════════════════════════════════
print("\n== Training LSTM baseline on full AG News ==")
lstm_model = LSTMClassifier(
    vocab_size=len(vocab), embed_dim=128, hidden_dim=128, n_layers=2, n_classes=4
)
lstm_losses, lstm_accs = train_model(
    lstm_model,
    "lstm_baseline",
    train_loader,
    val_loader,
    tracker,
    exp_name,
    epochs=EPOCHS_SCRATCH,
)

# ══════════════════════════════════════════════════════════════════
# DIAGNOSTIC CHECKPOINT — LSTM baseline (contrast with Transformer 02)
# ══════════════════════════════════════════════════════════════════
from kailash_ml import diagnose

print("\n── Diagnostic Report (LSTM baseline) ──")
report = diagnose(lstm_model, kind="dl", data=val_loader, show=False)
# ══════ EXPECTED OUTPUT (reference pattern — LSTM on AG News) ═══════
# ════════════════════════════════════════════════════════════════
#   DL Diagnostics Report — Prescription Pad
# ════════════════════════════════════════════════════════════════
#   [!] Gradient flow (WARNING): RMS ratio
#       `lstm.weight_ih_l0` : `head.weight` ≈ 1:50 —
#       gradients concentrated at the classifier head, modest
#       signal reaching the embedding layer. Classic mild form
#       of the vanishing-gradient-through-time problem.
#   [✓] Activations    (HEALTHY): tanh/sigmoid gates within
#       [-1, 1] and [0, 1] — no saturation.
#   [✓] Loss trend     (HEALTHY): train loss decreases steadily;
#       val acc plateaus ~0.86 by epoch 6.
# ════════════════════════════════════════════════════════════════
# Best val acc: ~0.86 after 8 epochs — within 2% of the Transformer
# on AG News (headlines are short, so the LSTM's sequential
# bottleneck is manageable).
#
# STUDENT INTERPRETATION GUIDE — reading the Prescription Pad:
#
#  [BLOOD TEST] Gradients concentrate in the CLASSIFIER HEAD.
#     The LSTM's recurrent weights receive ~50x less signal than
#     `head.weight`. On 25-token headlines this is tolerable; on
#     500-token documents the same architecture would collapse
#     (the first 400 tokens' embeddings never update). This IS
#     the "vanishing gradient through time" that slide 5.3 warns
#     about — the Transformer sidesteps it entirely via direct
#     attention + residual connections.
#     >> Prescription Pad: if deploying on long documents, switch
#        to Transformer (ex_4/02) or add gradient clipping +
#        truncated BPTT. Bidirectional helps but doesn't eliminate
#        the issue.
#
#  [X-RAY] LSTM gate activations healthy — sigmoid outputs in
#     [0, 1] without saturating at either end. Saturation would
#     mean the forget gate is stuck "always remember" or "always
#     forget" which kills the LSTM's ability to learn what to
#     keep. If the X-Ray flags this, lower the learning rate and
#     add LayerNorm inside the LSTM cell.
#     >> Prescription Pad: healthy — no action.
#
#  [STETHOSCOPE] Loss curve shows the classic LSTM training
#     shape: rapid descent for 3 epochs, then a plateau as the
#     model hits the sequential-bottleneck ceiling. The
#     Transformer (02) shows a similar trajectory but reaches
#     a lower plateau — its plateau is the "architectural
#     ceiling" of the data, not the model.
#     >> Prescription Pad: compare final val acc side-by-side
#        with the Transformer to quantify the headroom.
#
#  FIVE-INSTRUMENT TAKEAWAY: on SHORT text the LSTM is 95%
#  as good as the Transformer. The Prescription Pad shows WHY
#  you would still prefer the Transformer — the gradient-flow
#  WARNING will become CRITICAL on real production documents
#  (news articles, support tickets, legal briefs). Never choose
#  an architecture on toy-length inputs; profile on your real
#  sequence length distribution.
#
#  CONNECT TO SLIDE 5.3 (RNNs) + 5.4 (Transformers): slide 5.3's
#  unrolled-RNN diagram shows information squeezing through the
#  hidden state; the WARNING reading above is the quantitative
#  version of that diagram. Slide 5.4 claims attention "lets
#  every token reach every other token in one step" — compare
#  this WARNING with the HEALTHY Blood Test in ex_4/02 to see
#  the architectural payoff in numbers.
# ══════════════════════════════════════════════════════════════════

# ── Checkpoint 2 ─────────────────────────────────────────────────────
assert len(lstm_losses) == EPOCHS_SCRATCH, "LSTM should train for all epochs"
assert (
    max(lstm_accs) > 0.60
), f"LSTM should reach >60% accuracy, got {max(lstm_accs):.3f}"
# INTERPRETATION: The LSTM provides a strong baseline. On short headlines
# (avg ~10 words), the LSTM's sequential bottleneck isn't as severe as it
# would be on longer documents. The real gap between LSTM and Transformer
# widens as sequence length increases -- on 512-token documents, the
# Transformer's direct attention outperforms LSTM by a wider margin.
print(f"\n  LSTM best accuracy: {max(lstm_accs):.3f}")
print(f"  LSTM final loss: {lstm_losses[-1]:.4f}")
print("\n--- Checkpoint 2 passed --- LSTM baseline trained\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 4 — Visualise: LSTM training dynamics
# ════════════════════════════════════════════════════════════════════════
# The LSTM's training curve reveals its learning dynamics. Unlike the
# Transformer, which can leverage parallel attention from epoch 1, the
# LSTM must learn to propagate information through the hidden state chain.
from shared.mlfp05.ex_4 import get_viz

viz = get_viz()
fig_lstm = viz.training_history(
    metrics={
        "LSTM train_loss": lstm_losses,
        "LSTM val_accuracy": lstm_accs,
    },
    x_label="Epoch",
    y_label="Value",
)
fig_lstm.write_html("ex_4_3_lstm_training_curves.html")
print("  LSTM training curves saved to ex_4_3_lstm_training_curves.html")

# ── Checkpoint 3 ─────────────────────────────────────────────────────
assert len(lstm_accs) == EPOCHS_SCRATCH, "Should have accuracy for each epoch"
# INTERPRETATION: The LSTM learning curve typically shows:
#   - Rapid initial learning (epochs 1-3): learns common word-class associations
#   - Slower improvement (epochs 4-6): refining contextual understanding
#   - Plateau (epochs 7-8): limited by the sequential bottleneck
# The Transformer, by contrast, often shows faster initial convergence
# because attention provides immediate access to all positions.
print("\n--- Checkpoint 3 passed --- LSTM training dynamics visualised\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 5 — Apply: Customer Feedback Classification for Singapore Airlines
# ════════════════════════════════════════════════════════════════════════
# SCENARIO: Singapore Airlines (SQ) collects thousands of customer reviews
# monthly across Skytrax, Google Reviews, social media, and their own
# feedback portal. The customer experience team needs to classify each
# review by topic -- service, food, seat comfort, delays -- to route it
# to the right operational team for action.
#
# BUSINESS VALUE: Manual review classification takes 2-3 minutes per review.
# With ~5,000 reviews/month, that is 167-250 hours of analyst time. An
# automated LSTM classifier handles the first-pass routing, letting analysts
# focus on extracting actionable insights rather than sorting.
#
# WHY LSTM HERE: The LSTM baseline establishes the accuracy floor. If the
# LSTM achieves 85% routing accuracy, the Transformer needs to beat that
# to justify its higher computational cost. If the LSTM achieves 95%, the
# Transformer's marginal improvement may not justify the infrastructure
# investment. This is the fundamental question baselines answer.
#
# SPEED COMPARISON: On a single GPU, the LSTM processes ~2,000 reviews/second
# (sequential processing). The Transformer processes ~5,000 reviews/second
# (parallel attention). For 5,000 reviews/month, both are fast enough --
# the speed difference matters at scale (millions of reviews).
print("\n== Application: Customer Feedback at Singapore Airlines ==")

# Classify sample "customer reviews" (using AG News as proxy data)
sq_reviews = [
    "World class service from cabin crew on long haul flight",
    "New business class seat design wins innovation award",
    "Flight delayed three hours due to technical issues at Changi",
    "Award winning food menu designed by celebrity chef",
    "Technology upgrade to in-flight entertainment system completed",
]
review_topics = ["Service", "Seat", "Delay", "Food", "Tech"]

lstm_model.eval()
with torch.no_grad():
    sq_idx = torch.tensor(
        [text_to_indices(t, vocab, MAX_LEN) for t in sq_reviews],
        dtype=torch.long,
        device=DEVICE,
    )
    sq_logits = lstm_model(sq_idx)
    sq_probs = F.softmax(sq_logits, dim=-1)
    sq_preds = sq_logits.argmax(dim=-1).cpu().tolist()

print(f"\n  Singapore Airlines review classification (LSTM baseline):")
print(f"  {'Review':<55} {'Topic':<8} {'AG News Class':<12} {'Confidence':>10}")
print("  " + "-" * 87)
for text, topic, pred, probs in zip(
    sq_reviews, review_topics, sq_preds, sq_probs.cpu().tolist()
):
    cls_name = CLASS_NAMES[pred]
    confidence = max(probs)
    print(f"  {text[:53]:<55} {topic:<8} {cls_name:<12} {confidence:>10.1%}")

# Measure throughput
import time

lstm_model.eval()
batch_input = torch.randint(0, len(vocab), (128, MAX_LEN), device=DEVICE)
with torch.no_grad():
    t0 = time.perf_counter()
    for _ in range(10):
        _ = lstm_model(batch_input)
    t1 = time.perf_counter()
    throughput = (128 * 10) / (t1 - t0)
    print(f"\n  LSTM throughput: {throughput:,.0f} reviews/second")

# ── Checkpoint 4 ─────────────────────────────────────────────────────
assert len(sq_preds) == len(sq_reviews), "Should classify all reviews"
# INTERPRETATION: The LSTM provides a solid baseline for customer feedback
# classification. Even without domain-specific training, it captures
# topic-relevant patterns in text. The Transformer (next exercise) will
# typically match or exceed this accuracy while processing reviews faster.
#
# BUSINESS IMPACT for Singapore Airlines:
#   - 5,000 customer reviews/month
#   - 2-3 min manual classification per review -> seconds with LSTM
#   - Annual saving: 2,000-3,000 analyst hours
#   - Faster issue escalation: delay complaints reach operations within minutes
#   - The Transformer comparison (next) determines if the accuracy uplift
#     justifies the additional compute cost
print("\n--- Checkpoint 4 passed --- Singapore Airlines application complete\n")


# ══════════════════════════════════════════════════════════════════════
# REFLECTION
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  WHAT YOU'VE MASTERED — LSTM Baseline")
print("=" * 70)
print(
    f"""
  [x] Understood why baselines are essential for fair evaluation
  [x] Built a bidirectional LSTM text classifier
  [x] Contrasted sequential (LSTM) vs parallel (Transformer) processing
  [x] Trained on full AG News (120K headlines), best acc: {max(lstm_accs):.1%}
  [x] Measured inference throughput for production sizing
  [x] Applied to Singapore Airlines customer feedback classification

  KEY INSIGHT:
    The LSTM is a strong baseline, not a strawman. On short sequences
    (headlines, tweets), it's competitive with transformers. The gap
    widens on longer documents where the sequential bottleneck hurts.
    Always establish your baseline BEFORE claiming your new architecture
    is "better" -- the margin matters more than the absolute number.

  Next: In 04_bert_finetuning.py, you'll see what happens when you
  combine the Transformer architecture with massive pre-training.
  BERT doesn't just learn from your 120K headlines -- it brings
  knowledge from billions of words of pre-training.
"""
)
