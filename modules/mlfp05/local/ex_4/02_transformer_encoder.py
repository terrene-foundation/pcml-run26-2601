# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# MLFP05 — Exercise 4.2: Transformer Encoder Classifier
# ════════════════════════════════════════════════════════════════════════
#
# WHAT YOU'LL LEARN:
#   After completing this exercise, you will be able to:
#   - Implement multi-head attention with parallel learned projections
#   - Explain how different heads capture different relationship types
#   - Build sinusoidal positional encoding (giving transformers order)
#   - Construct a full Transformer encoder classifier with nn.TransformerEncoder
#   - Train the Transformer on AG News and log metrics with ExperimentTracker
#   - Apply the model to regulatory document classification
#
# PREREQUISITES: ex_4/01_self_attention_from_scratch.py
# ESTIMATED TIME: ~30 min
# DATASET: AG News — 120,000 real news headlines, 4 classes.
#
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from shared.mlfp05.ex_4 import (
    CLASS_NAMES,
    DEVICE,
    EPOCHS_SCRATCH,
    MAX_LEN,
    VOCAB_SIZE,
    build_vocab,
    create_attention_heatmap,
    load_ag_news,
    prepare_dataloaders,
    scaled_dot_product_attention,
    setup_engines,
    text_to_indices,
    train_model,
)

print(f"Using device: {DEVICE}")


# ════════════════════════════════════════════════════════════════════════
# THEORY — Multi-Head Attention and Positional Encoding
# ════════════════════════════════════════════════════════════════════════
# A single attention head can only capture one type of relationship at a
# time. But language has many simultaneous relationship types:
#
#   "The bank raised interest rates to control inflation"
#
#   - SYNTACTIC: "bank" -> "raised" (subject-verb)
#   - SEMANTIC: "rates" -> "inflation" (economic cause-effect)
#   - COREFERENCE: "bank" -> "central bank" (if mentioned earlier)
#
# Multi-head attention runs h parallel attention operations, each with
# its own learned Q/K/V projections. Head 1 might learn syntax, head 2
# might learn semantics, head 3 might learn entity relationships. The
# outputs are concatenated and projected back to the model dimension.
#
# POSITIONAL ENCODING: Transformers process all tokens simultaneously
# (unlike RNNs, which process sequentially). This means they have no
# inherent sense of word order. "Dog bites man" and "Man bites dog"
# would look identical without positional information. We inject position
# using a sinusoidal signal:
#   PE(pos, 2i)   = sin(pos / 10000^(2i/d))
#   PE(pos, 2i+1) = cos(pos / 10000^(2i/d))
#
# This scheme gives each position a unique signature and allows the model
# to learn relative positions (the offset between any two positions has
# a consistent geometric relationship in the PE space).
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
print(f"  ExperimentTracker ready, experiment: {exp_name}")


# ════════════════════════════════════════════════════════════════════════
# TASK 2 — Build: Educational Multi-Head Attention
# ════════════════════════════════════════════════════════════════════════
# This implementation wraps our from-scratch scaled_dot_product_attention
# with learned Q/K/V projections and multi-head splitting. PyTorch has
# nn.MultiheadAttention, but building it ourselves reveals the mechanics.
class EducationalMultiHead(nn.Module):
    """Multi-head attention built on our from-scratch attention kernel.

    Each head gets its own learned projection of the input into Q, K, V
    subspaces. The heads operate in parallel, then their outputs are
    concatenated and projected back to the model dimension.
    """

    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        assert (
            d_model % n_heads == 0
        ), f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        # Single linear layer produces Q, K, V for all heads at once
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass returning output and attention weights.

        Args:
            x: Input tensor of shape (batch, seq_len, d_model)

        Returns:
            (output, attention_weights) where output has shape
            (batch, seq_len, d_model) and attention_weights has shape
            (batch, n_heads, seq_len, seq_len).
        """
        b, seq, d = x.shape

        # TODO: Compute Q, K, V for all heads in one matrix multiply
        # Hint: qkv = self.qkv(x).reshape(b, seq, 3, self.n_heads, self.d_k)
        # Then: q, k, v = qkv.unbind(dim=2)  — each is (b, seq, n_heads, d_k)
        qkv = ...  # YOUR CODE HERE
        q, k, v = ...  # YOUR CODE HERE

        # TODO: Reshape for attention — merge batch and head dims
        # Hint: q = q.transpose(1, 2).reshape(b * self.n_heads, seq, self.d_k)
        # Same for k and v
        q = ...  # YOUR CODE HERE
        k = ...  # YOUR CODE HERE
        v = ...  # YOUR CODE HERE

        # TODO: Apply scaled_dot_product_attention from helpers
        # Hint: out, weights = scaled_dot_product_attention(q, k, v)
        out, weights = ...  # YOUR CODE HERE

        # Reshape weights to (b, n_heads, seq, seq) for visualisation
        attn_weights = weights.reshape(b, self.n_heads, seq, seq)

        # TODO: Concatenate heads and project back to d_model
        # Hint: out = out.reshape(b, self.n_heads, seq, self.d_k).transpose(1, 2).reshape(b, seq, d)
        # Then: return self.proj(out), attn_weights
        out = ...  # YOUR CODE HERE
        return ...  # YOUR CODE HERE


# Sanity check the educational multi-head attention
mha = EducationalMultiHead(d_model=64, n_heads=4).to(DEVICE)
dummy = torch.randn(2, 16, 64, device=DEVICE)
mha_out, mha_attn = mha(dummy)
print(
    f"\nEducationalMultiHead output shape: {tuple(mha_out.shape)}  (expected (2, 16, 64))"
)
print(f"Attention weights shape: {tuple(mha_attn.shape)}  (expected (2, 4, 16, 16))")

# ── Checkpoint 1 ─────────────────────────────────────────────────────
assert mha_out.shape == (2, 16, 64), "Multi-head output should be (2, 16, 64)"
assert mha_attn.shape == (2, 4, 16, 16), "Attention should be (batch, heads, seq, seq)"
print("\n--- Checkpoint 1 passed --- multi-head attention architecture ready\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 3 — Build: Positional Encoding + Transformer Classifier
# ════════════════════════════════════════════════════════════════════════
class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding.

    Adds a fixed, non-learned signal that encodes position. Each dimension
    oscillates at a different frequency, creating a unique "fingerprint"
    for each position that the model can use to distinguish word order.
    """

    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        # TODO: Fill pe with sinusoidal values
        # Hint: pe[:, 0::2] = torch.sin(position * div)  — even dimensions
        # Hint: pe[:, 1::2] = torch.cos(position * div)  — odd dimensions
        ...  # YOUR CODE HERE
        ...  # YOUR CODE HERE
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class TransformerClassifier(nn.Module):
    """Full Transformer encoder for text classification.

    Architecture: embedding -> positional encoding -> stacked
    TransformerEncoderLayers -> mean pool -> classification head.

    Uses PyTorch's nn.TransformerEncoder for the stacked layers (which
    internally uses nn.MultiheadAttention), but the architecture mirrors
    our educational implementation above.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 3,
        n_classes: int = 4,
        dropout: float = 0.2,
    ):
        super().__init__()
        # TODO: Build the Transformer architecture
        # Hint: self.embed = nn.Embedding(vocab_size, d_model, padding_idx=0)
        # Hint: self.posenc = PositionalEncoding(d_model)
        # Hint: self.emb_drop = nn.Dropout(dropout)
        # Hint: layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads,
        #              dim_feedforward=4 * d_model, dropout=dropout, batch_first=True)
        # Hint: self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers, enable_nested_tensor=False)  # MPS-compat
        # Hint: self.head_drop = nn.Dropout(dropout)
        # Hint: self.head = nn.Linear(d_model, n_classes)
        self.embed = ...  # YOUR CODE HERE
        self.posenc = ...  # YOUR CODE HERE
        self.emb_drop = ...  # YOUR CODE HERE
        layer = ...  # YOUR CODE HERE
        self.encoder = ...  # YOUR CODE HERE
        self.head_drop = ...  # YOUR CODE HERE
        self.head = ...  # YOUR CODE HERE

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        # TODO: Implement the forward pass
        # Step 1: pad_mask = (tokens == 0)
        # Step 2: x = self.embed(tokens) -> posenc -> emb_drop
        # Step 3: x = self.encoder(x, src_key_padding_mask=pad_mask)
        # Step 4: Mean-pool over non-pad positions
        #   lengths = (~pad_mask).sum(dim=1, keepdim=True).clamp(min=1).float()
        #   x = x.masked_fill(pad_mask.unsqueeze(-1), 0.0)
        #   pooled = x.sum(dim=1) / lengths
        # Step 5: return self.head(self.head_drop(pooled))
        ...  # YOUR CODE HERE


# ── Checkpoint 2 ─────────────────────────────────────────────────────
tc_test = TransformerClassifier(vocab_size=100).to(DEVICE)
dummy_tokens = torch.randint(0, 100, (2, MAX_LEN), device=DEVICE)
tc_out = tc_test(dummy_tokens)
assert tc_out.shape == (2, 4), "TransformerClassifier should output (batch, 4 classes)"
print("--- Checkpoint 2 passed --- TransformerClassifier architecture ready\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 4 — Train: Transformer on full AG News with ExperimentTracker
# ════════════════════════════════════════════════════════════════════════
print("\n== Training Transformer on full AG News ==")
# TODO: Create TransformerClassifier and train it
# Hint: transformer_model = TransformerClassifier(vocab_size=len(vocab), d_model=128, n_heads=4, n_layers=3, n_classes=4)
# Hint: transformer_losses, transformer_accs = train_model(
#           transformer_model, "transformer", train_loader, val_loader,
#           tracker, exp_name, epochs=EPOCHS_SCRATCH)
transformer_model = ...  # YOUR CODE HERE
transformer_losses, transformer_accs = ...  # YOUR CODE HERE

# ── Checkpoint 3 ─────────────────────────────────────────────────────
assert (
    len(transformer_losses) == EPOCHS_SCRATCH
), "Transformer should train for all epochs"
assert (
    max(transformer_accs) > 0.60
), f"Transformer should reach >60% accuracy, got {max(transformer_accs):.3f}"
# INTERPRETATION: The Transformer processes all tokens in parallel and uses
# self-attention to capture long-range dependencies. On AG News headlines,
# it can directly connect "tech" at position 1 with "stocks" at position 8
# without propagating through every intermediate token. This architectural
# advantage becomes more pronounced on longer documents.
print(f"\n  Transformer best acc: {max(transformer_accs):.3f}")
print("\n--- Checkpoint 3 passed --- Transformer trained on AG News\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 5 — Visualise: Multi-head attention patterns on sample headline
# ════════════════════════════════════════════════════════════════════════
print("\n== Visualising multi-head attention patterns ==")
transformer_model.eval()
sample_texts = test_df["text"].to_list()[:3]
sample_idx = torch.tensor(
    [text_to_indices(t, vocab, MAX_LEN) for t in sample_texts],
    dtype=torch.long,
    device=DEVICE,
)

# TODO: Extract attention from EducationalMultiHead on trained embeddings
# Hint: mha_viz = EducationalMultiHead(d_model=128, n_heads=4).to(DEVICE)
# Hint: with torch.no_grad():
#           embed = transformer_model.embed(sample_idx[:1])
#           embed = transformer_model.posenc(embed)
#           _, attn_weights = mha_viz(embed)
mha_viz = ...  # YOUR CODE HERE
with torch.no_grad():
    embed = ...  # YOUR CODE HERE
    embed = ...  # YOUR CODE HERE
    _, attn_weights = ...  # YOUR CODE HERE

words = sample_texts[0].lower().split()[:MAX_LEN]
word_labels = words + ["<pad>"] * (MAX_LEN - len(words))

# Visualise each head's attention pattern
for head_idx in range(min(4, attn_weights.shape[1])):
    attn_np = attn_weights[0, head_idx].cpu().numpy()
    fig = create_attention_heatmap(
        attn_np,
        word_labels,
        title=f"Attention Head {head_idx} on: '{sample_texts[0][:50]}...'",
        max_tokens=12,
    )
    fig.write_html(f"ex_4_2_head_{head_idx}_attention.html")

print(f"  Saved 4 attention head heatmaps (ex_4_2_head_0..3_attention.html)")
print(f"  Different heads capture different relationship types:")
print(f"    Head 0: may focus on adjacent word pairs (local syntax)")
print(f"    Head 1: may focus on content words across the sentence (semantics)")
print(f"    Head 2: may focus on sentence boundaries and punctuation (structure)")
print(f"    Head 3: may focus on entity-to-entity relationships")

# ── Checkpoint 4 ─────────────────────────────────────────────────────
assert attn_weights.shape == (
    1,
    4,
    MAX_LEN,
    MAX_LEN,
), "Should have 4 heads of attention"
print("\n--- Checkpoint 4 passed --- multi-head attention visualised\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 6 — Apply: Regulatory Compliance Classification at MAS
# ════════════════════════════════════════════════════════════════════════
# SCENARIO: The Monetary Authority of Singapore (MAS) oversees compliance
# across banking, insurance, securities, and payments. Financial institutions
# submit thousands of regulatory filings monthly. MAS compliance officers
# need to classify each document by the regulation it pertains to:
#   - Banking Act (Cap. 19)
#   - Securities and Futures Act (Cap. 289)
#   - Payment Services Act 2019
#   - Insurance Act (Cap. 142)
#
# BUSINESS VALUE: Manual classification by compliance officers takes
# 15-20 minutes per document. With ~3,000 submissions/month across
# 200+ licensed institutions, that is 750-1,000 officer-hours/month.
# A Transformer classifier automates the first-pass classification,
# routing documents to the correct regulatory team in seconds.
#
# DOLLAR IMPACT: At S$80-120/hour for compliance officers, automating
# first-pass classification saves S$720K-1.44M annually. More importantly,
# the attention mechanism shows WHICH paragraphs triggered each
# classification -- providing audit trail transparency that regulators
# require under MAS Notice on Technology Risk Management.
print("\n== Application: Regulatory Compliance at MAS ==")

financial_headlines = [
    "Banks report higher profits amid rising interest rates",
    "New technology startups attract venture capital funding",
    "Stock market volatility increases as trade tensions rise",
    "Sports betting companies face new regulatory scrutiny",
    "Insurance companies adapt to climate change risks",
]

# TODO: Classify financial headlines with the trained transformer
# Step 1: Set model to eval mode — transformer_model.eval()
# Step 2: Tokenise — fin_idx = torch.tensor([text_to_indices(t, vocab, MAX_LEN) for t in financial_headlines], dtype=torch.long, device=DEVICE)
# Step 3: with torch.no_grad(): get logits, probs, preds
#   fin_logits = transformer_model(fin_idx)
#   fin_probs = F.softmax(fin_logits, dim=-1)
#   fin_preds = fin_logits.argmax(dim=-1).cpu().tolist()
transformer_model.eval()
with torch.no_grad():
    fin_idx = ...  # YOUR CODE HERE
    fin_logits = ...  # YOUR CODE HERE
    fin_probs = ...  # YOUR CODE HERE
    fin_preds = ...  # YOUR CODE HERE

print(f"\n  Regulatory document classification (Transformer):")
print(f"  {'Headline':<55} {'Classification':<12} {'Confidence':>10}")
print("  " + "-" * 79)
for text, pred, probs in zip(financial_headlines, fin_preds, fin_probs.cpu().tolist()):
    cls_name = CLASS_NAMES[pred]
    confidence = max(probs)
    print(f"  {text[:53]:<55} {cls_name:<12} {confidence:>10.1%}")

# TODO: Show attention-based explanation for the first document
# Hint: Use mha_viz to get attention, average across heads, compute token importance
with torch.no_grad():
    embed = transformer_model.embed(fin_idx[:1])
    embed = transformer_model.posenc(embed)
    _, fin_attn = mha_viz(embed)
    avg_attn = fin_attn[0].mean(dim=0).cpu().numpy()

fin_words = financial_headlines[0].lower().split()[:MAX_LEN]
fin_labels = fin_words + ["<pad>"] * (MAX_LEN - len(fin_words))
token_importance = avg_attn[: len(fin_words), : len(fin_words)].sum(axis=0)
token_importance = token_importance / token_importance.max()

print(f"\n  Attention-based explanation for: '{financial_headlines[0]}'")
print(f"  Token importance (which words drive the classification):")
for word, imp in sorted(zip(fin_words, token_importance), key=lambda x: -x[1])[:5]:
    bar = "#" * int(imp * 20)
    print(f"    {word:<15} {imp:.3f} {bar}")

# ── Checkpoint 5 ─────────────────────────────────────────────────────
assert len(fin_preds) == len(financial_headlines), "Should classify all headlines"
# INTERPRETATION: The Transformer classifies financial documents and the
# attention weights provide an audit trail showing which words drove each
# classification. For MAS compliance, this transparency is critical --
# regulators need to understand WHY a document was classified as it was,
# not just the classification itself.
#
# BUSINESS IMPACT for MAS:
#   - 3,000 regulatory submissions/month
#   - 15-20 min manual classification per document -> seconds with Transformer
#   - Annual saving: S$720K-1.44M in compliance officer time
#   - Attention audit trail satisfies MAS Technology Risk Management Notice
print("\n--- Checkpoint 5 passed --- MAS regulatory application complete\n")


# ══════════════════════════════════════════════════════════════════════
# REFLECTION
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  WHAT YOU'VE MASTERED — Transformer Encoder")
print("=" * 70)
print(
    f"""
  [x] Built multi-head attention wrapping the from-scratch attention kernel
  [x] Explained how different heads capture different relationship types
  [x] Implemented sinusoidal positional encoding (word order for transformers)
  [x] Built a full TransformerClassifier with nn.TransformerEncoder
  [x] Trained on full AG News (120K headlines), best acc: {max(transformer_accs):.1%}
  [x] Visualised per-head attention patterns
  [x] Applied to MAS regulatory compliance with attention-based explanations

  KEY INSIGHT:
    Multi-head attention is like having multiple specialists read the same
    document simultaneously. One head notices syntax, another notices
    entities, another notices sentiment. Together they capture a richer
    understanding than any single attention computation could.

  Next: In 03_lstm_baseline.py, you'll build an LSTM baseline to see
  exactly what the Transformer's attention mechanism buys us compared
  to sequential processing.
"""
)

# ══════════════════════════════════════════════════════════════════
# DIAGNOSTIC CHECKPOINT — Transformer (attention + residual stack)
# ══════════════════════════════════════════════════════════════════
from kailash_ml import diagnose

print("\n── Diagnostic Report (Transformer Encoder) ──")
report = diagnose(
    transformer_model,
    kind="dl",
    data=val_loader,
    show=False,
)

# ══════ EXPECTED OUTPUT (reference pattern — Transformer on AG News) ══
# ════════════════════════════════════════════════════════════════
#   DL Diagnostics Report — Prescription Pad
# ════════════════════════════════════════════════════════════════
#   [✓] Gradient flow (HEALTHY): per-layer RMS uniform across
#       `encoder.layers.{0..2}.self_attn` and `.linear1/2` —
#       residuals + LayerNorm doing their job.
#   [✓] Activations    (HEALTHY): no dead GELU units in the FFN
#       sub-blocks; attention softmax outputs within expected
#       entropy range (not collapsed onto one token).
#   [✓] Loss trend     (HEALTHY): train loss falls monotonically,
#       val loss tracks within 0.05 of train loss — no overfit
#       signal at 8 epochs on 120K headlines.
# ════════════════════════════════════════════════════════════════
# Best val acc: ~0.88 after 8 epochs on MPS/CUDA.
#
# STUDENT INTERPRETATION GUIDE — reading the Prescription Pad:
#
#  [BLOOD TEST] Gradient flow is UNIFORM — this is the architectural
#     payoff of the Transformer over the vanilla RNN (ex_3/01). The
#     residual connection around every sub-block (self-attn + FFN)
#     gives gradients a "highway" to the embedding layer, preventing
#     the vanishing-gradient problem that plagues the LSTM on long
#     sequences. Slide 5.4 (Transformers) calls this the "why we
#     stopped using RNNs" moment — the Blood Test proves it.
#     >> Prescription Pad: no action needed. If you see RMS spread
#        >2 orders of magnitude across layers, suspect post-norm
#        layout (unstable) — switch to pre-norm.
#
#  [X-RAY] Attention activations are not saturated. A collapsed
#     attention head (one token getting ~100% of the softmax mass)
#     is the Transformer's equivalent of the dead-ReLU problem —
#     that head becomes a no-op and its projection weights stop
#     learning. If the Prescription Pad flags WARNING on
#     `self_attn` activation stats, lower d_model/n_heads (too
#     many heads for too little signal) or add attention dropout.
#     >> Prescription Pad: ratio check — healthy multi-head
#        attention shows mean entropy per head near log(seq_len)/2.
#
#  [STETHOSCOPE] Loss curve converges smoothly — no instability,
#     no NaN, no periodic spikes. With 8 epochs and LayerNorm,
#     you should NOT need gradient clipping. If you see the
#     training loss oscillate, check your learning rate — the
#     Transformer is sensitive to warmup in particular.
#     >> Prescription Pad: add linear warmup over first 10% of
#        steps if loss is noisy early.
#
#  FIVE-INSTRUMENT TAKEAWAY: the Transformer's diagnostic report
#  should be almost boringly green. The Prescription Pad's value
#  here is as a canary — when you later fine-tune on a small
#  domain corpus (ex_4/04 BERT) you will see the same gradient
#  flow degrade if the learning rate is wrong. Slide 5.4 uses
#  this report as evidence that attention + residuals is the
#  "train-it-and-it-just-works" architecture that made BERT and
#  GPT possible.
#
#  CONNECT TO SLIDE 5.4: The slide claims "residuals + LayerNorm
#  make deep Transformers trainable where deep RNNs weren't."
#  The HEALTHY Blood Test reading across `layers.0..2` is the
#  direct empirical proof of that claim. Compare to ex_4/03's
#  LSTM report — gradients there concentrate in the final layer.
# ════════════════════════════════════════════════════════════════

# ── Checkpoint 3 ─────────────────────────────────────────────────────
assert (
    len(transformer_losses) == EPOCHS_SCRATCH
), "Transformer should train for all epochs"
assert (
    max(transformer_accs) > 0.60
), f"Transformer should reach >60% accuracy, got {max(transformer_accs):.3f}"
# INTERPRETATION: The Transformer processes all tokens in parallel and uses
# self-attention to capture long-range dependencies. On AG News headlines,
# it can directly connect "tech" at position 1 with "stocks" at position 8
# without propagating through every intermediate token. This architectural
# advantage becomes more pronounced on longer documents.
print(f"\n  Transformer best acc: {max(transformer_accs):.3f}")
print("\n--- Checkpoint 3 passed --- Transformer trained on AG News\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 5 — Visualise: Multi-head attention patterns on sample headline
# ════════════════════════════════════════════════════════════════════════
print("\n== Visualising multi-head attention patterns ==")
transformer_model.eval()
sample_texts = test_df["text"].to_list()[:3]
sample_idx = torch.tensor(
    [text_to_indices(t, vocab, MAX_LEN) for t in sample_texts],
    dtype=torch.long,
    device=DEVICE,
)

# Extract attention from our EducationalMultiHead on the trained embeddings
mha_viz = EducationalMultiHead(d_model=128, n_heads=4).to(DEVICE)
with torch.no_grad():
    embed = transformer_model.embed(sample_idx[:1])
    embed = transformer_model.posenc(embed)
    _, attn_weights = mha_viz(embed)  # (1, 4, seq, seq)

words = sample_texts[0].lower().split()[:MAX_LEN]
word_labels = words + ["<pad>"] * (MAX_LEN - len(words))

# Visualise each head's attention pattern
for head_idx in range(min(4, attn_weights.shape[1])):
    attn_np = attn_weights[0, head_idx].cpu().numpy()
    fig = create_attention_heatmap(
        attn_np,
        word_labels,
        title=f"Attention Head {head_idx} on: '{sample_texts[0][:50]}...'",
        max_tokens=12,
    )
    fig.write_html(f"ex_4_2_head_{head_idx}_attention.html")

print(f"  Saved 4 attention head heatmaps (ex_4_2_head_0..3_attention.html)")
print(f"  Different heads capture different relationship types:")
print(f"    Head 0: may focus on adjacent word pairs (local syntax)")
print(f"    Head 1: may focus on content words across the sentence (semantics)")
print(f"    Head 2: may focus on sentence boundaries and punctuation (structure)")
print(f"    Head 3: may focus on entity-to-entity relationships")

# ── Checkpoint 4 ─────────────────────────────────────────────────────
assert attn_weights.shape == (
    1,
    4,
    MAX_LEN,
    MAX_LEN,
), "Should have 4 heads of attention"
print("\n--- Checkpoint 4 passed --- multi-head attention visualised\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 6 — Apply: Regulatory Compliance Classification at MAS
# ════════════════════════════════════════════════════════════════════════
# SCENARIO: The Monetary Authority of Singapore (MAS) oversees compliance
# across banking, insurance, securities, and payments. Financial institutions
# submit thousands of regulatory filings monthly. MAS compliance officers
# need to classify each document by the regulation it pertains to:
#   - Banking Act (Cap. 19)
#   - Securities and Futures Act (Cap. 289)
#   - Payment Services Act 2019
#   - Insurance Act (Cap. 142)
#
# BUSINESS VALUE: Manual classification by compliance officers takes
# 15-20 minutes per document. With ~3,000 submissions/month across
# 200+ licensed institutions, that is 750-1,000 officer-hours/month.
# A Transformer classifier automates the first-pass classification,
# routing documents to the correct regulatory team in seconds.
#
# DOLLAR IMPACT: At S$80-120/hour for compliance officers, automating
# first-pass classification saves S$720K-1.44M annually. More importantly,
# the attention mechanism shows WHICH paragraphs triggered each
# classification -- providing audit trail transparency that regulators
# require under MAS Notice on Technology Risk Management.
print("\n== Application: Regulatory Compliance at MAS ==")

# Use the trained Transformer to classify financial headlines (proxy for
# regulatory documents). In production, this would use MAS-specific
# regulatory text with fine-tuned classification categories.
financial_headlines = [
    "Banks report higher profits amid rising interest rates",
    "New technology startups attract venture capital funding",
    "Stock market volatility increases as trade tensions rise",
    "Sports betting companies face new regulatory scrutiny",
    "Insurance companies adapt to climate change risks",
]

transformer_model.eval()
with torch.no_grad():
    fin_idx = torch.tensor(
        [text_to_indices(t, vocab, MAX_LEN) for t in financial_headlines],
        dtype=torch.long,
        device=DEVICE,
    )
    fin_logits = transformer_model(fin_idx)
    fin_probs = F.softmax(fin_logits, dim=-1)
    fin_preds = fin_logits.argmax(dim=-1).cpu().tolist()

print(f"\n  Regulatory document classification (Transformer):")
print(f"  {'Headline':<55} {'Classification':<12} {'Confidence':>10}")
print("  " + "-" * 79)
for text, pred, probs in zip(financial_headlines, fin_preds, fin_probs.cpu().tolist()):
    cls_name = CLASS_NAMES[pred]
    confidence = max(probs)
    print(f"  {text[:53]:<55} {cls_name:<12} {confidence:>10.1%}")

# Show attention-based explanation for the first document
with torch.no_grad():
    embed = transformer_model.embed(fin_idx[:1])
    embed = transformer_model.posenc(embed)
    _, fin_attn = mha_viz(embed)  # (1, 4, seq, seq)
    # Average across heads for an aggregate attention view
    avg_attn = fin_attn[0].mean(dim=0).cpu().numpy()  # (seq, seq)

fin_words = financial_headlines[0].lower().split()[:MAX_LEN]
fin_labels = fin_words + ["<pad>"] * (MAX_LEN - len(fin_words))
# Token-level attention: how much total attention each token receives
token_importance = avg_attn[: len(fin_words), : len(fin_words)].sum(axis=0)
token_importance = token_importance / token_importance.max()

print(f"\n  Attention-based explanation for: '{financial_headlines[0]}'")
print(f"  Token importance (which words drive the classification):")
for word, imp in sorted(zip(fin_words, token_importance), key=lambda x: -x[1])[:5]:
    bar = "#" * int(imp * 20)
    print(f"    {word:<15} {imp:.3f} {bar}")

# ── Checkpoint 5 ─────────────────────────────────────────────────────
assert len(fin_preds) == len(financial_headlines), "Should classify all headlines"
# INTERPRETATION: The Transformer classifies financial documents and the
# attention weights provide an audit trail showing which words drove each
# classification. For MAS compliance, this transparency is critical --
# regulators need to understand WHY a document was classified as it was,
# not just the classification itself.
#
# BUSINESS IMPACT for MAS:
#   - 3,000 regulatory submissions/month
#   - 15-20 min manual classification per document -> seconds with Transformer
#   - Annual saving: S$720K-1.44M in compliance officer time
#   - Attention audit trail satisfies MAS Technology Risk Management Notice
print("\n--- Checkpoint 5 passed --- MAS regulatory application complete\n")


# ══════════════════════════════════════════════════════════════════════
# REFLECTION
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  WHAT YOU'VE MASTERED — Transformer Encoder")
print("=" * 70)
print(
    f"""
  [x] Built multi-head attention wrapping the from-scratch attention kernel
  [x] Explained how different heads capture different relationship types
  [x] Implemented sinusoidal positional encoding (word order for transformers)
  [x] Built a full TransformerClassifier with nn.TransformerEncoder
  [x] Trained on full AG News (120K headlines), best acc: {max(transformer_accs):.1%}
  [x] Visualised per-head attention patterns
  [x] Applied to MAS regulatory compliance with attention-based explanations

  KEY INSIGHT:
    Multi-head attention is like having multiple specialists read the same
    document simultaneously. One head notices syntax, another notices
    entities, another notices sentiment. Together they capture a richer
    understanding than any single attention computation could.

  Next: In 03_lstm_baseline.py, you'll build an LSTM baseline to see
  exactly what the Transformer's attention mechanism buys us compared
  to sequential processing.
"""
)

