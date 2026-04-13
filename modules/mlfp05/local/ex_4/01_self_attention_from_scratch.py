# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# MLFP05 — Exercise 4.1: Self-Attention from Scratch
# ════════════════════════════════════════════════════════════════════════
#
# WHAT YOU'LL LEARN:
#   After completing this exercise, you will be able to:
#   - Explain why RNNs struggle with long sequences (information bottleneck)
#   - Derive scaled dot-product attention step by step using torch.einsum
#   - Explain why we divide by sqrt(d_k) (softmax saturation on large dims)
#   - Visualise attention weight matrices as heatmaps
#   - Apply attention-weighted representations to a real business problem
#
# PREREQUISITES: M5/ex_3 (RNNs, sequence modelling, nn.Module training).
# ESTIMATED TIME: ~25 min
# DATASET: AG News — 120,000 real news headlines, 4 classes
#          (World / Sports / Business / Sci-Tech).
#
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from shared.mlfp05.ex_4 import (
    CLASS_NAMES,
    DEVICE,
    MAX_LEN,
    build_vocab,
    create_attention_heatmap,
    load_ag_news,
    text_to_indices,
)

print(f"Using device: {DEVICE}")


# ════════════════════════════════════════════════════════════════════════
# THEORY — Why RNNs Struggle with Long Sequences
# ════════════════════════════════════════════════════════════════════════
# Recurrent neural networks process tokens one at a time. Each token's
# representation depends on every previous token -- but that information
# must squeeze through a fixed-size hidden state vector. This creates
# two problems:
#
# 1. INFORMATION BOTTLENECK: By token 50, the hidden state has "forgotten"
#    the details of token 1. The model struggles to connect related words
#    that are far apart (e.g., "Singapore" at position 2 and "economy" at
#    position 30 in a news headline).
#
# 2. SEQUENTIAL COMPUTATION: RNNs process tokens one after another --
#    O(n) sequential steps. This means training cannot be parallelised
#    across sequence positions, making RNNs slow on long documents.
#
# Attention solves both problems. Instead of squeezing everything through
# a hidden state, attention lets each token directly look at every other
# token in a single step. This is the core insight of "Attention Is All
# You Need" (Vaswani et al., 2017): you don't need recurrence at all.
#
# For a single attention head:
#   Attention(Q, K, V) = softmax( Q K^T / sqrt(d_k) ) V
#
# Q (query): "what am I looking for?"   -- shape (B, L_q, d_k)
# K (key):   "what can I offer?"        -- shape (B, L_k, d_k)
# V (value): "what do I actually pass"  -- shape (B, L_k, d_v)
#
# The division by sqrt(d_k) prevents the dot products from growing with
# the embedding dimension, which would push softmax into saturation and
# kill gradients for all non-maximal keys.
# ════════════════════════════════════════════════════════════════════════


# ════════════════════════════════════════════════════════════════════════
# TASK 1 — Load AG News and build vocabulary
# ════════════════════════════════════════════════════════════════════════
train_df, test_df = load_ag_news()
vocab = build_vocab(train_df["text"].to_list())
print(f"  vocab size: {len(vocab)} (cap 15000, seq_len {MAX_LEN})")

# ── Checkpoint 1 ─────────────────────────────────────────────────────
assert len(train_df) >= 100000, (
    f"Expected full AG News train (~120K), got {len(train_df):,}. "
    "Use the complete dataset for meaningful model comparison."
)
assert len(test_df) >= 5000, f"Expected full AG News test (~7.6K), got {len(test_df):,}"
print("\n--- Checkpoint 1 passed --- AG News loaded, vocabulary built\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 2 — Build: Scaled dot-product attention from scratch
# ════════════════════════════════════════════════════════════════════════
# We derive the attention function step by step. The key insight is that
# torch.einsum makes the batched matrix multiplication explicit and
# readable. No magic -- just linear algebra.
def scaled_dot_product_attention(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute scaled dot-product attention.

    Args:
        q: Query tensor of shape (B, L_q, d_k)
        k: Key tensor of shape (B, L_k, d_k)
        v: Value tensor of shape (B, L_k, d_v)
        mask: Optional mask of shape (B, L_q, L_k). Positions with 0 are
              masked out (set to -inf before softmax).

    Returns:
        (output, attention_weights) where output has shape (B, L_q, d_v)
        and attention_weights has shape (B, L_q, L_k).
    """
    d_k = q.size(-1)

    # Step 1: Compute raw scores via batched matrix multiplication.
    # TODO: Use torch.einsum "bqd,bkd->bqk" to compute Q*K^T scores
    # Hint: scores = torch.einsum("bqd,bkd->bqk", q, k)
    scores = ...  # YOUR CODE HERE

    # Step 2: Scale by 1/sqrt(d_k). Without this, the dot products grow
    # proportionally to d_k, pushing softmax into regions where the gradient
    # is nearly zero.
    # TODO: Divide scores by math.sqrt(d_k)
    scores = ...  # YOUR CODE HERE

    # Step 3: Apply mask (if provided). Setting masked positions to -inf
    # ensures they get zero probability after softmax.
    # TODO: Use scores.masked_fill(mask == 0, float("-inf")) when mask is not None
    if mask is not None:
        ...  # YOUR CODE HERE

    # Step 4: Softmax over the key dimension (dim=-1).
    # TODO: Apply F.softmax to get attention weights — F.softmax(scores, dim=-1)
    weights = ...  # YOUR CODE HERE

    # Step 5: Weighted sum of values using einsum.
    # TODO: Use torch.einsum "bqk,bkd->bqd" to compute weighted values
    # Hint: out = torch.einsum("bqk,bkd->bqd", weights, v)
    out = ...  # YOUR CODE HERE

    return out, weights


# Sanity check: with scaled identity-style queries, each query should
# attend most strongly to its own key position.
q_demo = torch.eye(4, 8).unsqueeze(0) * 3.0
k_demo = q_demo.clone()
v_demo = torch.arange(4 * 8, dtype=torch.float32).reshape(1, 4, 8)
_, attn_demo = scaled_dot_product_attention(q_demo, k_demo, v_demo)
print("Demo attention weights (should peak on the diagonal):")
print(attn_demo.squeeze(0).round(decimals=2))

# ── Checkpoint 2 ─────────────────────────────────────────────────────
assert attn_demo.shape == (1, 4, 4), "Attention should be (1, 4, 4)"
diag_vals = torch.diag(attn_demo.squeeze(0))
assert diag_vals.min() > 0.5, "Diagonal should dominate (each query attends to itself)"
# INTERPRETATION: The attention matrix shows which queries attend to which
# keys. A strong diagonal means each position attends to itself -- exactly
# what we expect when Q and K are scaled identity matrices. In real text,
# the pattern is more interesting: "economy" might attend to "Singapore"
# and "growth" more than to "the" or "a".
print("\n--- Checkpoint 2 passed --- scaled dot-product attention works\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 3 — Visualise: Attention weight heatmap on example sentence
# ════════════════════════════════════════════════════════════════════════
# The attention heatmap is the transformer's "explanation" -- it reveals
# which words the model considers relevant to each other. This is not
# just a debugging tool; in production, attention visualisation helps
# domain experts validate that the model attends to the right features.
print("\n== Visualising attention on a real headline ==")

# Pick a sample headline and compute attention through a learned projection
sample_text = test_df["text"].to_list()[0]
sample_words = sample_text.lower().split()[:MAX_LEN]
sample_indices = torch.tensor(
    [text_to_indices(sample_text, vocab, MAX_LEN)], dtype=torch.long
)

# TODO: Create embedding layer and compute self-attention on a real headline
# Hint: embed_dim = 32; embedding = torch.nn.Embedding(len(vocab), embed_dim, padding_idx=0)
# Then: embedded = embedding(sample_indices) inside torch.no_grad()
# Then: call scaled_dot_product_attention(embedded, embedded, embedded)
embed_dim = 32
embedding = (
    ...
)  # YOUR CODE HERE — torch.nn.Embedding(len(vocab), embed_dim, padding_idx=0)
with torch.no_grad():
    embedded = ...  # YOUR CODE HERE — embedding(sample_indices)
    _, sample_attn = (
        ...
    )  # YOUR CODE HERE — scaled_dot_product_attention(embedded, embedded, embedded)
    sample_attn_np = sample_attn[0].numpy()  # (MAX_LEN, MAX_LEN)

# Build word labels for the heatmap
word_labels = sample_words + ["<pad>"] * (MAX_LEN - len(sample_words))

fig_attn = create_attention_heatmap(
    sample_attn_np,
    word_labels,
    title=f"Self-Attention on: '{sample_text[:60]}...'",
    max_tokens=15,
)
fig_attn.write_html("ex_4_1_attention_heatmap.html")
print(f"  Headline: {sample_text[:80]!r}")
print(f"  Attention heatmap saved to ex_4_1_attention_heatmap.html")
print(f"  True label: {CLASS_NAMES[test_df['label'].to_list()[0]]}")

# ── Checkpoint 3 ─────────────────────────────────────────────────────
assert sample_attn_np.shape == (
    MAX_LEN,
    MAX_LEN,
), "Attention should cover full sequence"
assert Path(
    "ex_4_1_attention_heatmap.html"
).exists(), "Attention heatmap should be saved"
# INTERPRETATION: The heatmap shows which words attend to which. Even with
# random embeddings, you can see structural patterns: content words attend
# to other content words, and padding positions form their own cluster.
# With trained embeddings, these patterns become meaningful -- "Singapore"
# would strongly attend to "economy", "growth", and "GDP".
print("\n--- Checkpoint 3 passed --- attention heatmap visualised\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 4 — Apply: Document Similarity for a Singapore Law Firm
# ════════════════════════════════════════════════════════════════════════
# SCENARIO: Rajah & Tann, one of Singapore's largest law firms, processes
# thousands of legal documents monthly. Lawyers need to find prior case
# precedents that are relevant to their current case. Traditional keyword
# search misses semantic connections (e.g., "breach of fiduciary duty" is
# related to "director's negligence" even though they share no keywords).
#
# BUSINESS VALUE: Attention-weighted document representations capture
# semantic similarity that keyword matching cannot. A lawyer searching
# for "breach of fiduciary duty" finds related cases about director
# negligence, trustee misconduct, and corporate governance failures --
# reducing precedent research from 4-6 hours to 15-30 minutes per case.
#
# DOLLAR IMPACT: At S$500-800/hour for senior associates, saving 3-5
# hours per case on a firm handling ~200 commercial litigation cases/year
# translates to S$300K-800K in recovered associate time annually.
print("\n== Application: Document Similarity for Rajah & Tann (Singapore Law) ==")

query_texts = [
    "Wall Street stocks fall as economy shows signs of weakness",
    "Technology companies report record profits in quarterly earnings",
    "Olympic athletes break world records in swimming competition",
]
query_labels = ["Business", "Sci/Tech", "Sports"]

embedding_legal = torch.nn.Embedding(len(vocab), embed_dim, padding_idx=0)

candidate_texts = test_df["text"].to_list()[:50]
candidate_labels = [CLASS_NAMES[l] for l in test_df["label"].to_list()[:50]]


def get_attention_representation(text: str) -> torch.Tensor:
    """Compute attention-weighted document representation.

    This is the core of attention-based similarity: instead of averaging
    all word embeddings equally, attention lets important words (determined
    by their relationship to all other words) contribute more.
    """
    indices = torch.tensor([text_to_indices(text, vocab, MAX_LEN)], dtype=torch.long)
    with torch.no_grad():
        # TODO: Compute embedding, apply self-attention, mean-pool over non-pad positions
        # Step 1: emb = embedding_legal(indices)
        # Step 2: attn_out, _ = scaled_dot_product_attention(emb, emb, emb)
        # Step 3: mask = (indices != 0).float().unsqueeze(-1)
        # Step 4: pooled = (attn_out * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        emb = ...  # YOUR CODE HERE
        attn_out, _ = ...  # YOUR CODE HERE
        mask = ...  # YOUR CODE HERE — (indices != 0).float().unsqueeze(-1)
        pooled = (
            ...
        )  # YOUR CODE HERE — (attn_out * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
    return pooled.squeeze(0)  # (embed_dim,)


# Compute representations for all candidates
candidate_reps = torch.stack([get_attention_representation(t) for t in candidate_texts])

print(f"\n  Query documents: {len(query_texts)}")
print(f"  Candidate pool: {len(candidate_texts)} headlines")

# TODO: For each query, compute its representation, find top-3 similar candidates
# Hint: q_rep = get_attention_representation(q_text)
# Hint: similarities = F.cosine_similarity(q_rep.unsqueeze(0), candidate_reps, dim=1)
# Hint: top_k = similarities.topk(3)
for qi, (q_text, q_label) in enumerate(zip(query_texts, query_labels)):
    q_rep = ...  # YOUR CODE HERE — get_attention_representation(q_text)
    similarities = (
        ...
    )  # YOUR CODE HERE — F.cosine_similarity(q_rep.unsqueeze(0), candidate_reps, dim=1)
    top_k = ...  # YOUR CODE HERE — similarities.topk(3)

    print(f"\n  Query ({q_label}): '{q_text[:60]}'")
    for rank, (score, idx) in enumerate(
        zip(top_k.values.tolist(), top_k.indices.tolist())
    ):
        match_label = candidate_labels[idx]
        match_text = candidate_texts[idx][:60]
        marker = "[correct]" if match_label == q_label else "[cross-topic]"
        print(f"    Top-{rank+1} (sim={score:.3f}) {marker}: '{match_text}'")

# ── Checkpoint 4 ─────────────────────────────────────────────────────
assert candidate_reps.shape == (
    50,
    embed_dim,
), "Should have 50 candidate representations"
# INTERPRETATION: Even with untrained embeddings, the attention mechanism
# captures structural patterns in text that aid similarity search. With
# trained embeddings (as in the Transformer and BERT exercises that follow),
# the similarity becomes semantically meaningful -- "breach of fiduciary duty"
# would cluster with "director's negligence" in the attention-weighted space.
#
# BUSINESS IMPACT for Rajah & Tann:
#   - 200 commercial litigation cases/year
#   - 3-5 hours saved per case on precedent research
#   - At S$500-800/hour senior associate rate
#   - Annual saving: S$300K-800K in recovered associate time
#   - Additional value: fewer missed precedents -> better case outcomes
print("\n--- Checkpoint 4 passed --- Singapore law firm application complete\n")


# ══════════════════════════════════════════════════════════════════════
# REFLECTION
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  WHAT YOU'VE MASTERED — Self-Attention from Scratch")
print("=" * 70)
print(
    """
  [x] Understood WHY RNNs struggle (information bottleneck, O(n) sequential)
  [x] Derived scaled dot-product attention step by step
  [x] Explained the 1/sqrt(d_k) scaling factor (prevents softmax saturation)
  [x] Visualised attention weights as an interpretable heatmap
  [x] Applied attention-weighted representations to document similarity
  [x] Evaluated business impact for Singapore legal industry

  KEY INSIGHT:
    Attention lets every token directly access every other token in one
    step. No information bottleneck. No sequential processing. This is
    the foundation that makes Transformers and BERT possible.

  Next: In 02_transformer_encoder.py, you'll build multi-head attention
  and a full Transformer encoder classifier that uses this attention
  mechanism with multiple parallel "heads" for richer representations.
"""
)
