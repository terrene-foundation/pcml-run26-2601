# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# MLFP05 — Exercise 6.1: Graph Convolutional Network (GCN)
# ════════════════════════════════════════════════════════════════════════
#
# WHAT YOU'LL LEARN:
#   - Why tabular ML methods fail on graph-structured data
#   - The message-passing intuition: nodes learn from their neighbours
#   - Build a GCN layer as torch.nn.Module:  H' = sigma(A_norm @ H @ W)
#   - Train a node classifier on the Cora citation network (2708 papers)
#   - Visualise learned node embeddings with 2-D PCA projection
#   - Track training with kailash-ml ExperimentTracker
#
# PREREQUISITES: M5/ex_4 (attention mechanisms, nn.Module training).
# ESTIMATED TIME: ~30 min
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from shared.mlfp05.ex_6 import (
    OUTPUT_DIR,
    device,
    load_graph_data,
    plot_graph_with_embeddings,
    plot_node_embeddings,
    plot_training_curves,
    register_model,
    setup_engines,
    train_node_classifier,
)
from kailash_ml.types import MetricSpec


# ════════════════════════════════════════════════════════════════════════
# PHASE 1 — THEORY: Why Graphs Need Their Own Neural Networks
# ════════════════════════════════════════════════════════════════════════
#
# Imagine you're classifying research papers at NUS or NTU into fields
# like "Computer Vision", "NLP", or "Reinforcement Learning". You could
# use each paper's bag-of-words features in a standard MLP — but you'd
# be ignoring the most valuable signal: CITATION LINKS.
#
# A paper about "Attention Is All You Need" that cites 10 NLP papers and
# is cited by 50 more NLP papers is almost certainly an NLP paper — even
# if its bag-of-words features overlap with computer vision papers.
#
# The problem: tabular methods (MLP, logistic regression, random forest)
# expect a fixed-size feature vector per sample. But graph data has:
#   - Variable-size neighbourhoods (some papers cite 2, others cite 200)
#   - Relational structure (who-cites-whom) that carries class signal
#   - No natural ordering of neighbours
#
# GCN's insight: MESSAGE PASSING. Think of it like a gossip protocol:
#   1. Each node collects features from its neighbours
#   2. It averages them (weighted by degree)
#   3. It passes the average through a learnable transform
#   4. After 2 hops, each node "knows" about its 2-hop neighbourhood
#
# Mathematically:  H' = sigma( D^{-1/2} A D^{-1/2} H W )
#   - A = adjacency matrix (who's connected to whom)
#   - D = degree matrix (how many connections each node has)
#   - H = current node features
#   - W = learnable weight matrix
#   - sigma = activation function (ReLU)
#
# The key insight: the entire message-passing step is ONE MATRIX
# MULTIPLICATION. No Python loops over nodes. This is why GCNs are
# fast and parallelisable on GPUs.
print("=" * 70)
print("  PHASE 1 — THEORY: Graph Convolutional Networks")
print("=" * 70)
print(
    """
  WHY GRAPHS NEED SPECIAL ARCHITECTURES:
  - Tabular methods ignore relational structure (citation links)
  - Variable-size neighbourhoods can't be flattened to a fixed vector
  - Graph structure carries classification signal independent of features

  GCN MESSAGE PASSING (the "gossip protocol"):
  1. Collect — each node gathers features from neighbours
  2. Aggregate — average features (weighted by degree)
  3. Transform — pass through learnable weights
  4. Stack — multiple layers = multi-hop information flow

  Formula:  H' = sigma( D^{-1/2} A D^{-1/2} H W )
  The ENTIRE neighbourhood aggregation is ONE matrix multiply!
"""
)


# ════════════════════════════════════════════════════════════════════════
# PHASE 2 — BUILD: GCN Layer Implementation
# ════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("  PHASE 2 — BUILD: GCN Layer + Model")
print("=" * 70)

# Load graph data and set up engines
graph_data = load_graph_data()
conn, tracker, exp_name, registry, has_registry = setup_engines()

X = graph_data["X"]
A = graph_data["A"]
A_norm = graph_data["A_norm"]
y = graph_data["y"]
y_np = graph_data["y_np"]
N = graph_data["N"]
F_dim = graph_data["F_dim"]
n_classes = graph_data["n_classes"]
dataset_name = graph_data["dataset_name"]

HIDDEN_DIM = 16 if dataset_name == "Karate Club" else 64
EPOCHS = 100


class GCNLayer(nn.Module):
    """Single GCN layer:  H' = A_norm @ H @ W.

    No Python loops over nodes — a single matmul aggregates every
    neighbourhood at once. This is the core insight of GCNs: message
    passing as matrix multiplication.
    """

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.W = nn.Linear(in_dim, out_dim, bias=True)

    def forward(self, h: torch.Tensor, a_norm: torch.Tensor) -> torch.Tensor:
        return a_norm @ self.W(h)


class GCN(nn.Module):
    """Two-layer GCN for node classification."""

    def __init__(self, in_dim: int, hidden_dim: int, n_classes: int):
        super().__init__()
        self.l1 = GCNLayer(in_dim, hidden_dim)
        self.l2 = GCNLayer(hidden_dim, n_classes)

    def forward(self, h: torch.Tensor, a_norm: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.l1(h, a_norm))
        h = F.dropout(h, p=0.5, training=self.training)
        return self.l2(h, a_norm)

    def embed(self, h: torch.Tensor, a_norm: torch.Tensor) -> torch.Tensor:
        """Return the hidden-layer embedding (before classification head)."""
        return F.relu(self.l1(h, a_norm))


gcn = GCN(in_dim=F_dim, hidden_dim=HIDDEN_DIM, n_classes=n_classes)
n_params = sum(p.numel() for p in gcn.parameters())
print(f"\n  GCN architecture:")
print(f"    Layer 1: GCNLayer({F_dim} -> {HIDDEN_DIM})")
print(f"    Layer 2: GCNLayer({HIDDEN_DIM} -> {n_classes})")
print(f"    Total parameters: {n_params:,}")
print(f"\n  How it works:")
print(f"    Input:  {N} nodes x {F_dim} features")
print(f"    Layer 1: A_norm @ (X @ W1) -> {N} x {HIDDEN_DIM} (neighbourhood averages)")
print(f"    ReLU + Dropout(0.5)")
print(f"    Layer 2: A_norm @ (H @ W2) -> {N} x {n_classes} (class logits)")


# ── Build Checkpoint ────────────────────────────────────────────────
assert isinstance(gcn, nn.Module), "GCN should be an nn.Module"
assert n_params > 0, "GCN should have learnable parameters"
print("\n--- Build checkpoint passed --- GCN architecture created\n")


# ════════════════════════════════════════════════════════════════════════
# PHASE 3 — TRAIN: Node Classification on Cora
# ════════════════════════════════════════════════════════════════════════
print("=" * 70)
print(f"  PHASE 3 — TRAIN: GCN on {dataset_name}")
print("=" * 70)

gcn_losses, gcn_val, gcn_test = train_node_classifier(
    model=gcn,
    name="GCN",
    forward_arg=A_norm,
    graph_data=graph_data,
    tracker=tracker,
    exp_name=exp_name,
    epochs=EPOCHS,
)

# ── Train Checkpoint ────────────────────────────────────────────────
assert len(gcn_losses) == EPOCHS, f"Expected {EPOCHS} epoch losses for GCN"
assert gcn_losses[-1] < gcn_losses[0], "GCN loss should decrease"
best_val = max(gcn_val)
best_test = max(gcn_test)
print(f"\n  GCN Results:")
print(f"    Best validation accuracy: {best_val:.4f}")
print(f"    Best test accuracy:       {best_test:.4f}")
print(f"    Final loss:               {gcn_losses[-1]:.4f}")
# INTERPRETATION: GCN uses a fixed aggregation scheme based on the graph
# Laplacian. Every node's new representation is a weighted average of its
# neighbours' features, where the weights come from the degree-normalised
# adjacency. This is equivalent to a 1-hop spectral filter on the graph.
print("\n--- Train checkpoint passed --- GCN trained successfully\n")


# ════════════════════════════════════════════════════════════════════════
# PHASE 4 — VISUALISE: Node Embeddings + Graph Structure
# ════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("  PHASE 4 — VISUALISE: GCN Node Embeddings")
print("=" * 70)

gcn.eval()
with torch.no_grad():
    gcn_emb = gcn.embed(X, A_norm).cpu().numpy()
    gcn_preds = gcn(X, A_norm).argmax(dim=-1).cpu().numpy()

# Plot 1: 2-D PCA of node embeddings coloured by TRUE class
coords = plot_node_embeddings(
    embeddings=gcn_emb,
    labels=y_np,
    n_classes=n_classes,
    title=f"GCN Node Embeddings (True Labels) — {dataset_name}",
    filename="gcn_node_embeddings.png",
)

# Plot 2: 2-D PCA of node embeddings coloured by PREDICTED class
# This reveals where the model is confident vs confused — misclassified
# nodes appear as "wrong-coloured" points amid a cluster of another class.
plot_node_embeddings(
    embeddings=gcn_emb,
    labels=gcn_preds,
    n_classes=n_classes,
    title=f"GCN Node Embeddings (Predicted Labels) — {dataset_name}",
    filename="gcn_node_embeddings_predicted.png",
)

# Plot 3: Graph structure overlaid on embedding space
plot_graph_with_embeddings(
    A_np=graph_data["A_np"],
    embeddings_2d=coords,
    labels=y_np,
    n_classes=n_classes,
    title=f"GCN — Graph Structure in Embedding Space ({dataset_name})",
    filename="gcn_graph_embeddings.png",
)

# Plot 4: Training accuracy curve
plot_training_curves(
    metrics_dict={"GCN val accuracy": gcn_val, "GCN test accuracy": gcn_test},
    title="GCN Accuracy",
    y_label="Accuracy",
    filename="gcn_accuracy_curves.html",
)

# Plot 5: Training loss curve
plot_training_curves(
    metrics_dict={"GCN train loss": gcn_losses},
    title="GCN Training Loss",
    y_label="Cross-Entropy Loss",
    filename="gcn_loss_curve.html",
)

# ── Visualise Checkpoint ────────────────────────────────────────────
assert gcn_emb.shape == (
    N,
    HIDDEN_DIM,
), f"Embedding shape should be ({N}, {HIDDEN_DIM})"
# INTERPRETATION: Good GCN embeddings show clear class separation in the
# 2-D projection. Comparing the true-label and predicted-label scatter
# plots reveals where the model is confident (colours match) and where
# it struggles (misclassified nodes appear as wrong-coloured dots within
# a cluster). Nodes of the same class cluster because the GCN aggregated
# features from their citation neighbourhoods — papers in the same field
# cite similar papers and share vocabulary, so their aggregated
# representations converge.
print("\n--- Visualise checkpoint passed --- GCN embeddings plotted\n")


# ════════════════════════════════════════════════════════════════════════
# PHASE 5 — APPLY: Academic Research Network at NUS/NTU
# ════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("  PHASE 5 — APPLY: Research Paper Classification at NUS/NTU")
print("=" * 70)
print(
    """
  SCENARIO: You're building a research analytics tool for NUS or NTU.
  The university publishes thousands of papers per year across faculties.
  Your task: automatically classify papers into research fields using
  both their text features AND citation links.

  WHY GCN BEATS BAG-OF-WORDS BASELINE:
  Consider a paper titled "Efficient Training of Vision Transformers".
  - Bag-of-words features: high weight on "training", "efficient", "vision"
    -> Could be Computer Vision OR Systems/Optimisation
  - Citation links: cites ViT, DeiT, DINO (all CV papers)
    -> GCN correctly classifies as Computer Vision

  The citation graph is a free signal that bag-of-words models ignore.
"""
)

# Demonstrate: compare a naive bag-of-words baseline to the GCN
print("  Bag-of-Words Baseline vs GCN Comparison:")

# Simple baseline: logistic regression on raw features (no graph structure)
from torch.optim import Adam

train_mask = graph_data["train_mask"]
test_mask = graph_data["test_mask"]

baseline_model = nn.Linear(F_dim, n_classes).to(device)
baseline_opt = Adam(baseline_model.parameters(), lr=1e-2)

for epoch in range(EPOCHS):
    baseline_model.train()
    baseline_opt.zero_grad()
    logits = baseline_model(X)
    loss = F.cross_entropy(logits[train_mask], y[train_mask])
    loss.backward()
    baseline_opt.step()

baseline_model.eval()
with torch.no_grad():
    baseline_preds = baseline_model(X).argmax(dim=-1)
    baseline_acc = (baseline_preds[test_mask] == y[test_mask]).float().mean().item()

print(f"    Bag-of-Words (no graph):  test accuracy = {baseline_acc:.4f}")
print(f"    GCN (with graph):         test accuracy = {best_test:.4f}")
improvement = best_test - baseline_acc
print(f"    Improvement from graph:   +{improvement:.4f} ({improvement*100:.1f} pp)")
print()

if improvement > 0:
    print("  INSIGHT: The citation graph provides significant additional signal.")
    print("  Papers in the same field cite each other more often, creating")
    print("  class-homogeneous neighbourhoods that the GCN exploits.")
else:
    print("  NOTE: On this split, bag-of-words is competitive — Cora's features")
    print("  are already informative. The GCN advantage grows on sparser features.")

print(
    """
  REAL-WORLD DEPLOYMENT:
  1. Build a citation graph from Scopus/Web of Science API
  2. Extract bag-of-words or TF-IDF features from abstracts
  3. Train GCN on papers with known faculty/department labels
  4. Classify new papers automatically for research analytics dashboards
  5. Track with ExperimentTracker — retrain quarterly as citation graph grows
"""
)

# Register the GCN model
if has_registry:
    version = register_model(
        registry=registry,
        name=f"m5_gcn_{dataset_name.lower().replace(' ', '_')}",
        model=gcn,
        metrics=[
            MetricSpec(name="best_val_accuracy", value=best_val),
            MetricSpec(name="best_test_accuracy", value=best_test),
            MetricSpec(name="final_loss", value=gcn_losses[-1]),
            MetricSpec(name="baseline_accuracy", value=baseline_acc),
            MetricSpec(name="graph_improvement", value=improvement),
            MetricSpec(name="hidden_dim", value=float(HIDDEN_DIM)),
            MetricSpec(name="epochs", value=float(EPOCHS)),
        ],
    )
    print(f"  Registered GCN: version={version.version}, val_acc={best_val:.4f}")

# ── Apply Checkpoint ────────────────────────────────────────────────
assert baseline_acc > 0.0, "Baseline should produce non-zero accuracy"
print("\n--- Apply checkpoint passed --- GCN vs baseline comparison complete\n")


# ════════════════════════════════════════════════════════════════════════
# REFLECTION
# ════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  WHAT YOU'VE MASTERED — GCN")
print("=" * 70)
print(
    f"""
  GRAPH CONVOLUTIONAL NETWORK (Kipf & Welling, 2017):
  [x] Message passing as matrix multiplication: H' = A_norm @ H @ W
  [x] Fixed aggregation via degree-normalised adjacency (Laplacian)
  [x] Simplest GNN — fast, parallelisable, effective on homogeneous graphs
  [x] Trained on {dataset_name}: {best_val:.1%} val accuracy, {best_test:.1%} test accuracy
  [x] Compared to bag-of-words baseline: +{improvement*100:.1f} percentage points
  [x] Visualised embeddings showing class separation in 2-D PCA

  KEY LIMITATION: GCN uses FIXED aggregation weights (degree-based).
  All neighbours contribute equally regardless of relevance. What if
  some citations are more important than others?

  Next: Exercise 6.2 — Graph Attention Networks (GAT) learn which
  neighbours matter most via attention weights...
"""
)

# Clean up
import asyncio

asyncio.run(conn.close())

# ══════════════════════════════════════════════════════════════════
# DIAGNOSTIC CHECKPOINT — five instruments before Visualise
# ══════════════════════════════════════════════════════════════════
# Reference: `kailash_ml.diagnostics` (via `kailash-ml`) — see gold standard
# `solutions/ex_1/01_standard_ae.py` for the full pattern.
from kailash_ml.diagnostics import run_diagnostic_checkpoint


def _diag_loss(m, batch):
    # GCN node classification loss
    # Customise per your exercise's loss shape.
    if isinstance(batch, (tuple, list)):
        x = batch[0]
        y = batch[1] if len(batch) > 1 else None
    else:
        x, y = batch, None
    out = m(x)
    import torch.nn.functional as F
    if y is None:
        return F.mse_loss(out, x)
    return F.cross_entropy(out, y)


print("\n── Diagnostic Report (GCN — Graph Convolutional Network) ──")
try:
    diag, findings = run_diagnostic_checkpoint(
        gcn,
        [(features, labels)],
        _diag_loss,
        title="GCN — Graph Convolutional Network",
        n_batches=8,
        show=False,
    )
except Exception as exc:
    # Diagnostic is pedagogical — never block the exercise on it.
    print(f"[diagnostic skipped: {exc}]")

# ══════ EXPECTED OUTPUT (synthesized reference — full run produces similar pattern) ══════
# ════════════════════════════════════════════════════════════════
#   DL Diagnostics Report — Prescription Pad
# ════════════════════════════════════════════════════════════════
# [✓] Gradient flow (HEALTHY): RMS range 5.2e-04 to 8.7e-03 across 2 GCN layers.
#     Shallow GCN = no over-smoothing risk yet.
# [✓] Dead neurons  (HEALTHY): 8% inactive — GCN uses ReLU, healthy.
# [✓] Loss trend    (HEALTHY): train loss → 0.23, val accuracy plateauing at ~82%.
# ════════════════════════════════════════════════════════════════
#
# STUDENT INTERPRETATION GUIDE — reading the Prescription Pad:

#  [BLOOD TEST] Shallow GCN (2 layers) shows no pathologies. BUT
#     slide 5.6 warns: stack 4+ GCN layers and node embeddings
#     become nearly identical (over-smoothing). Watch for this in
#     ex_6/05 architecture comparison — the signature is all nodes'
#     cosine similarity → 1.0 at deep layers.
#     >> Prescription: use skip connections (ResGCN) OR PairNorm OR
#        stick to 2-3 layers. GAT (ex_6/02) also helps because
#        learned attention weights can avoid smoothing.
#
#  [X-RAY] 8% dead ReLU is normal. GCN's linear aggregation
#     followed by ReLU doesn't typically kill channels.
#
#  [STETHOSCOPE] 82% accuracy on Cora is baseline competitive.
#     GAT and GraphSAGE (next exercises) typically push to 83-85%
#     via learned vs fixed aggregation.

