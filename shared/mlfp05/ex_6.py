# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
Shared infrastructure for Exercise 6 — Graph Neural Networks.

Contains: Cora dataset loading (with Karate Club fallback), graph
normalisation, train/val/test split, kailash-ml engine setup,
graph visualisation helpers, and training harness.

Technique-specific code (GCN, GAT, GraphSAGE layers) does NOT belong here.
"""
from __future__ import annotations

import asyncio
import pickle
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from kailash.db import ConnectionManager
from kailash_ml import ExperimentTracker, ModelVisualizer
from kailash_ml import ModelRegistry
from kailash_ml.types import MetricSpec

from shared.kailash_helpers import get_device, setup_environment

# ════════════════════════════════════════════════════════════════════════
# ENVIRONMENT SETUP
# ════════════════════════════════════════════════════════════════════════

setup_environment()
torch.manual_seed(42)
np.random.seed(42)
device = get_device()

OUTPUT_DIR = Path("outputs") / "ex6_gnns"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data" / "mlfp05" / "cora"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# ════════════════════════════════════════════════════════════════════════
# GRAPH DATA LOADING
# ════════════════════════════════════════════════════════════════════════


def load_cora() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, str, int]:
    """Cora — 2708 papers, 1433 bag-of-words features, 7 classes.

    Returns:
        X_np: node features (N, F)
        A_np: dense adjacency matrix (N, N)
        y_np: node labels (N,)
        edge_index_np: edge list (2, E) for link prediction
        dataset_name: "Cora"
        n_classes: 7
    """
    from torch_geometric.datasets import Planetoid

    dataset = Planetoid(root=str(DATA_DIR), name="Cora")
    # torch_geometric.data.Data has dynamic attributes (num_nodes/x/y/edge_index);
    # use Any to avoid type-checker false positives without losing runtime fidelity.
    data: Any = dataset[0]
    n = data.num_nodes
    X_np = data.x.numpy().astype(np.float32)
    y_np = data.y.numpy().astype(np.int64)

    # Build a dense adjacency matrix from the edge_index. Cora has ~10k
    # directed edges (5278 undirected) over 2708 nodes; the dense matrix
    # is ~7M entries which fits comfortably in CPU memory.
    A_np = np.zeros((n, n), dtype=np.float32)
    edge_index_np = data.edge_index.numpy()
    src = edge_index_np[0]
    dst = edge_index_np[1]
    A_np[src, dst] = 1.0
    A_np[dst, src] = 1.0  # symmetrise just in case
    n_classes = int(dataset.num_classes)
    return X_np, A_np, y_np, edge_index_np, "Cora", n_classes


def load_karate() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, str, int]:
    """Zachary's Karate Club — 34 nodes, 78 edges, 2 factions."""
    import networkx as nx

    G = nx.karate_club_graph()
    n = G.number_of_nodes()
    A_np = nx.to_numpy_array(G, dtype=np.float32)
    labels = np.array(
        [0 if G.nodes[i]["club"] == "Mr. Hi" else 1 for i in range(n)],
        dtype=np.int64,
    )
    # Karate has no node features; use one-hot identity (transductive)
    X_np = np.eye(n, dtype=np.float32)
    # Build edge_index from adjacency
    src, dst = np.where(A_np > 0)
    edge_index_np = np.stack([src, dst]).astype(np.int64)
    return X_np, A_np, labels, edge_index_np, "Karate Club", 2


def load_graph_data() -> dict:
    """Load Cora (with Karate fallback) and return all graph tensors.

    Returns a dict with keys:
        X, A, y, A_norm, A_hat — torch tensors on device
        X_np, A_np, y_np, edge_index_np — numpy arrays
        train_mask, val_mask, test_mask — boolean masks on device
        N, F_dim, n_classes, n_edges_undirected, dataset_name — scalars
    """
    try:
        X_np, A_np, y_np, edge_index_np, dataset_name, n_classes = load_cora()
    except Exception as exc:
        print(
            f"Could not load Cora ({type(exc).__name__}: {exc}); "
            "falling back to Karate Club"
        )
        X_np, A_np, y_np, edge_index_np, dataset_name, n_classes = load_karate()

    N = X_np.shape[0]
    F_dim = X_np.shape[1]
    n_edges_undirected = int(A_np.sum() // 2)
    print(
        f"Graph: {dataset_name} — {N} nodes, {n_edges_undirected} undirected edges, "
        f"feature_dim={F_dim}, classes={n_classes}"
    )
    class_counts = ", ".join(f"c{c}={int((y_np == c).sum())}" for c in range(n_classes))
    print(f"  per-class counts: {class_counts}")

    X = torch.from_numpy(X_np).to(device)
    A = torch.from_numpy(A_np).to(device)
    y = torch.from_numpy(y_np).to(device)

    # Add self-loops and build the symmetric Laplacian D^{-1/2} A D^{-1/2}
    A_hat = A + torch.eye(N, device=device)
    deg = A_hat.sum(dim=1)
    d_inv_sqrt = deg.pow(-0.5)
    d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.0
    A_norm = d_inv_sqrt.unsqueeze(1) * A_hat * d_inv_sqrt.unsqueeze(0)

    # Train/val/test split — 20% train, 20% val, 60% test (per class)
    train_mask = torch.zeros(N, dtype=torch.bool, device=device)
    val_mask = torch.zeros(N, dtype=torch.bool, device=device)
    rng = np.random.default_rng(0)
    for c in range(n_classes):
        idx = np.where(y_np == c)[0]
        if len(idx) == 0:
            continue
        rng.shuffle(idx)
        n_train = max(1, int(0.2 * len(idx)))
        n_val = max(1, int(0.2 * len(idx)))
        train_mask[idx[:n_train]] = True
        val_mask[idx[n_train : n_train + n_val]] = True
    test_mask = ~(train_mask | val_mask)
    print(
        f"  train: {int(train_mask.sum().item())}, "
        f"val: {int(val_mask.sum().item())}, "
        f"test: {int(test_mask.sum().item())}"
    )

    return {
        "X": X,
        "A": A,
        "y": y,
        "A_norm": A_norm,
        "A_hat": A_hat,
        "X_np": X_np,
        "A_np": A_np,
        "y_np": y_np,
        "edge_index_np": edge_index_np,
        "train_mask": train_mask,
        "val_mask": val_mask,
        "test_mask": test_mask,
        "N": N,
        "F_dim": F_dim,
        "n_classes": n_classes,
        "n_edges_undirected": n_edges_undirected,
        "dataset_name": dataset_name,
    }


# ════════════════════════════════════════════════════════════════════════
# KAILASH ENGINE SETUP
# ════════════════════════════════════════════════════════════════════════


async def _setup_engines():
    """Open kailash-ml 1.1.1 tracker + registry. 5-tuple preserved."""
    # Schema-conflict workaround (kailash-ml 1.5.x): ExperimentTracker
    # and ModelRegistry use incompatible _kml_model_versions schemas.
    # Route them to separate sqlite files until upstream fixes the conflict.
    db = "sqlite:///mlfp05_gnns.db"
    registry_db = "sqlite:///mlfp05_gnns_registry.db"
    tracker = await ExperimentTracker.create(store_url=db)
    conn = ConnectionManager(registry_db)
    await conn.initialize()
    registry = ModelRegistry(conn)
    return conn, tracker, "m5_gnns", registry, True


def setup_engines() -> tuple:
    """Synchronously set up kailash-ml engines."""
    return asyncio.run(_setup_engines())


# ════════════════════════════════════════════════════════════════════════
# TRAINING HARNESS
# ════════════════════════════════════════════════════════════════════════


def train_node_classifier(
    model: nn.Module,
    name: str,
    forward_arg: torch.Tensor,
    graph_data: dict,
    tracker: ExperimentTracker,
    exp_name: str,
    epochs: int = 100,
    lr: float = 1e-2,
    weight_decay: float = 5e-4,
) -> tuple[list[float], list[float], list[float]]:
    """Train a GNN for node classification and log metrics to ExperimentTracker.

    Returns:
        train_losses: per-epoch training loss
        val_accs: per-epoch validation accuracy
        test_accs: per-epoch test accuracy
    """
    return asyncio.run(
        _train_node_classifier_async(
            model,
            name,
            forward_arg,
            graph_data,
            tracker,
            exp_name,
            epochs,
            lr,
            weight_decay,
        )
    )


async def _train_node_classifier_async(
    model: nn.Module,
    name: str,
    forward_arg: torch.Tensor,
    graph_data: dict,
    tracker: ExperimentTracker,
    exp_name: str,
    epochs: int = 100,
    lr: float = 1e-2,
    weight_decay: float = 5e-4,
) -> tuple[list[float], list[float], list[float]]:
    """Async core — uses the kailash-ml 1.1.1 ``tracker.track(...)`` context manager."""
    X = graph_data["X"]
    y = graph_data["y"]
    train_mask = graph_data["train_mask"]
    val_mask = graph_data["val_mask"]
    test_mask = graph_data["test_mask"]
    N = graph_data["N"]
    n_edges = graph_data["n_edges_undirected"]
    dataset_name = graph_data["dataset_name"]
    hidden_dim = 16 if dataset_name == "Karate Club" else 64

    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  [{name}] parameters: {n_params:,}")

    train_losses: list[float] = []
    val_accs: list[float] = []
    test_accs: list[float] = []

    async with tracker.track(experiment=exp_name, run_name=name) as run:
        await run.log_params(
            {
                "model_type": name,
                "hidden_dim": str(hidden_dim),
                "epochs": str(epochs),
                "lr": str(lr),
                "weight_decay": str(weight_decay),
                "n_params": str(n_params),
                "dataset": dataset_name,
                "n_nodes": str(N),
                "n_edges": str(n_edges),
            }
        )

        for epoch in range(epochs):
            model.train()
            opt.zero_grad()
            logits = model(X, forward_arg)
            loss = F.cross_entropy(logits[train_mask], y[train_mask])
            loss.backward()
            opt.step()
            train_losses.append(loss.item())

            model.eval()
            with torch.no_grad():
                preds = model(X, forward_arg).argmax(dim=-1)
                v_acc = (preds[val_mask] == y[val_mask]).float().mean().item()
                t_acc = (preds[test_mask] == y[test_mask]).float().mean().item()
            val_accs.append(v_acc)
            test_accs.append(t_acc)

            await run.log_metrics(
                {
                    "train_loss": loss.item(),
                    "val_accuracy": v_acc,
                    "test_accuracy": t_acc,
                },
                step=epoch + 1,
            )

            if (epoch + 1) % 25 == 0:
                print(
                    f"  [{name}] epoch {epoch+1:3d}  "
                    f"loss={loss.item():.4f}  val_acc={v_acc:.3f}  test_acc={t_acc:.3f}"
                )

        await run.log_metrics(
            {
                "final_loss": train_losses[-1],
                "final_val_accuracy": val_accs[-1],
                "final_test_accuracy": test_accs[-1],
                "best_val_accuracy": max(val_accs),
                "best_test_accuracy": max(test_accs),
            }
        )

    return train_losses, val_accs, test_accs


# ════════════════════════════════════════════════════════════════════════
# VISUALISATION HELPERS
# ════════════════════════════════════════════════════════════════════════

viz = ModelVisualizer()


def plot_training_curves(
    metrics_dict: dict[str, list[float]],
    title: str,
    y_label: str,
    filename: str,
) -> None:
    """Plot overlaid training curves and save as HTML."""
    fig = viz.training_history(
        metrics=metrics_dict,
        x_label="Epoch",
        y_label=y_label,
    )
    fig.update_layout(title=title)
    filepath = OUTPUT_DIR / filename
    fig.write_html(str(filepath))
    print(f"  Saved: {filepath}")


def plot_node_embeddings(
    embeddings: np.ndarray,
    labels: np.ndarray,
    n_classes: int,
    title: str,
    filename: str,
) -> None:
    """2-D PCA projection of node embeddings coloured by class label.

    Uses SVD-based PCA (no sklearn dependency). Nodes of the same class
    should cluster together if the GNN learned meaningful representations.
    """
    emb_centered = embeddings - embeddings.mean(axis=0, keepdims=True)
    Vt = np.linalg.svd(emb_centered, full_matrices=False)[2]
    coords = emb_centered @ Vt.T[:, :2]

    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    cmap = plt.cm.get_cmap("tab10", n_classes)
    for c in range(n_classes):
        mask = labels == c
        ax.scatter(
            coords[mask, 0],
            coords[mask, 1],
            c=[cmap(c)],
            s=15,
            alpha=0.6,
            label=f"Class {c}",
        )
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("PC 1")
    ax.set_ylabel("PC 2")
    ax.legend(fontsize=8, markerscale=2, loc="best")
    plt.tight_layout()
    filepath = OUTPUT_DIR / filename
    plt.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {filepath}")

    # Text summary of first 3 nodes per class
    print(f"\n  {title} — first 3 nodes per class:")
    for c in range(min(n_classes, 7)):
        rows = coords[labels == c][:3]
        if len(rows) == 0:
            continue
        pretty = ", ".join(f"({r[0]:+.2f}, {r[1]:+.2f})" for r in rows)
        print(f"    class {c}: {pretty}")

    return coords


def plot_graph_with_embeddings(
    A_np: np.ndarray,
    embeddings_2d: np.ndarray,
    labels: np.ndarray,
    n_classes: int,
    title: str,
    filename: str,
    max_nodes: int = 200,
) -> None:
    """Draw graph edges on the 2-D embedding space, coloured by class.

    Subsamples to max_nodes for readability on large graphs.
    """
    N = A_np.shape[0]
    if N > max_nodes:
        rng = np.random.default_rng(42)
        subset = rng.choice(N, max_nodes, replace=False)
    else:
        subset = np.arange(N)

    coords = embeddings_2d[subset]
    sub_labels = labels[subset]
    sub_A = A_np[np.ix_(subset, subset)]

    fig, ax = plt.subplots(1, 1, figsize=(12, 9))
    cmap = plt.cm.get_cmap("tab10", n_classes)

    # Draw edges first (behind nodes)
    src_idx, dst_idx = np.where(np.triu(sub_A) > 0)
    for s, d in zip(src_idx, dst_idx):
        ax.plot(
            [coords[s, 0], coords[d, 0]],
            [coords[s, 1], coords[d, 1]],
            color="gray",
            alpha=0.08,
            linewidth=0.5,
        )

    # Draw nodes
    for c in range(n_classes):
        mask = sub_labels == c
        ax.scatter(
            coords[mask, 0],
            coords[mask, 1],
            c=[cmap(c)],
            s=25,
            alpha=0.7,
            label=f"Class {c}",
            zorder=2,
        )

    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(fontsize=8, markerscale=2, loc="best")
    ax.set_xlabel("PC 1")
    ax.set_ylabel("PC 2")
    plt.tight_layout()
    filepath = OUTPUT_DIR / filename
    plt.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {filepath}")


def plot_attention_weights(
    alpha: np.ndarray,
    A_np: np.ndarray,
    labels: np.ndarray,
    title: str,
    filename: str,
    node_idx: int = 0,
    top_k: int = 10,
) -> None:
    """Visualise attention weights for a single node's neighbourhood.

    Shows which neighbours the GAT layer attends to most strongly.
    """
    neighbours = np.where(A_np[node_idx] > 0)[0]
    if len(neighbours) == 0:
        print(f"  Node {node_idx} has no neighbours — skipping attention plot")
        return

    weights = alpha[node_idx, neighbours]
    order = np.argsort(-weights)[:top_k]
    top_neighbours = neighbours[order]
    top_weights = weights[order]

    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    bar_colors = [plt.cm.get_cmap("tab10")(labels[n] % 10) for n in top_neighbours]
    ax.barh(
        range(len(top_neighbours)),
        top_weights,
        color=bar_colors,
        edgecolor="white",
    )
    ax.set_yticks(range(len(top_neighbours)))
    ax.set_yticklabels(
        [f"Node {n} (class {labels[n]})" for n in top_neighbours],
        fontsize=9,
    )
    ax.set_xlabel("Attention Weight")
    ax.set_title(
        f"{title}\nNode {node_idx} (class {labels[node_idx]}) attending to top-{top_k} neighbours",
        fontsize=12,
        fontweight="bold",
    )
    ax.invert_yaxis()
    plt.tight_layout()
    filepath = OUTPUT_DIR / filename
    plt.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {filepath}")


# ════════════════════════════════════════════════════════════════════════
# MODEL REGISTRATION HELPER
# ════════════════════════════════════════════════════════════════════════


async def _register_model(
    registry: ModelRegistry,
    name: str,
    model: nn.Module,
    metrics: list[MetricSpec],
) -> object:
    """Register a model in the ModelRegistry."""
    model_bytes = pickle.dumps(model.state_dict())
    version = await registry.register_model(
        name=name,
        artifact=model_bytes,
        metrics=metrics,
    )
    return version


def register_model(
    registry: ModelRegistry,
    name: str,
    model: nn.Module,
    metrics: list[MetricSpec],
) -> object:
    """Synchronously register a model."""
    return asyncio.run(_register_model(registry, name, model, metrics))
