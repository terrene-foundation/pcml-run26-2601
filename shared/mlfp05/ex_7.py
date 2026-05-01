# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
Shared infrastructure for Exercise 7 — Transfer Learning.

Contains: CIFAR-10 data loading, feature visualisation helpers,
ExperimentTracker/ModelRegistry setup, training harness.
Technique-specific code does NOT belong here.
"""
from __future__ import annotations

import asyncio
import pickle
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader

import torchvision
import torchvision.transforms as T

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

OUTPUT_DIR = Path("outputs") / "ex7_transfer_learning"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ════════════════════════════════════════════════════════════════════════
# DATA LOADING — CIFAR-10 (full 50K, resized for ResNet-18)
# ════════════════════════════════════════════════════════════════════════
# ResNet-18 was designed for 224x224 ImageNet images. CIFAR-10 is 32x32.
# We resize to 96x96: ResNet's strided convolutions shrink spatial dims
# by 32x, so 32x32 would collapse to 1x1 before final pooling. 96x96
# gives a 3x3 final feature map — enough spatial information to learn.

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data" / "mlfp05" / "cifar10"
DATA_DIR.mkdir(parents=True, exist_ok=True)

INPUT_SIZE = 96
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
N_CLASSES = 10
BATCH_SIZE = 128
EPOCHS = 8

CLASS_NAMES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]

# Standard transforms: ImageNet normalisation + resize for ResNet
train_transform = T.Compose(
    [
        T.Resize((INPUT_SIZE, INPUT_SIZE)),
        T.RandomHorizontalFlip(),
        T.RandomCrop(INPUT_SIZE, padding=8),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ]
)
val_transform = T.Compose(
    [
        T.Resize((INPUT_SIZE, INPUT_SIZE)),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ]
)


def load_cifar10() -> tuple[
    torchvision.datasets.CIFAR10,
    torchvision.datasets.CIFAR10,
    DataLoader,
    DataLoader,
]:
    """Load CIFAR-10 with ImageNet-style transforms.

    Returns:
        train_set, val_set, train_loader, val_loader
    """
    train_set = torchvision.datasets.CIFAR10(
        root=str(DATA_DIR),
        train=True,
        download=True,
        transform=train_transform,
    )
    val_set = torchvision.datasets.CIFAR10(
        root=str(DATA_DIR),
        train=False,
        download=True,
        transform=val_transform,
    )

    train_loader = DataLoader(
        train_set,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
    )
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, num_workers=0)

    print(
        f"CIFAR-10 (full): train={len(train_set)}, val={len(val_set)}, "
        f"classes={N_CLASSES}"
    )
    print(f"  Input size: {INPUT_SIZE}x{INPUT_SIZE} (resized for ResNet-18)")
    print(f"  Classes: {CLASS_NAMES}")

    return train_set, val_set, train_loader, val_loader


# ════════════════════════════════════════════════════════════════════════
# KAILASH ENGINE SETUP
# ════════════════════════════════════════════════════════════════════════


async def _setup_engines():
    """Open kailash-ml 1.1.1 tracker + registry. 5-tuple preserved."""
    # Schema-conflict workaround (kailash-ml 1.5.x): ExperimentTracker
    # and ModelRegistry use incompatible _kml_model_versions schemas.
    # Route them to separate sqlite files until upstream fixes the conflict.
    db = "sqlite:///mlfp05_transfer.db"
    registry_db = "sqlite:///mlfp05_transfer_registry.db"
    tracker = await ExperimentTracker.create(store_url=db)
    conn = ConnectionManager(registry_db)
    await conn.initialize()
    registry = ModelRegistry(conn)
    return conn, tracker, "m5_transfer_learning", registry, True


def init_engines() -> tuple[
    ConnectionManager,
    ExperimentTracker,
    str,
    ModelRegistry | None,
    bool,
]:
    """Synchronously set up kailash-ml engines."""
    return asyncio.run(_setup_engines())


# ════════════════════════════════════════════════════════════════════════
# TRAINING HARNESS — shared by all technique files
# ════════════════════════════════════════════════════════════════════════


async def _train_model_async(
    model: nn.Module,
    name: str,
    tracker: ExperimentTracker,
    exp_name: str,
    tr_loader: DataLoader,
    vl_loader: DataLoader,
    epochs: int = EPOCHS,
    lr: float = 1e-3,
) -> tuple[list[float], list[float], list[float]]:
    """Train a model and log everything to ExperimentTracker.

    Returns:
        train_losses, val_accs, train_accs (per-epoch)
    """
    model.to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    n_trainable = sum(p.numel() for p in params)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"\n-- {name} --  trainable params: {n_trainable:,} / {n_total:,}")

    opt = torch.optim.Adam(params, lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    train_losses: list[float] = []
    val_accs: list[float] = []
    train_accs: list[float] = []

    async with tracker.track(experiment=exp_name, run_name=name) as run:
        await run.log_params(
            {
                "model_type": name,
                "trainable_params": str(n_trainable),
                "total_params": str(n_total),
                "epochs": str(epochs),
                "lr": str(lr),
                "batch_size": str(tr_loader.batch_size),
                "dataset_size": str(len(tr_loader.dataset)),
            }
        )

        for epoch in range(epochs):
            # -- Training --
            model.train()
            batch_losses = []
            correct = total = 0
            for xb, yb in tr_loader:
                xb, yb = xb.to(device), yb.to(device)
                opt.zero_grad()
                logits = model(xb)
                loss = F.cross_entropy(logits, yb)
                loss.backward()
                opt.step()
                batch_losses.append(loss.item())
                correct += int((logits.argmax(dim=-1) == yb).sum().item())
                total += int(yb.size(0))
            train_losses.append(float(np.mean(batch_losses)))
            train_accs.append(correct / total)
            scheduler.step()

            # -- Validation --
            model.eval()
            correct = total = 0
            with torch.no_grad():
                for xb, yb in vl_loader:
                    xb, yb = xb.to(device), yb.to(device)
                    preds = model(xb).argmax(dim=-1)
                    correct += int((preds == yb).sum().item())
                    total += int(yb.size(0))
            val_accs.append(correct / total)

            await run.log_metrics(
                {
                    "train_loss": train_losses[-1],
                    "train_acc": train_accs[-1],
                    "val_acc": val_accs[-1],
                },
                step=epoch + 1,
            )

            print(
                f"  epoch {epoch + 1}/{epochs}  "
                f"loss={train_losses[-1]:.4f}  "
                f"train_acc={train_accs[-1]:.3f}  "
                f"val_acc={val_accs[-1]:.3f}"
            )

        await run.log_metrics(
            {
                "final_val_acc": val_accs[-1],
                "best_val_acc": max(val_accs),
                "final_train_loss": train_losses[-1],
            }
        )

    return train_losses, val_accs, train_accs


def train_model(
    model: nn.Module,
    name: str,
    tracker: ExperimentTracker,
    exp_name: str,
    tr_loader: DataLoader,
    vl_loader: DataLoader,
    epochs: int = EPOCHS,
    lr: float = 1e-3,
) -> tuple[list[float], list[float], list[float]]:
    """Sync wrapper -- one asyncio.run per training call."""
    return asyncio.run(
        _train_model_async(
            model,
            name,
            tracker,
            exp_name,
            tr_loader,
            vl_loader,
            epochs,
            lr,
        )
    )


# ════════════════════════════════════════════════════════════════════════
# MODEL REGISTRATION
# ════════════════════════════════════════════════════════════════════════


async def _register_model(
    registry: ModelRegistry,
    name: str,
    model: nn.Module,
    val_acc: float,
    final_loss: float,
):
    """Register a trained model in the ModelRegistry."""
    model_bytes = pickle.dumps(model.state_dict())
    version = await registry.register_model(
        name=name,
        artifact=model_bytes,
        metrics=[
            MetricSpec(name="val_acc", value=val_acc),
            MetricSpec(name="final_loss", value=final_loss),
        ],
    )
    print(f"  Registered {name}: version={version.version}, acc={val_acc:.3f}")
    return version


def register_model(
    registry: ModelRegistry,
    name: str,
    model: nn.Module,
    val_acc: float,
    final_loss: float,
):
    """Sync wrapper for model registration."""
    return asyncio.run(_register_model(registry, name, model, val_acc, final_loss))


# ════════════════════════════════════════════════════════════════════════
# FEATURE EXTRACTION & VISUALISATION HELPERS
# ════════════════════════════════════════════════════════════════════════


def extract_features(
    model: nn.Module,
    loader: DataLoader,
    max_samples: int = 2000,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract features from the penultimate layer (before fc head).

    Works for both ResNet (avgpool hook) and Sequential models.
    """
    model.eval()
    hook_features: list[torch.Tensor] = []
    labels: list[np.ndarray] = []

    def hook_fn(module, inp, out):
        del module, inp  # PyTorch hook protocol args; not used here
        hook_features.append(out.flatten(1).detach().cpu())

    # ResNet: hook into avgpool; Sequential: second-to-last layer
    if hasattr(model, "avgpool"):
        avgpool = model.avgpool
        assert isinstance(
            avgpool, nn.Module
        ), f"expected nn.Module for avgpool, got {type(avgpool).__name__}"
        handle = avgpool.register_forward_hook(hook_fn)
    else:
        assert isinstance(
            model, nn.Sequential
        ), f"hook fallback requires nn.Sequential, got {type(model).__name__}"
        handle = model[-3].register_forward_hook(hook_fn)

    with torch.no_grad():
        collected = 0
        for xb, yb in loader:
            if collected >= max_samples:
                break
            xb = xb.to(device)
            model(xb)
            labels.append(yb.numpy())
            collected += len(yb)

    handle.remove()
    features_np = torch.cat(hook_features, dim=0).numpy()[:max_samples]
    labels_np = np.concatenate(labels)[:max_samples]
    return features_np, labels_np


def compute_tsne(features: np.ndarray, perplexity: int = 30) -> np.ndarray:
    """Run t-SNE dimensionality reduction to 2D."""
    # sklearn 1.5+ renamed n_iter → max_iter in TSNE.__init__
    tsne = TSNE(n_components=2, perplexity=perplexity, max_iter=500, random_state=42)
    return tsne.fit_transform(features)


def cluster_quality(coords: np.ndarray, labels: np.ndarray) -> float:
    """Cluster quality: ratio of intra-class to inter-class distance (lower = better)."""
    intra = []
    centroids = []
    for c in range(N_CLASSES):
        mask = labels == c
        if mask.sum() < 2:
            continue
        pts = coords[mask]
        centroid = pts.mean(axis=0)
        centroids.append(centroid)
        intra.append(np.mean(np.linalg.norm(pts - centroid, axis=1)))
    centroids_arr = np.array(centroids)
    inter = np.mean(
        [
            np.linalg.norm(centroids_arr[i] - centroids_arr[j])
            for i in range(len(centroids_arr))
            for j in range(i + 1, len(centroids_arr))
        ]
    )
    avg_intra = np.mean(intra)
    return float(avg_intra / inter) if inter > 0 else float("inf")


def plot_tsne(
    coords: np.ndarray,
    labels: np.ndarray,
    title: str,
    output_path: str | Path,
) -> None:
    """Create and save a t-SNE scatter plot coloured by class."""
    fig = go.Figure()
    for c in range(N_CLASSES):
        mask = labels == c
        fig.add_trace(
            go.Scatter(
                x=coords[mask, 0],
                y=coords[mask, 1],
                mode="markers",
                name=CLASS_NAMES[c],
                marker=dict(size=4, opacity=0.6),
            )
        )
    fig.update_layout(
        title=title,
        xaxis_title="t-SNE 1",
        yaxis_title="t-SNE 2",
        template="plotly_white",
    )
    fig.write_html(str(output_path))
    print(f"  Saved: {output_path}")


def create_visualizer() -> ModelVisualizer:
    """Return a configured ModelVisualizer instance."""
    return ModelVisualizer()


def save_training_plots(
    viz: ModelVisualizer,
    metrics: dict[str, list[float]],
    output_path: str | Path,
) -> None:
    """Save a training history plot to HTML."""
    fig = viz.training_history(metrics=metrics, x_label="Epoch", y_label="Value")
    fig.write_html(str(output_path))
    print(f"  Saved: {output_path}")


def count_params(model: nn.Module, trainable_only: bool = False) -> int:
    """Count parameters in a model."""
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())
