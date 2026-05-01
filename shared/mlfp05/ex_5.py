# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
Shared utilities for Exercise 5 — GANs and Generative Models.

Infrastructure only: data loading, visualisation helpers, metric computation,
and kailash-ml engine setup. No domain logic or business scenarios.
"""
from __future__ import annotations

import asyncio
import pickle
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import torchvision
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from kailash.db import ConnectionManager
from kailash_ml import ExperimentTracker
from kailash_ml import ModelRegistry
from kailash_ml.types import MetricSpec

from shared.kailash_helpers import get_device, setup_environment

if TYPE_CHECKING:
    from kailash_ml import ModelVersion

# ════════════════════════════════════════════════════════════════════════
# Constants
# ════════════════════════════════════════════════════════════════════════
LATENT_DIM = 64
IMG_DIM = 28 * 28
BATCH_SIZE = 128
try:
    _HERE = Path(__file__).resolve()
    REPO_ROOT = _HERE.parents[2]
    OUTPUT_DIR = _HERE.parent
except NameError:
    REPO_ROOT = Path.cwd()
    OUTPUT_DIR = Path.cwd()
DATA_DIR = REPO_ROOT / "data" / "mlfp05" / "mnist"


# ════════════════════════════════════════════════════════════════════════
# Environment and device
# ════════════════════════════════════════════════════════════════════════
def init_environment() -> torch.device:
    """Set up environment, seeds, and return the compute device."""
    setup_environment()
    torch.manual_seed(42)
    np.random.seed(42)
    device = get_device()
    print(f"Using device: {device}")
    return device


# ════════════════════════════════════════════════════════════════════════
# Data loading
# ════════════════════════════════════════════════════════════════════════
def load_mnist(device: torch.device) -> tuple[torch.Tensor, torch.Tensor, DataLoader]:
    """Load full MNIST (60K), scale to [-1, 1] for tanh generators.

    Returns:
        X_real: (60000, 1, 28, 28) tensor on device, range [-1, 1]
        y_real: (60000,) long tensor on device
        real_loader: DataLoader with batch_size=128, shuffle, drop_last
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    train_set = torchvision.datasets.MNIST(
        root=str(DATA_DIR),
        train=True,
        download=True,
        transform=torchvision.transforms.ToTensor(),
    )

    X_real = torch.stack([train_set[i][0] for i in range(len(train_set))])
    y_real = torch.tensor(
        [train_set[i][1] for i in range(len(train_set))], dtype=torch.long
    )
    X_real = (X_real * 2.0 - 1.0).to(device)
    y_real = y_real.to(device)

    print(
        f"MNIST: {len(X_real)} digits, shape {tuple(X_real.shape[1:])}, "
        f"pixel range [{X_real.min():.2f}, {X_real.max():.2f}]"
    )
    class_dist = ", ".join(f"{c}={int((y_real == c).sum())}" for c in range(10))
    print(f"  class distribution: {class_dist}")

    real_loader = DataLoader(
        TensorDataset(X_real), batch_size=BATCH_SIZE, shuffle=True, drop_last=True
    )
    return X_real, y_real, real_loader


# ════════════════════════════════════════════════════════════════════════
# Kailash engine setup
# ════════════════════════════════════════════════════════════════════════
def setup_engines() -> (
    tuple[ConnectionManager, ExperimentTracker, str, ModelRegistry | None]
):
    """Create ExperimentTracker (kailash-ml 1.1.1 factory) and ModelRegistry.

    Returns:
        conn, tracker, experiment_name, registry
    """

    async def _setup():
        # Schema-conflict workaround (kailash-ml 1.5.x): ExperimentTracker
        # and ModelRegistry use incompatible _kml_model_versions schemas.
        # Route them to separate sqlite files until upstream fixes the conflict.
        db = "sqlite:///mlfp05_gans.db"
        registry_db = "sqlite:///mlfp05_gans_registry.db"
        tracker = await ExperimentTracker.create(store_url=db)
        conn = ConnectionManager(registry_db)
        await conn.initialize()
        registry = ModelRegistry(conn)
        return conn, tracker, "m5_gans", registry

    return asyncio.run(_setup())


async def close_engines(conn: ConnectionManager) -> None:
    """Cleanly shut down the connection manager."""
    await conn.close()


# ════════════════════════════════════════════════════════════════════════
# Generator and Discriminator architectures
# ════════════════════════════════════════════════════════════════════════
class Generator(nn.Module):
    """MLP Generator: z -> 784-d -> (1, 28, 28).

    Uses BatchNorm + LeakyReLU (DCGAN best practices) and Tanh output
    to match the [-1, 1] image scaling.
    """

    def __init__(self, latent_dim: int = LATENT_DIM, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden),
            nn.BatchNorm1d(hidden),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden, hidden * 2),
            nn.BatchNorm1d(hidden * 2),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden * 2, IMG_DIM),
            nn.Tanh(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z).view(-1, 1, 28, 28)


class Discriminator(nn.Module):
    """MLP Discriminator: 28x28 -> scalar logit.

    Dropout prevents D from overfitting to real images (memorising
    instead of learning distributional features).
    """

    def __init__(self, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(IMG_DIM, hidden * 2),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(hidden * 2, hidden),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ════════════════════════════════════════════════════════════════════════
# FID feature extractor
# ════════════════════════════════════════════════════════════════════════
class LeNetFeatureExtractor(nn.Module):
    """CNN feature extractor for FID computation.

    Returns 64-dim feature vectors (analogous to InceptionV3 pool3 layer).
    We use a domain-specific extractor because InceptionV3 expects 299x299 RGB;
    for 28x28 grayscale MNIST a trained LeNet gives more meaningful distances.
    """

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(16, 32, 5, stride=2, padding=2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 64),
            nn.ReLU(),
        )
        self.classifier = nn.Linear(64, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.features(x)


def train_feature_extractor(
    X_real: torch.Tensor,
    y_real: torch.Tensor,
    device: torch.device,
    epochs: int = 5,
) -> LeNetFeatureExtractor:
    """Train the LeNet feature extractor on MNIST for FID computation.

    Returns the trained extractor in eval mode.
    """
    print("\n== Training feature extractor (for FID + mode coverage) ==")
    extractor = LeNetFeatureExtractor().to(device)
    opt = torch.optim.Adam(extractor.parameters(), lr=1e-3)
    X_01 = (X_real + 1.0) / 2.0  # [0, 1] for classifier

    for epoch in range(epochs):
        losses = []
        for xb, yb in DataLoader(
            TensorDataset(X_01, y_real), batch_size=256, shuffle=True
        ):
            loss = F.cross_entropy(extractor(xb), yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item())
        with torch.no_grad():
            acc = (extractor(X_01[:10000]).argmax(-1) == y_real[:10000]).float().mean()
        print(f"  epoch {epoch+1}/{epochs}  loss={np.mean(losses):.3f}  acc={acc:.3f}")

    extractor.eval()
    return extractor


# ════════════════════════════════════════════════════════════════════════
# FID computation
# ════════════════════════════════════════════════════════════════════════
def compute_fid(
    extractor: LeNetFeatureExtractor,
    real: torch.Tensor,
    generated: torch.Tensor,
) -> float:
    """Frechet Inception Distance between real and generated images.

    FID = ||mu_r - mu_g||^2 + Tr(Sigma_r + Sigma_g - 2*sqrt(Sigma_r @ Sigma_g))

    Lower FID = closer to real distribution = better generator.
    Uses eigendecomposition (no scipy dependency).
    """
    extractor.eval()

    def _extract(images: torch.Tensor) -> np.ndarray:
        feats = []
        with torch.no_grad():
            for i in range(0, len(images), 512):
                feats.append(
                    extractor.extract_features(images[i : i + 512]).cpu().numpy()
                )
        return np.concatenate(feats)

    rf, gf = _extract(real), _extract(generated)
    mu_r, mu_g = rf.mean(0), gf.mean(0)
    sig_r, sig_g = np.cov(rf, rowvar=False), np.cov(gf, rowvar=False)

    diff = mu_r - mu_g
    product = sig_r @ sig_g
    eigvals, eigvecs = np.linalg.eigh(product)
    eigvals = np.maximum(eigvals, 0.0)
    sqrt_prod = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T

    return float(diff @ diff + np.trace(sig_r + sig_g - 2 * sqrt_prod))


# ════════════════════════════════════════════════════════════════════════
# Mode coverage diagnostic
# ════════════════════════════════════════════════════════════════════════
def mode_coverage(
    G: Generator,
    classifier: LeNetFeatureExtractor,
    device: torch.device,
    n: int = 5000,
) -> tuple[int, dict[int, int], float]:
    """Measure mode coverage: how many of the 10 digit classes the generator produces.

    Returns:
        n_classes: number of unique classes generated
        per_class_counts: dict mapping class -> count
        shannon_entropy: diversity measure (max = log2(10) = 3.32 for uniform)
    """
    G.eval()
    classifier.eval()
    with torch.no_grad():
        fake_01 = (G(torch.randn(n, LATENT_DIM, device=device)) + 1) / 2
        preds = classifier(fake_01).argmax(-1).cpu().numpy()
    unique, counts = np.unique(preds, return_counts=True)
    probs = counts / counts.sum()
    entropy = float(-np.sum(probs * np.log2(probs + 1e-10)))
    return int(len(unique)), {int(k): int(v) for k, v in zip(unique, counts)}, entropy


# ════════════════════════════════════════════════════════════════════════
# Visualisation helpers
# ════════════════════════════════════════════════════════════════════════
def plot_image_grid(
    images: torch.Tensor,
    nrow: int = 8,
    ncol: int = 8,
    title: str = "Generated Images",
    save_path: str | None = None,
) -> Figure:
    """Plot an 8x8 grid of generated images.

    Args:
        images: (N, 1, 28, 28) tensor in [-1, 1] range
        nrow, ncol: grid dimensions
        title: figure title
        save_path: optional path to save the figure
    """
    n = min(nrow * ncol, len(images))
    fig, axes = plt.subplots(nrow, ncol, figsize=(12, 12))
    fig.suptitle(title, fontsize=16, fontweight="bold")

    for i in range(nrow * ncol):
        ax = axes[i // ncol][i % ncol]
        if i < n:
            img = images[i].detach().cpu().squeeze().numpy()
            img = (img + 1) / 2  # [-1,1] -> [0,1]
            ax.imshow(img, cmap="gray", vmin=0, vmax=1)
        ax.axis("off")

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {save_path}")
    return fig


def plot_latent_interpolation(
    G: Generator,
    device: torch.device,
    n_steps: int = 10,
    n_rows: int = 5,
    title: str = "Latent Space Interpolation",
    save_path: str | None = None,
) -> Figure:
    """Interpolate between pairs of random latent vectors.

    Shows smooth transitions between generated images — evidence that
    the generator has learned a continuous manifold, not memorised digits.
    """
    G.eval()
    fig, axes = plt.subplots(n_rows, n_steps, figsize=(n_steps * 1.2, n_rows * 1.4))
    fig.suptitle(title, fontsize=14, fontweight="bold")

    with torch.no_grad():
        for row in range(n_rows):
            z1 = torch.randn(1, LATENT_DIM, device=device)
            z2 = torch.randn(1, LATENT_DIM, device=device)
            for col in range(n_steps):
                alpha = col / (n_steps - 1)
                z = (1 - alpha) * z1 + alpha * z2
                img = G(z).squeeze().cpu().numpy()
                img = (img + 1) / 2
                axes[row][col].imshow(img, cmap="gray", vmin=0, vmax=1)
                axes[row][col].axis("off")

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {save_path}")
    return fig


def plot_training_progression(
    G: Generator,
    device: torch.device,
    epoch_snapshots: dict[int, dict],
    title: str = "Training Progression",
    save_path: str | None = None,
) -> Figure:
    """Plot generated images at different training epochs.

    Shows how generation quality improves over training — from random
    noise to recognisable digits.

    Args:
        epoch_snapshots: dict of {epoch: state_dict} captured during training
    """
    n_snapshots = len(epoch_snapshots)
    n_samples = 8
    fig, axes = plt.subplots(
        n_snapshots, n_samples, figsize=(n_samples * 1.4, n_snapshots * 1.6)
    )
    fig.suptitle(title, fontsize=14, fontweight="bold")

    fixed_z = torch.randn(n_samples, LATENT_DIM, device=device)

    for row, (epoch, state_dict) in enumerate(sorted(epoch_snapshots.items())):
        G.load_state_dict(state_dict)
        G.eval()
        with torch.no_grad():
            imgs = G(fixed_z)
        for col in range(n_samples):
            ax = axes[row][col] if n_snapshots > 1 else axes[col]
            img = imgs[col].squeeze().cpu().numpy()
            img = (img + 1) / 2
            ax.imshow(img, cmap="gray", vmin=0, vmax=1)
            ax.axis("off")
            if col == 0:
                ax.set_ylabel(f"Epoch {epoch}", fontsize=10, rotation=0, labelpad=50)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {save_path}")
    return fig


def plot_loss_curves(
    g_losses: list[float],
    d_losses: list[float],
    title: str = "Training Dynamics",
    g_label: str = "Generator",
    d_label: str = "Discriminator",
    save_path: str | None = None,
) -> Figure:
    """Plot G vs D loss curves across epochs."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(title, fontsize=14, fontweight="bold")

    epochs = range(1, len(g_losses) + 1)

    # Individual losses
    ax1.plot(epochs, g_losses, "b-", linewidth=2, label=g_label)
    ax1.plot(epochs, d_losses, "r-", linewidth=2, label=d_label)
    ax1.set_xlabel("Epoch", fontsize=12)
    ax1.set_ylabel("Loss", fontsize=12)
    ax1.set_title("G vs D Loss", fontsize=13)
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)

    # G/D ratio — healthy GAN has ratio near 1
    ratio = [g / (d + 1e-8) for g, d in zip(g_losses, d_losses)]
    ax2.plot(epochs, ratio, "g-", linewidth=2)
    ax2.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5, label="Balanced (1.0)")
    ax2.set_xlabel("Epoch", fontsize=12)
    ax2.set_ylabel("G/D Loss Ratio", fontsize=12)
    ax2.set_title("Training Balance", fontsize=13)
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {save_path}")
    return fig


# ════════════════════════════════════════════════════════════════════════
# Model registration helper
# ════════════════════════════════════════════════════════════════════════
def register_generator(
    registry: ModelRegistry | None,
    name: str,
    model: Generator,
    fid: float,
    coverage: int,
    entropy: float,
) -> "ModelVersion | None":
    """Register a trained generator in ModelRegistry with quality metrics."""
    if registry is None:
        print(f"  ModelRegistry not available — skipping {name}")
        return None

    async def _register():
        ver = await registry.register_model(
            name=f"m5_{name}",
            artifact=pickle.dumps(model.state_dict()),
            metrics=[
                MetricSpec(name="fid_score", value=fid),
                MetricSpec(name="mode_coverage", value=float(coverage)),
                MetricSpec(name="class_entropy", value=entropy),
            ],
        )
        print(
            f"  Registered {name}: v={ver.version}, FID={fid:.2f}, "
            f"coverage={coverage}/10"
        )
        return ver

    return asyncio.run(_register())
