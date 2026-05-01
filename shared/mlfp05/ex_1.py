# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
Shared infrastructure for Exercise 1 — The Complete Autoencoder Family.

Contains: data loading, visualisation helpers, training loop, engine setup.
Technique-specific code does NOT belong here.
"""
from __future__ import annotations

import asyncio
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import torchvision

from kailash.db import ConnectionManager
from kailash_ml import ExperimentTracker
from kailash_ml import ModelRegistry

from shared.kailash_helpers import get_device, setup_environment

# ════════════════════════════════════════════════════════════════════════
# ENVIRONMENT SETUP
# ════════════════════════════════════════════════════════════════════════

setup_environment()
torch.manual_seed(42)
np.random.seed(42)
device = get_device()

# Output directory for all visualisation artifacts
OUTPUT_DIR = Path("outputs") / "ex1_autoencoders"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ════════════════════════════════════════════════════════════════════════
# DATA LOADING — Fashion-MNIST (full 60K)
# ════════════════════════════════════════════════════════════════════════

REPO_ROOT = Path.cwd()
DATA_DIR = REPO_ROOT / "data" / "mlfp05" / "fashion_mnist"
DATA_DIR.mkdir(parents=True, exist_ok=True)

INPUT_DIM = 28 * 28
LATENT_DIM = 16
EPOCHS = 10


def load_fashion_mnist() -> (
    tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, DataLoader, DataLoader
    ]
):
    """Load Fashion-MNIST and return flat/image tensors + loaders.

    Returns:
        (X_flat, X_test_flat, X_img, X_test_img, flat_loader, img_loader)
    """
    train_set = torchvision.datasets.FashionMNIST(
        root=str(DATA_DIR),
        train=True,
        download=True,
        transform=torchvision.transforms.ToTensor(),
    )
    test_set = torchvision.datasets.FashionMNIST(
        root=str(DATA_DIR),
        train=False,
        download=True,
        transform=torchvision.transforms.ToTensor(),
    )

    X_img = torch.stack([train_set[i][0] for i in range(len(train_set))])
    X_img = X_img.to(device).float()
    X_flat = X_img.reshape(len(X_img), -1)

    X_test_img = torch.stack([test_set[i][0] for i in range(len(test_set))])
    X_test_img = X_test_img.to(device).float()
    X_test_flat = X_test_img.reshape(len(X_test_img), -1)

    flat_loader = DataLoader(TensorDataset(X_flat), batch_size=256, shuffle=True)
    img_loader = DataLoader(TensorDataset(X_img), batch_size=256, shuffle=True)

    print(
        f"Fashion-MNIST loaded: {len(X_img)} train + {len(X_test_img)} test images, "
        f"shape {tuple(X_img.shape[1:])}, pixel range [{X_img.min():.2f}, {X_img.max():.2f}]"
    )

    return X_flat, X_test_flat, X_img, X_test_img, flat_loader, img_loader


def get_fashion_mnist_labels() -> tuple[torch.Tensor, torch.Tensor]:
    """Return train and test label tensors."""
    train_set = torchvision.datasets.FashionMNIST(
        root=str(DATA_DIR), train=True, download=True
    )
    test_set = torchvision.datasets.FashionMNIST(
        root=str(DATA_DIR), train=False, download=True
    )
    train_labels = torch.tensor([train_set[i][1] for i in range(len(train_set))])
    test_labels = torch.tensor([test_set[i][1] for i in range(len(test_set))])
    return train_labels, test_labels


# ════════════════════════════════════════════════════════════════════════
# KAILASH ENGINE SETUP
# ════════════════════════════════════════════════════════════════════════


async def _setup_engines():
    """Open kailash-ml 1.1.1 tracker + registry. 5-tuple preserved for callers."""
    # Schema-conflict workaround (kailash-ml 1.5.x): ExperimentTracker
    # and ModelRegistry use incompatible _kml_model_versions schemas.
    # Route them to separate sqlite files until upstream fixes the conflict.
    db = "sqlite:///mlfp05_autoencoders.db"
    registry_db = "sqlite:///mlfp05_autoencoders_registry.db"
    tracker = await ExperimentTracker.create(store_url=db)
    conn = ConnectionManager(registry_db)
    await conn.initialize()
    registry = ModelRegistry(conn)
    return conn, tracker, "m5_autoencoders", registry, True


def setup_engines() -> tuple:
    """Synchronously set up kailash-ml engines."""
    return asyncio.run(_setup_engines())


# ════════════════════════════════════════════════════════════════════════
# VISUALISATION UTILITIES — "Seeing Is Believing"
# ════════════════════════════════════════════════════════════════════════


def show_reconstruction(model, test_data, title, n=10, is_conv=False):
    """Show original vs reconstructed images side by side."""
    model.eval()
    with torch.no_grad():
        x = test_data[:n].to(device)
        result = model(x)
        x_hat = result[0]

    fig, axes = plt.subplots(2, n, figsize=(15, 3))
    fig.suptitle(title, fontsize=14, fontweight="bold")

    for i in range(n):
        if is_conv:
            orig = x[i].cpu().squeeze()
            recon = x_hat[i].cpu().squeeze()
        else:
            orig = x[i].cpu().reshape(28, 28)
            recon = x_hat[i].cpu().reshape(28, 28)

        axes[0, i].imshow(orig, cmap="gray")
        axes[0, i].axis("off")
        if i == 0:
            axes[0, i].set_title("Original", fontsize=9)

        axes[1, i].imshow(recon.clamp(0, 1), cmap="gray")
        axes[1, i].axis("off")
        if i == 0:
            axes[1, i].set_title("Reconstructed", fontsize=9)

    plt.tight_layout()
    fname = (
        OUTPUT_DIR
        / f"ex1_{title.lower().replace(' ', '_').replace('(', '').replace(')', '')}.png"
    )
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"  Saved: {fname}")


def show_denoising_grid(model, clean_data, title, n=10, sigma=0.3):
    """3-row grid: original, noisy input, cleaned output."""
    model.eval()
    with torch.no_grad():
        clean = clean_data[:n].to(device)
        noisy = torch.clamp(clean + sigma * torch.randn_like(clean), 0.0, 1.0)
        result = model(noisy)
        cleaned = result[0]

    fig, axes = plt.subplots(3, n, figsize=(15, 4.5))
    fig.suptitle(title, fontsize=14, fontweight="bold")
    row_labels = ["Original", "Noisy Input", "Cleaned Output"]

    for i in range(n):
        for row, data in enumerate([clean, noisy, cleaned]):
            img = data[i].cpu().reshape(28, 28)
            axes[row, i].imshow(img.clamp(0, 1), cmap="gray")
            axes[row, i].axis("off")
            if i == 0:
                axes[row, i].set_title(row_labels[row], fontsize=9)

    plt.tight_layout()
    fname = OUTPUT_DIR / "ex1_denoising_ae.png"
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"  Saved: {fname}")


def show_activation_sparsity(model, test_data, title="Sparse AE Activations"):
    """Histogram of hidden-layer activations showing sparsity."""
    model.eval()
    with torch.no_grad():
        x = test_data[:1000].to(device)
        h = model.encoder(x)

    activations = h.cpu().numpy().flatten()

    _, ax = plt.subplots(1, 1, figsize=(8, 4))
    ax.hist(activations, bins=100, color="steelblue", edgecolor="white", alpha=0.8)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("Activation Value")
    ax.set_ylabel("Frequency")
    pct_near_zero = (np.abs(activations) < 0.1).mean() * 100
    ax.annotate(
        f"{pct_near_zero:.1f}% of activations near zero",
        xy=(0.95, 0.95),
        xycoords="axes fraction",
        ha="right",
        va="top",
        fontsize=11,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow"),
    )
    plt.tight_layout()
    fname = OUTPUT_DIR / "ex1_sparse_activations.png"
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"  Saved: {fname}")


def show_latent_interpolation(model, test_data, title, n_steps=10, is_conv=False):
    """Morph between two images via latent space interpolation."""
    model.eval()
    with torch.no_grad():
        x1 = test_data[0:1].to(device)
        x2 = test_data[5:6].to(device)
        z1 = model.encoder(x1)
        z2 = model.encoder(x2)

        alphas = torch.linspace(0, 1, n_steps).to(device)
        interpolated = []
        for alpha in alphas:
            z = (1 - alpha) * z1 + alpha * z2
            x_hat = model.decoder(z)
            interpolated.append(x_hat)

    fig, axes = plt.subplots(1, n_steps + 2, figsize=(16, 2))
    fig.suptitle(title, fontsize=13, fontweight="bold")

    src_img = x1[0].cpu().reshape(28, 28) if not is_conv else x1[0].cpu().squeeze()
    axes[0].imshow(src_img, cmap="gray")
    axes[0].set_title("Start", fontsize=8)
    axes[0].axis("off")

    for i, x_hat in enumerate(interpolated):
        img = x_hat[0].cpu()
        img = img.reshape(28, 28) if not is_conv else img.squeeze()
        axes[i + 1].imshow(img.clamp(0, 1), cmap="gray")
        axes[i + 1].set_title(f"{alphas[i]:.1f}", fontsize=7)
        axes[i + 1].axis("off")

    tgt_img = x2[0].cpu().reshape(28, 28) if not is_conv else x2[0].cpu().squeeze()
    axes[-1].imshow(tgt_img, cmap="gray")
    axes[-1].set_title("End", fontsize=8)
    axes[-1].axis("off")

    plt.tight_layout()
    fname = OUTPUT_DIR / f"ex1_{title.lower().replace(' ', '_')}.png"
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"  Saved: {fname}")


def show_generated_samples(model, title="VAE Generated Samples", grid_size=8):
    """Grid of images sampled from the VAE's learned prior N(0, I)."""
    model.eval()
    n = grid_size * grid_size
    with torch.no_grad():
        samples = model.sample(n).cpu()

    fig, axes = plt.subplots(grid_size, grid_size, figsize=(10, 10))
    fig.suptitle(title, fontsize=14, fontweight="bold")
    for i in range(grid_size):
        for j in range(grid_size):
            idx = i * grid_size + j
            axes[i, j].imshow(samples[idx].reshape(28, 28).clamp(0, 1), cmap="gray")
            axes[i, j].axis("off")
    plt.tight_layout()
    fname = OUTPUT_DIR / "ex1_vae_generated_samples.png"
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"  Saved: {fname}")


def show_latent_traversal(
    model, test_data, title="VAE Latent Traversal", n_dims=5, n_steps=11
):
    """Vary one latent dimension at a time and observe what changes."""
    model.eval()
    with torch.no_grad():
        x = test_data[0:1].to(device)
        mu, _ = model.encode(x)
        base_z = mu.clone()

    traversal_range = torch.linspace(-3, 3, n_steps)
    fig, axes = plt.subplots(n_dims, n_steps, figsize=(14, n_dims * 1.4))
    fig.suptitle(title, fontsize=14, fontweight="bold")

    for dim in range(n_dims):
        for step_idx, val in enumerate(traversal_range):
            z = base_z.clone()
            z[0, dim] = val
            with torch.no_grad():
                x_hat = model.decoder(z)
            img = x_hat[0].cpu().reshape(28, 28).clamp(0, 1)
            axes[dim, step_idx].imshow(img, cmap="gray")
            axes[dim, step_idx].axis("off")
            if dim == 0:
                axes[dim, step_idx].set_title(f"z={val:.1f}", fontsize=7)
        axes[dim, 0].set_ylabel(f"dim {dim}", fontsize=8, rotation=0, labelpad=30)

    plt.tight_layout()
    fname = OUTPUT_DIR / "ex1_vae_latent_traversal.png"
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"  Saved: {fname}")


def show_timeseries_reconstruction(model, test_data, title, n_series=4):
    """Overlay original vs reconstructed time series."""
    model.eval()
    with torch.no_grad():
        x = test_data[:n_series].to(device)
        x_hat, _ = model(x)

    fig, axes = plt.subplots(n_series, 1, figsize=(14, 3 * n_series))
    if n_series == 1:
        axes = [axes]
    fig.suptitle(title, fontsize=14, fontweight="bold")

    for i in range(n_series):
        orig = x[i].cpu().numpy()
        recon = x_hat[i].cpu().numpy()
        t = np.arange(len(orig))

        axes[i].plot(t, orig, "b-", linewidth=1.5, label="Original", alpha=0.8)
        axes[i].plot(t, recon, "r--", linewidth=1.5, label="Reconstructed", alpha=0.8)
        axes[i].set_ylabel(f"Series {i + 1}")
        axes[i].legend(loc="upper right", fontsize=8)
        axes[i].grid(True, alpha=0.3)

    axes[-1].set_xlabel("Time Step")
    plt.tight_layout()
    fname = OUTPUT_DIR / "ex1_recurrent_ae_timeseries.png"
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"  Saved: {fname}")


# ════════════════════════════════════════════════════════════════════════
# TRAINING LOOP — shared by all variants
# ════════════════════════════════════════════════════════════════════════

# Collect results across variants (populated by train_variant)
all_losses: dict[str, list[float]] = {}
all_models: dict[str, nn.Module] = {}


async def _train_variant_async(
    tracker: ExperimentTracker,
    exp_name: str,
    model: nn.Module,
    name: str,
    loader: DataLoader,
    loss_fn,
    epochs: int = EPOCHS,
    lr: float = 1e-3,
    extra_params: dict | None = None,
) -> list[float]:
    """Universal training loop for any AE variant."""
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    losses: list[float] = []

    params = {
        "model_type": name,
        "latent_dim": str(LATENT_DIM),
        "epochs": str(epochs),
        "lr": str(lr),
        "dataset_size": str(len(loader.dataset)),
        "batch_size": str(loader.batch_size),
    }
    if extra_params:
        params.update(extra_params)

    async with tracker.track(experiment=exp_name, run_name=name) as run:
        await run.log_params(params)

        for epoch in range(epochs):
            batch_losses = []
            for (xb,) in loader:
                opt.zero_grad()
                loss, _ = loss_fn(model, xb)
                loss.backward()
                opt.step()
                batch_losses.append(loss.item())
            epoch_loss = float(np.mean(batch_losses))
            losses.append(epoch_loss)
            await run.log_metric("loss", epoch_loss, step=epoch + 1)
            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(f"  [{name}] epoch {epoch + 1}/{epochs}  loss={epoch_loss:.4f}")
        await run.log_metric("final_loss", losses[-1])

    return losses


def train_variant(
    tracker: ExperimentTracker,
    exp_name: str,
    model: nn.Module,
    name: str,
    loader: DataLoader,
    loss_fn,
    epochs: int = EPOCHS,
    lr: float = 1e-3,
    extra_params: dict | None = None,
) -> list[float]:
    """Sync wrapper for training with ExperimentTracker integration."""
    losses = asyncio.run(
        _train_variant_async(
            tracker, exp_name, model, name, loader, loss_fn, epochs, lr, extra_params
        )
    )
    all_losses[name] = losses
    all_models[name] = model
    return losses


# ════════════════════════════════════════════════════════════════════════
# MODEL REGISTRATION
# ════════════════════════════════════════════════════════════════════════


async def _register_model(
    registry: ModelRegistry,
    name: str,
    model: nn.Module,
    final_loss: float,
):
    """Register a single model variant."""
    from kailash_ml.types import MetricSpec

    model_bytes = pickle.dumps(model.state_dict())
    version = await registry.register_model(
        name=f"m5_{name}",
        artifact=model_bytes,
        metrics=[
            MetricSpec(name="final_loss", value=final_loss),
            MetricSpec(name="latent_dim", value=float(LATENT_DIM)),
            MetricSpec(name="epochs", value=float(EPOCHS)),
        ],
    )
    print(f"  Registered {name}: version={version.version}, loss={final_loss:.4f}")
    return version


def register_model(
    registry: ModelRegistry, name: str, model: nn.Module, final_loss: float
):
    """Sync wrapper for model registration."""
    return asyncio.run(_register_model(registry, name, model, final_loss))
