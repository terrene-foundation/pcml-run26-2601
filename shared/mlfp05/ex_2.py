# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
Shared utilities for MLFP05 Exercise 2 — CNNs on CIFAR-10.

Common infrastructure used by all technique files:
  - CIFAR-10 data loading and normalisation
  - Device detection and precision settings
  - ExperimentTracker / ModelRegistry setup
  - LightningModule wrapper for training
  - Visualisation helpers
"""
from __future__ import annotations

import asyncio
import pickle
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

import pytorch_lightning as pl
import torchvision

from kailash.db import ConnectionManager
from kailash_ml import ExperimentTracker, ModelVisualizer
from kailash_ml import ModelRegistry
from kailash_ml.types import MetricSpec
from shared.kailash_helpers import get_device, setup_environment

# ── Environment & Reproducibility ────────────────────────────────────
setup_environment()
torch.manual_seed(42)
np.random.seed(42)
pl.seed_everything(42, workers=True)

DEVICE = get_device()
# Notebook-safe path resolution: __file__ is undefined when this module is
# inlined into a Colab notebook by scripts/generate_selfcontained_notebook.py.
try:
    _HERE = Path(__file__).resolve()
    REPO_ROOT = _HERE.parents[2]
    ARTIFACT_DIR = _HERE.parent
except NameError:
    REPO_ROOT = Path.cwd()
    ARTIFACT_DIR = Path.cwd()
DATA_DIR = REPO_ROOT / "data" / "mlfp05" / "cifar10"

N_CLASSES = 10
BATCH_SIZE = 128
EPOCHS = 8

# ── CIFAR-10 class labels (used for visualisation and apply sections) ─
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

# Per-channel normalisation statistics (CIFAR-10 population)
CIFAR_MEAN = torch.tensor([0.4914, 0.4822, 0.4465]).view(1, 3, 1, 1)
CIFAR_STD = torch.tensor([0.2470, 0.2435, 0.2616]).view(1, 3, 1, 1)


# ═══════════════════════════════════════════════════════════════════════
# Data Loading
# ═══════════════════════════════════════════════════════════════════════
def load_cifar10() -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    DataLoader,
    DataLoader,
]:
    """Load and normalise the full CIFAR-10 dataset.

    Returns:
        X_train, y_train, X_val, y_val, train_loader, val_loader
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    train_set = torchvision.datasets.CIFAR10(
        root=str(DATA_DIR),
        train=True,
        download=True,
        transform=torchvision.transforms.ToTensor(),
    )
    val_set = torchvision.datasets.CIFAR10(
        root=str(DATA_DIR),
        train=False,
        download=True,
        transform=torchvision.transforms.ToTensor(),
    )

    X_train = torch.stack([train_set[i][0] for i in range(len(train_set))])
    y_train = torch.tensor(
        [train_set[i][1] for i in range(len(train_set))],
        dtype=torch.long,
    )
    X_val = torch.stack([val_set[i][0] for i in range(len(val_set))])
    y_val = torch.tensor(
        [val_set[i][1] for i in range(len(val_set))],
        dtype=torch.long,
    )

    # Per-channel normalisation
    X_train = (X_train - CIFAR_MEAN) / CIFAR_STD
    X_val = (X_val - CIFAR_MEAN) / CIFAR_STD

    train_ds = TensorDataset(X_train, y_train)
    val_ds = TensorDataset(X_val, y_val)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

    print(
        f"CIFAR-10: train {tuple(X_train.shape)}, val {tuple(X_val.shape)}, "
        f"classes={N_CLASSES}: {CLASS_NAMES}"
    )
    return X_train, y_train, X_val, y_val, train_loader, val_loader


# ═══════════════════════════════════════════════════════════════════════
# Kailash Engine Setup
# ═══════════════════════════════════════════════════════════════════════
async def setup_engines(db_name: str = "mlfp05_cnns.db") -> tuple[
    ConnectionManager,
    ExperimentTracker,
    str,
    ModelRegistry | None,
    bool,
]:
    """Initialise ExperimentTracker (kailash-ml 1.1.1 factory) and ModelRegistry.

    Returns:
        conn, tracker, experiment_name, registry, has_registry
    """
    # Schema-conflict workaround (kailash-ml 1.5.x): ExperimentTracker and
    # ModelRegistry use incompatible _kml_model_versions schemas. Route them
    # to separate sqlite files until upstream fixes the conflict.
    db = f"sqlite:///{db_name}"
    registry_db_name = (
        db_name.replace(".db", "_registry.db")
        if db_name.endswith(".db")
        else db_name + "_registry.db"
    )
    registry_db = f"sqlite:///{registry_db_name}"
    tracker = await ExperimentTracker.create(store_url=db)
    conn = ConnectionManager(registry_db)
    await conn.initialize()
    registry = ModelRegistry(conn)
    return conn, tracker, "m5_cnns", registry, True


def init_engines(db_name: str = "mlfp05_cnns.db") -> tuple[
    ConnectionManager,
    ExperimentTracker,
    str,
    ModelRegistry | None,
    bool,
]:
    """Sync wrapper for setup_engines."""
    return asyncio.run(setup_engines(db_name))


# ═══════════════════════════════════════════════════════════════════════
# Lightning Training Infrastructure
# ═══════════════════════════════════════════════════════════════════════
class LitCNN(pl.LightningModule):
    """Lightning wrapper for any nn.Module classifier.

    Tracks per-epoch training loss and validation accuracy for later
    logging to ExperimentTracker and visualisation with ModelVisualizer.
    """

    def __init__(self, model: nn.Module, lr: float = 1e-3):
        super().__init__()
        self.model = model
        self.lr = lr
        self.train_losses: list[float] = []
        self.val_accs: list[float] = []
        self._batch_losses: list[float] = []
        self._val_correct = 0
        self._val_total = 0

    def training_step(self, batch, batch_idx):
        del batch_idx  # Lightning protocol arg; not used here
        x, y = batch
        logits = self.model(x)
        loss = F.cross_entropy(logits, y)
        self._batch_losses.append(loss.item())
        return loss

    def on_train_epoch_end(self):
        if self._batch_losses:
            self.train_losses.append(float(np.mean(self._batch_losses)))
            self._batch_losses = []

    def validation_step(self, batch, batch_idx):
        del batch_idx  # Lightning protocol arg; not used here
        x, y = batch
        logits = self.model(x)
        self._val_correct += int((logits.argmax(dim=-1) == y).sum().item())
        self._val_total += int(y.size(0))

    def on_validation_epoch_end(self):
        if self._val_total > 0:
            self.val_accs.append(self._val_correct / self._val_total)
            self._val_correct = 0
            self._val_total = 0

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.lr)


def get_precision_setting() -> str:
    """Return the optimal Lightning precision string for the current backend.

    Routes through ``kailash_ml.device()`` so MPS / CUDA / ROCm / Intel XPU /
    CPU are all selected by the same policy the rest of the platform uses.
    BackendInfo.precision is the canonical answer ("16-mixed" on Apple MPS
    + Tensor-Core GPUs, "32" on older GPUs and CPU).
    """
    import kailash_ml as km

    backend = km.device()
    print(f"  {backend.backend} detected -- using {backend.precision} precision")
    return backend.precision


def get_accelerator() -> str:
    """Return the Lightning accelerator string for the current backend."""
    import kailash_ml as km

    return km.device().accelerator


PRECISION = get_precision_setting()
ACCELERATOR = get_accelerator()


async def train_model_async(
    model: nn.Module,
    name: str,
    tracker: ExperimentTracker,
    exp_name: str,
    train_loader: DataLoader,
    val_loader: DataLoader,
    lr: float = 1e-3,
    epochs: int = EPOCHS,
) -> tuple[list[float], list[float]]:
    """Train a CNN with Lightning and log metrics to ExperimentTracker.

    Uses the kailash-ml 1.1.1 ``tracker.track(...)`` async context manager.
    On normal exit the run is marked FINISHED; on exception it is marked FAILED.
    """
    lit = LitCNN(model, lr=lr)
    trainer = pl.Trainer(
        max_epochs=epochs,
        accelerator=ACCELERATOR,
        precision=PRECISION,
        enable_progress_bar=False,
        enable_model_summary=False,
        logger=False,
        enable_checkpointing=False,
    )

    async with tracker.track(experiment=exp_name, run_name=name) as run:
        await run.log_params(
            {
                "architecture": name,
                "lr": str(lr),
                "epochs": str(epochs),
                "batch_size": str(BATCH_SIZE),
                "dataset_size": str(len(train_loader.dataset)),
                "precision": PRECISION,
                "accelerator": ACCELERATOR,
            }
        )

        trainer.fit(lit, train_loader, val_loader)

        for epoch_idx, loss in enumerate(lit.train_losses):
            await run.log_metric("train_loss", loss, step=epoch_idx + 1)
        for epoch_idx, acc in enumerate(lit.val_accs):
            await run.log_metric("val_accuracy", acc, step=epoch_idx + 1)

        await run.log_metrics(
            {
                "final_train_loss": lit.train_losses[-1],
                "final_val_accuracy": lit.val_accs[-1],
            }
        )

    print(
        f"  [{name}] final train loss={lit.train_losses[-1]:.4f}  "
        f"val acc={lit.val_accs[-1]:.3f}"
    )
    return lit.train_losses, lit.val_accs


def train_model(
    model: nn.Module,
    name: str,
    tracker: ExperimentTracker,
    exp_name: str,
    train_loader: DataLoader,
    val_loader: DataLoader,
    lr: float = 1e-3,
    epochs: int = EPOCHS,
) -> tuple[list[float], list[float]]:
    """Sync wrapper -- one asyncio.run per training call."""
    return asyncio.run(
        train_model_async(
            model,
            name,
            tracker,
            exp_name,
            train_loader,
            val_loader,
            lr,
            epochs,
        )
    )


# ═══════════════════════════════════════════════════════════════════════
# Model Registration
# ═══════════════════════════════════════════════════════════════════════
async def register_model_async(
    registry: ModelRegistry,
    name: str,
    model: nn.Module,
    final_loss: float,
    final_acc: float,
    epochs: int = EPOCHS,
):
    """Register a trained model with metrics in the ModelRegistry."""
    model_bytes = pickle.dumps(model.state_dict())
    version = await registry.register_model(
        name=name,
        artifact=model_bytes,
        metrics=[
            MetricSpec(name="final_loss", value=final_loss),
            MetricSpec(name="val_accuracy", value=final_acc),
            MetricSpec(name="epochs", value=float(epochs)),
            MetricSpec(name="batch_size", value=float(BATCH_SIZE)),
        ],
    )
    print(f"  Registered {name}: version={version.version}, acc={final_acc:.3f}")
    return version


def register_model(
    registry: ModelRegistry,
    name: str,
    model: nn.Module,
    final_loss: float,
    final_acc: float,
    epochs: int = EPOCHS,
):
    """Sync wrapper for register_model_async."""
    return asyncio.run(
        register_model_async(registry, name, model, final_loss, final_acc, epochs)
    )


# ═══════════════════════════════════════════════════════════════════════
# Visualisation Helpers
# ═══════════════════════════════════════════════════════════════════════
def create_visualizer() -> ModelVisualizer:
    """Return a configured ModelVisualizer instance."""
    return ModelVisualizer()


def save_training_plots(
    viz: ModelVisualizer,
    metrics: dict[str, list[float]],
    output_path: str | Path,
    x_label: str = "Epoch",
    y_label: str = "Value",
) -> None:
    """Save a training history plot to HTML."""
    fig = viz.training_history(
        metrics=metrics,
        x_label=x_label,
        y_label=y_label,
    )
    fig.write_html(str(output_path))
    print(f"  Saved plot: {output_path}")


def count_parameters(model: nn.Module) -> int:
    """Count total trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters())


def denormalise_cifar(img_tensor: torch.Tensor) -> torch.Tensor:
    """Reverse CIFAR-10 normalisation for display.

    Args:
        img_tensor: shape (C, H, W) or (B, C, H, W), normalised

    Returns:
        Tensor with pixel values clipped to [0, 1]
    """
    if img_tensor.dim() == 3:
        mean = CIFAR_MEAN.squeeze(0)
        std = CIFAR_STD.squeeze(0)
    else:
        mean = CIFAR_MEAN
        std = CIFAR_STD
    return (img_tensor * std + mean).clamp(0, 1)
