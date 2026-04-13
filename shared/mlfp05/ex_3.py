# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# MLFP05 — Exercise 3: Shared Utilities
# ════════════════════════════════════════════════════════════════════════
#
# Common data loading, windowing, training, visualisation, and experiment
# tracking utilities shared across all technique files in this exercise.
#
# This module is NOT meant to be run standalone — import it from the
# technique files (01_vanilla_rnn.py, 02_lstm.py, etc.).
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import asyncio
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from shared.kailash_helpers import get_device, setup_environment

from kailash.db import ConnectionManager
from kailash_ml import ModelVisualizer
from kailash_ml.engines.experiment_tracker import ExperimentTracker
from kailash_ml.engines.model_registry import ModelRegistry

# ── Constants ───────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data" / "mlfp05" / "stocks"
OUTPUT_DIR = REPO_ROOT / "outputs" / "mlfp05" / "ex_3"

TICKERS = {
    "^STI": "Straits Times Index",
    "DBS.SI": "DBS Group",
    "9988.HK": "Alibaba HK",
    "AAPL": "Apple",
    "005930.KS": "Samsung",
    "7203.T": "Toyota",
}

SEQ_LEN = 20  # 20-day lookback (4 trading weeks)
FORECAST_HORIZON = 5  # predict next 5 days
FEATURES = ["Close", "High", "Low", "Volume"]
HIDDEN_DIM = 64
EPOCHS = 15
LR = 1e-3
CLIP = 1.0
BATCH_SIZE = 64

def init_environment() -> torch.device:
    """Set up environment, seeds, device, and output directories."""
    setup_environment()
    torch.manual_seed(42)
    np.random.seed(42)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    device = get_device()
    print(f"Using device: {device}")
    return device

# ── Data Loading ────────────────────────────────────────────────────────
def fetch_ticker(symbol: str) -> pl.DataFrame:
    """Download daily OHLCV bars from yfinance, return polars DataFrame."""
    import yfinance as yf

    df = yf.download(
        symbol, start="2010-01-01", end="2024-12-31", progress=False, auto_adjust=True
    )
    if df is None or len(df) == 0:
        raise RuntimeError(f"yfinance returned empty frame for {symbol}")
    df = df.copy()
    df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
    return pl.from_pandas(df.reset_index())

def load_or_fetch(symbol: str) -> tuple[pl.DataFrame | None, str]:
    """Load from parquet cache, or download and cache."""
    cache = DATA_DIR / f"{symbol.replace('^', '').replace('.', '_')}.parquet"
    if cache.exists():
        return pl.read_parquet(cache), "cache"
    try:
        df = fetch_ticker(symbol)
        df.write_parquet(cache)
        return df, "yfinance"
    except Exception as exc:
        print(f"  {symbol} unavailable ({type(exc).__name__}: {exc})")
        return None, "failed"

def load_stock_data() -> tuple[dict[str, pl.DataFrame], str, pl.DataFrame]:
    """Load all tickers and return (stock_data, primary_symbol, primary_df)."""
    stock_data: dict[str, pl.DataFrame] = {}
    for symbol, name in TICKERS.items():
        df, source = load_or_fetch(symbol)
        if df is not None:
            stock_data[symbol] = df
            print(f"  {symbol} ({name}): {len(df)} days [{source}]")

    if "^STI" not in stock_data and "AAPL" not in stock_data:
        raise RuntimeError("Need at least ^STI or AAPL data to proceed")

    primary = "^STI" if "^STI" in stock_data else "AAPL"
    primary_df = stock_data[primary]
    print(
        f"\nPrimary: {primary} -- {len(primary_df)} days, "
        f"{primary_df['Date'].min()} -> {primary_df['Date'].max()}"
    )
    return stock_data, primary, primary_df

# ── Windowed Datasets ───────────────────────────────────────────────────
def build_dataset(
    df: pl.DataFrame,
    seq_len: int = SEQ_LEN,
    horizon: int = FORECAST_HORIZON,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    """Build (seq_len window) -> (next horizon closes) arrays with z-score normalisation.

    Returns: X, y, mean, std, n_train_windows
    """
    data = df.select(FEATURES).to_numpy().astype(np.float32)
    n = len(data)
    split_n = int(0.8 * n)
    train_data = data[:split_n]
    mean = train_data.mean(axis=0, keepdims=True)
    std = train_data.std(axis=0, keepdims=True) + 1e-8
    data_norm = (data - mean) / std

    n_windows = n - seq_len - horizon + 1
    X = np.stack([data_norm[i : i + seq_len] for i in range(n_windows)])
    y = np.stack(
        [data_norm[i + seq_len : i + seq_len + horizon, 0] for i in range(n_windows)]
    )
    split_idx = split_n - seq_len
    return X.astype(np.float32), y.astype(np.float32), mean, std, split_idx

def prepare_dataloaders(
    primary_df: pl.DataFrame,
    device: torch.device,
) -> tuple[
    DataLoader,
    DataLoader,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    np.ndarray,
    np.ndarray,
    int,
    int,
]:
    """Build train/val dataloaders and return raw tensors plus normalisation stats.

    Returns: train_loader, val_loader, X_train_t, y_train_t, X_val_t, y_val_t,
             norm_mean, norm_std, n_train_w, n_features
    """
    X_all, y_all, norm_mean, norm_std, n_train_w = build_dataset(primary_df)
    print(
        f"Built {len(X_all)} windows (seq_len={SEQ_LEN}, horizon={FORECAST_HORIZON}); "
        f"train {n_train_w}, val {len(X_all) - n_train_w}"
    )

    X_train_t = torch.from_numpy(X_all[:n_train_w]).to(device)
    y_train_t = torch.from_numpy(y_all[:n_train_w]).to(device)
    X_val_t = torch.from_numpy(X_all[n_train_w:]).to(device)
    y_val_t = torch.from_numpy(y_all[n_train_w:]).to(device)
    print(f"  X_train: {tuple(X_train_t.shape)}  y_train: {tuple(y_train_t.shape)}")
    print(f"  X_val:   {tuple(X_val_t.shape)}    y_val:   {tuple(y_val_t.shape)}")

    train_loader = DataLoader(
        TensorDataset(X_train_t, y_train_t), batch_size=BATCH_SIZE, shuffle=True
    )
    val_loader = DataLoader(TensorDataset(X_val_t, y_val_t), batch_size=BATCH_SIZE)
    n_features = X_train_t.shape[-1]

    return (
        train_loader,
        val_loader,
        X_train_t,
        y_train_t,
        X_val_t,
        y_val_t,
        norm_mean,
        norm_std,
        n_train_w,
        n_features,
    )

# ── Experiment Tracking ─────────────────────────────────────────────────
async def _setup_engines(
    primary: str,
    experiment_suffix: str = "",
) -> tuple[ConnectionManager, ExperimentTracker, str, ModelRegistry | None, bool]:
    """Create ExperimentTracker and ModelRegistry for a technique file."""
    conn = ConnectionManager("sqlite:///mlfp05_rnns.db")
    await conn.initialize()

    tracker = ExperimentTracker(conn)
    exp_name = await tracker.create_experiment(
        name=(
            f"m5_rnns_{experiment_suffix}"
            if experiment_suffix
            else "m5_rnns_sequence_models"
        ),
        description=(
            f"RNN variant on {primary} stock data. "
            f"Multi-step forecasting (next {FORECAST_HORIZON} days)."
        ),
    )

    try:
        registry = ModelRegistry(conn)
        has_registry = True
    except Exception as e:
        registry = None
        has_registry = False
        print(f"  Note: ModelRegistry setup skipped ({e})")

    return conn, tracker, exp_name, registry, has_registry

def setup_engines(
    primary: str,
    experiment_suffix: str = "",
) -> tuple[ConnectionManager, ExperimentTracker, str, ModelRegistry | None, bool]:
    """Sync wrapper for engine setup."""
    return asyncio.run(_setup_engines(primary, experiment_suffix))

# ── Training Harness ────────────────────────────────────────────────────
def compute_gradient_norm(model: nn.Module) -> float:
    """Compute the total L2 norm of all gradients (before clipping)."""
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total_norm += p.grad.data.norm(2).item() ** 2
    return total_norm**0.5

def _predict(model: nn.Module, x: torch.Tensor, attn: bool = False) -> torch.Tensor:
    """Forward pass, handling attention models that return a tuple."""
    out = model(x)
    return out[0] if attn else out

async def _train_model_async(
    model: nn.Module,
    name: str,
    tracker: ExperimentTracker,
    exp_name: str,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int = EPOCHS,
    lr: float = LR,
    clip: float = CLIP,
    attn: bool = False,
) -> dict[str, Any]:
    """Train with gradient tracking, log to ExperimentTracker."""
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    train_losses, val_losses, gradient_norms = [], [], []
    n_params = sum(p.numel() for p in model.parameters())

    async with tracker.run(experiment_name=exp_name, run_name=name) as ctx:
        await ctx.log_params(
            {
                "model_type": name,
                "hidden_dim": str(HIDDEN_DIM),
                "seq_len": str(SEQ_LEN),
                "forecast_horizon": str(FORECAST_HORIZON),
                "epochs": str(epochs),
                "lr": str(lr),
                "clip_norm": str(clip),
                "n_params": str(n_params),
            }
        )
        print(f"  [{name}] {n_params:,} parameters")

        for epoch in range(epochs):
            model.train()
            b_losses, e_grads = [], []
            for xb, yb in train_loader:
                opt.zero_grad()
                loss = F.mse_loss(_predict(model, xb, attn), yb)
                loss.backward()
                e_grads.append(compute_gradient_norm(model))
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip)
                opt.step()
                b_losses.append(loss.item())

            tl, gn = float(np.mean(b_losses)), float(np.mean(e_grads))
            train_losses.append(tl)
            gradient_norms.append(gn)

            model.eval()
            with torch.no_grad():
                vl = float(
                    np.mean(
                        [
                            F.mse_loss(_predict(model, xb, attn), yb).item()
                            for xb, yb in val_loader
                        ]
                    )
                )
            val_losses.append(vl)

            await ctx.log_metrics(
                {"train_loss": tl, "val_loss": vl, "gradient_norm": gn},
                step=epoch + 1,
            )
            print(
                f"  [{name}] epoch {epoch+1:2d}/{epochs}  "
                f"train={tl:.4f}  val={vl:.4f}  grad={gn:.4f}"
            )

        await ctx.log_metric("final_val_loss", val_losses[-1])

    return {
        "train_losses": train_losses,
        "val_losses": val_losses,
        "gradient_norms": gradient_norms,
        "final_val_loss": val_losses[-1],
    }

def train_model(
    model: nn.Module,
    name: str,
    tracker: ExperimentTracker,
    exp_name: str,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int = EPOCHS,
    lr: float = LR,
    clip: float = CLIP,
    attn: bool = False,
) -> dict[str, Any]:
    """Sync wrapper for training."""
    return asyncio.run(
        _train_model_async(
            model,
            name,
            tracker,
            exp_name,
            train_loader,
            val_loader,
            device,
            epochs,
            lr,
            clip,
            attn,
        )
    )

# ── Model Registry ──────────────────────────────────────────────────────
def register_best_model(
    model: nn.Module,
    model_name: str,
    val_loss: float,
    primary: str,
    registry: ModelRegistry | None,
    has_registry: bool,
) -> None:
    """Register a model in the ModelRegistry."""
    if not has_registry or registry is None:
        print("  ModelRegistry not available, skipping registration")
        return

    model_bytes = pickle.dumps(model.state_dict())
    try:
        reg_result = asyncio.run(
            registry.register(
                name=f"m5_rnn_{model_name.lower()}_{primary.replace('^', '')}",
                model_data=model_bytes,
                metadata={
                    "architecture": model_name,
                    "ticker": primary,
                    "hidden_dim": HIDDEN_DIM,
                    "seq_len": SEQ_LEN,
                    "forecast_horizon": FORECAST_HORIZON,
                    "val_loss": val_loss,
                    "epochs": EPOCHS,
                },
            )
        )
        print(f"  Registered: {reg_result}")
    except Exception as e:
        print(f"  ModelRegistry registration skipped ({type(e).__name__}: {e})")

# ── Visualisation Helpers ───────────────────────────────────────────────
def get_visualizer() -> ModelVisualizer:
    """Create a ModelVisualizer instance."""
    return ModelVisualizer()

def plot_training_curves(
    viz: ModelVisualizer,
    results: dict[str, Any],
    model_name: str,
    output_prefix: str,
) -> None:
    """Plot training/validation loss curves and gradient norms."""
    train_metrics = {
        f"{model_name} train": results["train_losses"],
        f"{model_name} val": results["val_losses"],
    }
    viz.training_history(
        metrics=train_metrics, x_label="Epoch", y_label="MSE Loss"
    ).write_html(str(OUTPUT_DIR / f"{output_prefix}_training_curves.html"))

    grad_metrics = {model_name: results["gradient_norms"]}
    viz.training_history(
        metrics=grad_metrics, x_label="Epoch", y_label="Gradient L2 Norm"
    ).write_html(str(OUTPUT_DIR / f"{output_prefix}_gradient_norms.html"))

def plot_predictions(
    viz: ModelVisualizer,
    model: nn.Module,
    X_val_t: torch.Tensor,
    y_val_t: torch.Tensor,
    norm_mean: np.ndarray,
    norm_std: np.ndarray,
    output_prefix: str,
    attn: bool = False,
) -> tuple[np.ndarray, np.ndarray, torch.Tensor | None]:
    """Generate prediction-vs-actual scatter and return denormalised arrays.

    Returns: preds_denorm, actual_denorm, attn_weights (or None)
    """
    model.eval()
    with torch.no_grad():
        if attn:
            val_preds, attn_weights = model(X_val_t)
        else:
            val_preds = model(X_val_t)
            attn_weights = None

    close_mean, close_std = norm_mean[0, 0], norm_std[0, 0]
    preds_denorm = val_preds.cpu().numpy() * close_std + close_mean
    actual_denorm = y_val_t.cpu().numpy() * close_std + close_mean

    pred_df = pl.DataFrame(
        {
            "actual": actual_denorm[:, 0].tolist(),
            "predicted": preds_denorm[:, 0].tolist(),
        }
    )
    viz.scatter(pred_df, x="actual", y="predicted").write_html(
        str(OUTPUT_DIR / f"{output_prefix}_pred_vs_actual.html")
    )

    return preds_denorm, actual_denorm, attn_weights

def plot_time_series_overlay(
    preds_denorm: np.ndarray,
    actual_denorm: np.ndarray,
    output_prefix: str,
    title: str = "Predicted vs Actual Close Price",
    n_points: int = 200,
) -> None:
    """Plot predicted vs actual as overlaid time-series lines."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = min(n_points, len(preds_denorm))
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(
        range(n), actual_denorm[:n, 0], label="Actual", color="#2196F3", linewidth=1.5
    )
    ax.plot(
        range(n),
        preds_denorm[:n, 0],
        label="Predicted",
        color="#FF5722",
        linewidth=1.5,
        linestyle="--",
        alpha=0.85,
    )
    ax.set_xlabel("Validation Window Index")
    ax.set_ylabel("Close Price")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(str(OUTPUT_DIR / f"{output_prefix}_time_series_overlay.png"), dpi=150)
    plt.close(fig)
    print(f"  Saved: {output_prefix}_time_series_overlay.png")

def plot_horizon_error(
    preds_denorm: np.ndarray,
    actual_denorm: np.ndarray,
    model_name: str,
) -> list[float]:
    """Print and return RMSE by forecast horizon day."""
    print(f"\n  Forecast Error by Horizon Day ({model_name}):")
    rmses = []
    for day in range(FORECAST_HORIZON):
        rmse = (
            float(np.mean((preds_denorm[:, day] - actual_denorm[:, day]) ** 2)) ** 0.5
        )
        rmses.append(rmse)
        print(f"    Day {day + 1}: RMSE={rmse:.2f}")
    return rmses
