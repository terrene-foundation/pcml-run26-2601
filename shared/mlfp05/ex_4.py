# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# MLFP05 — Exercise 4: Shared Utilities
# ════════════════════════════════════════════════════════════════════════
#
# Common infrastructure for all Exercise 4 technique files:
#   - AG News loading + caching (120K train, 7.6K test)
#   - Vocabulary building and text-to-index conversion
#   - ExperimentTracker + ModelRegistry + OnnxBridge setup
#   - Shared constants, device detection, visualisation helpers
#
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import asyncio
import math
import pickle
from collections import Counter
from pathlib import Path
import numpy as np
import polars as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from torch.utils.data import DataLoader, TensorDataset

import plotly.graph_objects as go

from kailash.db import ConnectionManager
from kailash_ml import ModelVisualizer
from kailash_ml.engines.experiment_tracker import ExperimentTracker
from kailash_ml.engines.model_registry import ModelRegistry
from kailash_ml.bridge.onnx_bridge import OnnxBridge
from shared.kailash_helpers import get_device, setup_environment

# ════════════════════════════════════════════════════════════════════════
# Environment + Device
# ════════════════════════════════════════════════════════════════════════
setup_environment()

torch.manual_seed(42)
np.random.seed(42)
DEVICE = get_device()

# ════════════════════════════════════════════════════════════════════════
# Constants
# ════════════════════════════════════════════════════════════════════════
CLASS_NAMES = ["World", "Sports", "Business", "Sci/Tech"]
MAX_LEN = 40
VOCAB_SIZE = 15000
EPOCHS_SCRATCH = 8
BERT_MODEL_NAME = "bert-base-uncased"
BERT_MAX_LEN = 64
BERT_EPOCHS = 3
BERT_LR = 2e-5
BERT_BATCH_SIZE = 32


# ════════════════════════════════════════════════════════════════════════
# Core Attention Function (shared across 01, 02, 05)
# ════════════════════════════════════════════════════════════════════════
def scaled_dot_product_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute scaled dot-product attention.

    Attention(Q, K, V) = softmax( Q K^T / sqrt(d_k) ) V

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
    scores = torch.einsum("bqd,bkd->bqk", q, k) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float("-inf"))
    weights = F.softmax(scores, dim=-1)
    out = torch.einsum("bqk,bkd->bqd", weights, v)
    return out, weights


# ════════════════════════════════════════════════════════════════════════
# AG News Loading
# ════════════════════════════════════════════════════════════════════════
def load_ag_news_split(split: str, cache_name: str) -> pl.DataFrame:
    """Load AG News split, caching to parquet for subsequent runs."""
    cache = Path(__file__).resolve().parents[2] / "data" / "mlfp05" / cache_name
    cache.parent.mkdir(parents=True, exist_ok=True)
    if cache.exists():
        return pl.read_parquet(cache)
    print(f"  Downloading AG News {split} from HuggingFace...")
    ds = load_dataset("fancyzhx/ag_news", split=split)
    df = pl.from_pandas(ds.to_pandas())
    df.write_parquet(cache)
    return df


def load_ag_news() -> tuple[pl.DataFrame, pl.DataFrame]:
    """Load full AG News train + test splits with status reporting."""
    print("\n== Loading FULL AG News (120K train + 7.6K test) ==")
    train_df = load_ag_news_split("train", "ag_news_full_train.parquet")
    test_df = load_ag_news_split("test", "ag_news_full_test.parquet")
    print(f"  train rows: {len(train_df):,}   test rows: {len(test_df):,}")
    print(f"  sample headline: {train_df['text'][0][:80]!r}")
    print(f"  class balance (train): {dict(Counter(train_df['label'].to_list()))}")
    return train_df, test_df


# ════════════════════════════════════════════════════════════════════════
# Vocabulary + Tokenisation (for from-scratch models: LSTM + Transformer)
# ════════════════════════════════════════════════════════════════════════
def build_vocab(texts: list[str], max_vocab: int = VOCAB_SIZE) -> dict[str, int]:
    """Build a word-level vocabulary from text list."""
    words: Counter[str] = Counter()
    for t in texts:
        words.update(t.lower().split())
    vocab = {"<pad>": 0, "<unk>": 1}
    for word, _ in words.most_common(max_vocab - 2):
        vocab[word] = len(vocab)
    return vocab


def text_to_indices(
    text: str, vocab: dict[str, int], max_len: int = MAX_LEN
) -> list[int]:
    """Convert text to padded index sequence."""
    tokens = text.lower().split()[:max_len]
    idxs = [vocab.get(t, 1) for t in tokens]
    return (idxs + [0] * (max_len - len(idxs)))[:max_len]


def prepare_dataloaders(
    train_df: pl.DataFrame,
    test_df: pl.DataFrame,
    vocab: dict[str, int],
    batch_size: int = 128,
) -> tuple[
    DataLoader, DataLoader, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
]:
    """Tokenise AG News and build DataLoaders for from-scratch models.

    Returns: (train_loader, val_loader, train_t, train_y, test_t, test_y)
    """
    train_tokens = np.array(
        [text_to_indices(t, vocab, MAX_LEN) for t in train_df["text"].to_list()],
        dtype=np.int64,
    )
    train_labels = np.array(train_df["label"].to_list(), dtype=np.int64)
    test_tokens = np.array(
        [text_to_indices(t, vocab, MAX_LEN) for t in test_df["text"].to_list()],
        dtype=np.int64,
    )
    test_labels = np.array(test_df["label"].to_list(), dtype=np.int64)

    train_t = torch.from_numpy(train_tokens).to(DEVICE)
    train_y = torch.from_numpy(train_labels).to(DEVICE)
    test_t = torch.from_numpy(test_tokens).to(DEVICE)
    test_y = torch.from_numpy(test_labels).to(DEVICE)

    train_loader = DataLoader(
        TensorDataset(train_t, train_y), batch_size=batch_size, shuffle=True
    )
    val_loader = DataLoader(TensorDataset(test_t, test_y), batch_size=batch_size)

    return train_loader, val_loader, train_t, train_y, test_t, test_y


# ════════════════════════════════════════════════════════════════════════
# Kailash-ML Engine Setup
# ════════════════════════════════════════════════════════════════════════
async def _setup_engines_async() -> (
    tuple[ConnectionManager, ExperimentTracker, str, ModelRegistry | None, bool]
):
    """Initialise ExperimentTracker + ModelRegistry (async)."""
    conn = ConnectionManager("sqlite:///mlfp05_transformers.db")
    await conn.initialize()

    tracker = ExperimentTracker(conn)
    exp_name = await tracker.create_experiment(
        name="m5_transformers",
        description="LSTM vs Transformer vs BERT on AG News (120K headlines)",
    )

    try:
        registry = ModelRegistry(conn)
        has_registry = True
    except Exception as e:
        registry = None
        has_registry = False
        print(f"  Note: ModelRegistry setup skipped ({e})")

    return conn, tracker, exp_name, registry, has_registry


def setup_engines() -> tuple[
    ConnectionManager,
    ExperimentTracker,
    str,
    ModelRegistry | None,
    bool,
    OnnxBridge,
]:
    """Set up all kailash-ml engines (sync wrapper).

    Returns: (conn, tracker, exp_name, registry, has_registry, bridge)
    """
    conn, tracker, exp_name, registry, has_registry = asyncio.run(
        _setup_engines_async()
    )
    bridge = OnnxBridge()
    return conn, tracker, exp_name, registry, has_registry, bridge


# ════════════════════════════════════════════════════════════════════════
# Training Utilities
# ════════════════════════════════════════════════════════════════════════
async def train_model_async(
    model: nn.Module,
    model_name: str,
    train_loader: DataLoader,
    val_loader: DataLoader,
    tracker: ExperimentTracker,
    exp_name: str,
    epochs: int = EPOCHS_SCRATCH,
    lr: float = 2e-3,
) -> tuple[list[float], list[float]]:
    """Train a PyTorch classifier and log every epoch to ExperimentTracker.

    Uses the modern ``tracker.run(...)`` async context manager -- bulk
    param logging, per-step metrics, automatic COMPLETED/FAILED state.
    """
    model.to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    train_losses: list[float] = []
    val_accs: list[float] = []
    best_acc = 0.0
    best_state: dict | None = None
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)

    async with tracker.run(experiment_name=exp_name, run_name=model_name) as ctx:
        await ctx.log_params(
            {
                "model_type": model_name,
                "epochs": str(epochs),
                "lr": str(lr),
                "dataset_size": str(len(train_loader.dataset)),
                "batch_size": str(train_loader.batch_size),
                "trainable_params": str(param_count),
            }
        )

        for epoch in range(epochs):
            model.train()
            batch_losses = []
            for xb, yb in train_loader:
                optimizer.zero_grad()
                logits = model(xb)
                loss = F.cross_entropy(logits, yb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                batch_losses.append(loss.item())
            scheduler.step()
            epoch_loss = float(np.mean(batch_losses))
            train_losses.append(epoch_loss)

            model.eval()
            with torch.no_grad():
                correct = 0
                total = 0
                for xb, yb in val_loader:
                    preds = model(xb).argmax(dim=-1)
                    correct += int((preds == yb).sum().item())
                    total += int(yb.size(0))
                acc = correct / total
                val_accs.append(acc)

            await ctx.log_metrics(
                {"train_loss": epoch_loss, "val_accuracy": acc},
                step=epoch + 1,
            )
            if acc > best_acc:
                best_acc = acc
                best_state = {
                    k: v.detach().clone() for k, v in model.state_dict().items()
                }
            print(
                f"  [{model_name}] epoch {epoch+1}/{epochs}  "
                f"loss={epoch_loss:.4f}  val_acc={acc:.3f}"
            )

        await ctx.log_metrics(
            {
                "best_val_accuracy": best_acc,
                "final_train_loss": train_losses[-1],
            }
        )

    if best_state is not None:
        model.load_state_dict(best_state)

    return train_losses, val_accs


def train_model(
    model: nn.Module,
    model_name: str,
    train_loader: DataLoader,
    val_loader: DataLoader,
    tracker: ExperimentTracker,
    exp_name: str,
    epochs: int = EPOCHS_SCRATCH,
    lr: float = 2e-3,
) -> tuple[list[float], list[float]]:
    """Sync wrapper -- one asyncio.run per training call."""
    return asyncio.run(
        train_model_async(
            model, model_name, train_loader, val_loader, tracker, exp_name, epochs, lr
        )
    )


# ════════════════════════════════════════════════════════════════════════
# Visualisation Helpers
# ════════════════════════════════════════════════════════════════════════
def create_attention_heatmap(
    attn_weights: np.ndarray,
    labels: list[str],
    title: str = "Self-Attention Heatmap",
    max_tokens: int = 15,
) -> go.Figure:
    """Create a plotly heatmap from an attention weight matrix.

    Args:
        attn_weights: 2D numpy array of shape (seq, seq)
        labels: token labels for axes
        title: chart title
        max_tokens: limit display to first N non-pad tokens

    Returns:
        plotly Figure
    """
    show_len = min(max_tokens, len([w for w in labels if w != "<pad>"]))
    fig = go.Figure(
        data=go.Heatmap(
            z=attn_weights[:show_len, :show_len],
            x=labels[:show_len],
            y=labels[:show_len],
            colorscale="Viridis",
            colorbar={"title": "Attention weight"},
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title="Key position",
        yaxis_title="Query position",
        width=600,
        height=500,
    )
    return fig


def get_viz() -> ModelVisualizer:
    """Return a ModelVisualizer instance."""
    return ModelVisualizer()
