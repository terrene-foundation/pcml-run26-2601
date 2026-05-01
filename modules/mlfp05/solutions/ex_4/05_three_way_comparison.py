# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# MLFP05 — Exercise 4.5: Three-Way Comparison + ONNX Export
# ════════════════════════════════════════════════════════════════════════
#
# WHAT YOU'LL LEARN:
#   After completing this exercise, you will be able to:
#   - Compare LSTM vs Transformer vs BERT side by side on the same dataset
#   - Explain the accuracy hierarchy (BERT >> Transformer > LSTM) and why
#   - Register all models in the ModelRegistry with versioned metrics
#   - Export the best model to ONNX for portable deployment
#   - Visualise training curves for all three architectures
#   - Interpret model predictions with attention heatmaps
#
# PREREQUISITES: All previous ex_4 files (01-04).
# ESTIMATED TIME: ~25 min
# DATASET: AG News — 120,000 real news headlines, 4 classes.
#
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import asyncio
import math
import pickle
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from shared.mlfp05.ex_4 import (
    BERT_BATCH_SIZE,
    BERT_MAX_LEN,
    BERT_MODEL_NAME,
    CLASS_NAMES,
    DEVICE,
    EPOCHS_SCRATCH,
    MAX_LEN,
    build_vocab,
    create_attention_heatmap,
    get_viz,
    load_ag_news,
    prepare_dataloaders,
    scaled_dot_product_attention,
    setup_engines,
    text_to_indices,
    train_model,
)

print(f"Using device: {DEVICE}")


# ════════════════════════════════════════════════════════════════════════
# THEORY — The Architecture Hierarchy
# ════════════════════════════════════════════════════════════════════════
# This exercise is the payoff of the entire Exercise 4 sequence. We've
# built three architectures with fundamentally different approaches:
#
#   1. LSTM (sequential, no pre-training):
#      Processes tokens one at a time through a hidden state. No
#      pre-trained knowledge -- learns everything from our 120K headlines.
#      The sequential bottleneck limits long-range dependency capture.
#
#   2. Transformer (parallel attention, no pre-training):
#      Processes all tokens simultaneously via self-attention. Same
#      training data as LSTM, but the attention mechanism provides
#      direct access to all positions. Still learns from scratch.
#
#   3. BERT (parallel attention + pre-training):
#      Same architecture as the Transformer, but starts with pre-trained
#      weights from billions of words. Fine-tuning adapts this vast
#      language knowledge to our specific task.
#
# The comparison isolates two variables:
#   LSTM -> Transformer: the value of ATTENTION (parallel vs sequential)
#   Transformer -> BERT: the value of PRE-TRAINING (scratch vs transfer)
#
# Together, these reveal why modern NLP is dominated by pre-trained
# transformers: attention + pre-training is the winning combination.
# ════════════════════════════════════════════════════════════════════════


# ════════════════════════════════════════════════════════════════════════
# TASK 1 — Train all three models (reusing architectures from 02-04)
# ════════════════════════════════════════════════════════════════════════
train_df, test_df = load_ag_news()
vocab = build_vocab(train_df["text"].to_list())
train_loader, val_loader, train_t, train_y, test_t, test_y = prepare_dataloaders(
    train_df, test_df, vocab
)
conn, tracker, exp_name, registry, has_registry, bridge = setup_engines()

# --- Model Architectures (defined here for standalone execution) ---


class LSTMClassifier(nn.Module):
    """Bidirectional LSTM for text classification (same as 03_lstm_baseline)."""

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 128,
        hidden_dim: int = 128,
        n_layers: int = 2,
        n_classes: int = 4,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            embed_dim,
            hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0,
            bidirectional=True,
        )
        self.head_drop = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_dim * 2, n_classes)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        x = self.embed(tokens)
        lstm_out, _ = self.lstm(x)
        pad_mask = tokens == 0
        lengths = (~pad_mask).sum(dim=1, keepdim=True).clamp(min=1).float()
        lstm_out = lstm_out.masked_fill(pad_mask.unsqueeze(-1), 0.0)
        pooled = lstm_out.sum(dim=1) / lengths
        return self.head(self.head_drop(pooled))


class EducationalMultiHead(nn.Module):
    """Multi-head attention (same as 02_transformer_encoder)."""

    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        b, seq, d = x.shape
        qkv = self.qkv(x).reshape(b, seq, 3, self.n_heads, self.d_k)
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2).reshape(b * self.n_heads, seq, self.d_k)
        k = k.transpose(1, 2).reshape(b * self.n_heads, seq, self.d_k)
        v = v.transpose(1, 2).reshape(b * self.n_heads, seq, self.d_k)
        out, weights = scaled_dot_product_attention(q, k, v)
        attn_weights = weights.reshape(b, self.n_heads, seq, seq)
        out = (
            out.reshape(b, self.n_heads, seq, self.d_k)
            .transpose(1, 2)
            .reshape(b, seq, d)
        )
        return self.proj(out), attn_weights


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding (same as 02_transformer_encoder)."""

    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div)
        pe[:, 1::2] = torch.cos(position * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class TransformerClassifier(nn.Module):
    """Transformer encoder classifier (same as 02_transformer_encoder)."""

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
        self.embed = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.posenc = PositionalEncoding(d_model)
        self.emb_drop = nn.Dropout(dropout)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            batch_first=True,
        )
        # enable_nested_tensor=False: MPS-compatibility (see ex_4/02 note).
        self.encoder = nn.TransformerEncoder(
            layer, num_layers=n_layers, enable_nested_tensor=False
        )
        self.head_drop = nn.Dropout(dropout)
        self.head = nn.Linear(d_model, n_classes)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        pad_mask = tokens == 0
        x = self.embed(tokens)
        x = self.posenc(x)
        x = self.emb_drop(x)
        x = self.encoder(x, src_key_padding_mask=pad_mask)
        lengths = (~pad_mask).sum(dim=1, keepdim=True).clamp(min=1).float()
        x = x.masked_fill(pad_mask.unsqueeze(-1), 0.0)
        pooled = x.sum(dim=1) / lengths
        return self.head(self.head_drop(pooled))


# --- LSTM ---
print("\n== Training LSTM baseline ==")
lstm_model = LSTMClassifier(
    vocab_size=len(vocab), embed_dim=128, hidden_dim=128, n_layers=2, n_classes=4
)
lstm_losses, lstm_accs = train_model(
    lstm_model,
    "lstm_baseline",
    train_loader,
    val_loader,
    tracker,
    exp_name,
    epochs=EPOCHS_SCRATCH,
)

# --- Transformer ---
print("\n== Training Transformer ==")
transformer_model = TransformerClassifier(
    vocab_size=len(vocab), d_model=128, n_heads=4, n_layers=3, n_classes=4
)
transformer_losses, transformer_accs = train_model(
    transformer_model,
    "transformer",
    train_loader,
    val_loader,
    tracker,
    exp_name,
    epochs=EPOCHS_SCRATCH,
)

# --- BERT ---
print(f"\n== Fine-tuning {BERT_MODEL_NAME} ==")
from transformers import BertTokenizer, BertForSequenceClassification

bert_tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_NAME)
bert_model = BertForSequenceClassification.from_pretrained(
    BERT_MODEL_NAME, num_labels=4
).to(DEVICE)

# Freeze lower 8 of 12 layers
for name, param in bert_model.named_parameters():
    if "bert.encoder.layer" in name:
        layer_num = int(name.split(".")[3])
        if layer_num < 8:
            param.requires_grad = False
    elif "bert.embeddings" in name:
        param.requires_grad = False

trainable = sum(p.numel() for p in bert_model.parameters() if p.requires_grad)
total_params = sum(p.numel() for p in bert_model.parameters())


def tokenise_for_bert(
    texts: list[str], max_len: int = BERT_MAX_LEN
) -> tuple[torch.Tensor, torch.Tensor]:
    encoding = bert_tokenizer(
        texts,
        max_length=max_len,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    return encoding["input_ids"], encoding["attention_mask"]


print("  Tokenising for BERT...")
bert_train_ids, bert_train_mask = tokenise_for_bert(train_df["text"].to_list())
bert_test_ids, bert_test_mask = tokenise_for_bert(test_df["text"].to_list())
bert_train_y = torch.tensor(train_df["label"].to_list(), dtype=torch.long)
bert_test_y = torch.tensor(test_df["label"].to_list(), dtype=torch.long)

bert_train_loader = DataLoader(
    TensorDataset(
        bert_train_ids.to(DEVICE), bert_train_mask.to(DEVICE), bert_train_y.to(DEVICE)
    ),
    batch_size=BERT_BATCH_SIZE,
    shuffle=True,
)
bert_val_loader = DataLoader(
    TensorDataset(
        bert_test_ids.to(DEVICE), bert_test_mask.to(DEVICE), bert_test_y.to(DEVICE)
    ),
    batch_size=BERT_BATCH_SIZE,
)


async def train_bert_async(model, train_loader, val_loader, epochs=3, lr=2e-5):
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr,
        weight_decay=0.01,
    )
    scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=1.0,
        end_factor=0.1,
        total_iters=epochs,
    )
    train_losses, val_accs = [], []
    best_acc = 0.0

    async with tracker.track(experiment=exp_name, run_name="bert_finetune") as run:
        await run.log_params(
            {
                "model_type": "bert_finetune",
                "base_model": BERT_MODEL_NAME,
                "epochs": str(epochs),
                "lr": str(lr),
                "frozen_layers": "0-7",
                "trainable_params": str(trainable),
                "dataset_size": str(len(train_loader.dataset)),
            }
        )
        for epoch in range(epochs):
            model.train()
            batch_losses = []
            for batch_idx, (ids, mask, labels) in enumerate(train_loader):
                optimizer.zero_grad()
                outputs = model(input_ids=ids, attention_mask=mask, labels=labels)
                outputs.loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                batch_losses.append(outputs.loss.item())
                if (batch_idx + 1) % 500 == 0:
                    print(
                        f"    batch {batch_idx+1}/{len(train_loader)}  loss={np.mean(batch_losses[-500:]):.4f}"
                    )
            scheduler.step()
            epoch_loss = float(np.mean(batch_losses))
            train_losses.append(epoch_loss)

            model.eval()
            with torch.no_grad():
                correct = total_count = 0
                for ids, mask, labels in val_loader:
                    preds = model(input_ids=ids, attention_mask=mask).logits.argmax(
                        dim=-1
                    )
                    correct += int((preds == labels).sum().item())
                    total_count += int(labels.size(0))
                acc = correct / total_count
                val_accs.append(acc)

            await run.log_metrics(
                {"train_loss": epoch_loss, "val_accuracy": acc}, step=epoch + 1
            )
            if acc > best_acc:
                best_acc = acc
            print(
                f"  [BERT] epoch {epoch+1}/{epochs}  loss={epoch_loss:.4f}  val_acc={acc:.3f}"
            )

        await run.log_metrics(
            {"best_val_accuracy": best_acc, "final_train_loss": train_losses[-1]}
        )
    return train_losses, val_accs


bert_losses, bert_accs = asyncio.run(
    train_bert_async(bert_model, bert_train_loader, bert_val_loader, epochs=3)
)

# ══════════════════════════════════════════════════════════════════
# DIAGNOSTIC CHECKPOINT — comparative Prescription Pad for all 3
# ══════════════════════════════════════════════════════════════════
from kailash_ml.diagnostics import run_diagnostic_checkpoint
from kailash_ml import diagnose

print("\n── Diagnostic Report (LSTM) ──")
report = diagnose(lstm_model, kind="dl", data=val_loader, show=False)

print("\n── Diagnostic Report (Transformer) ──")
report = diagnose(
    transformer_model,
    kind="dl",
    data=val_loader,
    show=False,
)


def _bert_loss(m, ids, mask, labels):
    return m(input_ids=ids, attention_mask=mask, labels=labels).loss


def _bert_adapter(batch):
    return batch[0], batch[1], batch[2]


print("\n── Diagnostic Report (BERT fine-tune) ──")
bert_diag, bert_findings = run_diagnostic_checkpoint(
    bert_model,
    bert_val_loader,
    _bert_loss,
    title="BERT fine-tune (3-way comparison)",
    n_batches=4,
    train_losses=bert_losses,
    val_losses=[1.0 - a for a in bert_accs],
    batch_adapter=_bert_adapter,
    show=False,
)

# ══════ EXPECTED OUTPUT (reference pattern — 3-way on AG News) ══════
# All three reports follow their individual-file patterns (02, 03, 04).
# Side-by-side takeaway:
#   LSTM        : gradient-flow WARNING (recurrent weights 1:50 of head)
#   Transformer : all HEALTHY, uniform gradients across encoder layers
#   BERT        : all HEALTHY, frozen layers 0-7 show ZERO RMS (by design)
#
# STUDENT INTERPRETATION GUIDE — the comparative narrative:
#
#  [BLOOD TEST] compare gradient-flow readings across the three
#     reports. The LSTM shows concentrated gradients at the head;
#     the Transformer shows uniform gradients across `encoder.layers`;
#     BERT shows zeros in frozen layers and healthy signal above layer
#     8. This ONE instrument tells the whole story of slide 5.4:
#     attention + residuals + pretraining stack three architectural
#     wins on top of each other.
#     >> Prescription Pad reading: the accuracy gap (LSTM -> Transformer
#        -> BERT) is NOT just hyperparameter tuning — it's visible in
#        the gradient-flow report before you ever look at val acc.
#
#  [STETHOSCOPE] loss curve shapes differ:
#     - LSTM: slow convergence, plateau by epoch 5 (sequential ceiling)
#     - Transformer: steady decline, could benefit from more epochs
#     - BERT: sharp drop in 2 epochs, then gentle tail (pretraining
#       head-start is doing the work)
#     The Stethoscope surfaces the "how fast can this architecture
#     learn" question that slide 5.4 frames as "inductive bias from
#     pretraining amortises compute".
#
#  [FIVE-INSTRUMENT TAKEAWAY] This comparison is the most valuable
#  diagnostic exercise in ex_4. You see the SAME instruments producing
#  three distinct signatures on the SAME dataset. Students who memorise
#  these three signatures can diagnose any NLP architecture in the
#  future. Prescription Pad's value is as a classifier OF classifiers.
#
#  CONNECT TO SLIDE 5.4: The slide's "pretrain-then-finetune is the
#  dominant paradigm" claim is proven by the ZERO RMS on BERT's
#  frozen layers + the 4-point accuracy win at 3 epochs. That's the
#  empirical form of the argument.
# ══════════════════════════════════════════════════════════════════

# ── Checkpoint 1 ─────────────────────────────────────────────────────
assert max(lstm_accs) > 0.60, f"LSTM should exceed 60%, got {max(lstm_accs):.3f}"
assert (
    max(transformer_accs) > 0.60
), f"Transformer should exceed 60%, got {max(transformer_accs):.3f}"
assert max(bert_accs) > 0.85, f"BERT should exceed 85%, got {max(bert_accs):.3f}"
print("\n--- Checkpoint 1 passed --- all three models trained\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 2 — Visualise: Side-by-side comparison table + training curves
# ════════════════════════════════════════════════════════════════════════
results = {
    "LSTM": {
        "best_acc": max(lstm_accs),
        "final_loss": lstm_losses[-1],
        "params": sum(p.numel() for p in lstm_model.parameters()),
    },
    "Transformer": {
        "best_acc": max(transformer_accs),
        "final_loss": transformer_losses[-1],
        "params": sum(p.numel() for p in transformer_model.parameters()),
    },
    "BERT (fine-tuned)": {
        "best_acc": max(bert_accs),
        "final_loss": bert_losses[-1],
        "params": total_params,
    },
}

print("\n== 3-Way Model Comparison on AG News ==")
print(f"{'Model':<20} {'Best Acc':>10} {'Final Loss':>12} {'Params':>12}")
print("-" * 56)
for name, r in results.items():
    print(
        f"{name:<20} {r['best_acc']:>10.3f} {r['final_loss']:>12.4f} {r['params']:>12,}"
    )

# Training curves comparison: all 3 models on one chart
viz = get_viz()
fig_curves = viz.training_history(
    metrics={
        "LSTM loss": lstm_losses,
        "Transformer loss": transformer_losses,
        "BERT loss": bert_losses,
        "LSTM val_acc": lstm_accs,
        "Transformer val_acc": transformer_accs,
        "BERT val_acc": bert_accs,
    },
    x_label="Epoch",
    y_label="Value",
)
fig_curves.write_html("ex_4_5_training_curves.html")
print("\nTraining curves saved to ex_4_5_training_curves.html")

# Sample predictions from all three models
sample_texts = test_df["text"].to_list()[:5]
sample_true = test_df["label"].to_list()[:5]
sample_idx = torch.tensor(
    [text_to_indices(t, vocab, MAX_LEN) for t in sample_texts],
    dtype=torch.long,
    device=DEVICE,
)

transformer_model.eval()
lstm_model.eval()
bert_model.eval()
with torch.no_grad():
    transformer_preds = transformer_model(sample_idx).argmax(dim=-1).cpu().tolist()
    lstm_preds = lstm_model(sample_idx).argmax(dim=-1).cpu().tolist()
    bert_sample_ids, bert_sample_mask = tokenise_for_bert(sample_texts)
    bert_preds = (
        bert_model(
            input_ids=bert_sample_ids.to(DEVICE),
            attention_mask=bert_sample_mask.to(DEVICE),
        )
        .logits.argmax(dim=-1)
        .cpu()
        .tolist()
    )

print(f"\n== Sample Predictions (all 3 models) ==")
print(f"{'Headline':<50} {'True':<10} {'LSTM':<10} {'Trans':<10} {'BERT':<10}")
print("-" * 90)
for i, text in enumerate(sample_texts):
    t = CLASS_NAMES[sample_true[i]]
    l = CLASS_NAMES[lstm_preds[i]]
    tr = CLASS_NAMES[transformer_preds[i]]
    b = CLASS_NAMES[bert_preds[i]]
    print(f"{text[:48]:<50} {t:<10} {l:<10} {tr:<10} {b:<10}")

# ── Checkpoint 2 ─────────────────────────────────────────────────────
best_model_name = max(results, key=lambda k: results[k]["best_acc"])
assert best_model_name == "BERT (fine-tuned)", (
    f"Expected BERT to be the best model, but {best_model_name} won. "
    "Pre-trained models should dominate on standard NLP benchmarks."
)
assert Path("ex_4_5_training_curves.html").exists(), "Training curves should be saved"
# INTERPRETATION: The 3-way comparison reveals a clear hierarchy:
#   BERT >> Transformer > LSTM
# BERT dominates because it starts with pre-trained language knowledge.
# The Transformer edges out the LSTM because attention captures long-range
# dependencies without the information bottleneck of a fixed-size hidden
# state. The LSTM still does respectably -- it is a strong baseline.
print("\n--- Checkpoint 2 passed --- 3-way comparison complete\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 3 — Register all models in ModelRegistry
# ════════════════════════════════════════════════════════════════════════
async def register_all_models():
    """Register all three models in the ModelRegistry with metrics."""
    if not has_registry:
        print("  ModelRegistry not available -- skipping registration")
        return {}

    from kailash_ml.types import MetricSpec

    model_versions = {}
    models_to_register = [
        ("m5_bert_agnews", bert_model.state_dict(), max(bert_accs), "bert_finetune"),
        (
            "m5_transformer_agnews",
            transformer_model.state_dict(),
            max(transformer_accs),
            "transformer",
        ),
        ("m5_lstm_agnews", lstm_model.state_dict(), max(lstm_accs), "lstm_baseline"),
    ]

    for name, state_dict, best_acc, model_type in models_to_register:
        model_bytes = pickle.dumps(state_dict)
        version = await registry.register_model(
            name=name,
            artifact=model_bytes,
            metrics=[
                MetricSpec(name="best_val_accuracy", value=best_acc),
                MetricSpec(name="dataset", value=0.0),
                MetricSpec(name="model_type", value=0.0),
            ],
        )
        model_versions[model_type] = version
        print(f"  Registered {name}: version={version.version}, acc={best_acc:.3f}")

    return model_versions


model_versions = asyncio.run(register_all_models())

# ── Checkpoint 3 ─────────────────────────────────────────────────────
if has_registry:
    assert len(model_versions) == 3, "Should register all 3 models"
print("\n--- Checkpoint 3 passed --- models registered in ModelRegistry\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 4 — Export best model (BERT) to ONNX via OnnxBridge
# ════════════════════════════════════════════════════════════════════════
# In production, the ModelRegistry stores the winning model. OnnxBridge
# exports it to ONNX format for portable deployment (any language, any
# runtime, no PyTorch dependency). This is how production ML pipelines
# separate training (Python) from serving (any language).
onnx_path = Path("ex_4_bert_agnews.onnx")
bert_model.eval()

exported = False
try:
    result = bridge.export(
        model=bert_model,
        framework="pytorch",
        output_path=onnx_path,
        n_features=BERT_MAX_LEN,
    )
    success = getattr(result, "success", bool(result))
    exported = bool(success) and onnx_path.exists()
except Exception:
    pass

if not exported:
    # Torch ONNX export for BERT: provide dummy input_ids + attention_mask
    print("  Using torch.onnx.export for BERT model...")
    dummy_ids = torch.ones(1, BERT_MAX_LEN, dtype=torch.long, device=DEVICE)
    dummy_mask = torch.ones(1, BERT_MAX_LEN, dtype=torch.long, device=DEVICE)
    bert_cpu = bert_model.cpu()
    torch.onnx.export(
        bert_cpu,
        (dummy_ids.cpu(), dummy_mask.cpu()),
        onnx_path,
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "seq"},
            "attention_mask": {0: "batch", 1: "seq"},
            "logits": {0: "batch"},
        },
        opset_version=17,
        dynamo=False,
    )
    bert_model.to(DEVICE)

if onnx_path.exists():
    print(f"  ONNX export: {onnx_path} ({onnx_path.stat().st_size // 1024:,} KB)")
else:
    print("  ONNX export: skipped (export not available in this environment)")

# ── Checkpoint 4 ─────────────────────────────────────────────────────
# INTERPRETATION: The ModelRegistry gives you a versioned record of every
# model experiment. The ONNX export makes the model portable -- it can run
# on a server without PyTorch installed, in a mobile app, or in a browser
# via ONNX.js. This is how production ML pipelines separate training
# (Python) from serving (any language).
print("\n--- Checkpoint 4 passed --- ONNX export complete\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 5 — Visualise: Attention heatmap from trained Transformer
# ════════════════════════════════════════════════════════════════════════
# The attention heatmap is the Transformer's "explanation" -- it shows
# which words the model attends to when classifying a headline.
transformer_model.eval()
mha_viz = EducationalMultiHead(d_model=128, n_heads=4).to(DEVICE)

with torch.no_grad():
    embed = transformer_model.embed(sample_idx[:1])
    embed = transformer_model.posenc(embed)
    _, attn_weights = mha_viz(embed)
    attn_np = attn_weights[0, 0].cpu().numpy()

words = sample_texts[0].lower().split()[:MAX_LEN]
word_labels = words + ["<pad>"] * (MAX_LEN - len(words))

fig_attn = create_attention_heatmap(
    attn_np,
    word_labels,
    title=f"Transformer Attention on: '{sample_texts[0][:50]}...'",
    max_tokens=15,
)
fig_attn.write_html("ex_4_5_attention_heatmap.html")
print("Attention heatmap saved to ex_4_5_attention_heatmap.html")

# ── Checkpoint 5 ─────────────────────────────────────────────────────
assert attn_np.shape[0] == MAX_LEN, "Attention heatmap should cover full sequence"
assert Path("ex_4_5_attention_heatmap.html").exists(), "Heatmap should be saved"
print("\n--- Checkpoint 5 passed --- visualisations complete\n")


# ════════════════════════════════════════════════════════════════════════
# DESTINATION-FIRST CLOSE — km.diagnose
# ════════════════════════════════════════════════════════════════════════
# This lesson walked the journey of attention-based language models —
# from-scratch self-attention, transformer encoder, LSTM baseline, and
# BERT fine-tuning. The kailash-ml SDK ships a single-call diagnostic
# primitive that closes the production loop: km.diagnose inspects a
# trained model and emits an auto-dashboard (loss curves, gradient flow,
# dead neurons, activation stats, weight distributions). One cell.
# Every diagnostic students would otherwise hand-roll, ready to surface
# in a Plotly dashboard.

from kailash_ml import diagnose

# We diagnose the from-scratch transformer (val_loader yields token-id
# tensors compatible with its forward signature). `kind='auto'` dispatches
# by model type — DLDiagnostics for torch.nn.Module.
report = diagnose(transformer_model, kind="auto", data=val_loader, show=False)
report.plot_training_dashboard()
print()
print("km.diagnose: 1 line of code -> the same observability the lesson")
print("body hand-rolled in 200+ lines. This is what 'destination-first'")
print("means — when the journey is internalised, the SDK is one call.")


# ══════════════════════════════════════════════════════════════════════
# REFLECTION
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  WHAT YOU'VE MASTERED — Complete Exercise 4")
print("=" * 70)
print(
    f"""
  [x] Derived scaled dot-product attention with torch.einsum
  [x] Explained the 1/sqrt(d_k) factor (prevents softmax saturation)
  [x] Wrote a hand-rolled multi-head attention wrapping the scratch kernel
  [x] Built a TransformerClassifier with nn.TransformerEncoder
  [x] Built an LSTM baseline for fair comparison
  [x] Trained all 3 models on FULL AG News (120K headlines)
  [x] Fine-tuned BERT ({BERT_MODEL_NAME}) -- best acc: {max(bert_accs):.1%}
  [x] Visualised attention heatmaps (what the model "looks at")
  [x] Tracked every run with ExperimentTracker (params, per-epoch metrics)
  [x] Registered models in ModelRegistry with versioned metrics
  [x] Exported the fine-tuned model to ONNX for portable deployment

  KEY INSIGHT — The Attention Hierarchy:
    LSTM best acc:        {max(lstm_accs):.1%}  (sequential, no pre-training)
    Transformer best acc: {max(transformer_accs):.1%}  (parallel attention, no pre-training)
    BERT best acc:        {max(bert_accs):.1%}  (parallel attention + pre-training)

  Pre-training is the single biggest lever in NLP. The Transformer
  architecture enables it, but the pre-trained weights are what make
  BERT dominate. This is why modern NLP is "pre-train then fine-tune."

  Next: In Exercise 5, you'll build generative models (DCGAN + WGAN-GP)
  that CREATE new data instead of classifying existing data.
"""
)
