# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
Shared infrastructure for Exercise 2 — LLM Fine-Tuning.

Contains: SFT dataset loader (IMDB -> instruction pairs), LoRA/adapter
parameter-counting helpers, kailash-align config factory, base
hyperparameters, and output directory. Technique-specific classes
(LoRALayer, AdapterLayer, etc.) live in the per-technique files.
"""
from __future__ import annotations

import os
from pathlib import Path

import polars as pl
import torch

from shared.kailash_helpers import get_device, setup_environment

# ════════════════════════════════════════════════════════════════════════
# ENVIRONMENT SETUP
# ════════════════════════════════════════════════════════════════════════

setup_environment()
torch.manual_seed(42)
device = get_device()

# Output directory for all visualisation and training artifacts
REPO_ROOT = Path.cwd()
OUTPUT_DIR = REPO_ROOT / "outputs" / "ex2_finetuning"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ════════════════════════════════════════════════════════════════════════
# SHARED HYPERPARAMETERS
# ════════════════════════════════════════════════════════════════════════

D_MODEL = 512  # synthetic demo hidden dim for from-scratch LoRA / adapter
LORA_RANK = 8  # default LoRA rank for the demos
LORA_ALPHA = 16.0  # typical alpha = 2 * rank
ADAPTER_BOTTLENECK = 64  # default adapter bottleneck

# ════════════════════════════════════════════════════════════════════════
# SFT DATA LOADING — IMDB instruction-response pairs
# ════════════════════════════════════════════════════════════════════════

DATA_DIR = REPO_ROOT / "data" / "mlfp06" / "imdb"
DATA_DIR.mkdir(parents=True, exist_ok=True)
CACHE_FILE = DATA_DIR / "imdb_sft_2k.parquet"


def load_imdb_sft(
    n_rows: int = 2000, train_frac: float = 0.9
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Load IMDB movie reviews and reformat as SFT instruction pairs.

    Returns:
        (full, train, eval) polars DataFrames with columns:
            instruction, response, text, label
    """
    if CACHE_FILE.exists():
        print(f"Loading cached IMDB SFT pairs from {CACHE_FILE}")
        sft_data = pl.read_parquet(CACHE_FILE)
    else:
        print("Downloading stanfordnlp/imdb from HuggingFace (first run)...")
        from datasets import load_dataset

        ds = load_dataset("stanfordnlp/imdb", split="train")
        ds = ds.shuffle(seed=42).select(range(min(n_rows, len(ds))))

        label_names = {0: "negative", 1: "positive"}
        rows = []
        for row in ds:
            review = row["text"][:1500]
            sentiment = label_names[row["label"]]
            rows.append(
                {
                    "instruction": (
                        "Classify the sentiment of the following movie review as "
                        "either 'positive' or 'negative', then briefly justify "
                        f"your answer.\n\nReview: {review}"
                    ),
                    "response": (
                        f"Sentiment: {sentiment}. The reviewer expresses a clearly "
                        f"{sentiment} reaction to the film."
                    ),
                    "text": review,
                    "label": sentiment,
                }
            )
        sft_data = pl.DataFrame(rows)
        sft_data.write_parquet(CACHE_FILE)
        print(f"Cached {sft_data.height} SFT pairs to {CACHE_FILE}")

    n_train = int(sft_data.height * train_frac)
    train_data = sft_data[:n_train]
    eval_data = sft_data[n_train:]
    print(
        f"IMDB SFT loaded: {sft_data.height} pairs "
        f"({train_data.height} train / {eval_data.height} eval)"
    )
    return sft_data, train_data, eval_data


# ════════════════════════════════════════════════════════════════════════
# PARAMETER COUNTING HELPERS
# ════════════════════════════════════════════════════════════════════════


def count_lora_params(
    d_model: int, rank: int, num_modules: int = 1, num_layers: int = 1
) -> int:
    """LoRA trainable params = 2 * d * r per target module per layer.

    For d=512, r=8, modules=1, layers=1 -> 8,192 params.
    """
    return (d_model * rank + rank * d_model) * num_modules * num_layers


def count_adapter_params(d_model: int, bottleneck: int, num_layers: int = 1) -> int:
    """Adapter params per layer: down(d*b+b) + up(b*d+d) + layernorm(2*d).

    Matches the AdapterLayer in 02_adapter_from_scratch.py.
    """
    return (
        d_model * bottleneck + bottleneck + bottleneck * d_model + d_model + 2 * d_model
    ) * num_layers


def full_finetune_params(
    d_model: int, num_modules: int = 1, num_layers: int = 1
) -> int:
    """Full fine-tuning baseline: d^2 per module per layer."""
    return d_model * d_model * num_modules * num_layers


# ════════════════════════════════════════════════════════════════════════
# KAILASH-ALIGN CONFIG FACTORY
# ════════════════════════════════════════════════════════════════════════


def get_base_model_name() -> str:
    """Return the HuggingFace repo id of the SFT base model.

    Resolution: ``SFT_BASE_MODEL`` env var → ``Qwen/Qwen2.5-0.5B-Instruct``.
    Note: Align/TRL training needs a HuggingFace repo id, NOT an Ollama tag —
    the Ollama-side ``OLLAMA_FT_BASE_MODEL`` is for inference of the
    fine-tuned model after GGUF export, not for training.
    """
    return os.environ.get("SFT_BASE_MODEL") or "Qwen/Qwen2.5-0.5B-Instruct"


def build_sft_config(
    base_model: str | None = None,
    lora_r: int = 16,
    lora_alpha: int = 32,
    num_epochs: int = 3,
    output_subdir: str = "sft_output",
):
    """Build a kailash-align AlignmentConfig for SFT+LoRA training.

    Imported lazily so technique files that do not call SFT are not
    forced to install the align extra at import time.

    kailash-align 0.6.0+ uses a composed AlignmentConfig: top-level
    method + base_model_id + experiment_dir, plus required LoRAConfig
    + SFTConfig + DPOConfig sub-configs. The DPOConfig is unused for
    SFT but the constructor still expects it (default values are fine).
    """
    from kailash_align import AlignmentConfig, DPOConfig, LoRAConfig, SFTConfig

    return AlignmentConfig(
        method="sft",
        base_model_id=base_model or get_base_model_name(),
        lora=LoRAConfig(
            rank=lora_r,
            alpha=lora_alpha,
            target_modules=("q_proj", "v_proj"),
            dropout=0.05,
        ),
        sft=SFTConfig(
            num_train_epochs=num_epochs,
            per_device_train_batch_size=4,
            gradient_accumulation_steps=4,
            learning_rate=2e-4,
            warmup_ratio=0.1,
            max_seq_length=512,
        ),
        dpo=DPOConfig(),
        experiment_dir=str(OUTPUT_DIR / output_subdir),
    )
