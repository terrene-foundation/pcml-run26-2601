# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
Shared infrastructure for MLFP06 Exercise 3 — DPO Preference Alignment.

Contains: UltraFeedback loading, DPO loss helper, GRPO advantage helper,
visualisation utilities, output paths. Technique-specific code does NOT
belong here.
"""
from __future__ import annotations

import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import torch
import torch.nn.functional as F

from shared.kailash_helpers import get_device, setup_environment

# ════════════════════════════════════════════════════════════════════════
# ENVIRONMENT SETUP
# ════════════════════════════════════════════════════════════════════════

setup_environment()
torch.manual_seed(42)
np.random.seed(42)
device = get_device()

from shared.mlfp06._ollama_bootstrap import DEFAULT_CHAT_MODEL

MODEL_NAME = DEFAULT_CHAT_MODEL
BASE_MODEL = os.environ.get("SFT_BASE_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")

# Output directories
OUTPUT_DIR = Path("outputs") / "ex3_dpo"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DATA_CACHE = Path("data") / "mlfp06" / "ultrafeedback"
DATA_CACHE.mkdir(parents=True, exist_ok=True)
CACHE_FILE = DATA_CACHE / "ultrafeedback_2k.parquet"

ADAPTER_OUTPUT_DIR = Path("./dpo_output")


# ════════════════════════════════════════════════════════════════════════
# DATA LOADING — UltraFeedback Binarized preference pairs
# ════════════════════════════════════════════════════════════════════════


def load_ultrafeedback(n_samples: int = 2000) -> pl.DataFrame:
    """Load UltraFeedback Binarized preference dataset as a polars DataFrame.

    Columns: prompt, chosen, rejected. Cached as parquet after first load.
    """
    if CACHE_FILE.exists():
        print(f"Loading cached preference pairs from {CACHE_FILE}")
        return pl.read_parquet(CACHE_FILE)

    print("Downloading UltraFeedback Binarized from HuggingFace (first run)...")
    from datasets import load_dataset

    ds = load_dataset("trl-lib/ultrafeedback_binarized", split="train")
    ds = ds.shuffle(seed=42).select(range(min(n_samples, len(ds))))

    def _extract(row: dict) -> dict:
        chosen_msgs = row["chosen"]
        rejected_msgs = row["rejected"]
        prompt = ""
        for msg in chosen_msgs:
            if msg.get("role") == "user":
                prompt = msg.get("content", "")
                break
        chosen_text = next(
            (m["content"] for m in chosen_msgs if m.get("role") == "assistant"),
            "",
        )
        rejected_text = next(
            (m["content"] for m in rejected_msgs if m.get("role") == "assistant"),
            "",
        )
        return {
            "prompt": prompt,
            "chosen": chosen_text,
            "rejected": rejected_text,
        }

    rows = [_extract(r) for r in ds]
    rows = [r for r in rows if r["prompt"] and r["chosen"] and r["rejected"]]
    pref_data = pl.DataFrame(rows)
    pref_data.write_parquet(CACHE_FILE)
    print(f"Cached {pref_data.height} preference pairs to {CACHE_FILE}")
    return pref_data


def split_preferences(
    pref_data: pl.DataFrame, train_frac: float = 0.9
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Split preference pairs into train/eval."""
    n_train = int(pref_data.height * train_frac)
    return pref_data[:n_train], pref_data[n_train:]


# ════════════════════════════════════════════════════════════════════════
# DPO LOSS — canonical implementation (re-used across technique files)
# ════════════════════════════════════════════════════════════════════════


def dpo_loss(
    policy_chosen_logps: torch.Tensor,
    policy_rejected_logps: torch.Tensor,
    ref_chosen_logps: torch.Tensor,
    ref_rejected_logps: torch.Tensor,
    beta: float = 0.1,
) -> torch.Tensor:
    """Compute the DPO loss from scratch.

    Args:
        policy_chosen_logps:   log P_policy(y_w | x)   [batch]
        policy_rejected_logps: log P_policy(y_l | x)   [batch]
        ref_chosen_logps:      log P_ref(y_w | x)      [batch]
        ref_rejected_logps:    log P_ref(y_l | x)      [batch]
        beta:                  Temperature controlling preference strength.

    Returns:
        Scalar DPO loss (averaged over batch).
    """
    chosen_log_ratio = policy_chosen_logps - ref_chosen_logps
    rejected_log_ratio = policy_rejected_logps - ref_rejected_logps
    logits = beta * (chosen_log_ratio - rejected_log_ratio)
    return -F.logsigmoid(logits).mean()


# ════════════════════════════════════════════════════════════════════════
# GRPO — group-relative advantage helper
# ════════════════════════════════════════════════════════════════════════


def grpo_advantages(rewards: torch.Tensor) -> torch.Tensor:
    """Compute GRPO advantages: r_i minus the group mean per prompt.

    rewards: [n_prompts, K] tensor of scalar rewards, K completions per prompt.
    Returns advantages of the same shape, each row summing to ~0.
    """
    group_mean = rewards.mean(dim=1, keepdim=True)
    return rewards - group_mean


# ════════════════════════════════════════════════════════════════════════
# VISUALISATION — "Seeing Is Believing" for alignment
# ════════════════════════════════════════════════════════════════════════


def show_preference_length_distribution(
    pref_data: pl.DataFrame, title: str = "Chosen vs Rejected Length"
) -> Path:
    """Histogram of chosen vs rejected response lengths (character count)."""
    chosen_lens = np.array([len(s) for s in pref_data["chosen"].to_list()])
    rejected_lens = np.array([len(s) for s in pref_data["rejected"].to_list()])

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(
        chosen_lens,
        bins=60,
        alpha=0.65,
        label=f"Chosen (median={int(np.median(chosen_lens))})",
        color="#2E7D32",
    )
    ax.hist(
        rejected_lens,
        bins=60,
        alpha=0.65,
        label=f"Rejected (median={int(np.median(rejected_lens))})",
        color="#C62828",
    )
    ax.set_xlabel("Response length (characters)")
    ax.set_ylabel("Count")
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.legend()
    fname = OUTPUT_DIR / "ex3_preference_lengths.png"
    plt.tight_layout()
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"  Saved: {fname}")
    return fname


def show_dpo_loss_curve_by_margin(
    beta: float = 0.1, title: str = "DPO loss vs policy preference margin"
) -> Path:
    """Visual proof: as the policy's preference margin grows, DPO loss drops."""
    margins = torch.linspace(-3, 3, 121)
    losses = [-F.logsigmoid(torch.tensor(beta * m)).item() for m in margins]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(margins.numpy(), losses, color="#1565C0", linewidth=2.2)
    ax.axvline(0, color="#616161", linestyle=":", label="No preference (margin=0)")
    ax.set_xlabel("Policy log-ratio margin  (chosen - rejected)")
    ax.set_ylabel("DPO loss")
    ax.set_title(f"{title}  (beta={beta})", fontsize=13, fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fname = OUTPUT_DIR / "ex3_dpo_loss_curve.png"
    plt.tight_layout()
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"  Saved: {fname}")
    return fname


def show_beta_sensitivity(betas: list[float], losses: list[float]) -> Path:
    """Bar chart of DPO loss for a synthetic batch across beta values."""
    fig, ax = plt.subplots(figsize=(9, 5))
    colors = [
        "#90CAF9" if b <= 0.1 else "#FFB74D" if b <= 0.3 else "#E57373" for b in betas
    ]
    bars = ax.bar([f"{b}" for b in betas], losses, color=colors, edgecolor="black")
    for bar, loss_val in zip(bars, losses):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.005,
            f"{loss_val:.3f}",
            ha="center",
            fontsize=9,
        )
    ax.set_xlabel("Beta (alignment temperature)")
    ax.set_ylabel("DPO loss (synthetic batch)")
    ax.set_title(
        "Beta sensitivity — higher beta, stronger preference pressure",
        fontsize=13,
        fontweight="bold",
    )
    fname = OUTPUT_DIR / "ex3_beta_sensitivity.png"
    plt.tight_layout()
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"  Saved: {fname}")
    return fname


def show_grpo_advantages(rewards: torch.Tensor, advantages: torch.Tensor) -> Path:
    """Visualise GRPO advantages for each prompt group."""
    n_prompts, K = rewards.shape
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    im0 = axes[0].imshow(rewards.numpy(), aspect="auto", cmap="viridis")
    axes[0].set_title("Rewards  r(x, y_i)", fontsize=12, fontweight="bold")
    axes[0].set_xlabel(f"Completion index (K={K})")
    axes[0].set_ylabel("Prompt index")
    fig.colorbar(im0, ax=axes[0])

    im1 = axes[1].imshow(
        advantages.numpy(),
        aspect="auto",
        cmap="RdBu_r",
        vmin=-abs(advantages).max(),
        vmax=abs(advantages).max(),
    )
    axes[1].set_title("GRPO advantages  r_i - mean(r)", fontsize=12, fontweight="bold")
    axes[1].set_xlabel(f"Completion index (K={K})")
    axes[1].set_ylabel("Prompt index")
    fig.colorbar(im1, ax=axes[1])

    fname = OUTPUT_DIR / "ex3_grpo_advantages.png"
    plt.tight_layout()
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"  Saved: {fname}")
    return fname


def show_safety_refusal_rates(
    base_rate: float, aligned_rate: float, n_prompts: int
) -> Path:
    """Bar chart comparing base-model vs DPO-aligned refusal rates."""
    fig, ax = plt.subplots(figsize=(8, 5))
    labels = ["Base (pre-DPO)", "DPO-aligned"]
    values = [base_rate * 100, aligned_rate * 100]
    colors = ["#9E9E9E", "#2E7D32"]
    bars = ax.bar(labels, values, color=colors, edgecolor="black")
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1,
            f"{val:.0f}%",
            ha="center",
            fontsize=12,
            fontweight="bold",
        )
    ax.set_ylim(0, 110)
    ax.set_ylabel("Refusal rate on adversarial prompts (%)")
    ax.set_title(
        f"DPO safety alignment — {n_prompts} adversarial prompts",
        fontsize=13,
        fontweight="bold",
    )
    fname = OUTPUT_DIR / "ex3_safety_refusal_rates.png"
    plt.tight_layout()
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"  Saved: {fname}")
    return fname
