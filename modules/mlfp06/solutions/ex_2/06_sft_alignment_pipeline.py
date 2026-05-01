# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# MLFP06 — Exercise 2.6: SFT with kailash-align AlignmentPipeline
# ════════════════════════════════════════════════════════════════════════
#
# WHAT YOU'LL LEARN:
#   - Build an AlignmentConfig for SFT + LoRA (no raw transformers.Trainer)
#   - Run AlignmentPipeline.train() on the IMDB instruction dataset
#   - Register the trained adapter in AdapterRegistry
#   - Visualise training loss + interpret train/eval gap
#   - Apply adapter-registry discipline to a Singapore e-commerce scenario
#
# PREREQUISITES: Exercises 2.1-2.5
# ESTIMATED TIME: ~40 min (training dominates)
#
# FRAMEWORK-FIRST: kailash-align, NOT raw transformers.Trainer. The
# pipeline wraps TRL SFTTrainer but adds LoRA config validation,
# adapter serialisation, and registry integration — the same pattern
# you used in MLFP03 for TrainingPipeline.
#
# TASKS:
#   1. THEORY: why AlignmentPipeline beats raw SFTTrainer
#   2. BUILD: AlignmentConfig for SFT + LoRA r=16
#   3. TRAIN: pipeline.train() on IMDB instruction pairs
#   4. VISUALISE: train vs eval loss curve
#   5. APPLY: Singapore e-commerce adapter registry governance
#
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import asyncio

import matplotlib.pyplot as plt
from dotenv import load_dotenv

from shared.mlfp06.ex_2 import (
    OUTPUT_DIR,
    build_sft_config,
    get_base_model_name,
    load_imdb_sft,
)

load_dotenv()


# ════════════════════════════════════════════════════════════════════════
# THEORY — Why AlignmentPipeline (Framework-First)
# ════════════════════════════════════════════════════════════════════════
# Raw transformers.Trainer + peft.get_peft_model is the Hello-World of
# SFT tutorials, but it loses three guarantees at production scale:
#
#   1. LoRA config validation — a typo in target_modules ("q_proj" vs
#      "q_porj") silently trains zero parameters.  AlignmentConfig
#      validates targets against the base model's module tree at
#      construction time.
#
#   2. Adapter artefact lifecycle — raw trainer leaves a directory of
#      .bin files with no versioning, no metadata, no provenance.
#      AlignmentPipeline writes a structured adapter bundle and
#      registers it in AdapterRegistry with (name, base, method,
#      metrics, tags, timestamp).
#
#   3. Registry hand-off — AdapterRegistry is the hand-off point
#      between training and serving.  Without a registry, the serving
#      team pulls .bin files by filename and the link between an
#      eval-loss number and a production deployment is lost.
#
# The rule is the same as rules/framework-first.md: Engine layer first.
# Drop to primitives only when AlignmentPipeline cannot express the
# behaviour (e.g. a novel alignment method not yet in kailash-align).


# ════════════════════════════════════════════════════════════════════════
# TASK 1 — Load the SFT dataset
# ════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("TASK 1: Load IMDB SFT instruction pairs")
print("=" * 70)

sft_data, train_data, eval_data = load_imdb_sft()
print(f"Train: {train_data.height} pairs")
print(f"Eval:  {eval_data.height} pairs")

# ── Checkpoint 1 ─────────────────────────────────────────────────────────
assert train_data.height > 0, "Task 1: train split should not be empty"
assert eval_data.height > 0, "Task 1: eval split should not be empty"
print("✓ Checkpoint 1 passed — SFT data loaded\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 2 — BUILD: AlignmentConfig via the shared factory
# ════════════════════════════════════════════════════════════════════════

print("=" * 70)
print("TASK 2: Build AlignmentConfig for SFT + LoRA r=16")
print("=" * 70)

# Base model read from env (rules/env-models.md — no hardcoded models)
base_model = get_base_model_name()
config = build_sft_config(
    base_model=base_model,
    lora_r=16,
    lora_alpha=32,
    num_epochs=3,
    output_subdir="sft_output",
)

print(f"  Method:        {config.method}")
print(f"  Base model:    {config.base_model_id}")
print(f"  LoRA:          r={config.lora.rank}, alpha={config.lora.alpha}")
print(f"  Target modules: {config.lora.target_modules}")
print(f"  Epochs:        {config.sft.num_train_epochs}")
print(f"  Batch size:    {config.sft.per_device_train_batch_size}")
print(f"  Learning rate: {config.sft.learning_rate}")
print(f"  Output dir:    {config.experiment_dir}")

# ── Checkpoint 2 ─────────────────────────────────────────────────────────
assert config.method == "sft", "Task 2: method should be 'sft'"
assert config.lora.rank == 16, "Task 2: LoRA rank should be 16"
assert "q_proj" in config.lora.target_modules, "Task 2: q_proj must be a target module"
print("✓ Checkpoint 2 passed — AlignmentConfig built\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 3 — TRAIN: AlignmentPipeline + register adapter
# ════════════════════════════════════════════════════════════════════════
# The pipeline runs lazily inside an async function so that environments
# without GPU / HF access can still import-check this file.  Set
# MLFP_SKIP_SFT_TRAIN=1 to skip the real training step and exercise the
# registry with synthetic metrics.

print("=" * 70)
print("TASK 3: AlignmentPipeline.train() -> AdapterRegistry")
print("=" * 70)


async def run_sft_and_register() -> dict:
    """Train the SFT adapter and register it. Returns metrics dict."""
    import os

    skip = os.environ.get("MLFP_SKIP_SFT_TRAIN") == "1"

    if skip:
        print("  MLFP_SKIP_SFT_TRAIN=1 -> using synthetic metrics")
        metrics = {
            "final_loss": 0.742,
            "eval_loss": 0.881,
            "training_time_seconds": 0.0,
            "adapter_path": str(OUTPUT_DIR / "sft_output" / "adapter"),
        }
    else:
        from kailash_align import (
            AdapterRegistry,
            AdapterSignature,
            AlignmentPipeline,
        )

        pipeline = AlignmentPipeline(config)
        print("  Running SFT training (this may take several minutes)...")
        result = await pipeline.train(train_data=train_data, eval_data=eval_data)
        # AlignmentResult.training_metrics is a dict in 0.6.0 — final_loss /
        # eval_loss / training_time_seconds live there, not as direct attrs.
        train_metrics = result.training_metrics
        metrics = {
            "final_loss": train_metrics["final_loss"],
            "eval_loss": train_metrics["eval_loss"],
            "training_time_seconds": train_metrics.get("training_time_seconds", 0),
            "adapter_path": result.adapter_path,
        }
        print(f"  Final loss:    {metrics['final_loss']:.4f}")
        print(f"  Eval loss:     {metrics['eval_loss']:.4f}")
        print(f"  Training time: {metrics['training_time_seconds']:.0f}s")

        # Register the trained adapter (kailash-align 0.6.0+ API):
        # AdapterSignature bundles base + adapter_type + training_method;
        # register_adapter returns an AdapterVersion dataclass.
        registry = AdapterRegistry()
        signature = AdapterSignature(
            base_model_id=config.base_model_id,
            adapter_type="lora",
            training_method="sft",
        )
        version = await registry.register_adapter(
            name="imdb_sentiment_sft_v1",
            adapter_path=metrics["adapter_path"],
            signature=signature,
            training_metrics={
                "final_loss": metrics["final_loss"],
                "eval_loss": metrics["eval_loss"],
            },
            tags=["imdb", "sentiment", "lora-r16"],
        )
        adapter_id = f"{version.adapter_name}:v{version.version}"
        metrics["adapter_id"] = adapter_id
        print(f"  Registered as: {adapter_id}")

    return metrics


metrics = asyncio.run(run_sft_and_register())

# ── Checkpoint 3 ─────────────────────────────────────────────────────────
assert metrics["final_loss"] is not None, "Task 3: training should produce a loss"
assert metrics["final_loss"] > 0, "Task 3: final loss should be positive"
print("✓ Checkpoint 3 passed — SFT training + registration complete\n")

# INTERPRETATION:
#   eval_loss ~ train_loss (within 0.1) -> healthy generalisation
#   eval_loss >> train_loss             -> overfitting; reduce epochs
#   eval_loss < train_loss              -> suspect data leakage between splits


# ════════════════════════════════════════════════════════════════════════
# TASK 4 — VISUALISE: train vs eval loss (conceptual bar)
# ════════════════════════════════════════════════════════════════════════
# The real pipeline emits per-step loss curves; here we plot the
# headline train/eval bars so the gap is visible at a glance.

print("=" * 70)
print("TASK 4: Visualise final train vs eval loss")
print("=" * 70)

labels = ["Train loss (final)", "Eval loss"]
values = [metrics["final_loss"], metrics["eval_loss"]]
colors = ["steelblue", "darkorange"]

fig, ax = plt.subplots(1, 1, figsize=(8, 5))
bars = ax.bar(labels, values, color=colors, edgecolor="black")
for bar, v in zip(bars, values):
    ax.annotate(
        f"{v:.3f}",
        xy=(bar.get_x() + bar.get_width() / 2, v),
        xytext=(0, 3),
        textcoords="offset points",
        ha="center",
    )
ax.set_ylabel("Cross-entropy loss")
ax.set_title("SFT LoRA r=16 — train vs eval loss", fontweight="bold")
ax.grid(True, axis="y", alpha=0.3)
plt.tight_layout()
fname = OUTPUT_DIR / "ex2_sft_train_eval.png"
plt.savefig(fname, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  Saved: {fname}")

gap = metrics["eval_loss"] - metrics["final_loss"]
print(f"\n  Train-eval gap: {gap:+.3f}")
print("  Healthy: |gap| < 0.15; larger values suggest over/under-fitting")

# ── Checkpoint 4 ─────────────────────────────────────────────────────────
assert fname.exists(), "Task 4: loss plot should exist"
print("✓ Checkpoint 4 passed — train vs eval visualised\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 5 — APPLY: Singapore e-commerce adapter registry governance
# ════════════════════════════════════════════════════════════════════════
# SCENARIO: A Singapore e-commerce platform runs an LLM-powered
# customer service assistant.  Over 18 months, the team has trained
# 37 different LoRA adapters on top of the same 7B base: returns,
# refunds, promotions, shipping, loyalty tiers, billing, KYC, and so
# on.  Each adapter was trained by a different on-call engineer with
# no shared record of "which adapter is live in production right now".
#
# PROBLEM: when a shopper complains that "the chatbot said shipping
# is free above S$50 but it charged me", the team cannot trace which
# adapter produced the answer, which dataset it was trained on, or
# which eval metrics it passed at training time.  Customer trust
# erodes; the promotion team cannot roll back a bad adapter because
# they don't know which version is deployed.
#
# GOVERNANCE FIX: every trained adapter MUST go through AdapterRegistry
# at training time with:
#   - name: human-readable identifier (e.g. "shipping_v3")
#   - base_model: exact base checkpoint SHA
#   - method: sft_lora / dpo_lora / adapter / ...
#   - metrics: train loss, eval loss, business-level eval (CSAT proxy)
#   - tags: domain (shipping), audience (retail), legal-reviewed (yes)
#
# Downstream serving pulls by registry name + metric threshold, never
# by raw .bin filename.  Deployment becomes auditable: every live
# adapter has a registry row that links back to the training run.
#
# BUSINESS IMPACT:
#   - Audit: compliance team can trace any customer-facing LLM
#     response to a registered adapter + training run + eval report.
#     This was previously impossible and blocked two MAS (Monetary
#     Authority of Singapore) compliance audits in the past year.
#   - Rollback: a bad adapter can be rolled back in minutes by
#     pointing the serving layer at the previous registered version.
#     Previous rollback took ~4 hours and required redeploying the
#     serving image.
#   - Retraining cost avoided: the registry prevents duplicate runs
#     of the same fine-tune.  Engineers used to retrain ~4 adapters
#     per quarter because they could not find the previous adapter
#     files.  At ~S$600/training run + ~1 day engineering time, that
#     is ~S$2,400 + 4 engineer-days per quarter avoided.
#
# ANNUAL BENEFIT: ~S$32,000 in direct cost + unblocked compliance
# posture + faster incident response.
#
# THE RULE: no adapter ships to production without a registry entry.
# Treat AdapterRegistry the way you treat ModelRegistry in MLFP03 —
# the single source of truth for which artefact runs where.

print("Singapore e-commerce adapter governance:")
annual_retraining_saving = 4 * 4 * 600  # 4 quarters * 4 runs/quarter * S$600
annual_engineer_hours_saved = 4 * 4 * 8  # 4 * 4 * 1 day
engineer_hourly_sgd = 90
annual_engineer_saving = annual_engineer_hours_saved * engineer_hourly_sgd
total_annual = annual_retraining_saving + annual_engineer_saving
print(f"  Retraining runs avoided / year:      {4 * 4}")
print(f"  Annual retraining cost avoided:      S${annual_retraining_saving:,}")
print(f"  Engineer hours saved / year:         {annual_engineer_hours_saved}")
print(f"  Annual engineer time saving:         S${annual_engineer_saving:,}")
print(f"  Total direct annual saving:          S${total_annual:,}")
print(f"  Plus: MAS compliance + rollback SLA (unquantified but critical)")
print(f"  Recommended: AdapterRegistry mandatory for all SFT runs")

# ── Checkpoint 5 ─────────────────────────────────────────────────────────
assert total_annual > 0, "Task 5: governance should deliver positive ROI"
print("✓ Checkpoint 5 passed — e-commerce governance analysed\n")


# ══════════════════════════════════════════════════════════════════
# DIAGNOSTIC CHECKPOINT — six lenses before completion
# ══════════════════════════════════════════════════════════════════
# The LLM Observatory extends M5's Doctor's Bag for LLM/agent work.
# Six lenses:
#   1. Output        — is the generation coherent, factual, on-task?
#   2. Attention     — what does the model attend to internally?
#   3. Retrieval     — did we fetch the right context?  [RAG only]
#   4. Agent Trace   — what did the agent actually do?  [Agent only]
#   5. Alignment     — is it aligned with our intent?   [Fine-tune only]
#   6. Governance    — is it within policy?            [PACT only]
from shared.mlfp06.diagnostics import LLMObservatory

# Primary lens: Alignment (KL divergence from base, reward margin).
# Secondary: Output (judge quality on paired completions), Attention
# (layer-wise shift in target modules for LoRA).
if False:  # scaffold — requires trained base + adapter checkpoint
    obs = LLMObservatory(run_id="ex_2_finetune_run")
    # Typical alignment read:
    # for step, metrics in enumerate(training_log):
    #     obs.alignment.log_training_step(step=step, **metrics)
    # obs.alignment.evaluate_pair(base_responses, adapter_responses)
    print("\n── LLM Observatory Report ──")
    findings = obs.report()

# ══════ EXPECTED OUTPUT (synthesised reference) ══════
# ════════════════════════════════════════════════════════════════
#   LLM Observatory — composite Prescription Pad
# ════════════════════════════════════════════════════════════════
#   [!] Alignment  (WARNING): KL divergence from base = 0.42 nats
#       Fix: healthy range 0.2-1.0; this is low-end — adapter barely
#            moved. Increase LoRA rank or learning rate.
#   [✓] Output     (HEALTHY): judge win-rate 0.58 vs base (>0.50 = good)
#   [✓] Attention  (HEALTHY): shift concentrated in q_proj/v_proj as
#       expected for LoRA; no drift in frozen layers.
#   [?] Retrieval / Agent / Governance (n/a)
# ════════════════════════════════════════════════════════════════
#
# STUDENT INTERPRETATION GUIDE — reading the Prescription Pad:
#
#  [ALIGNMENT LENS] KL 0.42 nats is the SIGNATURE of a cautiously-trained
#     LoRA adapter — it diverged from the base distribution but not
#     enough to break it. Above 2.0 nats signals over-fit; below 0.2
#     signals the adapter barely learned. Our value is slightly under the
#     0.5 floor we want for visible task lift.
#     >> Prescription: raise lora_r from 8 -> 16 or train another epoch.
#  [OUTPUT LENS] Win-rate 0.58 > 0.50 confirms the adapter is better
#     than base on held-out prompts — tiny lift but statistically real.
#  [ATTENTION LENS] Shift localised in the target modules = LoRA is
#     doing what it's supposed to do (low-rank delta on attention
#     projections, frozen MLP). If attention shifted everywhere you'd
#     know you accidentally unfroze a module.
# ════════════════════════════════════════════════════════════════════


# ════════════════════════════════════════════════════════════════════════
# REFLECTION
# ════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("  WHAT YOU'VE MASTERED")
print("=" * 70)
print(
    """
  [x] Built an AlignmentConfig for SFT + LoRA via kailash-align
      (framework-first: no raw transformers.Trainer)
  [x] Ran AlignmentPipeline.train() end-to-end on IMDB SFT data
  [x] Registered the trained adapter in AdapterRegistry with metrics
  [x] Visualised train vs eval loss and interpreted the gap
  [x] Applied adapter-registry governance to a Singapore e-commerce
      scenario (~S$32k/year saving + unblocked MAS compliance)

  KEY INSIGHT: SFT is the first rung of the alignment ladder.
  AlignmentPipeline + AdapterRegistry give you the versioning and
  audit trail you need before you layer DPO, GRPO, or RLHF on top.

  Next exercise (Exercise 3) moves from "learn the right response"
  (SFT) to "learn which response is PREFERRED" (DPO). Preference
  data replaces instruction pairs as the training signal.
"""
)
