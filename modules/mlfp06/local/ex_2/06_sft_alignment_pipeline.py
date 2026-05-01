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
# FRAMEWORK-FIRST: kailash-align, NOT raw transformers.Trainer.
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
# Raw transformers.Trainer + peft.get_peft_model loses three guarantees:
#   1. LoRA config validation against the base model's module tree
#   2. Structured adapter artefact lifecycle (not a bag of .bin files)
#   3. AdapterRegistry hand-off between training and serving
# Framework-first rule: Engine layer first, primitives only when the
# engine cannot express the behaviour.


# ════════════════════════════════════════════════════════════════════════
# TASK 1 — Load the SFT dataset
# ════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("TASK 1: Load IMDB SFT instruction pairs")
print("=" * 70)

# TODO: sft_data, train_data, eval_data = load_imdb_sft()
sft_data, train_data, eval_data = ____
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

# TODO: Read the base model name from the environment via get_base_model_name()
base_model = ____
# TODO: Build an AlignmentConfig via build_sft_config(base_model=base_model,
# lora_r=16, lora_alpha=32, num_epochs=3, output_subdir="sft_output")
config = ____

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
# Set MLFP_SKIP_SFT_TRAIN=1 to exercise the registry with synthetic
# metrics on environments without GPU / HF access.

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

        # TODO: Instantiate AlignmentPipeline(config) and await
        # pipeline.train(train_data=..., eval_data=...)
        pipeline = ____
        print("  Running SFT training (this may take several minutes)...")
        result = ____

        metrics = {
            "final_loss": result.final_loss,
            "eval_loss": result.eval_loss,
            "training_time_seconds": result.training_time_seconds,
            "adapter_path": result.adapter_path,
        }
        print(f"  Final loss:    {metrics['final_loss']:.4f}")
        print(f"  Eval loss:     {metrics['eval_loss']:.4f}")
        print(f"  Training time: {metrics['training_time_seconds']:.0f}s")

        # TODO: Instantiate registry = AdapterRegistry()
        registry = ____
        # TODO: Build an AdapterSignature with base_model_id=config.base_model_id,
        #   adapter_type="lora", training_method="sft"
        signature = ____
        # TODO: await registry.register_adapter(
        #     name="imdb_sentiment_sft_v1",
        #     adapter_path=metrics["adapter_path"],
        #     signature=signature,
        #     training_metrics={"final_loss": ..., "eval_loss": ...},
        #     tags=["imdb", "sentiment", "lora-r16"])
        # register_adapter returns an AdapterVersion (not a string).
        version = ____
        # TODO: Build a stable adapter_id string of the form
        #   f"{version.adapter_name}:v{version.version}"
        adapter_id = ____
        metrics["adapter_id"] = adapter_id
        print(f"  Registered as: {adapter_id}")

    return metrics


metrics = asyncio.run(run_sft_and_register())

# ── Checkpoint 3 ─────────────────────────────────────────────────────────
assert metrics["final_loss"] is not None, "Task 3: training should produce a loss"
assert metrics["final_loss"] > 0, "Task 3: final loss should be positive"
print("✓ Checkpoint 3 passed — SFT training + registration complete\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 4 — VISUALISE: train vs eval loss bars
# ════════════════════════════════════════════════════════════════════════

print("=" * 70)
print("TASK 4: Visualise final train vs eval loss")
print("=" * 70)

labels = ["Train loss (final)", "Eval loss"]
values = [metrics["final_loss"], metrics["eval_loss"]]

# TODO: Bar chart with train and eval loss, annotations on each bar.
# Save to OUTPUT_DIR / "ex2_sft_train_eval.png"
____
fname = OUTPUT_DIR / "ex2_sft_train_eval.png"
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
# A Singapore e-commerce platform has 37 LoRAs trained over 18 months
# with no shared record of "which adapter is live right now". Customer
# complaints cannot be traced back to a training run, compliance
# audits fail, rollback takes hours. Fix: mandate AdapterRegistry
# entries for every training run with name, base SHA, method, metrics,
# and tags. Serving pulls by registry name, not filename.

print("Singapore e-commerce adapter governance:")
# TODO: Compute annual_retraining_saving (16 runs * S$600), engineer hours
# (16 * 8), annual_engineer_saving at S$90/hr, and total_annual
annual_retraining_saving = ____
annual_engineer_hours_saved = ____
engineer_hourly_sgd = 90
annual_engineer_saving = ____
total_annual = ____
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
  audit trail you need before you layer DPO / GRPO / RLHF on top.

  Next: Exercise 3 (DPO Alignment) moves from "learn the right
  response" to "learn which response is PREFERRED".
"""
)

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
