# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# MLFP06 — Exercise 1.1: Zero-Shot Classification with Kaizen Delegate
# ════════════════════════════════════════════════════════════════════════
#
# WHAT YOU'LL LEARN:
#   - Call an LLM with zero examples using Kaizen Delegate
#   - Write a minimal classification prompt (task + categories + input)
#   - Normalise free-form LLM text into a discrete label
#   - Measure accuracy, cost, and latency across a sample
#
# PREREQUISITES: M5 (transformers, attention). Understanding that LLMs
# predict the next token — prompts shift which tokens become likely.
#
# ESTIMATED TIME: ~25 min
#
# TASKS:
#   1. Theory — why zero-shot works
#   2. Build — write the zero-shot prompt
#   3. Train — there is no training; we EVALUATE on SST-2 eval docs
#   4. Visualise — per-doc predictions + headline metrics
#   5. Apply — Singapore DBS multilingual review triage
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import asyncio

from dotenv import load_dotenv

from shared.mlfp06.ex_1 import (
    CATEGORIES,
    get_eval_docs,
    normalise_label,
    print_summary,
    run_delegate,
)

load_dotenv()


# ════════════════════════════════════════════════════════════════════════
# THEORY — Why Zero-Shot Works
# ════════════════════════════════════════════════════════════════════════
# A large LLM has been pre-trained on trillions of tokens. Every sentiment
# word ("wonderful", "tedious", "masterpiece") already has a representation
# in the model. Zero-shot exploits that prior — no examples, no fine-tuning.
#
# Cost/quality trade-off: cheapest, fastest, least consistent. Use it as
# your baseline before climbing the prompting ladder.


# ════════════════════════════════════════════════════════════════════════
# TASK 2 — BUILD the zero-shot classifier
# ════════════════════════════════════════════════════════════════════════


async def zero_shot_classify(text: str) -> tuple[str, float, float]:
    """Classify sentiment with zero examples. Returns (label, tokens, elapsed)."""
    # TODO: Build a prompt that (a) names the task, (b) lists the categories
    # from CATEGORIES, (c) includes the review text, (d) asks for ONLY the
    # category name. Truncate text to 800 chars.
    prompt = ____

    # TODO: Call run_delegate(prompt) and unpack (response, tokens, elapsed)
    response, tokens, elapsed = ____

    # TODO: Normalise the free-form response into a discrete label via
    # normalise_label(). Return (label, tokens, elapsed).
    ____


# ════════════════════════════════════════════════════════════════════════
# TASK 3 — TRAIN (evaluate across SST-2 eval docs)
# ════════════════════════════════════════════════════════════════════════


async def evaluate() -> list[dict]:
    docs = get_eval_docs()
    results: list[dict] = []
    # TODO: Iterate over zip(docs["text"].to_list(), docs["label"].to_list()),
    # call zero_shot_classify for each, and append a dict with keys:
    # text, pred, true, correct, tokens, elapsed. Print the first 5.
    ____
    return results


print("\n" + "=" * 70)
print("  Zero-Shot Classification on SST-2")
print("=" * 70)
zero_shot_results = asyncio.run(evaluate())


# ── Checkpoint ──────────────────────────────────────────────────────────
assert len(zero_shot_results) > 0, "Task 3: zero-shot should produce results"
assert all(
    r["pred"] in CATEGORIES or r["pred"] == "unknown" for r in zero_shot_results
), "Predictions must be in CATEGORIES or 'unknown'"
print("\n[ok] Checkpoint passed — zero-shot evaluation complete\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 4 — VISUALISE — headline metrics
# ════════════════════════════════════════════════════════════════════════
print_summary(zero_shot_results, "Zero-Shot")


# ════════════════════════════════════════════════════════════════════════
# TASK 5 — APPLY: DBS Bank Multilingual Review Triage
# ════════════════════════════════════════════════════════════════════════
# DBS Bank receives ~40K app-store reviews/month across English, Mandarin,
# Malay, Tamil. Zero-shot is the right tool: no labelled data exists for
# Malay/Tamil, the LLM already knows sentiment, and cost matters at scale.
#
# BUSINESS IMPACT: Each viral complaint prevented is worth ~S$8K. Catching
# 20 extra negatives/month = S$160K/mo, vs S$120/mo in LLM cost. 1,300x ROI.
#
# LIMITATIONS: Sarcasm and mixed reviews are hard — those need CoT (Ex 1.3).


# ════════════════════════════════════════════════════════════════════════
# REFLECTION
# ════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  WHAT YOU'VE MASTERED")
print("=" * 70)
print(
    """
  [x] Invoked an LLM via Kaizen Delegate with a cost budget
  [x] Wrote a minimal zero-shot classification prompt
  [x] Normalised free-form LLM output into discrete labels
  [x] Measured accuracy, cost, and latency on a real SST-2 sample
  [x] Identified a production scenario where zero-shot is optimal

  Next: 02_few_shot.py — add a handful of examples and watch accuracy improve.
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

# Primary lens: Output (LLM-as-judge over the classifier's predictions).
# We'd pass the predicted label + true label as (prompt, response) pairs
# and ask a judge to score coherence/faithfulness. Attention is optional
# here — only meaningful for open-weight models.
if False:  # scaffold — requires OPENAI_API_KEY + judge budget
    obs = LLMObservatory(run_id="ex_1_prompting_run")
    # Build (prompt, response) pairs from the exercise results:
    # prompts = [r["text"] for r in zero_shot_results]
    # responses = [r["pred"] for r in zero_shot_results]
    # obs.output.evaluate(prompts, responses, criteria="coherence,label_fidelity")
    print("\n── LLM Observatory Report ──")
    findings = obs.report()
    # Optional: obs.plot_dashboard().show()

# ══════ EXPECTED OUTPUT (synthesised reference) ══════
# ════════════════════════════════════════════════════════════════
#   LLM Observatory — composite Prescription Pad
# ════════════════════════════════════════════════════════════════
#   [✓] Output     (HEALTHY): judge coherence 0.91, label_fidelity 0.84
#   [?] Attention  (n/a): API-only model — lens short-circuits to UNKNOWN
#   [?] Retrieval  (n/a): no retrieval in this exercise
#   [?] Agent      (n/a): no tool-using agent in this exercise
#   [?] Alignment  (n/a): no fine-tuning signal to compare
#   [?] Governance (n/a): no PACT engine attached
# ════════════════════════════════════════════════════════════════
#
# STUDENT INTERPRETATION GUIDE — reading the Prescription Pad:
#
#  [OUTPUT LENS] judge coherence 0.91 is HEALTHY (>0.80). label_fidelity
#     0.84 means the judge thought 84% of predictions were coherent
#     labels in the allowed category set. The remaining 16% are where
#     the LLM drifted off-template ("Positive sentiment, I think" instead
#     of "positive") — a signature of under-constrained zero-shot.
#     >> Prescription: tighten the prompt (structured output in ex_1.6)
#        or add few-shot exemplars (ex_1.2).
#
#  [ATTENTION LENS] GPT-class models are API-only — the Attention lens
#     short-circuits to UNKNOWN. To actually inspect attention, switch to
#     an open-weight model (e.g. Qwen2-0.5B via transformers) and call
#     obs.attention.logit_lens(prompt=..., answer_token=...).
#
#  [OTHER LENSES] All n/a — prompting has no retrieval, no agent loop, no
#     fine-tuning pair, no governance envelope. This is exactly the
#     signature the design doc predicts for Lesson 6.1.
# ════════════════════════════════════════════════════════════════════


# ════════════════════════════════════════════════════════════════════════
