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
    compute_metrics,
    get_eval_docs,
    normalise_label,
    plot_accuracy_bars,
    print_summary,
    run_delegate,
)

load_dotenv()


# ════════════════════════════════════════════════════════════════════════
# THEORY — Why Zero-Shot Works
# ════════════════════════════════════════════════════════════════════════
# A large LLM has been pre-trained on trillions of tokens that include
# book reviews, product reviews, news commentary, and movie criticism.
# Every one of those documents contains sentiment signals ("wonderful",
# "dreadful", "tedious", "masterpiece"). The LLM has already learned the
# mapping from language -> sentiment WITHOUT ever being told that task.
#
# Zero-shot prompting exploits that: you describe the task in plain
# English, name the categories, and ask for the label. No examples, no
# fine-tuning, no training data. The LLM generalises from its pre-training
# distribution to your task in a single forward pass.
#
# COST / QUALITY TRADE-OFF:
#   + cheapest (shortest prompt = fewest input tokens)
#   + fastest (single LLM call per item, no reasoning)
#   - inconsistent output format (may return "Positive" vs "positive")
#   - lowest accuracy on ambiguous or domain-shifted inputs
#   - no examples to steer the model's output style
#
# WHEN TO USE: Well-known tasks, large-capability models, low-stakes
# classification, high-volume cheap triage.


# ════════════════════════════════════════════════════════════════════════
# TASK 2 — BUILD the zero-shot classifier
# ════════════════════════════════════════════════════════════════════════


async def zero_shot_classify(text: str) -> tuple[str, float, float]:
    """Classify sentiment with zero examples. Returns (label, total_tokens, elapsed_s)."""
    prompt = f"""Classify the sentiment of the following movie review snippet
into exactly one category.

Categories: {', '.join(CATEGORIES)}

Review: "{text[:800]}"

Respond with ONLY the category name, nothing else."""

    response, tokens, elapsed = await run_delegate(prompt)
    return normalise_label(response), tokens, elapsed


# ════════════════════════════════════════════════════════════════════════
# TASK 3 — TRAIN (evaluate across SST-2 eval docs)
# ════════════════════════════════════════════════════════════════════════
# Zero-shot has no training loop. Instead we run the classifier over a
# held-out slice of SST-2 and measure accuracy/cost/latency directly.


async def evaluate() -> list[dict]:
    docs = get_eval_docs()
    results: list[dict] = []
    texts = docs["text"].to_list()
    labels = docs["label"].to_list()
    for i, (text, true_label) in enumerate(zip(texts, labels)):
        pred, tokens, elapsed = await zero_shot_classify(text)
        correct = pred == true_label
        results.append(
            {
                "text": text,
                "pred": pred,
                "true": true_label,
                "correct": correct,
                "tokens": tokens,
                "elapsed": elapsed,
            }
        )
        if i < 5:
            mark = "[ok]" if correct else "[miss]"
            print(f"  Doc {i+1}: pred={pred}, true={true_label} {mark}")
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
# TASK 4 — VISUALISE — headline metrics + per-category accuracy chart
# ════════════════════════════════════════════════════════════════════════
print_summary(zero_shot_results, "Zero-Shot")

# R9A: visual proof — per-category accuracy bar chart
plot_accuracy_bars(
    zero_shot_results,
    CATEGORIES,
    title="Zero-Shot Accuracy by Category (SST-2)",
    filename="ex1_01_zero_shot_accuracy.png",
)

# INTERPRETATION: Zero-shot gives you a baseline with zero engineering
# effort. If the number here is "good enough" for your use case, STOP —
# every technique below this one costs more tokens, more latency, and
# more prompt-engineering effort. Only move up the ladder when zero-shot
# fails your accuracy bar.
# The bar chart reveals whether errors are SYMMETRIC (equal miss rate on
# both categories) or SKEWED (e.g. the model defaults to "positive" on
# ambiguous inputs). Skewed errors signal that zero-shot's prior from
# pre-training is biased — few-shot examples (Exercise 1.2) can fix it.


# ════════════════════════════════════════════════════════════════════════
# TASK 5 — APPLY: DBS Bank Multilingual Review Triage
# ════════════════════════════════════════════════════════════════════════
# SCENARIO: DBS Bank (Singapore) receives ~40,000 app-store reviews per
# month across English, Mandarin, Malay, and Tamil. The CX team wants
# every review tagged positive/negative within 10 minutes of posting so
# complaints can be routed to the on-call support lead.
#
# Why zero-shot is the right tool here:
#   - The LLM already understands sentiment across all four languages
#   - No labelled training data exists for Malay/Tamil reviews
#   - Cost matters — 40K reviews/mo at CoT cost would be ~S$2,000/mo;
#     zero-shot is ~S$120/mo (roughly 15x cheaper)
#   - The downstream action (route to agent) is reversible, so occasional
#     misclassifications are recoverable
#
# BUSINESS IMPACT: A 10-minute detection window on negative reviews lets
# DBS's support team reach frustrated customers BEFORE they escalate to
# social media. Industry data from SGX-listed banks shows each prevented
# viral complaint is worth ~S$8,000 in avoided remediation + PR cost.
# Catching even 20 extra negative reviews/month = S$160K/mo in avoided
# brand damage — vs S$120/mo in LLM inference cost. 1,300x ROI.
#
# LIMITATIONS:
#   - Sarcasm is hard ("wow, another outage, just what I needed")
#   - Mixed reviews (4-star with a complaint) may be misrouted
#   - For these edge cases, Exercise 1.3 (chain-of-thought) does better


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
  [x] Identified a production scenario (DBS review triage) where
      zero-shot is the economically optimal choice

  KEY INSIGHT: Zero-shot is the first rung of the prompting ladder.
  Every subsequent technique costs more per call — only climb higher
  when the business outcome needs the accuracy.

  Next: 02_few_shot.py — add a handful of examples and watch accuracy
  improve without changing the model or the task.
"""
)
