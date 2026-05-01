# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# MLFP06 — Exercise 1.2: Few-Shot Prompting with Curated Examples
# ════════════════════════════════════════════════════════════════════════
#
# WHAT YOU'LL LEARN:
#   - Construct a few-shot prompt with 3-8 curated examples
#   - Understand why examples steer output format AND decision boundary
#   - Trade longer prompts for better consistency and accuracy
#   - Compare few-shot against the zero-shot baseline
#
# PREREQUISITES: 01_zero_shot.py
# ESTIMATED TIME: ~30 min
#
# TASKS:
#   1. Theory — what examples do to LLM behaviour
#   2. Build — prompt with curated positive/negative exemplars
#   3. Train — evaluate on SST-2 eval docs
#   4. Visualise — few-shot metrics vs zero-shot expectation
#   5. Apply — MAS supervisory report triage
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
# THEORY — What Examples Do
# ════════════════════════════════════════════════════════════════════════
# In-context learning: the LLM reads examples in the prompt and patterns
# its response on them. Examples (a) demonstrate output format, (b) narrow
# the decision boundary, (c) anchor category names. 3-8 examples is the
# sweet spot; diverse, balanced, representative.


# TODO: Define FEW_SHOT_EXAMPLES as a list of 4 dicts with keys "text"
# and "category". Use 2 positive + 2 negative, all distinct styles.
FEW_SHOT_EXAMPLES = ____


# ════════════════════════════════════════════════════════════════════════
# TASK 2 — BUILD the few-shot classifier
# ════════════════════════════════════════════════════════════════════════


async def few_shot_classify(text: str) -> tuple[str, float, float]:
    """Classify sentiment with 4 curated examples. Returns (label, tokens, elapsed)."""
    # TODO: Build an examples_text string by joining each example as:
    #   Review: "<text>"
    #   Sentiment: <category>
    examples_text = ____

    # TODO: Build the full prompt: intro line, examples_text, "Now classify:",
    # the new review (truncated 800 chars), and "Sentiment:" trailing.
    prompt = ____

    # TODO: run_delegate, normalise, return
    ____


# ════════════════════════════════════════════════════════════════════════
# TASK 3 — TRAIN (evaluate)
# ════════════════════════════════════════════════════════════════════════


async def evaluate() -> list[dict]:
    docs = get_eval_docs()
    results: list[dict] = []
    # TODO: Loop over docs, call few_shot_classify, build the results list
    # with the same keys as 01_zero_shot.py. Print the first 5.
    ____
    return results


print("\n" + "=" * 70)
print("  Few-Shot Classification on SST-2 (4 exemplars)")
print("=" * 70)
few_shot_results = asyncio.run(evaluate())


# ── Checkpoint ──────────────────────────────────────────────────────────
assert len(few_shot_results) > 0, "Task 3: few-shot should produce results"
assert all(
    r["pred"] in CATEGORIES or r["pred"] == "unknown" for r in few_shot_results
), "Predictions must be in CATEGORIES or 'unknown'"
print("\n[ok] Checkpoint passed — few-shot evaluation complete\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 4 — VISUALISE
# ════════════════════════════════════════════════════════════════════════
print_summary(few_shot_results, "Few-Shot (4 examples)")


# ════════════════════════════════════════════════════════════════════════
# TASK 5 — APPLY: MAS Supervisory Report Triage
# ════════════════════════════════════════════════════════════════════════
# MAS receives ~800 supervisory incident reports/week that need tagging
# as "material" or "routine". Few-shot fits because: domain-specific
# vocabulary, MAS's OWN definition of "material" differs from the textbook,
# and 6 examples encode that definition without weight updates.
#
# BUSINESS IMPACT: Senior examiners cost S$250/hr. A 5% routing improvement
# on 800 reports/week saves ~S$5K/week = S$260K/year, vs S$1,560/year in
# LLM cost. 167x ROI.


# ════════════════════════════════════════════════════════════════════════
# REFLECTION
# ════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  WHAT YOU'VE MASTERED")
print("=" * 70)
print(
    """
  [x] Built a few-shot prompt with curated positive/negative examples
  [x] Understood in-context learning — LLMs learn patterns from the prompt
  [x] Traded longer prompts for better accuracy and output consistency

  Next: 03_chain_of_thought.py — make the model show its reasoning.
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
