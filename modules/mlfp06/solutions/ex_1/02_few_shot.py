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
# PREREQUISITES: 01_zero_shot.py (baseline metrics for comparison)
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
    compute_metrics,
    get_eval_docs,
    normalise_label,
    plot_comparison_bars,
    print_summary,
    run_delegate,
)

load_dotenv()


# ════════════════════════════════════════════════════════════════════════
# THEORY — What Examples Do
# ════════════════════════════════════════════════════════════════════════
# A few-shot prompt places examples INSIDE the prompt, before the real
# question. The LLM reads the examples the same way it reads any other
# context: as a pattern to continue. Because autoregressive LLMs are
# trained to predict the next token given everything before it, they
# effectively "learn" the task from the in-context examples without any
# weight updates (this is called in-context learning).
#
# Examples do three things at once:
#   1. DEMONSTRATE the output format — the LLM mimics "Sentiment: positive"
#      rather than "I think it's positive because..."
#   2. NARROW the decision boundary — ambiguous cases pattern-match against
#      the closest example
#   3. ANCHOR the category names — prevents the LLM from inventing categories
#      (e.g. "mixed", "neutral", "tentatively positive")
#
# SELECTION TIPS:
#   - 3-8 examples is the usual sweet spot; more is diminishing returns
#   - DIVERSE examples (don't use 4 similar ones)
#   - BALANCED classes (2 pos + 2 neg for binary)
#   - REPRESENTATIVE of the distribution you'll see at test time
#   - ORDER MATTERS but there's no universal best order — experiment


FEW_SHOT_EXAMPLES = [
    {
        "text": "an absolute masterpiece of storytelling and visual style.",
        "category": "positive",
    },
    {
        "text": "a tedious and predictable mess from start to finish.",
        "category": "negative",
    },
    {
        "text": "delightfully clever, with performances that elevate every scene.",
        "category": "positive",
    },
    {
        "text": "fails to land a single emotional beat in over two hours.",
        "category": "negative",
    },
]


# ════════════════════════════════════════════════════════════════════════
# TASK 2 — BUILD the few-shot classifier
# ════════════════════════════════════════════════════════════════════════


async def few_shot_classify(text: str) -> tuple[str, float, float]:
    """Classify sentiment with 4 curated examples. Returns (label, tokens, elapsed)."""
    examples_text = "\n".join(
        f'Review: "{ex["text"]}"\nSentiment: {ex["category"]}\n'
        for ex in FEW_SHOT_EXAMPLES
    )
    prompt = f"""Classify movie review snippets by sentiment.

{examples_text}
Now classify:
Review: "{text[:800]}"
Sentiment:"""

    response, tokens, elapsed = await run_delegate(prompt)
    return normalise_label(response), tokens, elapsed


# ════════════════════════════════════════════════════════════════════════
# TASK 3 — TRAIN (evaluate)
# ════════════════════════════════════════════════════════════════════════


async def evaluate() -> list[dict]:
    docs = get_eval_docs()
    results: list[dict] = []
    for i, (text, true_label) in enumerate(
        zip(docs["text"].to_list(), docs["label"].to_list())
    ):
        pred, tokens, elapsed = await few_shot_classify(text)
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

# R9A: visual proof — few-shot vs zero-shot accuracy comparison
# We use expected zero-shot baselines so this file is independently runnable
# (R10: no runtime chaining between technique files).
zero_shot_expected = {
    "strategy": "Zero-Shot",
    "accuracy": 0.80,
    "total_tokens": 1500,
    "avg_latency_s": 1.0,
    "n": 20,
}
few_shot_metrics = compute_metrics(few_shot_results, "Few-Shot")
plot_comparison_bars(
    [zero_shot_expected, few_shot_metrics],
    title="Few-Shot vs Zero-Shot — Accuracy / Cost / Latency",
    filename="ex1_02_few_shot_comparison.png",
)

# INTERPRETATION: Few-shot typically gains 3-8 percentage points over
# zero-shot in exchange for ~3x the input tokens. The cost rises; the
# output format gets more reliable; ambiguous edge cases improve.
# If your zero-shot baseline was 80% and your accuracy bar is 90%,
# few-shot is the first thing to try.
# The bar chart makes the trade-off visible: how many dollars of extra
# inference cost does each percentage point of accuracy cost you?


# ════════════════════════════════════════════════════════════════════════
# TASK 5 — APPLY: MAS Supervisory Report Triage
# ════════════════════════════════════════════════════════════════════════
# SCENARIO: The Monetary Authority of Singapore (MAS) receives ~800
# supervisory incident reports per week from regulated financial
# institutions. Each report needs tagging as "material" or "routine"
# so senior examiners prioritise the material ones.
#
# Zero-shot struggles because:
#   - The domain is specialised (banking operational risk vocabulary)
#   - The distinction between "material" and "routine" is organisation-
#     specific — MAS's definition differs from a textbook definition
#   - Examiner time is expensive; misclassification costs senior time
#
# Few-shot fits because:
#   - MAS can provide 6 examples from their historical log that
#     encode THEIR definition of "material" (not a generic one)
#   - The LLM mimics those examples instead of relying on its
#     pre-training prior
#   - Cost is still bounded (6 examples ~~ 600 extra input tokens per call;
#     800 calls/week ~~ S$30/week)
#
# BUSINESS IMPACT: Senior examiners cost ~S$250/hour. Each avoided
# misrouting saves ~30 minutes of senior triage time = S$125/incident.
# At 800 reports/week and a 5% improvement over zero-shot routing
# (40 incidents/week), that's ~S$5,000/week = S$260,000/year in
# reclaimed senior capacity, against S$1,560/year in LLM cost.
# 167x ROI.
#
# OPERATIONAL NOTE: Store the examples in a version-controlled repo
# (not in the Python file). When MAS's definition evolves, the examples
# are updated by the compliance team without touching code.


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
  [x] Built a few-shot prompt with curated positive/negative examples
  [x] Understood in-context learning — LLMs learn patterns from the prompt
  [x] Traded longer prompts for better accuracy and output consistency
  [x] Sized the approach against a real MAS supervisory triage use case
  [x] Computed the ROI of examples vs zero-shot inference cost

  KEY INSIGHT: Examples are the cheapest form of "training" an LLM.
  You don't update weights — you update the prompt. When the
  definition changes, you edit the examples, not retrain the model.

  Next: 03_chain_of_thought.py — make the model show its reasoning
  before answering, and watch accuracy climb on ambiguous inputs.
"""
)
