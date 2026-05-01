# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# MLFP06 — Exercise 1.4: Zero-Shot CoT ("Let's Think Step by Step")
# ════════════════════════════════════════════════════════════════════════
#
# WHAT YOU'LL LEARN:
#   - Trigger step-by-step reasoning WITHOUT a hand-crafted template
#   - Understand why one magic phrase replaces 4 reasoning steps
#   - Compare zero-shot CoT against full CoT and zero-shot
#   - Grasp the cost/quality ratio sweet spot
#
# PREREQUISITES: 03_chain_of_thought.py (for the full-CoT comparison)
# ESTIMATED TIME: ~25 min
#
# TASKS:
#   1. Theory — Kojima et al. 2022 and why the trigger phrase works
#   2. Build — the tiny prompt change
#   3. Train — evaluate on SST-2
#   4. Visualise — compare cost/accuracy vs full CoT
#   5. Apply — SingPost delivery-complaint triage
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
    plot_tokens_vs_accuracy,
    print_summary,
    run_delegate,
)

load_dotenv()


# ════════════════════════════════════════════════════════════════════════
# THEORY — The Magic Phrase
# ════════════════════════════════════════════════════════════════════════
# Kojima et al. (2022, "Large Language Models are Zero-Shot Reasoners")
# discovered that simply appending "Let's think step by step." to a
# prompt unlocks most of the CoT benefit without any hand-crafted
# reasoning template.
#
# Why it works: the training distribution contains millions of StackOverflow,
# textbook, and tutoring exchanges where "let's think step by step" precedes
# a careful, sequential explanation. The phrase is a PROMPT SHORTCUT into
# that style of generation. No examples, no template, just one sentence.
#
# TRADE-OFFS VS FULL COT:
#   + cheaper (no 4-step template to pay input tokens for)
#   + faster (slightly shorter output, same order of magnitude)
#   - less consistent reasoning format (harder to parse deterministically)
#   - slightly lower accuracy than hand-crafted CoT on tricky tasks
#
# Zero-shot CoT is the "best bang for the buck" reasoning technique for
# moderate-stakes tasks that don't need a strict audit format.


# ════════════════════════════════════════════════════════════════════════
# TASK 2 — BUILD
# ════════════════════════════════════════════════════════════════════════


async def zero_shot_cot_classify(text: str) -> tuple[str, str, float, float]:
    """Classify by appending the Kojima trigger phrase. Returns (label, reasoning, tokens, elapsed)."""
    prompt = f"""Classify the sentiment of this movie review as "positive" or "negative".

Review: "{text[:800]}"

Let's think step by step."""

    response, tokens, elapsed = await run_delegate(prompt)
    reasoning = response.strip()
    return normalise_label(reasoning), reasoning, tokens, elapsed


# ════════════════════════════════════════════════════════════════════════
# TASK 3 — TRAIN (evaluate)
# ════════════════════════════════════════════════════════════════════════


async def evaluate() -> list[dict]:
    docs = get_eval_docs()
    results: list[dict] = []
    for i, (text, true_label) in enumerate(
        zip(docs["text"].to_list(), docs["label"].to_list())
    ):
        pred, reasoning, tokens, elapsed = await zero_shot_cot_classify(text)
        correct = pred == true_label
        results.append(
            {
                "text": text,
                "pred": pred,
                "true": true_label,
                "correct": correct,
                "tokens": tokens,
                "elapsed": elapsed,
                "reasoning": reasoning,
            }
        )
        if i < 3:
            mark = "[ok]" if correct else "[miss]"
            print(f"  Doc {i+1}: pred={pred}, true={true_label} {mark}")
    return results


print("\n" + "=" * 70)
print("  Zero-Shot CoT — 'Let's think step by step'")
print("=" * 70)
zs_cot_results = asyncio.run(evaluate())


# ── Checkpoint ──────────────────────────────────────────────────────────
assert len(zs_cot_results) > 0, "Task 3: zero-shot CoT should produce results"
assert all(
    r["pred"] in CATEGORIES or r["pred"] == "unknown" for r in zs_cot_results
), "Predictions must be in CATEGORIES or 'unknown'"
print("\n[ok] Checkpoint passed — zero-shot CoT evaluation complete\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 4 — VISUALISE
# ════════════════════════════════════════════════════════════════════════
print_summary(zs_cot_results, "Zero-Shot CoT")

# R9A: visual proof — 4-method comparison chart (the full prompting ladder)
# Expected baselines (R10: independently runnable)
zero_shot_expected = {
    "strategy": "Zero-Shot",
    "accuracy": 0.80,
    "total_tokens": 1500,
    "avg_latency_s": 1.0,
    "n": 20,
}
few_shot_expected = {
    "strategy": "Few-Shot",
    "accuracy": 0.85,
    "total_tokens": 5400,
    "avg_latency_s": 1.2,
    "n": 20,
}
cot_expected = {
    "strategy": "CoT",
    "accuracy": 0.90,
    "total_tokens": 10500,
    "avg_latency_s": 3.5,
    "n": 20,
}
zs_cot_metrics = compute_metrics(zs_cot_results, "ZS-CoT")
all_methods = [zero_shot_expected, few_shot_expected, cot_expected, zs_cot_metrics]

plot_comparison_bars(
    all_methods,
    title="Prompting Ladder — All 4 Methods Compared",
    filename="ex1_04_method_comparison.png",
)

plot_tokens_vs_accuracy(
    all_methods,
    title="Cost vs Accuracy — Full Prompting Ladder",
    filename="ex1_04_cost_vs_accuracy.png",
)

# INTERPRETATION: Accuracy usually lands between zero-shot and full CoT,
# with a cost profile much closer to zero-shot. For tasks where 1-2
# percentage points of accuracy matter less than cost/latency, this is
# the default choice.
# The 4-method chart is the decision tool: pick the cheapest method that
# clears your accuracy bar. ZS-CoT often sits at the Pareto-optimal
# "knee" — best accuracy per dollar for non-regulated tasks.


# ════════════════════════════════════════════════════════════════════════
# TASK 5 — APPLY: SingPost Delivery-Complaint Triage
# ════════════════════════════════════════════════════════════════════════
# SCENARIO: SingPost receives ~6,000 delivery-complaint messages per day
# via its mobile app. Each message must be tagged "urgent" (missed
# delivery, wrong address, damaged parcel) or "informational" (status
# query, general feedback). Urgent messages feed a priority queue the
# dispatch team works every 30 minutes.
#
# Why zero-shot CoT fits:
#   - 6,000 msgs/day is a high volume where latency matters — a 3x
#     slowdown vs zero-shot at full CoT would mean messages pile up
#     faster than the team can process
#   - The task is moderately ambiguous — "my parcel is late" could be
#     urgent (2 days overdue) or informational (asking for an ETA)
#   - There is no regulatory audit requirement — unlike SGH (Ex 1.3)
#
# The trigger phrase gives SingPost ~80% of the CoT accuracy gain at
# ~40% of the CoT cost — the Pareto frontier for this class of task.
#
# BUSINESS IMPACT: Each urgent message triaged within 30 minutes
# (versus 4 hours baseline) saves an average S$22 in re-delivery cost
# and customer appeasement credits. At 6,000 msgs/day with ~8% urgent
# (480 urgent msgs/day) and a 6% accuracy lift over zero-shot = 29
# extra urgent msgs caught/day = S$633/day = S$230K/year in avoided
# cost. LLM cost at zero-shot-CoT rate: ~S$18K/year. 13x ROI.


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
  [x] Triggered reasoning with one sentence instead of a 4-step template
  [x] Understood why "Let's think step by step" generalises across tasks
  [x] Compared the cost/accuracy Pareto vs zero-shot and full CoT
  [x] Applied it to a high-volume, moderately-ambiguous triage task

  KEY INSIGHT: The trigger phrase is a cultural shortcut. The LLM knows
  what "think step by step" SHOULD look like because it saw a million
  tutors write that phrase before careful explanations.

  Next: 05_self_consistency.py — when one reasoning path isn't enough,
  sample N of them and vote.
"""
)
