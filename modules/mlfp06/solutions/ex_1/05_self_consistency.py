# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# MLFP06 — Exercise 1.5: Self-Consistency (Sample N Paths, Majority Vote)
# ════════════════════════════════════════════════════════════════════════
#
# WHAT YOU'LL LEARN:
#   - Sample multiple INDEPENDENT CoT paths for the same input
#   - Aggregate them with majority vote
#   - Understand when variance across paths beats single-path accuracy
#   - See the linear cost scaling (N samples = N x cost)
#
# PREREQUISITES: 03_chain_of_thought.py (reuses the CoT classifier)
# ESTIMATED TIME: ~30 min
#
# TASKS:
#   1. Theory — why independent samples help
#   2. Build — the sampling loop + vote aggregator
#   3. Train — evaluate on a small subset for cost reasons
#   4. Visualise — vote distributions + majority outcomes
#   5. Apply — PACT ethics-review ensemble for legal research
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import asyncio
from collections import Counter

from dotenv import load_dotenv

from shared.mlfp06.ex_1 import (
    CATEGORIES,
    compute_metrics,
    get_eval_docs,
    normalise_label,
    plot_tokens_vs_accuracy,
    plot_vote_agreement,
    print_summary,
    run_delegate,
)

load_dotenv()

N_SAMPLES = 3  # independent CoT paths per query; production uses 5-9


# ════════════════════════════════════════════════════════════════════════
# THEORY — Why Independent Samples Help
# ════════════════════════════════════════════════════════════════════════
# LLM generation is stochastic at nonzero temperature — sampling the
# same prompt twice produces different reasoning traces. Self-consistency
# (Wang et al. 2023) exploits that: run the CoT prompt N times, collect
# N answers, return the majority vote.
#
# Intuition: if the reasoning is correct, most paths converge on the
# same answer. If the reasoning is noisy, the votes spread across
# categories and the majority still lands on the most-likely-correct
# label. This is the LLM equivalent of an ensemble model.
#
# WHEN IT HELPS:
#   - Ambiguous inputs where a single CoT gets "talked into" a wrong
#     answer by a single bad reasoning step
#   - Tasks with skewed error distribution (most paths are right; a
#     few confidently wrong paths would sink a single-sample baseline)
#
# COST: N times everything — N times input tokens, N times output
# tokens, N times latency (unless you parallelise the calls). Beyond
# N=5 returns diminish for binary classification; N=7-9 for multi-class.


# ════════════════════════════════════════════════════════════════════════
# TASK 2 — BUILD
# ════════════════════════════════════════════════════════════════════════


async def cot_once(text: str) -> tuple[str, float, float]:
    """One CoT sample. (We inline the prompt so this file is independently
    runnable without importing from 03_chain_of_thought.py.)"""
    prompt = f"""Classify the sentiment of this movie review as positive or negative.

Think step by step about the opinion words, tone, and any sarcasm.
End with your final classification as exactly "positive" or "negative".

Review: "{text[:800]}"

Step-by-step reasoning:"""
    response, tokens, elapsed = await run_delegate(prompt)
    return normalise_label(response), tokens, elapsed


async def self_consistency_classify(text: str) -> tuple[str, list[str], float, float]:
    """Sample N_SAMPLES CoT paths, return (majority_label, votes, total_tokens, total_elapsed).

    Samples run in parallel via asyncio.gather to avoid N x latency.
    """
    tasks = [cot_once(text) for _ in range(N_SAMPLES)]
    results = await asyncio.gather(*tasks)
    votes = [r[0] for r in results]
    total_tokens = sum(r[1] for r in results)
    # Parallel max latency, not sum — gather runs concurrently
    max_elapsed = max(r[2] for r in results)
    majority = Counter(votes).most_common(1)[0][0]
    return majority, votes, total_tokens, max_elapsed


# ════════════════════════════════════════════════════════════════════════
# TASK 3 — TRAIN (evaluate on a small subset for cost reasons)
# ════════════════════════════════════════════════════════════════════════


async def evaluate() -> list[dict]:
    docs = get_eval_docs().head(10)  # subset — self-consistency is N times expensive
    results: list[dict] = []
    for i, (text, true_label) in enumerate(
        zip(docs["text"].to_list(), docs["label"].to_list())
    ):
        pred, votes, tokens, elapsed = await self_consistency_classify(text)
        correct = pred == true_label
        results.append(
            {
                "text": text,
                "pred": pred,
                "true": true_label,
                "correct": correct,
                "tokens": tokens,
                "elapsed": elapsed,
                "votes": votes,
            }
        )
        if i < 5:
            mark = "[ok]" if correct else "[miss]"
            print(
                f"  Doc {i+1}: votes={votes} -> majority={pred}, "
                f"true={true_label} {mark}"
            )
    return results


print("\n" + "=" * 70)
print(f"  Self-Consistency — {N_SAMPLES} parallel CoT samples + majority vote")
print("=" * 70)
sc_results = asyncio.run(evaluate())


# ── Checkpoint ──────────────────────────────────────────────────────────
assert len(sc_results) > 0, "Task 3: self-consistency should produce results"
assert all(
    "votes" in r and len(r["votes"]) == N_SAMPLES for r in sc_results
), f"Each result must record exactly {N_SAMPLES} votes"
assert all(
    r["pred"] in CATEGORIES or r["pred"] == "unknown" for r in sc_results
), "Predictions must be in CATEGORIES or 'unknown'"
print("\n[ok] Checkpoint passed — self-consistency evaluation complete\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 4 — VISUALISE
# ════════════════════════════════════════════════════════════════════════
print_summary(sc_results, f"Self-Consistency (N={N_SAMPLES})")

# Distribution of vote agreement — how often did all N paths agree?
unanimous = sum(1 for r in sc_results if len(set(r["votes"])) == 1)
split = len(sc_results) - unanimous
print(f"\n  Vote agreement: {unanimous}/{len(sc_results)} unanimous, {split} split")
print(
    "  Unanimous = high-confidence prediction; split = hard case where "
    "the majority vote saved us from a bad single-sample answer."
)

# R9A: visual proof — vote agreement histogram + cost-vs-accuracy curve
plot_vote_agreement(
    sc_results,
    N_SAMPLES,
    title=f"Self-Consistency Vote Agreement (N={N_SAMPLES})",
    filename="ex1_05_vote_agreement.png",
)

# N-samples vs accuracy: show how the prompting ladder scales with cost
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
sc_metrics = compute_metrics(sc_results, f"SC (N={N_SAMPLES})")
plot_tokens_vs_accuracy(
    [zero_shot_expected, few_shot_expected, cot_expected, sc_metrics],
    title="Cost vs Accuracy — Self-Consistency on the Pareto Frontier",
    filename="ex1_05_cost_vs_accuracy.png",
)

# INTERPRETATION: The vote-agreement histogram is the confidence signal.
# Unanimous votes (1 distinct label) are high confidence. Split votes
# (2+ labels) flag hard cases. In production, route split-vote items
# to a human reviewer — the model is telling you it's unsure.


# ════════════════════════════════════════════════════════════════════════
# TASK 5 — APPLY: PACT Ethics Review for Legal Research
# ════════════════════════════════════════════════════════════════════════
# SCENARIO: A Singapore Big-4 law firm uses an LLM to screen discovery
# documents for "potentially privileged" content. Each document passes
# through a CoT classifier. Misclassification is extremely expensive:
#   - False negative: privileged material leaked to opposing counsel,
#     leading to malpractice exposure (~S$500K+ per incident)
#   - False positive: non-privileged material withheld, court
#     sanctions for obstruction (~S$50K per incident)
#
# Why self-consistency is mandatory here:
#   - The stakes are catastrophic on one side of the error distribution
#   - Single CoT paths have ~3% error rate on ambiguous legal text —
#     that's 30 errors per 1,000 documents, unacceptable
#   - Self-consistency at N=7 drops that to <0.5% — a 6x improvement
#   - The firm's PACT governance policy REQUIRES multi-sample consensus
#     for any action with >S$100K downside risk (see rules/tenant-isolation
#     and Exercise 6 governance)
#
# BUSINESS IMPACT: At 12,000 discovery documents per matter and
# ~3 matters/month, that's 36,000 classifications/month. Moving from
# single CoT (3% error) to N=7 self-consistency (<0.5%) avoids ~900
# errors/month. Even at the lower-bound S$5K expected-value cost per
# error (blended FN/FP), that's S$4.5M/month in avoided exposure.
# LLM cost at 7x CoT rate: ~S$28K/month. 160x ROI.
#
# Note: the cost is only justified because the downside is catastrophic.
# For the SingPost triage task (Ex 1.4), self-consistency would be
# overkill and you'd waste 6x the inference budget for <1% accuracy gain.


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
  [x] Sampled N independent CoT paths and aggregated with majority vote
  [x] Parallelised sampling with asyncio.gather (N x cost, 1x latency)
  [x] Observed vote agreement as a confidence signal
  [x] Sized the N x cost against a catastrophic-downside legal scenario

  KEY INSIGHT: Self-consistency is the ensemble method for LLMs. Like
  every ensemble, the right question is: "does the downside of being
  wrong justify N times the cost of being more right?" For most tasks,
  no. For high-stakes tasks, absolutely.

  Next: 06_structured_output.py — ditch free-form text and get
  type-safe structured responses from the LLM.
"""
)
