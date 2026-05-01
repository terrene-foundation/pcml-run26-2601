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
# PREREQUISITES: 03_chain_of_thought.py
# ESTIMATED TIME: ~30 min
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import asyncio
from collections import Counter

from dotenv import load_dotenv

from shared.mlfp06.ex_1 import (
    CATEGORIES,
    get_eval_docs,
    normalise_label,
    print_summary,
    run_delegate,
)

load_dotenv()

N_SAMPLES = 3  # independent CoT paths per query


# ════════════════════════════════════════════════════════════════════════
# THEORY — Why Independent Samples Help
# ════════════════════════════════════════════════════════════════════════
# LLM generation is stochastic. Sample N times, vote. If most paths are
# right, the majority converges. If one path goes astray, the others
# overrule it. This is the LLM equivalent of an ensemble.
# Cost: N x everything. Returns diminish beyond N=5 for binary tasks.


# ════════════════════════════════════════════════════════════════════════
# TASK 2 — BUILD
# ════════════════════════════════════════════════════════════════════════


async def cot_once(text: str) -> tuple[str, float, float]:
    """One CoT sample."""
    # TODO: Build a CoT prompt (positive/negative, think step by step).
    prompt = ____
    # TODO: run_delegate, normalise, return (label, tokens, elapsed)
    ____


async def self_consistency_classify(
    text: str,
) -> tuple[str, list[str], float, float]:
    """Sample N_SAMPLES CoT paths in parallel, return majority vote.

    Returns (majority_label, votes, total_tokens, max_elapsed).
    """
    # TODO: Build a list of N_SAMPLES cot_once coroutines and await them
    # in parallel with asyncio.gather. Collect votes, sum costs, take max elapsed.
    ____

    # TODO: Use collections.Counter to find the majority vote
    ____


# ════════════════════════════════════════════════════════════════════════
# TASK 3 — TRAIN (evaluate on a small subset for cost reasons)
# ════════════════════════════════════════════════════════════════════════


async def evaluate() -> list[dict]:
    docs = get_eval_docs().head(10)
    results: list[dict] = []
    # TODO: Loop, call self_consistency_classify, record dict including "votes"
    ____
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

unanimous = sum(1 for r in sc_results if len(set(r["votes"])) == 1)
split = len(sc_results) - unanimous
print(f"\n  Vote agreement: {unanimous}/{len(sc_results)} unanimous, {split} split")


# ════════════════════════════════════════════════════════════════════════
# TASK 5 — APPLY: PACT Ethics Review for Legal Research
# ════════════════════════════════════════════════════════════════════════
# Big-4 law firm screens discovery documents for privileged content.
# Single-CoT error rate (~3%) is unacceptable — privileged leaks cost
# S$500K+ per incident. N=7 self-consistency drops error to <0.5%.
# PACT governance policy REQUIRES multi-sample consensus for decisions
# with >S$100K downside.
#
# BUSINESS IMPACT: 36K classifications/mo, 2.5% error reduction avoids
# S$4.5M/mo in exposure, vs S$28K/mo in LLM cost. 160x ROI.


# ════════════════════════════════════════════════════════════════════════
# REFLECTION
# ════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  WHAT YOU'VE MASTERED")
print("=" * 70)
print(
    """
  [x] Sampled N independent CoT paths and aggregated with majority vote
  [x] Parallelised with asyncio.gather (N x cost, 1x latency)
  [x] Sized N x cost against catastrophic-downside scenarios

  Next: 06_structured_output.py — ditch string parsing, use Kaizen Signatures.
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
