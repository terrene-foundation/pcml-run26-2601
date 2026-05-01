# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# MLFP06 — Exercise 5.4: Critic Agent — Iterative Refinement
# ════════════════════════════════════════════════════════════════════════
#
# WHAT YOU'LL LEARN:
#   - Compose multiple BaseAgents into a quality-assurance loop
#   - Define a critic Signature that returns should_revise: bool
#   - Build a refinement agent that consumes critic feedback
#   - Run the full Analyse -> Critique -> Refine cycle
#   - Distinguish critic refinement from self-consistency sampling
#
# PREREQUISITES: 03_structured_agent.py (Signature + BaseAgent basics)
# ESTIMATED TIME: ~45 min
#
# TASKS:
#   1. Load data + summary
#   2. Declare Critic and Refined signatures + agents
#   3. Run the Analyse -> Critique -> Refine loop
#   4. Visualise the delta between initial and refined output
#   5. Apply: Singapore regulatory compliance drafting scenario
#
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass

import matplotlib.pyplot as plt

from kaizen import InputField, OutputField, Signature
from kaizen.core.base_agent import BaseAgent

from shared.mlfp06._ollama_bootstrap import OLLAMA_BASE_URL
from shared.mlfp06.ex_5 import (
    MODEL,
    OUTPUT_DIR,
    data_summary,
    load_hotpotqa,
    make_tools,
)

# ════════════════════════════════════════════════════════════════════════
# TASK 1 — Data + summary
# ════════════════════════════════════════════════════════════════════════

qa_data = load_hotpotqa()
tools = make_tools(qa_data)
summary_text = data_summary()
print(f"Loaded {qa_data.height} examples\n")

# ── Checkpoint 1 ─────────────────────────────────────────────────────────
assert qa_data.height > 0
assert len(summary_text) > 100
print("✓ Checkpoint 1 passed — infra ready\n")


# ════════════════════════════════════════════════════════════════════════
# THEORY — Critic agents vs self-consistency
# ════════════════════════════════════════════════════════════════════════
# The critic pattern is a QA loop over LLM output:
#
#   Step 1: Analyst agent produces an analysis
#   Step 2: Critic agent reviews it and returns strengths, weaknesses,
#           suggestions, and a should_revise boolean
#   Step 3: If should_revise, a refiner agent produces an improved
#           analysis that incorporates the critic's suggestions
#
# This is NOT the same as self-consistency (MLFP06 Ex 1 Task 6).
# Self-consistency samples N INDEPENDENT answers and picks the majority.
# The critic REVIEWS a specific answer and suggests targeted fixes.
#
# When to use which:
#   Self-consistency : correctness-sensitive, short answers, parallelisable
#   Critic loop      : long-form analysis, quality-sensitive, can afford
#                      2-3x more LLM calls for substantially better output
#
# ANALOGY: Self-consistency is "ask 5 doctors, take the majority diagnosis."
# Critic loop is "junior doctor writes the diagnosis, senior doctor reviews
# and says what to change."  Both improve quality; they do it differently.


# ════════════════════════════════════════════════════════════════════════
# TASK 2 — Declare signatures and agents
# ════════════════════════════════════════════════════════════════════════
#
# We reuse the DataAnalysisSignature structure as the analyst.  Then we
# add a CriticSignature and a RefinedAnalysisSignature.


class DataAnalysisSignature(Signature):
    """Analyse a dataset summary and produce structured insights."""

    dataset_summary: str = InputField(description="Statistical summary")
    analysis_question: str = InputField(description="Question to investigate")

    key_findings: list[str] = OutputField(description="Top 3-5 findings")
    recommended_approach: str = OutputField(description="Best ML approach")
    data_quality_issues: list[str] = OutputField(description="Quality concerns")
    next_steps: list[str] = OutputField(description="3-5 next steps")
    confidence: float = OutputField(description="Confidence 0.0-1.0")


class CriticSignature(Signature):
    """Critique an analysis and suggest improvements."""

    original_analysis: str = InputField(description="The analysis to critique")
    analysis_question: str = InputField(description="The original question asked")

    strengths: list[str] = OutputField(description="What the analysis does well")
    weaknesses: list[str] = OutputField(description="Gaps or errors in the analysis")
    suggestions: list[str] = OutputField(description="Specific improvement suggestions")
    quality_score: float = OutputField(description="Overall quality 0.0 to 1.0")
    should_revise: bool = OutputField(description="Whether the analysis needs revision")


class RefinedAnalysisSignature(Signature):
    """Produce an improved analysis incorporating critic feedback."""

    dataset_summary: str = InputField(description="Dataset summary")
    analysis_question: str = InputField(description="Question to analyse")
    critic_feedback: str = InputField(description="Critic's improvement suggestions")

    improved_findings: list[str] = OutputField(description="Revised findings")
    methodology_note: str = OutputField(description="How the analysis was improved")
    confidence: float = OutputField(description="Confidence after revision (0-1)")


# Domain configs — dataclass pattern (kaizen 2.7.3).  Each agent type
# owns its budget so the critic loop can meter each step independently.


@dataclass
class DataAnalysisConfig:
    llm_provider: str = os.environ.get("LLM_PROVIDER", "ollama")

    base_url: str = os.environ.get("OLLAMA_BASE_URL", OLLAMA_BASE_URL)
    model: str = MODEL  # resolved from .env
    temperature: float = 0.2
    budget_limit_usd: float = 1.0


@dataclass
class CriticConfig:
    llm_provider: str = os.environ.get("LLM_PROVIDER", "ollama")

    base_url: str = os.environ.get("OLLAMA_BASE_URL", OLLAMA_BASE_URL)
    model: str = MODEL
    temperature: float = 0.2
    budget_limit_usd: float = 1.0


@dataclass
class RefinedAnalysisConfig:
    llm_provider: str = os.environ.get("LLM_PROVIDER", "ollama")

    base_url: str = os.environ.get("OLLAMA_BASE_URL", OLLAMA_BASE_URL)
    model: str = MODEL
    temperature: float = 0.2
    budget_limit_usd: float = 1.0


class DataAnalysisAgent(BaseAgent):
    def __init__(self, config: DataAnalysisConfig):
        super().__init__(config=config, signature=DataAnalysisSignature())


class CriticAgent(BaseAgent):
    def __init__(self, config: CriticConfig):
        super().__init__(config=config, signature=CriticSignature())


class RefinedAnalysisAgent(BaseAgent):
    def __init__(self, config: RefinedAnalysisConfig):
        super().__init__(config=config, signature=RefinedAnalysisSignature())


# ── Checkpoint 2 ─────────────────────────────────────────────────────────
# The class-level `signature = XxxSignature` pattern is a SILENT bug in
# kaizen 2.7.3 — BaseAgent ignores class-level attrs and falls back to
# DefaultSignature.  We check that each agent's INSTANCE signature is
# an instance of its declared Signature subclass, which is the only
# observable proof that the canonical pattern is wired correctly.
_analyst_probe = DataAnalysisAgent(DataAnalysisConfig())
_critic_probe = CriticAgent(CriticConfig())
_refiner_probe = RefinedAnalysisAgent(RefinedAnalysisConfig())
assert isinstance(_analyst_probe.signature, DataAnalysisSignature)
assert isinstance(_critic_probe.signature, CriticSignature)
assert isinstance(_refiner_probe.signature, RefinedAnalysisSignature)
print("✓ Checkpoint 2 passed — 3 signatures + 3 agents declared\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 3 — Run the Analyse -> Critique -> Refine loop
# ════════════════════════════════════════════════════════════════════════

QUESTION = "What makes multi-hop QA harder than single-hop QA?"


async def iterative_refinement():
    print("Step 1: initial analysis...")
    analyst = DataAnalysisAgent(DataAnalysisConfig())
    initial = await analyst.run_async(
        dataset_summary=summary_text,
        analysis_question=QUESTION,
    )
    initial_text = (
        f"Findings: {initial['key_findings']}\n"
        f"Approach: {initial['recommended_approach']}\n"
        f"Next steps: {initial['next_steps']}"
    )
    print(f"  Initial confidence: {initial['confidence']:.2f}")
    print(f"  Findings: {initial['key_findings'][:2]}")

    print("\nStep 2: critic reviews...")
    critic = CriticAgent(CriticConfig())
    critique = await critic.run_async(
        original_analysis=initial_text,
        analysis_question=QUESTION,
    )
    print(f"  Quality score: {critique['quality_score']:.2f}")
    print(f"  Should revise: {critique['should_revise']}")
    print(f"  Weaknesses:    {critique['weaknesses'][:2]}")

    if critique["should_revise"]:
        print("\nStep 3: refining based on critic feedback...")
        refiner = RefinedAnalysisAgent(RefinedAnalysisConfig())
        refined = await refiner.run_async(
            dataset_summary=summary_text,
            analysis_question=QUESTION,
            critic_feedback=str(critique["suggestions"]),
        )
        print(f"  Refined confidence: {refined['confidence']:.2f}")
        return initial, critique, refined

    print("\nStep 3: critic approves — no revision needed.")
    return initial, critique, None


initial, critique, refined = asyncio.run(iterative_refinement())

# ── Checkpoint 3 ─────────────────────────────────────────────────────────
assert initial is not None, "Task 3: initial analysis required"
assert critique is not None, "Task 3: critic output required"
assert "should_revise" in critique
assert "quality_score" in critique
assert 0 <= critique["quality_score"] <= 1
print("\n✓ Checkpoint 3 passed — refinement loop executed\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 4 — Visualise the delta between initial and refined
# ════════════════════════════════════════════════════════════════════════

print("=" * 70)
print("  Refinement Delta")
print("=" * 70)

print(f"\nINITIAL confidence:  {initial['confidence']:.2f}")
print(f"Critic quality:      {critique['quality_score']:.2f}")
print(f"Critic decision:     {'REVISE' if critique['should_revise'] else 'APPROVE'}")
if refined is not None:
    print(f"REFINED confidence:  {refined['confidence']:.2f}")
    delta = refined["confidence"] - initial["confidence"]
    direction = "↑" if delta >= 0 else "↓"
    print(f"Confidence delta:    {direction} {abs(delta):.2f}")
    print(f"\nMethodology note: {refined['methodology_note'][:200]}...")
    print(f"\nImproved findings ({len(refined['improved_findings'])}):")
    for i, f in enumerate(refined["improved_findings"][:5], 1):
        print(f"  {i}. {f}")
else:
    print("(No refinement — critic approved the initial analysis.)")

# ── Checkpoint 4 ─────────────────────────────────────────────────────────
assert isinstance(critique["strengths"], list)
assert isinstance(critique["weaknesses"], list)
assert isinstance(critique["suggestions"], list)
if refined is not None:
    assert isinstance(refined["improved_findings"], list)
    assert 0 <= refined["confidence"] <= 1
print("\n✓ Checkpoint 4 passed — refinement delta visualised\n")

# INTERPRETATION: The critic's quality_score is the first gate.  If the
# score is high and should_revise is False, you save one LLM call and
# ship the initial analysis.  If the score is low and should_revise is
# True, the refiner fires and produces a second-pass result.  The
# confidence delta is your quality signal — if refined.confidence is
# not meaningfully higher than initial.confidence, the critic pattern
# isn't helping on this task shape and you should disable it.


# ════════════════════════════════════════════════════════════════════════
# TASK 5 — Apply: Singapore regulatory compliance drafting
# ════════════════════════════════════════════════════════════════════════
# SCENARIO: A Singapore fintech drafts Monetary Authority of Singapore
# (MAS) compliance memos every time a new product ships.  Memos are
# long-form, high-stakes, and must reference specific MAS notices.
# One bad memo = regulatory finding = S$50K+ fine.
#
# BEFORE: Senior compliance officers (S$150K/year) draft every memo
# personally.  Each memo takes ~3 hours.  400 memos/year * 3h = 1200h
# of senior time = ~S$90,000/year in compliance-officer hours.
#
# NAIVE LLM APPROACH: Hand the product brief to an LLM, get a memo back,
# ship it.  Problem: the LLM hallucinates MAS notice numbers, miscites
# regulations, and the compliance team has to rewrite everything.  Net
# time saved: near zero.  Net risk: high (hallucinated citations).
#
# WITH CRITIC LOOP:
#   1. DraftAgent     writes the first-pass memo
#   2. ComplianceCritic reviews against a checklist (cite MAS notices,
#      reference correct licence tier, flag consumer-protection gaps)
#   3. RefinedDraftAgent produces a second pass incorporating fixes
#   4. Human compliance officer reviews the REFINED draft (not the
#      raw first pass) — 20 minutes instead of 3 hours
#
# BUSINESS IMPACT:
#   - Senior time per memo: 3h -> 20 min  (9x reduction)
#   - Annual hours saved:   1200h -> 133h = 1067h ~ S$80,000/year
#   - Quality floor:        the critic enforces the checklist; first
#                           drafts that miss items never reach the
#                           human reviewer
#   - Audit trail:          critic JSON output persists as evidence
#                           that the checklist was applied — this is
#                           itself a regulatory asset
#
# THE GENERAL PATTERN: Critic loops shine wherever the cost of a human
# review is high and the cost of two extra LLM calls is negligible.


print("\n" + "=" * 70)
print("  KEY TAKEAWAY: Critic loops trade LLM spend for human time")
print("=" * 70)
print(
    """
  Each critic loop adds 2-3 LLM calls (~$0.10) but saves 1-3 hours of
  senior human review (~S$100).  A 1000x ROI on every memo.  The
  pattern is boring to read and devastating to deploy.
"""
)


# ════════════════════════════════════════════════════════════════════════
# VISUALISATION — Quality improvement: initial vs refined
# ════════════════════════════════════════════════════════════════════════

stages = ["Initial\nAnalysis", "Critic\nScore", "Refined\nAnalysis"]
scores = [
    initial["confidence"],
    critique["quality_score"],
    refined["confidence"] if refined is not None else initial["confidence"],
]
colors = ["#90CAF9", "#FFE082", "#A5D6A7"]

fig, ax = plt.subplots(figsize=(6, 4))
bars = ax.bar(stages, scores, color=colors, edgecolor="#333", linewidth=0.8)
for bar, score in zip(bars, scores):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.02,
        f"{score:.2f}",
        ha="center",
        va="bottom",
        fontweight="bold",
    )
ax.set_ylim(0, 1.15)
ax.set_ylabel("Confidence / Quality Score")
ax.set_title("Critic Loop: Quality Improvement Across Stages")
ax.axhline(y=0.8, color="green", linestyle="--", alpha=0.5, label="Quality threshold")
ax.legend()
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "04_critic_improvement.png", dpi=150)
plt.close(fig)
print(f"\nSaved: {OUTPUT_DIR / '04_critic_improvement.png'}")


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

# Primary lens: Agent Trace (TAOD capture, tool-call success, stuck-loop
# detection). Secondary: Output (final answer quality).
if False:  # scaffold — requires a live Delegate + API key
    obs = LLMObservatory(delegate=react_agent, run_id="ex_5_agent_run")
    # Re-run the agent under the lens:
    # import asyncio
    # trace = asyncio.run(obs.agent.capture_run(react_agent, task=prompt))
    # obs.output.evaluate(prompts=[prompt], responses=[trace.final_answer])
    print("\n── LLM Observatory Report ──")
    findings = obs.report()

# ══════ EXPECTED OUTPUT (synthesised reference) ══════
# ════════════════════════════════════════════════════════════════
#   LLM Observatory — composite Prescription Pad
# ════════════════════════════════════════════════════════════════
#   [✓] Agent      (HEALTHY): 5 TAOD steps, tool-call success 1.00,
#       no stuck loops, total cost $0.017 (budget $2.00).
#   [✓] Output     (HEALTHY): judge faithfulness 0.89 on final answer.
#   [?] Retrieval / Alignment / Governance / Attention (n/a)
# ════════════════════════════════════════════════════════════════
#
# STUDENT INTERPRETATION GUIDE — reading the Prescription Pad:
#
#  [AGENT LENS] 5 TAOD steps for a multi-hop question is the healthy
#     signature — general (data_summary) -> specific (run_query) ->
#     targeted (search_documents) -> grounded (lookup_answer) ->
#     synthesis. The BAD signature would be the same tool called with
#     the same args 3+ times ("stuck loop") or a step count of 1
#     (skipped the tools entirely). The loop detector in AgentDiagnostics
#     flags both.
#     >> Prescription (if stuck): tighten the tool docstrings, the LLM
#        is guessing because the tools don't advertise what they do.
#  [OUTPUT LENS] Faithfulness 0.89 on the final answer confirms the
#     agent's synthesis used the observations rather than fabricating.
# ════════════════════════════════════════════════════════════════════


# ════════════════════════════════════════════════════════════════════════
# REFLECTION
# ════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("  WHAT YOU'VE MASTERED")
print("=" * 70)
print(
    """
  [x] Composed 3 BaseAgents (analyst, critic, refiner) in a loop
  [x] Declared should_revise as a typed bool that gates the refiner
  [x] Ran the Analyse -> Critique -> Refine cycle end-to-end
  [x] Measured the confidence delta as a quality signal
  [x] Distinguished critic refinement from self-consistency sampling

  KEY INSIGHT: Three agents doing one job, each with a single
  responsibility: analyse, critique, refine.  This is the same pattern
  as a code-review workflow — a junior writes, a senior reviews, the
  junior revises.  You just automated both halves of it.

  MODULE WRAPUP: Exercise 5 covered four agent patterns — ReAct for
  tool exploration, budgeted agents for cost safety, structured agents
  for pipeline integration, and critic loops for quality assurance.
  Together they are the production agent toolkit.  Exercise 6
  (Multi-Agent Systems) composes specialists with a supervisor for
  fan-out / fan-in orchestration.
"""
)
