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
# PREREQUISITES: 03_structured_agent.py
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

from kaizen import InputField, OutputField, Signature
from kaizen.core.base_agent import BaseAgent

from shared.mlfp06._ollama_bootstrap import OLLAMA_BASE_URL
from shared.mlfp06.ex_5 import MODEL, data_summary, load_hotpotqa, make_tools

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
# Critic pattern:
#   Step 1: Analyst produces an analysis
#   Step 2: Critic reviews and returns suggestions + should_revise: bool
#   Step 3: If should_revise, Refiner produces a second-pass analysis
#
# Self-consistency samples N independent answers and picks the majority.
# The critic REVIEWS one answer and suggests targeted fixes.  Use
# self-consistency for short correctness-sensitive answers, critic loop
# for long-form quality-sensitive analysis.


# ════════════════════════════════════════════════════════════════════════
# TASK 2 — Declare signatures and agents
# ════════════════════════════════════════════════════════════════════════


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
    weaknesses: list[str] = OutputField(description="Gaps or errors")
    suggestions: list[str] = OutputField(description="Improvement suggestions")
    quality_score: float = OutputField(description="Overall quality 0.0 to 1.0")
    # TODO: Declare should_revise: bool as an OutputField with description
    #       "Whether the analysis needs revision"
    should_revise: bool = ____


class RefinedAnalysisSignature(Signature):
    """Produce an improved analysis incorporating critic feedback."""

    dataset_summary: str = InputField(description="Dataset summary")
    analysis_question: str = InputField(description="Question to analyse")
    critic_feedback: str = InputField(description="Critic's suggestions")

    improved_findings: list[str] = OutputField(description="Revised findings")
    methodology_note: str = OutputField(description="How it was improved")
    confidence: float = OutputField(description="Confidence after revision")


# The canonical kaizen 2.7.3 pattern: dataclass config + instance
# signature in super().__init__.  Each agent type owns its own config
# so the critic loop can meter each step's budget independently.


@dataclass
class DataAnalysisConfig:
    llm_provider: str = os.environ.get("LLM_PROVIDER", "ollama")

    base_url: str = os.environ.get("OLLAMA_BASE_URL", OLLAMA_BASE_URL)
    model: str = MODEL
    temperature: float = 0.2
    budget_limit_usd: float = 1.0


@dataclass
class CriticConfig:
    llm_provider: str = os.environ.get("LLM_PROVIDER", "ollama")

    base_url: str = os.environ.get("OLLAMA_BASE_URL", OLLAMA_BASE_URL)
    model: str = MODEL
    temperature: float = 0.2
    # TODO: Set budget_limit_usd = 1.0
    budget_limit_usd: float = ____


@dataclass
class RefinedAnalysisConfig:
    llm_provider: str = os.environ.get("LLM_PROVIDER", "ollama")

    base_url: str = os.environ.get("OLLAMA_BASE_URL", OLLAMA_BASE_URL)
    model: str = MODEL
    temperature: float = 0.2
    # TODO: Set budget_limit_usd = 1.0
    budget_limit_usd: float = ____


class DataAnalysisAgent(BaseAgent):
    def __init__(self, config: DataAnalysisConfig):
        super().__init__(config=config, signature=DataAnalysisSignature())


class CriticAgent(BaseAgent):
    def __init__(self, config: CriticConfig):
        # TODO: Call super().__init__(config=config,
        #       signature=CriticSignature()) — signature is an INSTANCE.
        super().__init__(____)


class RefinedAnalysisAgent(BaseAgent):
    def __init__(self, config: RefinedAnalysisConfig):
        # TODO: Call super().__init__(config=config,
        #       signature=RefinedAnalysisSignature())
        super().__init__(____)


# ── Checkpoint 2 ─────────────────────────────────────────────────────────
# We check the INSTANCE signature, not the class attribute — BaseAgent
# 2.7.3 ignores class-level signature attrs.  The instance probe below
# is the only observable proof that the canonical pattern is wired.
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
    # TODO: Await analyst.run_async with dataset_summary=summary_text,
    #       analysis_question=QUESTION.  Note: .run_async(...), not .run().
    initial = ____
    initial_text = (
        f"Findings: {initial['key_findings']}\n"
        f"Approach: {initial['recommended_approach']}\n"
        f"Next steps: {initial['next_steps']}"
    )
    print(f"  Initial confidence: {initial['confidence']:.2f}")

    print("\nStep 2: critic reviews...")
    critic = CriticAgent(CriticConfig())
    # TODO: Await critic.run_async with original_analysis=initial_text,
    #       analysis_question=QUESTION
    critique = ____
    print(f"  Quality score: {critique['quality_score']:.2f}")
    print(f"  Should revise: {critique['should_revise']}")

    if critique["should_revise"]:
        print("\nStep 3: refining based on critic feedback...")
        refiner = RefinedAnalysisAgent(RefinedAnalysisConfig())
        # TODO: Await refiner.run_async with dataset_summary=summary_text,
        #       analysis_question=QUESTION,
        #       critic_feedback=str(critique["suggestions"])
        refined = ____
        print(f"  Refined confidence: {refined['confidence']:.2f}")
        return initial, critique, refined

    print("\nStep 3: critic approves — no revision needed.")
    return initial, critique, None


initial, critique, refined = asyncio.run(iterative_refinement())

# ── Checkpoint 3 ─────────────────────────────────────────────────────────
assert initial is not None
assert critique is not None
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


# ════════════════════════════════════════════════════════════════════════
# TASK 5 — Apply: Singapore regulatory compliance drafting
# ════════════════════════════════════════════════════════════════════════
# SCENARIO: Singapore fintech drafts MAS compliance memos.  Senior
# officers at S$150K/year draft each memo in 3h (1200h/year = S$90K).
# With critic loop: Draft -> ComplianceCritic (MAS notice checklist) ->
# Refiner -> 20-minute human review.  ~S$80K/year saved AND the critic
# output is itself a regulatory audit artifact.  1000x ROI per memo.


print("\n" + "=" * 70)
print("  KEY TAKEAWAY: Critic loops trade LLM spend for human time")
print("=" * 70)


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

  MODULE WRAPUP: Exercise 5 covered four agent patterns — ReAct for
  tool exploration, budgeted agents for cost safety, structured agents
  for pipeline integration, and critic loops for quality assurance.
  Together they are the production agent toolkit.  Exercise 6
  (Multi-Agent Systems) composes specialists with a supervisor.
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
