# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# MLFP06 — Exercise 5.3: Structured Agents (BaseAgent + Signature)
# ════════════════════════════════════════════════════════════════════════
#
# WHAT YOU'LL LEARN:
#   - Define a Kaizen Signature with typed InputField / OutputField
#   - Build a custom BaseAgent that declares signature + model + budget
#   - Run the agent and receive typed attributes instead of raw strings
#   - Validate outputs (types, ranges, list lengths)
#   - Know when to choose BaseAgent vs ReActAgent
#
# PREREQUISITES: 01_react_agent.py, 02_cost_budget_agent.py
# ESTIMATED TIME: ~40 min
#
# TASKS:
#   1. Load data + compute the dataset summary to feed the agent
#   2. Declare the DataAnalysisSignature (inputs + typed outputs)
#   3. Build and run the DataAnalysisAgent
#   4. Visualise the typed result
#   5. Apply: Singapore HR analytics pipeline scenario
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
# TASK 1 — Data and summary
# ════════════════════════════════════════════════════════════════════════

qa_data = load_hotpotqa()
tools = make_tools(qa_data)  # binds qa_data to tool closures
# TODO: Call data_summary() to get the text summary the agent will analyse
summary_text = ____
print("Dataset summary (first 200 chars):")
print(summary_text[:200], "...\n")

# ── Checkpoint 1 ─────────────────────────────────────────────────────────
assert qa_data.height > 0
assert len(summary_text) > 100
print("✓ Checkpoint 1 passed — summary ready to feed agent\n")


# ════════════════════════════════════════════════════════════════════════
# THEORY — Why structured output changes everything
# ════════════════════════════════════════════════════════════════════════
# ReActAgent returns a free-form string; downstream code must parse it.
# BaseAgent + Signature declares the OUTPUT SHAPE up front and returns
# a typed object (result.key_findings, result.confidence).  No JSON
# parsing, no "the LLM forgot the findings field."  The signature is a
# contract the model MUST satisfy.
#
# WHEN TO USE WHICH:
#   ReActAgent            : open-ended exploration, tool use, unknown steps
#   BaseAgent + Signature : known output shape, feeds pipelines, audit


# ════════════════════════════════════════════════════════════════════════
# TASK 2 — Declare the DataAnalysisSignature
# ════════════════════════════════════════════════════════════════════════


class DataAnalysisSignature(Signature):
    """Analyse a dataset summary and produce structured insights."""

    # TODO: Declare two InputField entries:
    #   dataset_summary: str   — "Statistical summary of the dataset"
    #   analysis_question: str — "Specific question to investigate"
    dataset_summary: str = ____
    analysis_question: str = ____

    # TODO: Declare five OutputField entries:
    #   key_findings: list[str]       — "Top 3-5 findings from the analysis"
    #   recommended_approach: str     — "Best ML approach for this data"
    #   data_quality_issues: list[str]— "Potential data quality concerns"
    #   next_steps: list[str]         — "3-5 recommended next analysis steps"
    #   confidence: float             — "Confidence in findings (0.0 to 1.0)"
    key_findings: list[str] = ____
    recommended_approach: str = ____
    data_quality_issues: list[str] = ____
    next_steps: list[str] = ____
    confidence: float = ____


# ── Checkpoint 2 ─────────────────────────────────────────────────────────
assert DataAnalysisSignature.__doc__, "Signature needs a docstring"
print("✓ Checkpoint 2 passed — DataAnalysisSignature declared\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 3 — Build and run the structured agent
# ════════════════════════════════════════════════════════════════════════


# The canonical kaizen 2.7.3 pattern: dataclass config + instance
# signature. The OLD pattern (class-level `signature = XxxSignature`)
# is a silent bug — BaseAgent ignores class-level attrs and falls back
# to DefaultSignature. Pass the signature as an INSTANCE in super().
@dataclass
class DataAnalysisConfig:
    llm_provider: str = os.environ.get("LLM_PROVIDER", "ollama")

    base_url: str = os.environ.get("OLLAMA_BASE_URL", OLLAMA_BASE_URL)
    model: str = MODEL
    temperature: float = 0.2
    # TODO: Set budget_limit_usd = 1.0 (replaces legacy max_llm_cost_usd)
    budget_limit_usd: float = ____


class DataAnalysisAgent(BaseAgent):
    """Structured data analysis agent using the typed Signature."""

    def __init__(self, config: DataAnalysisConfig):
        # TODO: Call super().__init__(config=config,
        #       signature=DataAnalysisSignature()) — the signature is
        #       passed as an INSTANCE, not a class.
        super().__init__(____)


async def run_structured_agent():
    agent = DataAnalysisAgent(DataAnalysisConfig())
    # TODO: Await agent.run_async with:
    #   dataset_summary=summary_text
    #   analysis_question="What patterns distinguish comparison questions from bridge questions in this dataset?"
    # Note: the entry point is .run_async(...), NOT .run(...).
    result = ____
    return result


structured_result = asyncio.run(run_structured_agent())

# ── Checkpoint 3 ─────────────────────────────────────────────────────────
assert structured_result is not None, "Task 3: agent should return a result"
assert "key_findings" in structured_result
assert "confidence" in structured_result
assert len(structured_result["key_findings"]) > 0
assert 0 <= structured_result["confidence"] <= 1
print("✓ Checkpoint 3 passed — structured result validated\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 4 — Visualise the typed result
# ════════════════════════════════════════════════════════════════════════

print("=" * 70)
print("  Structured Analysis Output")
print("=" * 70)
print(f"Key findings ({len(structured_result['key_findings'])}):")
for i, f in enumerate(structured_result["key_findings"], 1):
    print(f"  {i}. {f}")
print(f"\nRecommended approach: {structured_result['recommended_approach']}")
print(f"\nData quality issues:  {structured_result['data_quality_issues']}")
print(f"\nNext steps:")
for i, s in enumerate(structured_result["next_steps"], 1):
    print(f"  {i}. {s}")
print(f"\nConfidence: {structured_result['confidence']:.2f}")

# ── Checkpoint 4 ─────────────────────────────────────────────────────────
assert isinstance(structured_result["key_findings"], list)
assert isinstance(structured_result["recommended_approach"], str)
assert isinstance(structured_result["data_quality_issues"], list)
assert isinstance(structured_result["next_steps"], list)
assert isinstance(structured_result["confidence"], float)
print("\n✓ Checkpoint 4 passed — every field matches its declared type\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 5 — Apply: Singapore HR analytics pipeline
# ════════════════════════════════════════════════════════════════════════
# SCENARIO: 50 business units, monthly attrition briefs.  Before:
# S$12K/month analyst time.  With Signature: S$15 LLM + 30 min ops.
# The typed fields feed straight into a dashboard — no parsing, no
# schema drift, and every brief has the same shape because the
# Signature enforces it.


print("\n" + "=" * 70)
print("  KEY TAKEAWAY: Signatures turn LLMs into pipeline components")
print("=" * 70)


# ════════════════════════════════════════════════════════════════════════
# REFLECTION
# ════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("  WHAT YOU'VE MASTERED")
print("=" * 70)
print(
    f"""
  [x] Declared a typed Signature with 2 inputs and 5 outputs
  [x] Built a BaseAgent that enforces the signature contract
  [x] Received typed attributes instead of raw strings
  [x] Validated types, ranges, and non-empty lists
  [x] Chose BaseAgent vs ReActAgent based on output shape

  Next: 04_critic_agent.py uses TWO structured agents in a loop —
  one produces analysis, one critiques and requests revision...
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
