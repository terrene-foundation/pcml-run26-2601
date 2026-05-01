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
# PREREQUISITES: 01_react_agent.py (agent basics), 02_cost_budget_agent.py
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
# TASK 1 — Data and summary
# ════════════════════════════════════════════════════════════════════════

qa_data = load_hotpotqa()
tools = make_tools(qa_data)  # binds the module-level qa_data for tools
summary_text = data_summary()
print("Dataset summary (first 200 chars):")
print(summary_text[:200], "...\n")

# ── Checkpoint 1 ─────────────────────────────────────────────────────────
assert qa_data.height > 0
assert len(summary_text) > 100, "Task 1: summary should be non-trivial"
print("✓ Checkpoint 1 passed — summary ready to feed agent\n")


# ════════════════════════════════════════════════════════════════════════
# THEORY — Why structured output changes everything
# ════════════════════════════════════════════════════════════════════════
# ReActAgent is great for exploration — you hand it tools and ask a
# question and it figures out how to answer.  The downside: the output
# is a free-form string.  Downstream code has to parse it.  String
# parsing is fragile and breaks the moment the LLM rephrases.
#
# BaseAgent + Signature fixes this by declaring the OUTPUT SHAPE up
# front.  The agent returns a typed dict:
#
#   result = await agent.run_async(...)
#   result["key_findings"]         # list[str] — validated, never None
#   result["confidence"]           # float 0.0-1.0 — validated range
#   result["recommended_approach"] # str
#
# No JSON parsing.  No "the LLM forgot to include the findings field."
# The signature is enforced — the LLM MUST produce the declared shape
# or the call fails loudly.
#
# ANALOGY: ReActAgent is "tell me what you think" — a conversation.
# BaseAgent + Signature is "fill out this form" — a contract.  Forms
# are boring but they feed data pipelines without a translation layer.
#
# WHEN TO USE WHICH:
#
#   ReActAgent            : open-ended exploration, tool use required,
#                           number of steps unknown
#   BaseAgent + Signature : known output shape, feeds downstream code,
#                           audit trail, pipeline integration
#   Hybrid                : ReAct tools to GATHER, Signature to FORMAT


# ════════════════════════════════════════════════════════════════════════
# TASK 2 — Declare the DataAnalysisSignature
# ════════════════════════════════════════════════════════════════════════


class DataAnalysisSignature(Signature):
    """Analyse a dataset summary and produce structured insights."""

    dataset_summary: str = InputField(description="Statistical summary of the dataset")
    analysis_question: str = InputField(description="Specific question to investigate")

    key_findings: list[str] = OutputField(
        description="Top 3-5 findings from the analysis"
    )
    recommended_approach: str = OutputField(
        description="Best ML approach for this data (classification, clustering, etc.)"
    )
    data_quality_issues: list[str] = OutputField(
        description="Potential data quality concerns (missing values, bias, etc.)"
    )
    next_steps: list[str] = OutputField(
        description="3-5 recommended next analysis steps"
    )
    confidence: float = OutputField(description="Confidence in findings (0.0 to 1.0)")


# ── Checkpoint 2 ─────────────────────────────────────────────────────────
assert DataAnalysisSignature.__doc__, "Signature needs a docstring"
print("✓ Checkpoint 2 passed — DataAnalysisSignature declared\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 3 — Build and run the structured agent
# ════════════════════════════════════════════════════════════════════════


@dataclass
class DataAnalysisConfig:
    """Domain config — BaseAgent auto-converts to BaseAgentConfig.

    The dataclass holds the LLM wiring (provider, model, temperature) and
    the safety budget in ONE place — the canonical kaizen 2.7.3 pattern
    replaces the old `model = ...` / `max_llm_cost_usd = ...` class-level
    attributes, which kaizen silently ignored (see 04 critic notes).
    """

    llm_provider: str = os.environ.get("LLM_PROVIDER", "ollama")
    base_url: str = os.environ.get("OLLAMA_BASE_URL", OLLAMA_BASE_URL)
    model: str = MODEL  # resolved from .env in shared/mlfp06/ex_5.py
    temperature: float = 0.2
    budget_limit_usd: float = 1.0  # replaces legacy max_llm_cost_usd


class DataAnalysisAgent(BaseAgent):
    """Structured data analysis agent using the typed Signature.

    The Signature is passed as an INSTANCE to super().__init__ — NOT as a
    class-level attribute. Class-level `signature = DataAnalysisSignature`
    is silently ignored by BaseAgent and falls back to DefaultSignature,
    which is why the old code produced generic output instead of typed
    fields. The instance pattern is the canonical fix.
    """

    def __init__(self, config: DataAnalysisConfig):
        super().__init__(config=config, signature=DataAnalysisSignature())


async def run_structured_agent():
    agent = DataAnalysisAgent(DataAnalysisConfig())
    result = await agent.run_async(
        dataset_summary=summary_text,
        analysis_question=(
            "What patterns distinguish comparison questions from bridge "
            "questions in this dataset?"
        ),
    )
    return result


structured_result = asyncio.run(run_structured_agent())

# ── Checkpoint 3 ─────────────────────────────────────────────────────────
assert structured_result is not None, "Task 3: agent should return a result"
assert "key_findings" in structured_result, "Result needs key_findings"
assert "confidence" in structured_result, "Result needs confidence"
assert len(structured_result["key_findings"]) > 0, "Should have at least one finding"
assert 0 <= structured_result["confidence"] <= 1, "Confidence must be in [0, 1]"
print("✓ Checkpoint 3 passed — structured result validated\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 4 — Visualise the typed result
# ════════════════════════════════════════════════════════════════════════

print("=" * 70)
print("  Structured Analysis Output")
print("=" * 70)
print(f"Key findings ({len(structured_result["key_findings"])}):")
for i, f in enumerate(structured_result["key_findings"], 1):
    print(f"  {i}. {f}")
print(f"\nRecommended approach: {structured_result["recommended_approach"]}")
print(f"\nData quality issues:  {structured_result["data_quality_issues"]}")
print(f"\nNext steps:")
for i, s in enumerate(structured_result["next_steps"], 1):
    print(f"  {i}. {s}")
print(f"\nConfidence: {structured_result["confidence"]:.2f}")

# ── Checkpoint 4 ─────────────────────────────────────────────────────────
assert isinstance(structured_result["key_findings"], list)
assert isinstance(structured_result["recommended_approach"], str)
assert isinstance(structured_result["data_quality_issues"], list)
assert isinstance(structured_result["next_steps"], list)
assert isinstance(structured_result["confidence"], float)
print("\n✓ Checkpoint 4 passed — every field matches its declared type\n")

# INTERPRETATION: Compare this to the ReActAgent output from technique 1.
# Technique 1 returned a string that a human reads.  This output feeds
# directly into a dashboard, a report pipeline, or a downstream agent
# with zero parsing.  The confidence field alone lets you build a
# "route to human review if confidence < 0.7" branch trivially.


# ════════════════════════════════════════════════════════════════════════
# VISUALISE — Field coverage heatmap for structured output
# ════════════════════════════════════════════════════════════════════════
# Visual proof that every Signature field was populated. A missing field
# in production breaks downstream pipelines — this heatmap is the
# at-a-glance QA check that every slot in the contract was filled.

fields = [
    "key_findings",
    "recommended_approach",
    "data_quality_issues",
    "next_steps",
    "confidence",
]
completeness = []
for f in fields:
    val = structured_result.get(f)
    if isinstance(val, list):
        completeness.append(min(len(val) / 3.0, 1.0))  # 3+ items = full
    elif isinstance(val, str):
        completeness.append(1.0 if len(val) > 10 else 0.5)
    elif isinstance(val, float):
        completeness.append(val)  # confidence is already 0-1
    else:
        completeness.append(0.0)

fig, ax = plt.subplots(figsize=(8, 3))
ax.barh(
    fields,
    completeness,
    color=["#2ecc71" if c >= 0.7 else "#e74c3c" for c in completeness],
)
ax.set_xlim(0, 1.1)
ax.set_xlabel("Completeness (0 = empty, 1 = fully populated)")
ax.set_title("Structured Agent — Field Coverage Heatmap", fontweight="bold")
for i, c in enumerate(completeness):
    ax.text(c + 0.02, i, f"{c:.0%}", va="center", fontsize=9)
ax.axvline(0.7, color="gray", linestyle="--", alpha=0.5, label="Threshold (70%)")
ax.legend(loc="lower right")
plt.tight_layout()
fname = OUTPUT_DIR / "ex5_structured_field_coverage.png"
plt.savefig(fname, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"\n  Saved: {fname}")


# ════════════════════════════════════════════════════════════════════════
# TASK 5 — Apply: Singapore HR analytics pipeline
# ════════════════════════════════════════════════════════════════════════
# SCENARIO: A Singapore conglomerate runs monthly attrition analysis
# across 50 business units.  The HR team wants: key drivers of attrition,
# recommended interventions, and a confidence score — all 50 units, in
# one PDF, every month.
#
# BEFORE: 2 analysts spend 8 days each month reading 50 data dumps,
# writing 50 briefs, assembling a master report.  Cost: ~S$12,000/month
# in analyst time.  Latency: 8 days.  Inconsistency: every brief has a
# different structure, so the master report needs re-formatting.
#
# WITH DataAnalysisAgent (Signature):
#   for unit in business_units:
#       summary = build_unit_summary(unit)
#       result = await agent.run_async(
#           dataset_summary=summary,
#           analysis_question=f"What drives attrition in {unit.name}?",
#       )
#       dashboard.add_row(unit, result["key_findings"], result["confidence"],
#                        result["recommended_approach"])
#
# The typed fields go STRAIGHT into the dashboard.  No parsing.  No
# "the LLM phrased it differently this month."  Every brief has the
# same shape because the Signature enforces it.
#
# BUSINESS IMPACT:
#   - Cost per month:     S$12,000 -> S$15 (LLM) + 30 min ops
#   - Latency:            8 days -> 15 minutes
#   - Consistency:        free-form briefs -> uniform schema
#   - Human reviewers:    route units with confidence < 0.7 to the
#                         HR business partner for manual review
#
# THE UPGRADE: Pair this with the critic pattern (04_critic_agent.py)
# to automatically re-run low-confidence units with a feedback loop.


print("\n" + "=" * 70)
print("  KEY TAKEAWAY: Signatures turn LLMs into pipeline components")
print("=" * 70)
print(
    """
  Free-form text is a conversation.  Typed structured output is a
  contract.  Pipelines need contracts.  If your LLM call feeds
  downstream code, use Signature.  If it feeds a human reader,
  free-form is fine.
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

  KEY INSIGHT: result.confidence = {structured_result["confidence"]:.2f}.
  That float exists because we asked for it in the Signature.  The
  LLM had no choice but to produce it.  This is the power of the
  contract: the model serves the pipeline, not the other way around.

  Next: 04_critic_agent.py uses TWO structured agents in a loop —
  one produces analysis, one critiques and requests revision...
"""
)
