# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
Shared infrastructure for Exercise 6 — Multi-Agent Orchestration and MCP.

Contains: SQuAD 2.0 corpus loading, specialist Signature definitions,
specialist BaseAgent classes, synthesis agent, output directory setup.
Technique-specific orchestration logic lives in the per-technique files.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import polars as pl

from kaizen import InputField, OutputField, Signature
from kaizen.core.base_agent import BaseAgent

from shared.kailash_helpers import setup_environment

# ════════════════════════════════════════════════════════════════════════
# ENVIRONMENT SETUP
# ════════════════════════════════════════════════════════════════════════

setup_environment()

from shared.mlfp06._ollama_bootstrap import DEFAULT_CHAT_MODEL, OLLAMA_BASE_URL

MODEL = DEFAULT_CHAT_MODEL
LLM_PROVIDER_DEFAULT = os.environ.get("LLM_PROVIDER", "ollama")
LLM_BASE_URL_DEFAULT = os.environ.get("OLLAMA_BASE_URL", OLLAMA_BASE_URL)

# Output directory for all visualisation/trace artifacts
OUTPUT_DIR = Path("outputs") / "ex6_multi_agent"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ════════════════════════════════════════════════════════════════════════
# DATA LOADING — SQuAD 2.0 Multi-Domain Corpus
# ════════════════════════════════════════════════════════════════════════

CACHE_DIR = Path("data") / "mlfp06" / "squad"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
CACHE_FILE = CACHE_DIR / "squad_v2_300.parquet"


def load_squad_corpus(n_rows: int = 300) -> pl.DataFrame:
    """Load (or download + cache) SQuAD 2.0 validation split.

    SQuAD 2.0 is a multi-domain reading-comprehension benchmark with
    100K+ questions across hundreds of Wikipedia titles. We take a
    shuffled slice of 300 rows so exercises run in bounded time while
    still exercising the "multi-domain" property.

    Returns a polars DataFrame with columns: title, text, question, answer.
    """
    if CACHE_FILE.exists():
        print(f"Loading cached SQuAD 2.0 from {CACHE_FILE}")
        return pl.read_parquet(CACHE_FILE)

    print("Downloading rajpurkar/squad_v2 from HuggingFace...")
    from datasets import load_dataset

    ds = load_dataset("rajpurkar/squad_v2", split="validation")
    ds = ds.shuffle(seed=42).select(range(min(n_rows, len(ds))))
    rows = []
    for row in ds:
        answers = row["answers"]["text"]
        rows.append(
            {
                "title": row["title"],
                "text": row["context"],
                "question": row["question"],
                "answer": answers[0] if answers else "",
            }
        )
    passages = pl.DataFrame(rows)
    passages.write_parquet(CACHE_FILE)
    print(f"Cached {passages.height} SQuAD 2.0 rows to {CACHE_FILE}")
    return passages


# ════════════════════════════════════════════════════════════════════════
# SPECIALIST SIGNATURES — Domain-Specific Input/Output Contracts
# ════════════════════════════════════════════════════════════════════════


class FactualAnalysisSignature(Signature):
    """Extract factual claims and verify them against the passage."""

    document: str = InputField(description="Source passage text")
    question: str = InputField(description="Question to analyse")
    factual_claims: list[str] = OutputField(
        description="Key factual claims in the passage"
    )
    evidence_quality: str = OutputField(
        description="Quality of evidence: strong/moderate/weak"
    )
    answer_supported: bool = OutputField(
        description="Whether the passage supports an answer"
    )


class SemanticAnalysisSignature(Signature):
    """Analyse meaning, context, and implications beyond literal text."""

    document: str = InputField(description="Source passage text")
    question: str = InputField(description="Question to analyse")
    main_themes: list[str] = OutputField(description="Central themes in the passage")
    implicit_info: list[str] = OutputField(
        description="Implied but not stated information"
    )
    contextual_relevance: str = OutputField(
        description="How relevant context is to the question"
    )


class StructuralAnalysisSignature(Signature):
    """Analyse the structure, organisation, and argumentation pattern."""

    document: str = InputField(description="Source passage text")
    question: str = InputField(description="Question to analyse")
    structure_type: str = OutputField(
        description="Text structure: narrative/expository/argumentative"
    )
    key_entities: list[str] = OutputField(description="Named entities mentioned")
    relationships: list[str] = OutputField(description="Relationships between entities")


class SynthesisSignature(Signature):
    """Synthesise multiple specialist analyses into a unified answer."""

    document: str = InputField(description="Original passage")
    question: str = InputField(description="Original question")
    factual_analysis: str = InputField(description="Factual specialist output")
    semantic_analysis: str = InputField(description="Semantic specialist output")
    structural_analysis: str = InputField(description="Structural specialist output")
    unified_answer: str = OutputField(
        description="Comprehensive answer drawing on all analyses"
    )
    confidence: float = OutputField(description="Answer confidence 0-1")
    reasoning_chain: list[str] = OutputField(description="Step-by-step reasoning used")


class InterpretationSignature(Signature):
    """Interpret factual claims in context (sequential pipeline stage 2)."""

    factual_claims: str = InputField(description="Raw factual claims from prior stage")
    document: str = InputField(description="Original passage for context")
    question: str = InputField(description="Question being answered")
    interpreted_facts: list[str] = OutputField(
        description="Facts with contextual interpretation"
    )
    relevance_ranking: list[str] = OutputField(
        description="Facts ranked by relevance to question"
    )


# ════════════════════════════════════════════════════════════════════════
# SPECIALIST AGENT CLASSES
# ════════════════════════════════════════════════════════════════════════
#
# Canonical kaizen 2.7.3 pattern: dataclass config + instance signature
# in super().__init__.  Class-level `signature = XxxSig` is a silent
# bug — BaseAgent ignores class-level attrs and falls back to
# DefaultSignature, so every "specialist" was producing generic output
# before this migration.  See workspaces/mlfp06-migration/api-cheatsheet.md
# for the background.
#
# `description` remains a class-level attribute because it's read by
# Pipeline.router() and by the exercise's audit-trail logging — it's
# agent metadata, not part of the LLM-wiring contract.


@dataclass
class FactualConfig:
    llm_provider: str = LLM_PROVIDER_DEFAULT
    model: str = MODEL
    base_url: str = LLM_BASE_URL_DEFAULT
    temperature: float = 0.2
    budget_limit_usd: float = 1.0


@dataclass
class SemanticConfig:
    llm_provider: str = LLM_PROVIDER_DEFAULT
    model: str = MODEL
    base_url: str = LLM_BASE_URL_DEFAULT
    temperature: float = 0.2
    budget_limit_usd: float = 1.0


@dataclass
class StructuralConfig:
    llm_provider: str = LLM_PROVIDER_DEFAULT
    model: str = MODEL
    base_url: str = LLM_BASE_URL_DEFAULT
    temperature: float = 0.2
    budget_limit_usd: float = 1.0


@dataclass
class SynthesisConfig:
    llm_provider: str = LLM_PROVIDER_DEFAULT
    model: str = MODEL
    base_url: str = LLM_BASE_URL_DEFAULT
    temperature: float = 0.2
    # Supervisor gets a larger budget — it reasons over all specialist outputs.
    budget_limit_usd: float = 2.0


@dataclass
class InterpretationConfig:
    llm_provider: str = LLM_PROVIDER_DEFAULT
    model: str = MODEL
    base_url: str = LLM_BASE_URL_DEFAULT
    temperature: float = 0.2
    budget_limit_usd: float = 1.0


class FactualAgent(BaseAgent):
    description = "Specialist in factual analysis: claims, evidence, verification"

    def __init__(self, config: FactualConfig | None = None):
        super().__init__(
            config=config or FactualConfig(),
            signature=FactualAnalysisSignature(),
        )


class SemanticAgent(BaseAgent):
    description = "Specialist in semantic analysis: themes, implications, context"

    def __init__(self, config: SemanticConfig | None = None):
        super().__init__(
            config=config or SemanticConfig(),
            signature=SemanticAnalysisSignature(),
        )


class StructuralAgent(BaseAgent):
    description = (
        "Specialist in structural analysis: entities, relationships, organisation"
    )

    def __init__(self, config: StructuralConfig | None = None):
        super().__init__(
            config=config or StructuralConfig(),
            signature=StructuralAnalysisSignature(),
        )


class SynthesisAgent(BaseAgent):
    description = (
        "Supervisor that synthesises specialist analyses into unified decisions"
    )

    def __init__(self, config: SynthesisConfig | None = None):
        super().__init__(
            config=config or SynthesisConfig(),
            signature=SynthesisSignature(),
        )


class InterpretationAgent(BaseAgent):
    description = "Stage-2 interpreter: contextualises raw factual claims for synthesis"

    def __init__(self, config: InterpretationConfig | None = None):
        super().__init__(
            config=config or InterpretationConfig(),
            signature=InterpretationSignature(),
        )


def build_specialists() -> tuple[FactualAgent, SemanticAgent, StructuralAgent]:
    """Return fresh instances of the three analysis specialists.

    Uses the default Config for each specialist.  Callers that need to
    tweak model, temperature, or budget can instantiate directly:

        factual = FactualAgent(FactualConfig(budget_limit_usd=5.0))
    """
    return FactualAgent(), SemanticAgent(), StructuralAgent()


def build_synthesis() -> SynthesisAgent:
    """Return a fresh synthesis (supervisor) agent with its default budget."""
    return SynthesisAgent()
