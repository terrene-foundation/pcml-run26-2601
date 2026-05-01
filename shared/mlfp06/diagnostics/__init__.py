# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""LLM Observatory — six clinical lenses for LLM / agent / RAG / governance systems.

Mirrors the M5 Doctor's Bag (``kailash_ml.diagnostics.DLDiagnostics``) for
the M6 problem domain. A single :class:`LLMObservatory` facade composes six
lens classes that each answer exactly one diagnostic question:

    1. Output Lens       — is the generation coherent, faithful, on-task?
    2. Attention Lens    — what does the model attend to; what circuit answers?
    3. Retrieval Lens    — did we fetch the right context, did we use it?
    4. Agent Trace Lens  — what did the agent do, and where did it spend budget?
    5. Alignment Lens    — is the fine-tuning signal rewarding the right thing?
    6. Governance Lens   — is the system inside its envelope; is the audit intact?

Quick start::

    from shared.mlfp06.diagnostics import LLMObservatory

    obs = LLMObservatory(delegate=my_delegate)
    obs.output.faithfulness(response, context)
    obs.retrieval.recall_at_k(queries, relevant_ids, k=5)
    obs.agent.tool_usage_summary(trace)
    obs.governance.verify_chain(supervisor.audit.to_list())
    print(obs.report())

All DataFrames are Polars. All plots are Plotly. LLM calls route through
Kaizen :class:`~kaizen_agents.Delegate` (never raw ``openai.*``).
"""
from __future__ import annotations

from .agent import AgentDiagnostics, AgentTrace
from .alignment import AlignmentDiagnostics
from .governance import GovernanceDiagnostics
from .interpretability import InterpretabilityDiagnostics
from .observatory import LLMObservatory
from .output import JudgeVerdict, LLMDiagnostics
from .retrieval import RAGDiagnostics

__all__ = [
    "LLMObservatory",
    "LLMDiagnostics",
    "InterpretabilityDiagnostics",
    "RAGDiagnostics",
    "AgentDiagnostics",
    "AgentTrace",
    "AlignmentDiagnostics",
    "GovernanceDiagnostics",
    "JudgeVerdict",
]
