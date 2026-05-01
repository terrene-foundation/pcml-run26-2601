# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
Shared infrastructure for MLFP06 Exercise 8 — Capstone: Full Production Platform.

Contains: LLM model resolution, MMLU evaluation data loader, modern PACT
governance YAML (D/T/R grammar) for the MLFP Capstone org, canonical
Signature/Agent classes (dataclass config + instance signature — fixes the
silent ``DefaultSignature`` fallback), the ``handle_qa`` router used by every
technique file, ``build_capstone_stack(engine)`` (dedupes the 4-file 3-tier
boilerplate per ``workspaces/mlfp06-migration/decisions.md`` § 2), and small
middleware stubs (``SimpleJWTAuth``, ``RateLimiter``) used by the serving
technique file.

Technique-specific code (adapter loading, nexus registration, drift analysis,
compliance reporting) does NOT belong here — it lives in the per-technique
files under ``modules/mlfp06/solutions/ex_8/``.

Import from any cwd after ``uv sync``:

    from shared.mlfp06.ex_8 import (
        MODEL, OUTPUT_DIR, load_mmlu_eval, write_org_yaml,
        CapstoneQASignature, CapstoneQAConfig, CapstoneQAAgent,
        build_capstone_stack, handle_qa,
        SimpleJWTAuth, RateLimiter, run_async,
    )
"""
from __future__ import annotations

import asyncio
import os
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import polars as pl
from dotenv import load_dotenv

from kaizen import InputField, OutputField, Signature
from kaizen.core.base_agent import BaseAgent
from kaizen_agents import GovernedSupervisor

from shared.kailash_helpers import setup_environment

if TYPE_CHECKING:  # pragma: no cover — type-only imports
    from pact import GovernanceEngine

# ════════════════════════════════════════════════════════════════════════
# ENVIRONMENT SETUP
# ════════════════════════════════════════════════════════════════════════

setup_environment()
load_dotenv()

from shared.mlfp06._ollama_bootstrap import DEFAULT_CHAT_MODEL, OLLAMA_BASE_URL

MODEL = DEFAULT_CHAT_MODEL
LLM_PROVIDER_DEFAULT = os.environ.get("LLM_PROVIDER", "ollama")
LLM_BASE_URL_DEFAULT = os.environ.get("OLLAMA_BASE_URL", OLLAMA_BASE_URL)

if not MODEL:  # pragma: no cover — bootstrap default never returns empty
    raise EnvironmentError("OLLAMA_CHAT_MODEL or DEFAULT_LLM_MODEL must be set")

# Output + cache directories
OUTPUT_DIR = Path("outputs") / "ex8_capstone"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

EVAL_CACHE_DIR = Path("data/mlfp06/mmlu")
EVAL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
EVAL_CACHE_FILE = EVAL_CACHE_DIR / "mmlu_100.parquet"


# ════════════════════════════════════════════════════════════════════════
# DATA LOADING — MMLU (Massive Multitask Language Understanding)
# ════════════════════════════════════════════════════════════════════════


def load_mmlu_eval(n_rows: int = 100) -> pl.DataFrame:
    """Load MMLU evaluation data as a polars DataFrame, cached to parquet.

    Schema:
        instruction (str) — question + A/B/C/D choices as plain text
        response    (str) — correct letter (A/B/C/D)
        subject     (str) — MMLU subject area

    Returns:
        A polars DataFrame with at most ``n_rows`` shuffled MMLU questions.
    """
    if EVAL_CACHE_FILE.exists():
        print(f"Loading cached MMLU from {EVAL_CACHE_FILE}")
        return pl.read_parquet(EVAL_CACHE_FILE)

    print("Downloading cais/mmlu from HuggingFace...")
    from datasets import load_dataset

    ds = load_dataset("cais/mmlu", "all", split="validation")
    ds = ds.shuffle(seed=42).select(range(min(n_rows, len(ds))))
    rows: list[dict[str, Any]] = []
    for row in ds:
        choices = row["choices"]
        answer_idx = row["answer"]
        rows.append(
            {
                "instruction": (
                    f"{row['question']}\n\n"
                    f"A) {choices[0]}\nB) {choices[1]}\n"
                    f"C) {choices[2]}\nD) {choices[3]}"
                ),
                "response": ["A", "B", "C", "D"][answer_idx],
                "subject": row["subject"],
            }
        )
    eval_data = pl.DataFrame(rows)
    eval_data.write_parquet(EVAL_CACHE_FILE)
    print(f"Cached {eval_data.height} MMLU rows to {EVAL_CACHE_FILE}")
    return eval_data


# ════════════════════════════════════════════════════════════════════════
# SHARED SIGNATURE & BASE AGENT (canonical kaizen 2.7.3 pattern)
# ════════════════════════════════════════════════════════════════════════
#
# Canonical pattern:
#   1. `@dataclass` config carries model + budget_limit_usd
#   2. Signature is PASSED as an instance to `super().__init__(signature=...)`
#      — omitting the `signature=` keyword silently falls back to
#      `DefaultSignature()` and the declared output schema is ignored.


class CapstoneQASignature(Signature):
    """Answer questions with a governed, audited, confidence-scored response."""

    question: str = InputField(description="User's question")
    answer: str = OutputField(description="Detailed, grounded answer")
    confidence: float = OutputField(description="Confidence score 0-1")
    sources: list[str] = OutputField(description="Knowledge sources referenced")
    reasoning_steps: list[str] = OutputField(description="Step-by-step reasoning")


@dataclass
class CapstoneQAConfig:
    """Domain config — BaseAgent auto-converts to BaseAgentConfig.

    The ``model`` and ``budget_limit_usd`` fields are the canonical knobs in
    kaizen 2.7.3. The legacy class-level ``max_llm_cost_usd`` has moved here.
    """

    llm_provider: str = LLM_PROVIDER_DEFAULT
    model: str = MODEL
    base_url: str = LLM_BASE_URL_DEFAULT
    temperature: float = 0.2
    budget_limit_usd: float = 5.0


class CapstoneQAAgent(BaseAgent):
    """Capstone QA agent: wraps the fine-tuned model behind a typed signature."""

    def __init__(self, config: CapstoneQAConfig | None = None) -> None:
        super().__init__(
            config=config or CapstoneQAConfig(),
            signature=CapstoneQASignature(),
        )


# ════════════════════════════════════════════════════════════════════════
# PACT GOVERNANCE — shared org yaml (modern D/T/R grammar)
# ════════════════════════════════════════════════════════════════════════
#
# The MLFP Capstone org has a single department (AI Services) headed by
# an ML Director, and three delegated tasks with three Responsible
# agents:
#   qa    (Responder, internal clearance)      — customer-facing answers
#   admin (Operator,  confidential clearance)  — model lifecycle / metrics
#   audit (Auditor,   restricted clearance)    — full compliance access
#
# Each delegation carries a constraint envelope (financial + operational
# + data_access) matching the 3-tier stack Shard 7's technique files
# wire via `build_capstone_stack(engine)`.

ORG_YAML: str = """
# MLFP Capstone ML Platform — PACT Governance Definition
# D/T/R: every agent action traces to a human Delegator

org_id: "mlfp_capstone"
name: "MLFP Capstone ML Platform"

# One department, headed by the ML Director (Delegator).
departments:
  - id: "ai_services"
    name: "AI Services"

# Three teams — one per delegated task.
teams:
  - id: "qa_team"
    name: "Question Answering"
  - id: "ops_team"
    name: "Model Operations"
  - id: "audit_team"
    name: "Compliance Audit"

# Roles: ml_director (Delegator) + three Responsibles (agents).
roles:
  # ── Delegator (human) ──
  - id: "ml_director"
    name: "ML Director"
    heads: "ai_services"

  # ── Responsibles (agents) ──
  - id: "qa_agent"
    name: "QA Agent"
    reports_to: "ml_director"
    heads: "qa_team"
  - id: "admin_agent"
    name: "Admin Agent"
    reports_to: "ml_director"
    heads: "ops_team"
  - id: "audit_agent"
    name: "Audit Agent"
    reports_to: "ml_director"
    heads: "audit_team"

# Clearance lattice — canonical pact strings (public | restricted |
# confidential | secret | top_secret). The course's 4-level mental model
# (public < internal < confidential < restricted) maps to canonical as:
# "internal" -> RESTRICTED (historical alias), so the qa tier's
# teaching-level "internal" is expressed here as "restricted". The
# course's distinct "internal" vs "restricted" rungs are preserved at
# the kaizen_agents data_clearance string interface in
# build_capstone_stack below. See ex_7/02_envelopes.py sidebar for the
# 5-level canonical hierarchy.
clearances:
  - role: "ml_director"
    level: "confidential"
  - role: "qa_agent"
    level: "restricted"
  - role: "admin_agent"
    level: "confidential"
  - role: "audit_agent"
    level: "restricted"

# Envelopes = delegations. One envelope per Responsible.
envelopes:
  - target: "qa_agent"
    defined_by: "ml_director"
    financial:
      max_spend_usd: 1.0
    operational:
      allowed_actions: ["generate_answer", "search_context"]
    communication:
      max_response_length: 2000

  - target: "admin_agent"
    defined_by: "ml_director"
    financial:
      max_spend_usd: 10.0
    operational:
      allowed_actions:
        - "generate_answer"
        - "search_context"
        - "update_model"
        - "view_metrics"
        - "monitor_drift"

  - target: "audit_agent"
    defined_by: "ml_director"
    financial:
      max_spend_usd: 50.0
    operational:
      allowed_actions:
        - "generate_answer"
        - "search_context"
        - "view_metrics"
        - "access_audit_log"
        - "generate_report"
"""


def write_org_yaml(path: str | Path | None = None) -> str:
    """Write the shared capstone org YAML to a temp file and return the path."""
    if path is None:
        path = os.path.join(tempfile.gettempdir(), "capstone_org.yaml")
    with open(path, "w") as f:
        f.write(ORG_YAML)
    return str(path)


# ════════════════════════════════════════════════════════════════════════
# BUILD_CAPSTONE_STACK — shared 3-tier GovernedSupervisor builder
# ════════════════════════════════════════════════════════════════════════
#
# Per `workspaces/mlfp06-migration/decisions.md` § 2, the 4 ex_8 technique
# files each used to start with a near-identical 40-LOC 3-tier construction
# block. This helper deduplicates that block. Each technique file imports
# the helper and calls it at the top, then runs its per-file narrative
# unchanged.
#
# Role → tier mapping (teaching-coherent for the capstone narrative):
#   qa    → public   (low budget, narrow tools, internal clearance)
#   admin → internal (mid budget,  ops tools,     confidential)
#   audit → restricted (high budget, audit tools, restricted)
#
# The helper also attaches a `ConstraintEnvelopeConfig` to each Responsible
# role address via `engine.set_role_envelope(...)` so
# `engine.verify_action()` has a real envelope to enforce against — per
# Shard 5's finding that `verify_action` on a role with no envelope
# auto-approves. Parent-head envelopes are CONFIDENTIAL so
# `RoleEnvelope.validate_tightening()` accepts the RESTRICTED/PUBLIC
# children (Shard 4 finding: canonical clearance order is
# CONFIDENTIAL > RESTRICTED > PUBLIC, so parents need to be at or above
# the tightest child level the tree will hold).


# The three Responsible role addresses under ml_director (D1-R1). Each
# Responsible sits under a team (T<n>) under the single department (D1).
_QA_ADDR = "D1-R1-T1-R1"
_ADMIN_ADDR = "D1-R1-T2-R1"
_AUDIT_ADDR = "D1-R1-T3-R1"
_DIRECTOR_ADDR = "D1-R1"


@dataclass
class CapstoneTier:
    """Metadata for one GovernedSupervisor tier in the capstone stack."""

    role: str
    address: str
    budget_usd: float
    tools: list[str]
    clearance: str  # kaizen_agents data_clearance string
    description: str


CAPSTONE_TIERS: list[CapstoneTier] = [
    CapstoneTier(
        role="qa",
        address=_QA_ADDR,
        budget_usd=1.0,
        tools=["generate_answer", "search_context"],
        clearance="public",
        description="Retail-facing QA — narrow tools, low budget",
    ),
    CapstoneTier(
        role="admin",
        address=_ADMIN_ADDR,
        budget_usd=10.0,
        tools=[
            "generate_answer",
            "search_context",
            "update_model",
            "view_metrics",
            "monitor_drift",
        ],
        clearance="internal",
        description="Model ops — update + metrics + drift",
    ),
    CapstoneTier(
        role="audit",
        address=_AUDIT_ADDR,
        budget_usd=50.0,
        tools=[
            "generate_answer",
            "search_context",
            "view_metrics",
            "access_audit_log",
            "generate_report",
        ],
        clearance="restricted",
        description="Compliance audit — full audit log + reports",
    ),
]


def _attach_envelopes(engine: "GovernanceEngine") -> None:
    """Attach ConstraintEnvelopeConfig to every Responsible role address.

    Without this, `engine.verify_action()` on any of the Responsible
    addresses auto-approves (no envelope constraints). The envelope is
    the source of restriction — attach it so runtime enforcement has
    something to enforce against.
    """
    from pact import (
        CommunicationConstraintConfig,
        ConfidentialityLevel,
        ConstraintEnvelopeConfig,
        DataAccessConstraintConfig,
        FinancialConstraintConfig,
        OperationalConstraintConfig,
        RoleEnvelope,
        TemporalConstraintConfig,
    )

    # Map the course's 4-level teaching strings to canonical
    # ConfidentialityLevel. "internal" collides with "restricted" at
    # RESTRICTED per the historical alias in kaizen_agents; we keep
    # them distinct as teaching rungs but they resolve to the same
    # canonical level.
    clearance_map = {
        "public": ConfidentialityLevel.PUBLIC,
        "internal": ConfidentialityLevel.RESTRICTED,
        "confidential": ConfidentialityLevel.CONFIDENTIAL,
        "restricted": ConfidentialityLevel.RESTRICTED,
    }

    for tier in CAPSTONE_TIERS:
        envelope = ConstraintEnvelopeConfig(
            id=f"{tier.role}_envelope",
            description=tier.description,
            confidentiality_clearance=clearance_map[tier.clearance],
            financial=FinancialConstraintConfig(max_spend_usd=tier.budget_usd),
            operational=OperationalConstraintConfig(
                allowed_actions=list(tier.tools),
                blocked_actions=[],
            ),
            temporal=TemporalConstraintConfig(blackout_periods=[]),
            data_access=DataAccessConstraintConfig(
                read_paths=["/public/*"],
                write_paths=[],
                blocked_data_types=[],
            ),
            communication=CommunicationConstraintConfig(allowed_channels=["internal"]),
            max_delegation_depth=3,
        )
        engine.set_role_envelope(
            RoleEnvelope(
                id=f"{tier.role}_role_envelope",
                defining_role_address=_DIRECTOR_ADDR,
                target_role_address=tier.address,
                envelope=envelope,
            )
        )


def build_capstone_stack(
    engine: "GovernanceEngine",
) -> tuple[dict[str, GovernedSupervisor], list[CapstoneTier]]:
    """Build the shared 3-tier GovernedSupervisor stack.

    Returns:
        A 2-tuple ``(agents_by_role, tiers)`` where:
          - ``agents_by_role`` maps ``"qa" | "admin" | "audit"`` to the
            corresponding ``GovernedSupervisor`` — this is the shape
            ``handle_qa(question, role, agents_by_role)`` expects.
          - ``tiers`` is the list of ``CapstoneTier`` metadata entries
            (order: qa, admin, audit) so technique files can read
            budget / tools / clearance without re-deriving them.

    Side effects:
        Attaches a ``ConstraintEnvelopeConfig`` to every Responsible
        role address on ``engine`` via ``engine.set_role_envelope(...)``.
        After this call, ``engine.verify_action(tier.address, ...)``
        enforces the tier's envelope.
    """
    _attach_envelopes(engine)

    agents_by_role: dict[str, GovernedSupervisor] = {}
    for tier in CAPSTONE_TIERS:
        agents_by_role[tier.role] = GovernedSupervisor(
            model=MODEL,
            budget_usd=tier.budget_usd,
            tools=list(tier.tools),
            data_clearance=tier.clearance,
        )
    return agents_by_role, list(CAPSTONE_TIERS)


# ════════════════════════════════════════════════════════════════════════
# SHARED QA HANDLER — used by Nexus deployment AND monitoring/test files
# ════════════════════════════════════════════════════════════════════════
#
# `handle_qa()` preserves the original return dict shape so the four
# Shard 7 technique files (02_governance_pipeline, 03_multichannel_serving,
# 04_drift_monitoring, 05_compliance_audit) continue to read
# `result["answer"]`, `result["confidence"]`, etc. unchanged.
#
# Internally the call routes through the selected tier's
# `GovernedSupervisor.run(objective=question, execute_node=...)`, where
# the executor wraps a shared `CapstoneQAAgent` instance. The executor
# returns the four keys GovernedSupervisor expects (`result`, `cost`,
# `prompt_tokens`, `completion_tokens`) per the contract established
# by `shared.mlfp06.ex_7.make_fake_executor`.


# One module-level agent instance is shared by every handle_qa call.
# Constructing a CapstoneQAAgent per call would waste the compiled
# signature and the LLM client setup on the hot path.
_shared_qa_agent: CapstoneQAAgent | None = None


def _get_shared_agent() -> CapstoneQAAgent:
    global _shared_qa_agent
    if _shared_qa_agent is None:
        _shared_qa_agent = CapstoneQAAgent(CapstoneQAConfig())
    return _shared_qa_agent


async def _capstone_execute_node(_spec: Any, inputs: dict[str, Any]) -> dict[str, Any]:
    """Executor callback for GovernedSupervisor.run().

    Runs the shared `CapstoneQAAgent` on the question carried in
    ``inputs``. The ``_spec`` positional is required by the
    ``GovernedSupervisor.run(execute_node=...)`` interface but is not
    consumed here — the supervisor passes the question in ``inputs``.
    Returns the four-key dict GovernedSupervisor expects.

    Live mode uses ``agent.run_async(question=...)``; on any exception
    (missing key, rate limit, network error) we fall back to a
    deterministic offline stub so the teaching narrative runs end-to-end.
    """
    del _spec  # interface-required positional, not consumed
    agent = _get_shared_agent()
    objective = (
        inputs.get("objective")
        or inputs.get("question")
        or inputs.get("prompt")
        or str(inputs)
    )

    try:
        out = await agent.run_async(question=str(objective))
        answer = out.get("answer") if isinstance(out, dict) else str(out)
        return {
            "result": str(answer),
            "cost": 0.01,
            "prompt_tokens": 120,
            "completion_tokens": 80,
        }
    except Exception as exc:  # offline fallback — log + deterministic stub
        snippet = str(objective)[:60].replace("\n", " ")
        return {
            "result": f"[offline-fallback ({type(exc).__name__})] {snippet}",
            "cost": 0.005,
            "prompt_tokens": 80,
            "completion_tokens": 40,
        }


async def handle_qa(
    question: str,
    role: str,
    agents_by_role: dict[str, GovernedSupervisor],
) -> dict[str, Any]:
    """Route a question to the governed supervisor matching ``role``.

    Args:
        question: The user's question.
        role: Access role — keys of ``agents_by_role`` (``qa`` | ``admin``
            | ``audit``).
        agents_by_role: Mapping of role name to a ``GovernedSupervisor``
            instance, as returned by ``build_capstone_stack(engine)[0]``.

    Returns:
        A response dict with the shape the 4 ex_8 technique files read:

            {
                "answer":          str,
                "confidence":      float,
                "sources":         list[str],
                "reasoning_steps": list[str],
                "latency_ms":      float,
                "governed":        True,
                "role":            str,
            }

        On governance / budget / clearance denial, returns a dict with
        ``"error"``, ``"blocked": True``, ``"governed": True``, and
        ``"role"`` instead of the answer shape.
    """
    gs = agents_by_role.get(role) or next(iter(agents_by_role.values()))
    start = time.time()

    try:
        result = await gs.run(
            objective=question,
            execute_node=_capstone_execute_node,
        )
        latency = (time.time() - start) * 1000

        # Pull the answer out of the supervisor's plan output. In offline
        # mode the executor stub returns a short string; in live mode
        # it returns the real LLM output. Either way, extract a string.
        answer_text = ""
        if result.audit_trail:
            last = result.audit_trail[-1]
            if isinstance(last, dict):
                answer_text = str(last.get("result") or last.get("output") or "")
        if not answer_text:
            answer_text = f"[capstone:{role}] {question}"

        return {
            "answer": answer_text,
            "confidence": 0.85 if result.success else 0.0,
            "sources": ["capstone_model", "pact_governance"],
            "reasoning_steps": [
                f"tier={role}",
                f"budget_consumed=${result.budget_consumed:.4f}",
                f"success={result.success}",
            ],
            "latency_ms": latency,
            "governed": True,
            "role": role,
        }
    except Exception as e:  # governance / budget / clearance denial
        return {
            "error": str(e),
            "governed": True,
            "blocked": True,
            "role": role,
        }


# ════════════════════════════════════════════════════════════════════════
# SHARED MIDDLEWARE UTILITIES
# ════════════════════════════════════════════════════════════════════════


class SimpleJWTAuth:
    """Stub JWT validator — production uses RS256 signed tokens.

    This exists so the capstone can demonstrate the auth surface without
    requiring a real JWKS endpoint. Every token maps to a role claim.
    """

    VALID_TOKENS: dict[str, dict[str, str]] = {
        "token_viewer_001": {"sub": "alice", "role": "qa"},
        "token_operator_001": {"sub": "bob", "role": "admin"},
        "token_auditor_001": {"sub": "carol", "role": "audit"},
    }

    @classmethod
    def validate(cls, token: str) -> dict[str, str] | None:
        """Return token claims, or None if the token is invalid."""
        return cls.VALID_TOKENS.get(token)


class RateLimiter:
    """Sliding-window rate limiter: ``max_requests`` per ``window_seconds``."""

    def __init__(self, max_requests: int, window_seconds: int) -> None:
        self.max_requests = max_requests
        self.window = window_seconds
        self.requests: dict[str, list[float]] = {}

    def allow(self, client_id: str) -> bool:
        now = time.time()
        bucket = self.requests.setdefault(client_id, [])
        bucket[:] = [t for t in bucket if now - t < self.window]
        if len(bucket) >= self.max_requests:
            return False
        bucket.append(now)
        return True


# ════════════════════════════════════════════════════════════════════════
# RUN HELPER — for technique files that use asyncio.run() at module scope
# ════════════════════════════════════════════════════════════════════════


def run_async(coro):  # noqa: ANN001 — coroutine
    """Run an async coroutine, tolerating already-running event loops."""
    try:
        return asyncio.run(coro)
    except RuntimeError:
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(coro)
        finally:
            loop.close()
