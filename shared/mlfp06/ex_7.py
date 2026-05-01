# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
Shared infrastructure for MLFP06 Exercise 7 — AI Governance with PACT.

Contains: adversarial-prompt loading, canonical Singapore FinTech org YAML,
clearance hierarchy, teaching budget tracker, GovernanceEngine compile helper,
CompiledOrgAdapter (preserves the `.n_agents / .n_delegations / .n_departments`
caller contract for technique files), and `make_fake_executor()` — a
deterministic offline executor for `GovernedSupervisor.run(execute_node=...)`
so the runtime-enforcement narrative runs end-to-end without an LLM key.

Technique-specific code does NOT belong here — each technique file builds
its own scenario on top.

Import from any cwd after `uv sync`:

    from shared.mlfp06.ex_7 import (
        CLEARANCE_LEVELS, ORG_YAML, load_adversarial_prompts,
        write_org_yaml, compile_governance, TeachingBudgetTracker,
        CompiledOrgAdapter, default_model_name, make_fake_executor,
    )
"""
from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Awaitable, Callable

import polars as pl

from shared.kailash_helpers import setup_environment

setup_environment()

if TYPE_CHECKING:  # pragma: no cover — type-only imports
    from pact import CompiledOrg, GovernanceEngine

# ════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ════════════════════════════════════════════════════════════════════════

# MLFP06 teaches a 4-level clearance hierarchy (public < internal <
# confidential < restricted). Canonical pact is 5-level; see the sidebar
# in ex_7/02_envelopes.py for the mapping to PUBLIC/RESTRICTED/
# CONFIDENTIAL/SECRET/TOP_SECRET. The course's mental model keeps
# "internal" and "restricted" as distinct teaching rungs; "internal"
# is an historical alias of RESTRICTED at the string interface.
CLEARANCE_LEVELS: dict[str, int] = {
    "public": 0,
    "internal": 1,
    "confidential": 2,
    "restricted": 3,
}


# Default LLM (lazy-resolved; agents read at construction time)
def default_model_name() -> str | None:
    from shared.mlfp06._ollama_bootstrap import DEFAULT_CHAT_MODEL

    return DEFAULT_CHAT_MODEL


# ════════════════════════════════════════════════════════════════════════
# ADVERSARIAL PROMPT DATASET
# ════════════════════════════════════════════════════════════════════════

CACHE_DIR = Path("data/mlfp06/toxicity")
CACHE_FILE = CACHE_DIR / "real_toxicity_50.parquet"


def load_adversarial_prompts(n: int = 50) -> pl.DataFrame:
    """Load (and cache) the allenai/real-toxicity-prompts adversarial slice.

    Filters to prompts with toxicity > 0.5, shuffles with a fixed seed,
    and returns the first `n` rows as a polars DataFrame with columns
    `prompt_text` and `toxicity_score`.
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    if CACHE_FILE.exists():
        return pl.read_parquet(CACHE_FILE)

    from datasets import load_dataset

    ds = load_dataset("allenai/real-toxicity-prompts", split="train")
    ds = ds.filter(
        lambda r: r["prompt"]["toxicity"] is not None and r["prompt"]["toxicity"] > 0.5
    )
    ds = ds.shuffle(seed=42).select(range(min(n, len(ds))))
    rows = [
        {
            "prompt_text": row["prompt"]["text"],
            "toxicity_score": row["prompt"]["toxicity"],
        }
        for row in ds
    ]
    df = pl.DataFrame(rows)
    df.write_parquet(CACHE_FILE)
    return df


# ════════════════════════════════════════════════════════════════════════
# CANONICAL SINGAPORE FINTECH ORG YAML (D/T/R GRAMMAR)
# ════════════════════════════════════════════════════════════════════════
#
# Every technique file uses the same organisation so students can track
# how envelopes, budgets, and access decisions evolve as they add more
# governance structure. The D/T/R grammar is:
#   D (Delegator):   Human authority who authorises the task
#   T (Task):        Bounded scope of work (team)
#   R (Responsible): The agent that executes within the envelope
#
# Modern pact's load_org_yaml expects a flat top-level schema:
#   org_id, name
#   departments[] teams[] roles[] clearances[] envelopes[]
#
# Roles attach to units via `heads: <dept_or_team_id>` and chain via
# `reports_to: <role_id>`. Envelopes carry the delegation contract
# (financial + operational + ...); each `envelopes[]` entry is one
# D/T/R delegation in the compiled org.

ORG_YAML: str = """
# Singapore FinTech AI Organisation — PACT Governance Definition
# D/T/R: every agent action traces to a human Delegator

org_id: "sg_fintech_ai"
name: "SG FinTech AI Division"

# Three departments — each headed by a named human authority.
departments:
  - id: "ml_eng"
    name: "ML Engineering"
  - id: "risk_compliance"
    name: "Risk and Compliance"
  - id: "customer_intel"
    name: "Customer Intelligence"

# One team per delegated task (bounded scope of work).
teams:
  - id: "data_team"
    name: "Data Analysis"
  - id: "training_team"
    name: "Model Training"
  - id: "deploy_team"
    name: "Model Deployment"
  - id: "risk_team"
    name: "Risk Assessment"
  - id: "bias_team"
    name: "Bias Audit"
  - id: "customer_team"
    name: "Customer Interaction"

# Department heads (3 Delegators) + agents (6 Responsibles).
roles:
  # ── Delegators (humans) ──
  - id: "chief_ml_officer"
    name: "Chief ML Officer"
    heads: "ml_eng"
  - id: "chief_risk_officer"
    name: "Chief Risk Officer"
    heads: "risk_compliance"
  - id: "vp_customer"
    name: "VP Customer"
    heads: "customer_intel"

  # ── Responsibles (agents) ──
  - id: "data_analyst"
    name: "Data Analyst"
    reports_to: "chief_ml_officer"
    heads: "data_team"
  - id: "model_trainer"
    name: "Model Trainer"
    reports_to: "chief_ml_officer"
    heads: "training_team"
  - id: "model_deployer"
    name: "Model Deployer"
    reports_to: "chief_ml_officer"
    heads: "deploy_team"
  - id: "risk_assessor"
    name: "Risk Assessor"
    reports_to: "chief_risk_officer"
    heads: "risk_team"
  - id: "bias_checker"
    name: "Bias Checker"
    reports_to: "chief_risk_officer"
    heads: "bias_team"
  - id: "customer_agent"
    name: "Customer Agent"
    reports_to: "vp_customer"
    heads: "customer_team"

# Clearance lattice — canonical pact levels (lowercase strings).
clearances:
  - role: "chief_ml_officer"
    level: "restricted"
  - role: "chief_risk_officer"
    level: "restricted"
  - role: "vp_customer"
    level: "confidential"
  - role: "data_analyst"
    level: "restricted"
  - role: "model_trainer"
    level: "confidential"
  - role: "model_deployer"
    level: "confidential"
  - role: "risk_assessor"
    level: "restricted"
  - role: "bias_checker"
    level: "confidential"
  - role: "customer_agent"
    level: "public"

# Envelopes = delegations. Each entry binds a task scope (target agent)
# to the authorising human (defined_by) and the constraint set the
# agent runs within.
envelopes:
  - target: "data_analyst"
    defined_by: "chief_ml_officer"
    financial:
      max_spend_usd: 20.0
    operational:
      allowed_actions: ["read_data", "summarise_data", "generate_report"]
    data_access:
      max_rows: 500000

  - target: "model_trainer"
    defined_by: "chief_ml_officer"
    financial:
      max_spend_usd: 100.0
    operational:
      allowed_actions: ["train_model", "evaluate_model", "read_data"]
    data_access:
      max_rows: 1000000

  - target: "model_deployer"
    defined_by: "chief_ml_officer"
    financial:
      max_spend_usd: 50.0
    operational:
      allowed_actions: ["deploy_model", "monitor_model", "rollback_model"]

  - target: "risk_assessor"
    defined_by: "chief_risk_officer"
    financial:
      max_spend_usd: 200.0
    operational:
      allowed_actions:
        - "read_data"
        - "audit_model"
        - "generate_report"
        - "access_audit_log"

  - target: "bias_checker"
    defined_by: "chief_risk_officer"
    financial:
      max_spend_usd: 75.0
    operational:
      allowed_actions: ["read_data", "audit_model", "run_fairness_check"]

  - target: "customer_agent"
    defined_by: "vp_customer"
    financial:
      max_spend_usd: 5.0
    operational:
      allowed_actions: ["answer_question", "search_faq"]
    communication:
      max_response_length: 500
"""


def write_org_yaml(path: str | Path | None = None) -> str:
    """Write the canonical org YAML to a temp file and return the path."""
    if path is None:
        path = os.path.join(tempfile.gettempdir(), "sg_fintech_org.yaml")
    with open(path, "w") as f:
        f.write(ORG_YAML)
    return str(path)


# ════════════════════════════════════════════════════════════════════════
# COMPILED ORG ADAPTER — caller-contract shim
# ════════════════════════════════════════════════════════════════════════
#
# Modern pact's `CompiledOrg` exposes `org_id` and `nodes` — a flat
# dict of addresses (e.g. "D1-R1-T1-R1") to `OrgNode` objects with a
# `node_type` enum (DEPARTMENT | TEAM | ROLE). The MLFP06 course code
# has always used friendlier counters: `org.n_agents`,
# `org.n_delegations`, `org.n_departments`. Rather than rewrite every
# technique file, the adapter computes those counters from the flat
# nodes dict + the original envelopes list.


@dataclass
class CompiledOrgAdapter:
    """Thin facade over `pact.CompiledOrg` preserving the course's counter API.

    Agents in MLFP06 are the non-vacant ROLE nodes that report to a
    department head — the 6 Responsibles in the SG FinTech org. The 3
    department heads are Delegators, not agents. We count agents as
    "non-vacant ROLE nodes that are not themselves a department head".
    Delegations are envelopes; there is one envelope per delegation.
    """

    _compiled: "CompiledOrg"
    _n_envelopes: int

    @property
    def n_departments(self) -> int:
        from pact import NodeType

        return sum(
            1
            for n in self._compiled.nodes.values()
            if n.node_type == NodeType.DEPARTMENT
        )

    @property
    def n_teams(self) -> int:
        from pact import NodeType

        return sum(
            1 for n in self._compiled.nodes.values() if n.node_type == NodeType.TEAM
        )

    @property
    def n_agents(self) -> int:
        """Count Responsible roles (team-anchored ROLE nodes, non-vacant).

        A "Responsible" role is a ROLE node whose address sits under a
        TEAM (has a `-T<n>-R<n>` suffix) — distinguishing it from the
        department-head ROLE nodes that sit directly under a DEPARTMENT
        address (e.g. "D1-R1"). We also exclude vacant placeholders so
        the count reflects real agents.
        """
        from pact import NodeType

        count = 0
        for addr, node in self._compiled.nodes.items():
            if node.node_type != NodeType.ROLE:
                continue
            if node.is_vacant:
                continue
            # Department-head addresses are "D<n>-R<n>" — two segments.
            # Agent (Responsible) addresses sit under a team and have
            # the "-T<n>-R<n>" suffix, giving four or more segments.
            if "-T" in addr:
                count += 1
        return count

    @property
    def n_delegations(self) -> int:
        """One envelope == one D/T/R delegation contract."""
        return self._n_envelopes

    @property
    def org_id(self) -> str:
        return self._compiled.org_id


# ════════════════════════════════════════════════════════════════════════
# GOVERNANCE ENGINE COMPILATION
# ════════════════════════════════════════════════════════════════════════


def compile_governance(
    yaml_path: str | None = None,
) -> tuple["GovernanceEngine", CompiledOrgAdapter]:
    """Compile the canonical org YAML. Returns (engine, adapter).

    Modern pact flow:
        LoadedOrg      <- load_org_yaml(path)
        GovernanceEngine(loaded.org_definition)
        compiled = engine.get_org()
        adapter  = CompiledOrgAdapter(compiled, n_envelopes=len(loaded.envelopes))

    Compilation validates (via `load_org_yaml` + engine construction):
      - Every role references a known unit via `heads`
      - `reports_to` chains resolve to declared roles
      - Clearance levels are in the canonical lattice
      - Envelopes reference real roles on both `target` and `defined_by`
      - D/T/R grammar: every Department/Team is followed by exactly one Role
    What compilation does NOT validate:
      - Content safety of LLM outputs (needs adversarial testing)
      - Runtime budget consumption (needs the runtime wrapper in ex_7/04)
    """
    from pact import GovernanceEngine, load_org_yaml

    if yaml_path is None:
        yaml_path = write_org_yaml()
    loaded = load_org_yaml(yaml_path)
    engine = GovernanceEngine(loaded.org_definition)
    compiled = engine.get_org()
    adapter = CompiledOrgAdapter(_compiled=compiled, _n_envelopes=len(loaded.envelopes))
    return engine, adapter


# ════════════════════════════════════════════════════════════════════════
# TEACHING BUDGET TRACKER
# ════════════════════════════════════════════════════════════════════════
#
# Named `TeachingBudgetTracker` to avoid collision with internal
# `pact.BudgetTracker` / `pact.CostTracker` primitives. This is a
# simple pedagogical object for illustrating parent-to-child budget
# cascading in ex_7/03_budget_access — NOT a production substitute
# for the governance engine's own financial envelope enforcement.


class TeachingBudgetTracker:
    """Track budget allocation and consumption across an agent hierarchy.

    Parent allocates to children; children cannot spend more than their
    allocation. Used by ex_7/03_budget_access to demonstrate the
    monotonic-tightening property of financial envelopes in a form
    students can step through by hand.
    """

    def __init__(self, total_budget: float) -> None:
        self.total_budget = total_budget
        self.consumed: dict[str, float] = {}
        self.allocations: dict[str, float] = {}

    def allocate(self, agent_id: str, amount: float) -> bool:
        """Allocate budget to an agent. Returns False if insufficient."""
        total_allocated = sum(self.allocations.values())
        if total_allocated + amount > self.total_budget:
            return False
        self.allocations[agent_id] = self.allocations.get(agent_id, 0) + amount
        return True

    def spend(self, agent_id: str, amount: float) -> bool:
        """Record spending. Returns False if exceeds allocation."""
        allocation = self.allocations.get(agent_id, 0)
        current = self.consumed.get(agent_id, 0)
        if current + amount > allocation:
            return False
        self.consumed[agent_id] = current + amount
        return True

    def remaining(self, agent_id: str) -> float:
        return self.allocations.get(agent_id, 0) - self.consumed.get(agent_id, 0)

    def summary(self) -> pl.DataFrame:
        agents = set(self.allocations.keys()) | set(self.consumed.keys())
        rows = []
        for a in sorted(agents):
            rows.append(
                {
                    "agent": a,
                    "allocated": self.allocations.get(a, 0),
                    "consumed": self.consumed.get(a, 0),
                    "remaining": self.remaining(a),
                }
            )
        return pl.DataFrame(rows)


# ════════════════════════════════════════════════════════════════════════
# FAKE EXECUTOR — offline `GovernedSupervisor.run(execute_node=...)`
# ════════════════════════════════════════════════════════════════════════
#
# `GovernedSupervisor.run(objective, execute_node=...)` decomposes an
# objective into a plan and invokes `execute_node(spec, inputs)` for
# each node in that plan. The executor is where the real LLM call
# (or a stub) lives. Governance is enforced AROUND the callback —
# budget checked before, spend recorded after, audit trail appended
# automatically.
#
# `make_fake_executor()` returns a deterministic async callable that
# satisfies the `ExecuteNodeFn` type (`(AgentSpec, dict[str, Any])
# -> Awaitable[dict[str, Any]]`) so the runtime-enforcement narrative
# in ex_7/04_runtime_audit.py runs end-to-end offline. The fake
# short-circuits the LLM at the callback boundary but preserves the
# governance wiring exactly.


def make_fake_executor(
    *,
    base_cost: float = 0.01,
    prompt_tokens: int = 80,
    completion_tokens: int = 40,
    reply_prefix: str = "[offline-fake]",
) -> Callable[[Any, dict[str, Any]], Awaitable[dict[str, Any]]]:
    """Build a deterministic async executor for GovernedSupervisor.run.

    The returned callable accepts `(spec, inputs)` where `spec` is a
    `kaizen_agents.types.AgentSpec` and `inputs` is a plan-node input
    dict. It returns a dict with the four keys the supervisor expects:

        {
            "result": str,           # the node's "output"
            "cost": float,           # USD consumed by this node
            "prompt_tokens": int,    # input token count
            "completion_tokens": int,# output token count
        }

    All values are constant per call — the fake is intentionally
    deterministic so tests are reproducible. Use this in offline mode
    when no LLM key is available; swap for a real LLM-backed executor
    in production.
    """

    async def _fake(spec: Any, inputs: dict[str, Any]) -> dict[str, Any]:
        # Extract a human-readable label for the stubbed response so
        # the audit trail is still meaningful when read later.
        node_id = getattr(spec, "node_id", None) or "node"
        objective = inputs.get("objective") or inputs.get("prompt") or str(inputs)
        snippet = str(objective)[:60].replace("\n", " ")
        return {
            "result": f"{reply_prefix} {node_id}: {snippet}",
            "cost": float(base_cost),
            "prompt_tokens": int(prompt_tokens),
            "completion_tokens": int(completion_tokens),
        }

    return _fake
