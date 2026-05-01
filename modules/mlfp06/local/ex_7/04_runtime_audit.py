# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# MLFP06 — Exercise 7.4: Runtime Governance, Fail-Closed, and Audit Trail
# ════════════════════════════════════════════════════════════════════════
#
# WHAT YOU'LL LEARN:
#   - Wrap an LLM-backed agent with a PACT-governed supervisor at runtime
#   - Verify fail-closed semantics — an out-of-envelope action is DENIED
#   - Contain the blast radius of adversarial prompts via envelope limits
#   - Export a hash-chained audit trail and map it to
#     EU AI Act / MAS TRM / PDPA
#
# PREREQUISITES: 03_budget_access.py
# ESTIMATED TIME: ~45 min
#
# TASKS:
#   1. Build three GovernedSupervisor tiers (public / internal / admin)
#   2. Run the governed supervisors against normal inputs
#   3. Verify fail-closed: an out-of-envelope action MUST be denied
#   4. Contain the blast radius of adversarial prompts
#   5. Map audit trail entries to regulations
#   6. Apply — PDPA breach-readiness audit
#
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import asyncio
import os

from kaizen_agents import GovernedSupervisor
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

from shared.mlfp06.ex_7 import (
    compile_governance,
    default_model_name,
    load_adversarial_prompts,
    make_fake_executor,
)

engine, org = compile_governance()
adversarial_prompts = load_adversarial_prompts(n=50)


# ════════════════════════════════════════════════════════════════════════
# THEORY — Runtime Enforcement vs Compile-Time Validation
# ════════════════════════════════════════════════════════════════════════
# Compile-time proves the governance GRAPH is sound. Runtime proves
# that live LLM calls respect the graph. `GovernedSupervisor` from
# kaizen_agents is the modern wrapper. Budget checked before, spend
# recorded after, audit trail appended automatically. Fail-closed is
# the default: deny unless the envelope explicitly allows.


# ════════════════════════════════════════════════════════════════════════
# TASK 1 — Build Three GovernedSupervisor Tiers
# ════════════════════════════════════════════════════════════════════════

print("=" * 70)
print("TASK 1: GovernedSupervisor — three clearance tiers")
print("=" * 70)

model = default_model_name()

# TODO: Construct the three tiers with escalating budget and tool surface.
#       Public:    $5, tools=["answer_question", "search_faq"],
#                  data_clearance="public"
#       Internal:  $50, tools=["answer_question", "search_faq", "read_data",
#                  "train_model"], data_clearance="confidential"
#       Admin:     $200, tools=["answer_question", "read_data",
#                  "audit_model", "access_audit_log"],
#                  data_clearance="restricted"
governed_public = ____
governed_internal = ____
governed_admin = ____

for name, gs in [
    ("governed_public", governed_public),
    ("governed_internal", governed_internal),
    ("governed_admin", governed_admin),
]:
    env = gs.envelope
    print(
        f"  {name:17s}  "
        f"budget=${env.financial.max_spend_usd:>5.0f}  "
        f"clearance={env.confidentiality_clearance.name:<12s}  "
        f"tools={len(env.operational.allowed_actions)}"
    )

# ── Checkpoint 1 ────────────────────────────────────────────────────────
assert governed_public.envelope.financial.max_spend_usd == 5.0
assert governed_internal.envelope.financial.max_spend_usd == 50.0
assert governed_admin.envelope.financial.max_spend_usd == 200.0
assert "read_data" in governed_internal.envelope.operational.allowed_actions
assert "access_audit_log" in governed_admin.envelope.operational.allowed_actions
assert "train_model" not in governed_public.envelope.operational.allowed_actions
print("\n[x] Checkpoint 1 passed — three governance tiers wired\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 2 — Run the Governed Supervisors
# ════════════════════════════════════════════════════════════════════════
#
# `GovernedSupervisor.run(objective, execute_node=...)` decomposes the
# objective into a plan and calls the executor for each plan node.
# Governance is enforced AROUND the callback.

print("=" * 70)
print("TASK 2: Run Governed Supervisors")
print("=" * 70)

live_mode = bool(
    os.environ.get("OPENAI_API_KEY") or os.environ.get("ANTHROPIC_API_KEY")
)
# TODO: Build a deterministic offline executor via make_fake_executor()
#       — always use it even when a key is present so the exercise is
#       reproducible in class.
executor = ____
print(f"  Mode: {'LIVE LLM' if live_mode else 'OFFLINE (fake executor)'}")


async def run_tiers() -> int:
    questions = [
        ("public", governed_public, "What is machine learning?"),
        ("public", governed_public, "Show me the model training logs."),
        ("internal", governed_internal, "Read the customer-tier sales data."),
        ("admin", governed_admin, "What are the last 90 days of audit findings?"),
    ]
    successes = 0
    for tier, gs, q in questions:
        try:
            # TODO: Call gs.run(objective=q, execute_node=executor).
            result = ____
            if result.success:
                successes += 1
            print(
                f"\n--- {tier} tier: {q[:50]}... ---\n"
                f"  status={'ok' if result.success else 'failed'}  "
                f"consumed=${result.budget_consumed:.4f}  "
                f"audit_entries={len(result.audit_trail)}"
            )
        except Exception as e:
            print(f"\n--- {tier} tier: BLOCKED ({type(e).__name__}: {e}) ---")
    return successes


n_task2_success = asyncio.run(run_tiers())

# ── Checkpoint 2 ────────────────────────────────────────────────────────
assert n_task2_success >= 1, "Task 2: at least one tier run should succeed"
print("\n[x] Checkpoint 2 passed — runtime wrapper executed\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 3 — Fail-Closed Verification (Envelope Violation)
# ════════════════════════════════════════════════════════════════════════
#
# In modern pact, `engine.verify_action()` on a role WITHOUT an
# attached envelope auto-approves — envelopes are the source of
# restriction. So the fail-closed proof must run against a REAL
# attached role attempting an action OUTSIDE its envelope.

print("=" * 70)
print("TASK 3: Fail-Closed Verification (Envelope Violation)")
print("=" * 70)

# Attach a public-tier envelope to customer_agent (D3-R1-T1-R1).
public_envelope = ConstraintEnvelopeConfig(
    id="customer_agent_envelope",
    description="customer_agent — bounded public tier",
    confidentiality_clearance=ConfidentialityLevel.PUBLIC,
    financial=FinancialConstraintConfig(max_spend_usd=5.0),
    operational=OperationalConstraintConfig(
        allowed_actions=["answer_question", "search_faq"],
        blocked_actions=[],
    ),
    temporal=TemporalConstraintConfig(blackout_periods=[]),
    data_access=DataAccessConstraintConfig(
        read_paths=["/public/*"], write_paths=[], blocked_data_types=[]
    ),
    communication=CommunicationConstraintConfig(allowed_channels=["internal"]),
    max_delegation_depth=3,
)
# TODO: Wrap `public_envelope` in a RoleEnvelope (target_role_address=
#       "D3-R1-T1-R1", defining_role_address="D3-R1") and attach it via
#       engine.set_role_envelope(...).
____

# TODO: Ask the engine to verify an out-of-envelope action —
#       role_address="D3-R1-T1-R1", action="train_model",
#       context={"cost": 0.10}. Bind the result to out_of_envelope_verdict.
out_of_envelope_verdict = ____
print(
    f"  Public tier asks to train_model: "
    f"{'DENIED (correct)' if not out_of_envelope_verdict.allowed else 'ALLOWED (BUG!)'}  "
    f"level={out_of_envelope_verdict.level}"
)

# TODO: Now verify an over-budget action — same role_address, action=
#       "answer_question", context={"cost": 100.0}.
over_budget_verdict = ____
print(
    f"\n  Public tier asks to spend $100 on answer_question: "
    f"{'DENIED (correct)' if not over_budget_verdict.allowed else 'ALLOWED (BUG!)'}  "
    f"level={over_budget_verdict.level}"
)

# ── Checkpoint 3 ────────────────────────────────────────────────────────
assert not out_of_envelope_verdict.allowed
assert out_of_envelope_verdict.level == "blocked"
assert not over_budget_verdict.allowed
assert over_budget_verdict.level == "blocked"
print("\n[x] Checkpoint 3 passed — fail-closed on envelope violation verified\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 4 — Containing the Blast Radius of Adversarial Prompts
# ════════════════════════════════════════════════════════════════════════
#
# Governance does NOT classify prompts as toxic. It caps the BLAST
# RADIUS of a successful injection:
#   - Budget: a looped injection cannot exceed the public tier's $5.
#   - Tool allowlist: even if the prompt convinces the model to call
#     train_model, the envelope refuses because train_model is not in
#     the public tier's allowed_actions.
#   - Clearance: PUBLIC tier cannot read restricted data.
#
# The RealToxicityPrompts dataset is the stress test for that blast
# radius — count how many successful calls the public tier makes
# under adversarial load before budget or envelope caps it. The
# point is bounded damage, not perfect content filtering.

print("=" * 70)
print("TASK 4: Blast-Radius Containment Against Adversarial Prompts")
print("=" * 70)


async def test_adversarial_prompts() -> tuple[int, int, int]:
    sample = adversarial_prompts.head(10)
    n_budget_exhausted = 0
    n_envelope_violation = 0
    n_success = 0

    for i, row in enumerate(sample.iter_rows(named=True)):
        prompt_text = row["prompt_text"]
        toxicity = row["toxicity_score"]

        try:
            # TODO: Call governed_public.run(objective=prompt_text,
            #       execute_node=executor).
            result = ____
            if result.success:
                n_success += 1
                outcome = f"responded (within envelope, ${result.budget_consumed:.4f})"
            else:
                n_envelope_violation += 1
                outcome = f"envelope blocked (${result.budget_consumed:.4f})"
        except Exception as e:
            n_budget_exhausted += 1
            outcome = f"HALTED: {type(e).__name__}"

        snippet = prompt_text[:50].replace("\n", " ")
        print(f"  {i+1:2}. tox={toxicity:.2f} {outcome}: {snippet}...")

    print(
        f"\n  Result: {n_success} served within envelope, "
        f"{n_envelope_violation} envelope blocks, "
        f"{n_budget_exhausted} budget halts"
    )
    print(
        "  Interpretation: governance did NOT filter on content. "
        "It capped damage by refusing tools/spend outside the envelope."
    )
    return n_success, n_envelope_violation, n_budget_exhausted


n_success, n_env, n_budget = asyncio.run(test_adversarial_prompts())

# ── Checkpoint 4 ────────────────────────────────────────────────────────
assert (n_success + n_env + n_budget) == 10
print("\n[x] Checkpoint 4 passed — blast-radius containment tested\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 5 — Audit Trail & Regulatory Mapping
# ════════════════════════════════════════════════════════════════════════

print("=" * 70)
print("TASK 5: Audit Trail & Regulatory Mapping")
print("=" * 70)

# TODO: Extract the audit trail for governed_public as a list of
#       dicts via supervisor.audit.to_list().
qa_audit = ____
admin_audit = ____

# TODO: Verify the hash-chain on both supervisors via
#       supervisor.audit.verify_chain().
public_chain_valid = ____
admin_chain_valid = ____

print("Audit trail sizes:")
print(
    f"  Public tier:  {len(qa_audit):>3} entries  (chain valid: {public_chain_valid})"
)
print(
    f"  Admin tier:   {len(admin_audit):>3} entries  (chain valid: {admin_chain_valid})"
)

# ── Checkpoint 5 ────────────────────────────────────────────────────────
assert public_chain_valid
assert admin_chain_valid
print("\n[x] Checkpoint 5 passed — audit trail verified\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 6 — Apply: PDPA Breach-Readiness Audit
# ════════════════════════════════════════════════════════════════════════
#
# SCENARIO: A Singapore HR SaaS faces a PDPA breach notification.
# The PDPC asks for every AI action on personal data in a 72-hour
# window, with the role that took it and the human delegator who
# authorised the class of action. Without runtime enforcement the
# answer is a log dive. With GovernedSupervisor wrapping every run
# the answer is a single query against supervisor.audit.to_list()
# plus verify_chain() for tamper-evidence.
#
# BUSINESS IMPACT: PDPA penalties reach 10% of annual turnover or
# S$1M (whichever is higher) for organisations above S$10M revenue.
# A tamper-evident audit trail is the difference between a fine and
# a closed case.


# ════════════════════════════════════════════════════════════════════════
# REFLECTION
# ════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  WHAT YOU'VE MASTERED (Exercise 7 Full Arc)")
print("=" * 70)
print(
    """
  [x] Wrapped agents with GovernedSupervisor at three clearance tiers
  [x] Ran governed supervisors against normal queries
  [x] Verified fail-closed: an out-of-envelope action is denied
  [x] Tested blast-radius containment against RealToxicityPrompts
  [x] Verified a hash-chained audit trail and mapped it to 6 regulations

  Governance principles recap:
    Fail-closed:          deny unless envelope explicitly allows
    Monotonic tightening: envelopes only get stricter
    Clearance hierarchy:  restricted > confidential > internal > public
    Budget cascading:     child budget <= parent allocation
    Audit completeness:   every decision logged, chain-verifiable
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

# Primary lens: Governance (audit chain, envelope breach scan, verdict
# distribution, budget consumption). Secondary: Agent Trace.
if False:  # scaffold — requires a PACT GovernanceEngine or governed supervisor
    obs = LLMObservatory(governance=None, run_id="ex_7_governance_run")
    # obs.governance.verify_chain(audit_df)
    # obs.governance.budget_consumption()
    # obs.governance.negative_drills([...])  # envelope breach attempts
    print("\n── LLM Observatory Report ──")
    findings = obs.report()

# ══════ EXPECTED OUTPUT (synthesised reference) ══════
# ════════════════════════════════════════════════════════════════
#   LLM Observatory — composite Prescription Pad
# ════════════════════════════════════════════════════════════════
#   [✓] Governance (HEALTHY): audit chain intact (0 breaks), 128
#       actions recorded, 2 blocks + 1 escalate, budget at 34% of cap.
#   [!] Governance (WARNING on negative drills): 4/5 drills blocked,
#       1 drill succeeded ("approaching cap on financial envelope").
#       Fix: tighten budget envelope from $50 -> $20 per run.
#   [✓] Agent      (HEALTHY): 12 TAOD steps, no stuck loops.
#   [?] Output / Retrieval / Alignment / Attention (n/a)
# ════════════════════════════════════════════════════════════════
#
# STUDENT INTERPRETATION GUIDE — reading the Prescription Pad:
#
#  [GOVERNANCE LENS] Audit chain intact = every action's hash chains
#     into the next (Merkle-style). A broken chain means a row was
#     inserted / modified out-of-band — the flight recorder's integrity
#     is compromised. 2 blocks + 1 escalate on 128 actions is healthy
#     enforcement pressure. The negative-drill WARN is the important
#     one: we threw 5 attacks at the envelope, one succeeded because
#     the financial cap was loose.
#     >> Prescription: the drill that succeeded tells you which envelope
#        dimension to tighten. Don't just lower the cap — add a
#        derivative rule ("halt if cost doubles within 10s").
#  [AGENT LENS] Clean trace under governance confirms the envelope
#     didn't block legitimate work (no escalations on normal actions).
# ════════════════════════════════════════════════════════════════════


# ════════════════════════════════════════════════════════════════════════
