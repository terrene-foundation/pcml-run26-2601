# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""Bias-mitigated LLM-as-judge primitives built on Kaizen ``Delegate``.

All lens classes that need an LLM judge (Output, Retrieval, Alignment) share
this module. The single ``JudgeCallable`` abstraction wraps a Kaizen Delegate
with:

    * position-swap bias mitigation for pairwise preference judging
    * length-normalised pairwise scoring
    * a hard budget (``max_judge_calls``) so diagnostic runs stay cheap
    * structured INFO logs with correlation IDs per :mod:`rules.observability`

Per ``rules/framework-first.md`` MANDATORY: raw ``openai.chat.completions.create``
is BLOCKED — every LLM call goes through ``Delegate.run_sync`` which honours
the configured cost envelope.

Per ``rules/env-models.md``: ``judge_model`` defaults to
``OPENAI_JUDGE_MODEL`` → ``DEFAULT_LLM_MODEL`` → ``OPENAI_PROD_MODEL``. No
hardcoded model names.
"""
from __future__ import annotations

import logging
import os
import re
import threading
import time
import uuid
from dataclasses import dataclass
from typing import Any

from dotenv import load_dotenv

logger = logging.getLogger(__name__)


# ════════════════════════════════════════════════════════════════════════
# Model resolution — rules/env-models.md
# ════════════════════════════════════════════════════════════════════════


def resolve_judge_model(explicit: str | None = None) -> str:
    """Return the judge model name per ``rules/env-models.md``.

    Resolution order: explicit arg → ``OLLAMA_JUDGE_MODEL`` →
    ``OLLAMA_CHAT_MODEL`` → bootstrap default. The previous OpenAI fall-throughs
    (``OPENAI_JUDGE_MODEL`` / ``OPENAI_PROD_MODEL``) are honoured if set so
    legacy ``.env`` files still resolve, but the bootstrap default is the
    primary path.
    """
    load_dotenv()
    if explicit:
        return explicit
    for key in (
        "OLLAMA_JUDGE_MODEL",
        "OLLAMA_CHAT_MODEL",
        "OPENAI_JUDGE_MODEL",
        "DEFAULT_LLM_MODEL",
        "OPENAI_PROD_MODEL",
    ):
        val = os.environ.get(key)
        if val:
            return val
    # Final fallback: the M6 bootstrap chat default.
    from shared.mlfp06._ollama_bootstrap import DEFAULT_CHAT_MODEL

    return DEFAULT_CHAT_MODEL


# ════════════════════════════════════════════════════════════════════════
# Verdict schema
# ════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class JudgeVerdict:
    """Structured judgement output.

    Attributes:
        score: Numeric verdict in ``[0.0, 1.0]``. Higher is better.
        rationale: Plain-language explanation from the judge.
        criteria: The criteria string the judge was asked to score against.
        judge_model: The model that produced the verdict.
        mode: ``"real"`` when a live LLM was called; ``"fake"`` when the
            budget was exhausted or no delegate was available. Per
            ``rules/observability.md`` §3, every data call tags ``mode``.
        latency_ms: Wall-clock duration of the judge call.
    """

    score: float
    rationale: str
    criteria: str
    judge_model: str
    mode: str
    latency_ms: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "score": self.score,
            "rationale": self.rationale,
            "criteria": self.criteria,
            "judge_model": self.judge_model,
            "mode": self.mode,
            "latency_ms": self.latency_ms,
        }


# ════════════════════════════════════════════════════════════════════════
# Judge runner
# ════════════════════════════════════════════════════════════════════════


_SCORE_REGEX = re.compile(r"score\s*[:=]\s*([0-9]*\.?[0-9]+)", re.IGNORECASE)


def _parse_score(raw: str) -> float:
    """Extract a float score in ``[0, 1]`` from a judge's reply.

    The judge prompt asks for ``SCORE: <float in 0..1>``. A well-behaved
    judge returns e.g. ``SCORE: 0.82``. Fallback: look for the first
    0..1-looking float in the text. If no number is found, return ``0.0``
    and emit a WARN per ``rules/observability.md`` §1 (WARN on fallback).
    """
    match = _SCORE_REGEX.search(raw)
    if match:
        try:
            val = float(match.group(1))
            if val > 1.0 and val <= 10.0:
                # Judges sometimes produce 0..10; normalise.
                val = val / 10.0
            return max(0.0, min(1.0, val))
        except ValueError:
            pass
    # Second attempt — any float 0..1 in the text.
    for token in re.findall(r"[0-9]*\.?[0-9]+", raw):
        try:
            f = float(token)
            if 0.0 <= f <= 1.0:
                return f
        except ValueError:
            continue
    logger.warning(
        "judge.parse_fallback",
        extra={"raw_preview": raw[:120], "parsed_score": 0.0},
    )
    return 0.0


class JudgeCallable:
    """Bias-mitigated LLM-as-judge backed by a Kaizen ``Delegate``.

    The judge is constructed lazily: no network call happens at
    ``__init__`` time. The first ``__call__`` instantiates the Delegate.

    Args:
        judge_model: Model name. Resolved via :func:`resolve_judge_model`
            if ``None``.
        max_judge_calls: Hard cap on live judge calls per instance.
            Defaults to ``50`` (≈ <$2 for typical eval). When exhausted,
            subsequent calls return ``mode="fake"`` verdicts and emit WARN.
        delegate: Optional pre-built Delegate. If ``None``, a minimal one
            is constructed lazily on first call.
        sensitive: When ``True``, the prompt / response pair is NOT logged —
            only hashes. Per ``rules/observability.md`` §4 (no secrets).

    Example::

        judge = JudgeCallable(max_judge_calls=10)
        verdict = judge.score(
            "Paris is the capital of France.",
            criteria="factual_accuracy",
            context="User asked: what is the capital of France?",
        )
        print(verdict.score, verdict.rationale)
    """

    _JUDGE_SYSTEM = (
        "You are a rigorous LLM-as-judge. Score the given response against "
        "the stated criteria. Be strict, cite evidence, and output a single "
        "line of the form 'SCORE: <float in 0..1>' followed by a one-sentence "
        "rationale. Do NOT hedge with 0.5 — commit to a verdict."
    )

    def __init__(
        self,
        *,
        judge_model: str | None = None,
        max_judge_calls: int = 50,
        delegate: Any = None,
        sensitive: bool = False,
    ) -> None:
        if max_judge_calls < 0:
            raise ValueError("max_judge_calls must be >= 0")
        self._model_name = resolve_judge_model(judge_model)
        self._max_calls = max_judge_calls
        self._call_count = 0
        self._sensitive = sensitive
        self._delegate = delegate
        self._lock = threading.Lock()

    @property
    def model(self) -> str:
        return self._model_name

    @property
    def calls_consumed(self) -> int:
        return self._call_count

    @property
    def calls_remaining(self) -> int:
        return max(0, self._max_calls - self._call_count)

    def _ensure_delegate(self) -> Any:
        if self._delegate is not None:
            return self._delegate
        # Lazy import — bootstrap is not needed at module import time.
        from shared.mlfp06._ollama_bootstrap import make_delegate

        self._delegate = make_delegate(
            model=self._model_name,
            system_prompt=self._JUDGE_SYSTEM,
            temperature=0.0,
        )
        logger.info(
            "judge.delegate_constructed",
            extra={"judge_model": self._model_name, "provider": "ollama"},
        )
        return self._delegate

    def close(self) -> None:
        """Release the underlying Delegate's network resources."""
        if self._delegate is not None and hasattr(self._delegate, "close"):
            try:
                self._delegate.close()
            except Exception:
                # Cleanup path — zero-tolerance Rule 3 carve-out.
                pass
        self._delegate = None

    # ── Core scoring API ────────────────────────────────────────────────

    def score(
        self,
        response: str,
        *,
        criteria: str,
        context: str = "",
        run_id: str | None = None,
    ) -> JudgeVerdict:
        """Score a single response against a criteria string.

        Args:
            response: The model output under evaluation.
            criteria: Short phrase describing what to score for
                (e.g. ``"factual_accuracy"``, ``"coherence,helpfulness"``).
            context: Optional context — source document, user prompt, etc.
            run_id: Correlation ID per ``rules/observability.md`` §2.
                Auto-generated if ``None``.

        Returns:
            A :class:`JudgeVerdict`.
        """
        if run_id is None:
            run_id = f"judge-{uuid.uuid4().hex[:12]}"
        t0 = time.monotonic()

        with self._lock:
            over_budget = self._call_count >= self._max_calls
            if not over_budget:
                self._call_count += 1

        if over_budget:
            logger.warning(
                "judge.budget_exhausted",
                extra={
                    "run_id": run_id,
                    "max_calls": self._max_calls,
                    "calls": self._call_count,
                },
            )
            return JudgeVerdict(
                score=0.0,
                rationale=(
                    f"BUDGET EXHAUSTED after {self._max_calls} calls. "
                    "Increase max_judge_calls to score further."
                ),
                criteria=criteria,
                judge_model=self._model_name,
                mode="fake",
                latency_ms=(time.monotonic() - t0) * 1000.0,
            )

        prompt = _build_judge_prompt(response, criteria=criteria, context=context)
        log_extra = {
            "run_id": run_id,
            "judge_model": self._model_name,
            "criteria": criteria,
            "source": "kaizen_delegate",
            "mode": "real",
        }
        if not self._sensitive:
            log_extra["response_preview"] = response[:80]
        logger.info("judge.score.start", extra=log_extra)

        try:
            delegate = self._ensure_delegate()
            raw = delegate.run_sync(prompt)
        except Exception as exc:
            logger.exception(
                "judge.score.error",
                extra={"run_id": run_id, "error": str(exc)},
            )
            raise

        score_val = _parse_score(raw)
        # Rationale = everything after the SCORE line (or whole reply if none).
        rationale = _extract_rationale(raw)
        latency = (time.monotonic() - t0) * 1000.0

        logger.info(
            "judge.score.ok",
            extra={
                "run_id": run_id,
                "judge_model": self._model_name,
                "score": score_val,
                "latency_ms": latency,
                "mode": "real",
            },
        )
        return JudgeVerdict(
            score=score_val,
            rationale=rationale,
            criteria=criteria,
            judge_model=self._model_name,
            mode="real",
            latency_ms=latency,
        )

    # ── Pairwise (win-rate) ────────────────────────────────────────────

    def pairwise(
        self,
        a: str,
        b: str,
        *,
        prompt: str,
        criteria: str = "helpfulness,harmlessness,correctness",
        swap_positions: bool = True,
        length_normalise: bool = True,
        run_id: str | None = None,
    ) -> dict[str, Any]:
        """Pairwise preference with position-swap bias mitigation.

        Runs the judge twice (A-then-B and B-then-A) and averages the
        preference. When ``length_normalise`` is ``True``, the prompt
        instructs the judge to discount verbosity.

        Returns:
            A dict with keys ``winner`` (``"a"`` / ``"b"`` / ``"tie"``),
            ``score_a``, ``score_b``, ``margin``, ``mode``, ``latency_ms``.
        """
        if run_id is None:
            run_id = f"pairwise-{uuid.uuid4().hex[:12]}"
        t0 = time.monotonic()

        crit = criteria
        if length_normalise:
            crit = f"{criteria},length_normalised"

        # Forward pass — A vs B.
        forward = self.score(
            response=_pairwise_blob(a=a, b=b, prompt=prompt),
            criteria=crit,
            context="Which response better satisfies the prompt? SCORE >0.5 if A wins, <0.5 if B wins.",
            run_id=f"{run_id}-ab",
        )
        pref_a_forward = forward.score

        if swap_positions:
            reverse = self.score(
                response=_pairwise_blob(a=b, b=a, prompt=prompt),
                criteria=crit,
                context="Which response better satisfies the prompt? SCORE >0.5 if A wins, <0.5 if B wins.",
                run_id=f"{run_id}-ba",
            )
            # reverse.score > 0.5 means "b was preferred" in reversed framing,
            # i.e. original A was preferred → same direction.
            pref_a = 0.5 * pref_a_forward + 0.5 * (1.0 - reverse.score)
            mode = (
                "real"
                if (forward.mode == "real" and reverse.mode == "real")
                else "fake"
            )
        else:
            pref_a = pref_a_forward
            mode = forward.mode

        margin = pref_a - 0.5
        winner = "a" if pref_a > 0.55 else ("b" if pref_a < 0.45 else "tie")
        latency = (time.monotonic() - t0) * 1000.0
        return {
            "winner": winner,
            "pref_a": pref_a,
            "score_a": pref_a,
            "score_b": 1.0 - pref_a,
            "margin": margin,
            "mode": mode,
            "latency_ms": latency,
            "judge_model": self._model_name,
        }


def _build_judge_prompt(response: str, *, criteria: str, context: str = "") -> str:
    """Assemble the judge prompt. Pure string formatting — no LLM reasoning here."""
    return (
        f"[CONTEXT]\n{context or '(none)'}\n\n"
        f"[RESPONSE]\n{response}\n\n"
        f"[CRITERIA]\n{criteria}\n\n"
        "Output format (exact):\n"
        "SCORE: <float in 0..1>\n"
        "REASON: <one sentence>"
    )


def _pairwise_blob(*, a: str, b: str, prompt: str) -> str:
    return f"[PROMPT]\n{prompt}\n\n" f"[RESPONSE A]\n{a}\n\n" f"[RESPONSE B]\n{b}\n"


def _extract_rationale(raw: str) -> str:
    match = re.search(r"reason\s*[:=]\s*(.+)", raw, re.IGNORECASE | re.DOTALL)
    if match:
        return match.group(1).strip().splitlines()[0][:400]
    # Fall back to last 200 chars.
    return raw.strip()[-400:]
