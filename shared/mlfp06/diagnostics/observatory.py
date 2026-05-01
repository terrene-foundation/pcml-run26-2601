# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""LLM Observatory — central facade composing all six clinical lenses.

Mirrors ``kailash_ml.diagnostics.DLDiagnostics`` (the M5 Doctor's Bag)
for the M6 problem domain. The facade gives M6 exercises one import and
one constructor call that wires up every lens:

    * :class:`~shared.mlfp06.diagnostics.output.LLMDiagnostics` — Output
    * :class:`~shared.mlfp06.diagnostics.interpretability.InterpretabilityDiagnostics`
      — Attention
    * :class:`~shared.mlfp06.diagnostics.retrieval.RAGDiagnostics` — Retrieval
    * :class:`~shared.mlfp06.diagnostics.agent.AgentDiagnostics` — Agent trace
    * :class:`~shared.mlfp06.diagnostics.alignment.AlignmentDiagnostics`
      — Alignment
    * :class:`~shared.mlfp06.diagnostics.governance.GovernanceDiagnostics`
      — Governance

Typical use::

    from shared.mlfp06.diagnostics import LLMObservatory

    with LLMObservatory(delegate=my_delegate,
                        governance=supervisor.audit) as obs:
        obs.output.evaluate(prompts, responses)
        obs.retrieval.evaluate(queries, contexts, answers)
        obs.agent.capture_run(stream_events, run_id="demo")
        print(obs.report())
        obs.plot_dashboard().show()

All lenses degrade gracefully when their inputs are absent — the facade
never fabricates fake readings. When a method that requires a live
dependency (e.g. judge delegate, governance engine) is called without
one, the underlying lens raises a loud error per
``rules/observability.md`` §3.1 (no silent fake verdicts).
"""
from __future__ import annotations

import logging
from typing import Any

import plotly.graph_objects as go

from . import _plots
from .agent import AgentDiagnostics
from .alignment import AlignmentDiagnostics
from .governance import GovernanceDiagnostics
from .interpretability import InterpretabilityDiagnostics
from .output import LLMDiagnostics
from .retrieval import RAGDiagnostics

logger = logging.getLogger(__name__)

__all__ = ["LLMObservatory"]


# Severity thresholds — centralised here so every lens scores on the
# same axis when the facade composes a dashboard. Per
# ``rules/agent-reasoning.md``: these are NOT agent decisions. They are
# deterministic quality thresholds over already-computed metrics, which
# is explicit exception 4 (safety guards) in the permitted-conditional
# list.
_SEVERITY_HEALTHY = "HEALTHY"
_SEVERITY_WARNING = "WARNING"
_SEVERITY_CRITICAL = "CRITICAL"
_SEVERITY_UNKNOWN = "UNKNOWN"


class LLMObservatory:
    """Compose all six lenses — central entry point for M6 exercises.

    Args:
        delegate: Kaizen ``Delegate`` under observation. Re-used as the
            judge for the Output and Retrieval lenses. Optional — lenses
            that need a delegate raise loudly when called without one.
        judge_model: Explicit judge model. ``None`` resolves via
            ``OPENAI_JUDGE_MODEL`` → ``DEFAULT_LLM_MODEL`` →
            ``OPENAI_PROD_MODEL`` (see ``rules/env-models.md``).
        governance: ``GovernedSupervisor.audit`` or a PACT
            ``GovernanceEngine`` — whatever the Governance lens should
            observe.
        run_id: Correlation ID propagated to every lens for
            observability. Auto-generated if ``None``.
        max_judge_calls: Budget cap shared by Output and Retrieval
            judges. Defaults to 50 per ``_judges.JudgeCallable``.
        attention_model: Open-weight model name for the attention lens
            (defaults to the lens's own default). API-only models short-
            circuit with a ``not_applicable`` verdict.
    """

    def __init__(
        self,
        *,
        delegate: Any = None,
        judge_model: str | None = None,
        governance: Any = None,
        run_id: str | None = None,
        max_judge_calls: int = 50,
        attention_model: str | None = None,
    ) -> None:
        self._run_id = run_id

        self.output = LLMDiagnostics(
            delegate=delegate,
            judge_model=judge_model,
            max_judge_calls=max_judge_calls,
        )

        # InterpretabilityDiagnostics accepts model_name + run_id_prefix;
        # route run_id through the prefix for correlation.
        attention_kwargs: dict[str, Any] = {}
        if attention_model is not None:
            attention_kwargs["model_name"] = attention_model
        if run_id is not None:
            attention_kwargs["run_id_prefix"] = run_id
        self.attention = InterpretabilityDiagnostics(**attention_kwargs)

        self.retrieval = RAGDiagnostics(
            delegate=delegate,
            judge_model=judge_model,
            max_judge_calls=max_judge_calls,
        )

        self.agent = AgentDiagnostics()

        # AlignmentDiagnostics uses label= not run_id=.
        self.alignment = AlignmentDiagnostics(
            label=run_id if run_id is not None else "run"
        )

        self.governance = GovernanceDiagnostics(
            governance=governance,
            run_id=run_id,
        )

        self._delegate = delegate
        self._governance_engine = governance
        logger.info(
            "observatory.init",
            extra={
                "run_id": run_id,
                "has_delegate": delegate is not None,
                "has_governance": governance is not None,
                "judge_model": judge_model,
                "max_judge_calls": max_judge_calls,
            },
        )

    # ── Context manager ────────────────────────────────────────────────

    def __enter__(self) -> "LLMObservatory":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def close(self) -> None:
        """Release all lens handles — judge delegates, attention hooks, traces."""
        for lens_name in (
            "output",
            "attention",
            "retrieval",
            "agent",
            "alignment",
            "governance",
        ):
            lens = getattr(self, lens_name, None)
            if lens is not None and hasattr(lens, "close"):
                try:
                    lens.close()
                except Exception as exc:
                    # Cleanup path — zero-tolerance Rule 3 carve-out.
                    logger.warning(
                        "observatory.close.lens_error",
                        extra={"lens": lens_name, "error": str(exc)},
                    )

    # ── Composite report ──────────────────────────────────────────────

    def report(self, *, format: str = "rich") -> Any:
        """Composite Prescription Pad — one entry per lens.

        Args:
            format: ``"rich"`` (default) returns a Rich ``Table`` that
                renders as a colour-coded six-row table when printed in
                a notebook or terminal. ``"dict"`` returns the
                programmatic ``{lens: {summary, severity}}`` dict for
                callers that want to consume the data directly. ``"text"``
                returns a plain-text rendering for environments without
                Rich support.

        Each lens's ``report()`` returns a plain-text summary. The facade
        wraps each summary into a ``{"summary": str, "severity":
        HEALTHY/WARNING/CRITICAL/UNKNOWN}`` dict so downstream code (e.g.
        a dashboard widget) can colour-code per lens without parsing
        prose.

        The default switched from ``dict`` to ``rich`` in the M6 Ollama
        migration: students running ``obs.report()`` interactively in a
        notebook get a readable table instead of a wall of nested-dict
        text. Programmatic callers should pass ``format="dict"``
        explicitly.
        """
        result: dict[str, Any] = {}
        for lens_name in (
            "output",
            "attention",
            "retrieval",
            "agent",
            "alignment",
            "governance",
        ):
            lens = getattr(self, lens_name)
            try:
                summary = lens.report()
            except Exception as exc:
                logger.exception(
                    "observatory.report.lens_error",
                    extra={"lens": lens_name, "error": str(exc)},
                )
                result[lens_name] = {
                    "summary": f"{lens_name}-lens: error — {exc}",
                    "severity": _SEVERITY_UNKNOWN,
                }
                continue
            result[lens_name] = {
                "summary": summary,
                "severity": _derive_severity(lens_name, summary, lens),
            }
        logger.info(
            "observatory.report",
            extra={
                "run_id": self._run_id,
                "lenses": list(result.keys()),
                "source": "local_metric",
                "mode": "real",
                "format": format,
            },
        )

        if format == "dict":
            return result
        if format == "text":
            return _format_report_text(result)
        if format == "rich":
            return _format_report_rich(result)
        raise ValueError(
            f"unknown report format {format!r} — use 'rich', 'dict', or 'text'"
        )

    # ── Composite dashboard ───────────────────────────────────────────

    def plot_dashboard(self) -> go.Figure:
        """2x3 subplot — one panel per lens.

        Each lens provides its own plot method. The facade stitches the
        traces together into a single figure so M6 exercises can call
        ``obs.plot_dashboard().show()`` and see every lens at once.

        Lenses without data render the shared ``empty_figure`` placeholder.
        """
        from plotly.subplots import make_subplots

        fig = make_subplots(
            rows=2,
            cols=3,
            subplot_titles=(
                "Output (Stethoscope)",
                "Attention (X-Ray)",
                "Retrieval (Endoscope)",
                "Agent (Black Box)",
                "Alignment (ECG)",
                "Governance (Flight Recorder)",
            ),
            specs=[
                [{"type": "xy"}, {"type": "xy"}, {"type": "xy"}],
                [{"type": "xy"}, {"type": "xy"}, {"type": "xy"}],
            ],
        )

        # Attempt per-lens plotting; any failure yields a muted annotation
        # rather than crashing the whole dashboard.
        panel_specs = [
            ("output", 1, 1, self._output_panel),
            ("attention", 1, 2, self._attention_panel),
            ("retrieval", 1, 3, self._retrieval_panel),
            ("agent", 2, 1, self._agent_panel),
            ("alignment", 2, 2, self._alignment_panel),
            ("governance", 2, 3, self._governance_panel),
        ]
        for lens_name, row, col, builder in panel_specs:
            try:
                traces = builder()
            except Exception as exc:
                logger.warning(
                    "observatory.plot.panel_error",
                    extra={"lens": lens_name, "error": str(exc)},
                )
                traces = []
            for tr in traces:
                fig.add_trace(tr, row=row, col=col)

        fig.update_layout(
            title="LLM Observatory — six-lens dashboard",
            template=_plots.TEMPLATE,
            showlegend=False,
            height=800,
        )
        return fig

    # ── Per-lens panel builders (lightweight — just return traces) ────

    def _output_panel(self) -> list[Any]:
        import plotly.graph_objects as go  # local import — lazy

        df = self.output.results_df()
        if not df.height:
            return []
        return [
            go.Histogram(
                x=df["score"].to_list(),
                marker_color=_plots.PRIMARY,
                nbinsx=20,
                showlegend=False,
            )
        ]

    def _attention_panel(self) -> list[Any]:
        import plotly.graph_objects as go

        # Attention lens stores logits sweeps + probe readings; just count.
        n_attn = len(getattr(self.attention, "_attention_log", []))
        n_logit = len(getattr(self.attention, "_logit_log", []))
        n_probe = len(getattr(self.attention, "_probe_log", []))
        n_sae = len(getattr(self.attention, "_sae_log", []))
        return [
            go.Bar(
                x=["attention", "logit_lens", "probe", "sae"],
                y=[n_attn, n_logit, n_probe, n_sae],
                marker_color=_plots.ACCENT,
                showlegend=False,
            )
        ]

    def _retrieval_panel(self) -> list[Any]:
        import plotly.graph_objects as go

        df = self.retrieval.eval_df() if hasattr(self.retrieval, "eval_df") else None
        if df is None or not df.height:
            return []
        return [
            go.Scatter(
                x=list(range(df.height)),
                y=df["recall_at_k"].to_list(),
                mode="lines+markers",
                marker=dict(color=_plots.PRIMARY),
                showlegend=False,
            )
        ]

    def _agent_panel(self) -> list[Any]:
        import plotly.graph_objects as go

        traces = getattr(self.agent, "_traces", {})
        if not traces:
            return []
        rows = [(rid, trace.total_cost_usd()) for rid, trace in traces.items()]
        return [
            go.Bar(
                x=[r[0] for r in rows],
                y=[r[1] for r in rows],
                marker_color=_plots.PRIMARY,
                showlegend=False,
            )
        ]

    def _alignment_panel(self) -> list[Any]:
        import plotly.graph_objects as go

        df = (
            self.alignment.training_df()
            if hasattr(self.alignment, "training_df")
            else None
        )
        if df is None or not df.height:
            return []
        return [
            go.Scatter(
                x=df["step"].to_list(),
                y=df["reward"].to_list(),
                mode="lines+markers",
                line=dict(color=_plots.PRIMARY),
                showlegend=False,
            )
        ]

    def _governance_panel(self) -> list[Any]:
        import plotly.graph_objects as go

        if self._governance_engine is None:
            return []
        try:
            df = self.governance.budget_consumption()
        except Exception:
            return []
        if not df.height:
            return []
        return [
            go.Bar(
                x=df["dimension"].to_list(),
                y=df["consumed"].to_list(),
                marker_color=_plots.ACCENT,
                showlegend=False,
            )
        ]


# ════════════════════════════════════════════════════════════════════════
# Severity derivation — deterministic thresholds over metric summaries
# ════════════════════════════════════════════════════════════════════════


_SEVERITY_STYLE = {
    _SEVERITY_HEALTHY: "green",
    _SEVERITY_WARNING: "yellow",
    _SEVERITY_CRITICAL: "red bold",
    _SEVERITY_UNKNOWN: "dim",
}

_LENS_TITLE = {
    "output": "Output (Stethoscope)",
    "attention": "Attention (X-Ray)",
    "retrieval": "Retrieval (Endoscope)",
    "agent": "Agent (Black Box)",
    "alignment": "Alignment (ECG)",
    "governance": "Governance (Flight Recorder)",
}


def _format_report_text(result: dict[str, Any]) -> str:
    """Render the per-lens report dict as plain text (no Rich dependency)."""
    lines = ["LLM Observatory Prescription Pad", "=" * 40]
    for lens_name, payload in result.items():
        title = _LENS_TITLE.get(lens_name, lens_name)
        sev = payload.get("severity", _SEVERITY_UNKNOWN)
        summary = payload.get("summary", "")
        lines.append(f"  [{sev:<8}] {title}")
        for line in summary.splitlines() or [""]:
            lines.append(f"             {line}")
    return "\n".join(lines)


def _format_report_rich(result: dict[str, Any]) -> Any:
    """Render the per-lens report dict as a Rich ``Table``.

    Falls back to plain text if Rich is not importable so notebook
    environments without the optional dep still get a readable result.
    """
    try:
        from rich.table import Table
        from rich.text import Text
    except ImportError:
        return _format_report_text(result)

    table = Table(
        title="LLM Observatory — Prescription Pad",
        show_header=True,
        header_style="bold cyan",
        title_style="bold",
        expand=False,
    )
    table.add_column("Lens", style="bold", no_wrap=True)
    table.add_column("Severity", justify="center", no_wrap=True)
    table.add_column("Summary", overflow="fold")
    for lens_name, payload in result.items():
        title = _LENS_TITLE.get(lens_name, lens_name)
        sev = payload.get("severity", _SEVERITY_UNKNOWN)
        summary = payload.get("summary", "")
        sev_text = Text(sev, style=_SEVERITY_STYLE.get(sev, ""))
        table.add_row(title, sev_text, summary)
    return table


def _derive_severity(lens_name: str, summary: str, lens: Any) -> str:
    """Map a lens's report summary into one of four severity levels.

    This is deterministic plumbing (explicit exception 4 — safety guards
    — in ``rules/agent-reasoning.md``): the LLM already produced the
    metrics; this helper turns the tabular reading into a single label.
    """
    # "No readings yet" is the universal empty-state marker across lenses.
    # Cover both phrasings: "no readings recorded yet." (output/interp/
    # retrieval/alignment) and "no runs captured yet." (agent).
    if not summary:
        return _SEVERITY_UNKNOWN
    lowered = summary.lower()
    if (
        "no readings recorded yet" in lowered
        or "no runs captured yet" in lowered
        or "no engine configured" in lowered
    ):
        return _SEVERITY_UNKNOWN

    # Lens-specific escalation signals.
    lower = summary.lower()
    if lens_name == "output":
        if "faithfulness below" in lower or "scored below 0.5" in lower:
            return _SEVERITY_WARNING
        return _SEVERITY_HEALTHY
    if lens_name == "retrieval":
        if "recall" in lower and "below" in lower:
            return _SEVERITY_WARNING
        return _SEVERITY_HEALTHY
    if lens_name == "agent":
        if "stuck loop detected" in lower:
            return _SEVERITY_CRITICAL
        if "error event" in lower:
            return _SEVERITY_WARNING
        return _SEVERITY_HEALTHY
    if lens_name == "alignment":
        if "drifted far" in lower or "hacking scan" in lower:
            return _SEVERITY_CRITICAL
        if "too weak" in lower:
            return _SEVERITY_WARNING
        return _SEVERITY_HEALTHY
    if lens_name == "governance":
        if "chain broken" in lower:
            return _SEVERITY_CRITICAL
        if "approaching cap" in lower or "some drills passed" in lower:
            return _SEVERITY_WARNING
        if "no engine configured" in lower:
            return _SEVERITY_UNKNOWN
        return _SEVERITY_HEALTHY
    if lens_name == "attention":
        if "not applicable" in lower:
            return _SEVERITY_UNKNOWN
        return _SEVERITY_HEALTHY
    return _SEVERITY_HEALTHY
