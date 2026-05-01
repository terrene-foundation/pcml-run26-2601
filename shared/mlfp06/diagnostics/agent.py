# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""Lens 4 — Agent Trace Diagnostics (the MRI).

Question answered: *What did the agent do, and where did it spend its budget?*

Consumes the event stream from a Kaizen :class:`~kaizen_agents.Delegate`
(the TAOD loop — Thought/Action/Observation/Decision) and produces:

    * a TAOD-structured :class:`~shared.mlfp06.diagnostics._traces.AgentTrace`
    * tool-usage, cost, and latency breakdowns as Polars DataFrames
    * stuck-loop and oscillation detection
    * a timeline Plotly figure suitable for both Colab and VS Code
    * optional Langfuse export when ``LANGFUSE_HOST`` is set

Raw ``openai.*`` calls are BLOCKED (``rules/framework-first.md``). All
agent execution goes through ``Delegate.run()`` — we only observe its
event stream.

Quick start::

    from shared.mlfp06.diagnostics import AgentDiagnostics
    from shared.mlfp06._ollama_bootstrap import make_delegate

    agent_diag = AgentDiagnostics()
    delegate = make_delegate(tools=[...])  # Ollama-backed; no API keys

    trace = await agent_diag.capture_run(delegate, "Research and summarise ...")
    agent_diag.plot_trace(trace.run_id).show()
    print(agent_diag.report(trace.run_id))
"""
from __future__ import annotations

import inspect
import logging
import os
import time
import uuid
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import plotly.graph_objects as go
import polars as pl

from . import _plots
from ._traces import AgentTrace, TraceEvent, kaizen_events_to_trace

logger = logging.getLogger(__name__)

__all__ = ["AgentDiagnostics", "AgentTrace"]


# Minimum number of repeated (tool, args-hash) pairs in a window that we
# consider a "loop" for stuck-pattern detection.
_LOOP_MIN_REPEATS = 3


@dataclass
class _LoopFinding:
    run_id: str
    tool: str
    signature: str
    repeats: int
    first_idx: int
    last_idx: int


class AgentDiagnostics:
    """Agent-lens diagnostics — TAOD trace capture, timeline, stuck detection.

    Args:
        traces_dir: Directory where JSONL traces are persisted. When
            ``None`` traces live in-memory only (Colab default).
        langfuse_host: If set (or ``LANGFUSE_HOST`` env var), captured
            traces are exported after each run. When the Langfuse SDK is
            missing, export is skipped with a WARN log.
        langfuse_public_key / langfuse_secret_key: Auth for Langfuse; fall
            back to ``LANGFUSE_PUBLIC_KEY`` / ``LANGFUSE_SECRET_KEY``.
    """

    def __init__(
        self,
        *,
        traces_dir: Path | str | None = None,
        langfuse_host: str | None = None,
        langfuse_public_key: str | None = None,
        langfuse_secret_key: str | None = None,
    ) -> None:
        self._traces: dict[str, AgentTrace] = {}
        self._loops: list[_LoopFinding] = []
        self._traces_dir: Path | None = (
            Path(traces_dir) if traces_dir is not None else None
        )
        if self._traces_dir is not None:
            self._traces_dir.mkdir(parents=True, exist_ok=True)

        self._langfuse_host = langfuse_host or os.environ.get("LANGFUSE_HOST")
        self._langfuse_public = langfuse_public_key or os.environ.get(
            "LANGFUSE_PUBLIC_KEY"
        )
        self._langfuse_secret = langfuse_secret_key or os.environ.get(
            "LANGFUSE_SECRET_KEY"
        )
        self._langfuse_client: Any = None

        logger.info(
            "agent_diagnostics.init",
            extra={
                "traces_dir": str(self._traces_dir) if self._traces_dir else None,
                "langfuse_enabled": bool(self._langfuse_host),
            },
        )

    def __enter__(self) -> "AgentDiagnostics":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    def close(self) -> None:
        for t in self._traces.values():
            t.close()
        if self._langfuse_client is not None:
            try:
                # Langfuse uses ``flush`` to drain queued spans.
                self._langfuse_client.flush()
            except Exception:
                # Cleanup path — zero-tolerance Rule 3 carve-out.
                pass

    # ── Run capture ────────────────────────────────────────────────────

    async def capture_run(
        self,
        delegate: Any,
        prompt: str,
        *,
        run_id: str | None = None,
        tags: Sequence[str] = (),
    ) -> AgentTrace:
        """Execute ``delegate.run(prompt)`` and record its event stream.

        The delegate's ``run()`` is expected to be an async iterator of
        Kaizen ``StreamEvent`` objects (per ``kaizen_agents`` 0.9 API).
        Each event is routed through
        :func:`~shared.mlfp06.diagnostics._traces.kaizen_events_to_trace`
        to produce a TAOD :class:`AgentTrace`.

        Args:
            delegate: A constructed ``kaizen_agents.Delegate`` (or a mock
                exposing the same ``run(prompt)`` signature).
            prompt: The user prompt to feed the agent.
            run_id: Correlation ID; auto-generated when ``None``.
            tags: Optional tags forwarded to Langfuse on export.

        Returns:
            The :class:`AgentTrace` recorded for the run.
        """
        run_id = run_id or f"agent-{uuid.uuid4().hex[:12]}"
        t0 = time.monotonic()
        logger.info(
            "agent.capture_run.start",
            extra={"run_id": run_id, "prompt_preview": prompt[:80], "mode": "real"},
        )

        path: Path | None = None
        if self._traces_dir is not None:
            path = self._traces_dir / f"{run_id}.jsonl"

        events = await _collect_delegate_events(delegate, prompt)
        trace = kaizen_events_to_trace(events, run_id=run_id, path=path)
        self._traces[run_id] = trace

        # Post-capture analysis.
        for finding in _detect_tool_loops(trace):
            self._loops.append(finding)

        latency = (time.monotonic() - t0) * 1000.0
        logger.info(
            "agent.capture_run.ok",
            extra={
                "run_id": run_id,
                "n_events": len(trace),
                "cost_usd": trace.total_cost_usd(),
                "latency_ms": latency,
                "mode": "real",
            },
        )

        if self._langfuse_host:
            self._export_to_langfuse(trace, prompt=prompt, tags=tuple(tags))
        return trace

    def register_trace(self, trace: AgentTrace) -> AgentTrace:
        """Register an externally-produced trace for analysis.

        Useful when the trace was captured out-of-band (Langfuse SDK,
        OTEL exporter) and the user wants the lens's analysis methods to
        operate on it.
        """
        self._traces[trace.run_id] = trace
        for finding in _detect_tool_loops(trace):
            self._loops.append(finding)
        return trace

    # ── Retrieval / queries ────────────────────────────────────────────

    def get(self, run_id: str) -> AgentTrace:
        if run_id not in self._traces:
            raise KeyError(f"No trace registered for run_id={run_id!r}")
        return self._traces[run_id]

    def run_ids(self) -> list[str]:
        return list(self._traces.keys())

    def tool_usage(self, run_id: str) -> pl.DataFrame:
        """Tool-call counts with mean latency + total cost per tool."""
        trace = self.get(run_id)
        rows: dict[str, dict[str, float]] = {}
        for ev in trace.events:
            if ev.kind not in {"tool_start", "tool_end", "error"}:
                continue
            name = ev.tool or "<unknown>"
            bucket = rows.setdefault(
                name,
                {
                    "calls": 0.0,
                    "errors": 0.0,
                    "latency_ms_sum": 0.0,
                    "cost_usd_sum": 0.0,
                },
            )
            if ev.kind == "tool_start":
                bucket["calls"] += 1
            else:
                bucket["latency_ms_sum"] += ev.latency_ms
                bucket["cost_usd_sum"] += ev.cost_usd
                if ev.kind == "error" or ev.error:
                    bucket["errors"] += 1
        if not rows:
            return pl.DataFrame(
                schema={
                    "tool": pl.Utf8,
                    "calls": pl.Int64,
                    "errors": pl.Int64,
                    "mean_latency_ms": pl.Float64,
                    "total_cost_usd": pl.Float64,
                }
            )
        out = []
        for name, b in rows.items():
            calls = int(b["calls"])
            out.append(
                {
                    "tool": name,
                    "calls": calls,
                    "errors": int(b["errors"]),
                    "mean_latency_ms": (b["latency_ms_sum"] / calls if calls else 0.0),
                    "total_cost_usd": b["cost_usd_sum"],
                }
            )
        return pl.DataFrame(out).sort("calls", descending=True)

    def cost_breakdown(self, run_id: str) -> pl.DataFrame:
        """Per-event cost + cumulative cost for the run."""
        trace = self.get(run_id)
        running = 0.0
        rows = []
        for idx, ev in enumerate(trace.events):
            running += ev.cost_usd
            rows.append(
                {
                    "idx": idx,
                    "kind": ev.kind,
                    "tool": ev.tool,
                    "cost_usd": ev.cost_usd,
                    "cum_cost_usd": running,
                    "latency_ms": ev.latency_ms,
                }
            )
        if not rows:
            return pl.DataFrame(
                schema={
                    "idx": pl.Int64,
                    "kind": pl.Utf8,
                    "tool": pl.Utf8,
                    "cost_usd": pl.Float64,
                    "cum_cost_usd": pl.Float64,
                    "latency_ms": pl.Float64,
                }
            )
        return pl.DataFrame(rows)

    def detect_loops(self, run_id: str | None = None) -> pl.DataFrame:
        """Find repeated (tool, args) action sequences.

        A "loop" is a ``(tool, args-signature)`` pair that appears at
        least ``_LOOP_MIN_REPEATS`` times consecutively in the trace.
        """
        findings = (
            [f for f in self._loops if f.run_id == run_id]
            if run_id is not None
            else list(self._loops)
        )
        if not findings:
            return pl.DataFrame(
                schema={
                    "run_id": pl.Utf8,
                    "tool": pl.Utf8,
                    "signature": pl.Utf8,
                    "repeats": pl.Int64,
                    "first_idx": pl.Int64,
                    "last_idx": pl.Int64,
                }
            )
        return pl.DataFrame(
            [
                {
                    "run_id": f.run_id,
                    "tool": f.tool,
                    "signature": f.signature[:80],
                    "repeats": f.repeats,
                    "first_idx": f.first_idx,
                    "last_idx": f.last_idx,
                }
                for f in findings
            ]
        )

    # ── Plots ──────────────────────────────────────────────────────────

    def plot_trace(self, run_id: str) -> go.Figure:
        """2-row timeline: (tool calls on a time axis) + (cumulative cost)."""
        trace = self.get(run_id)
        if not len(trace):
            return _plots.empty_figure(f"Agent Trace {run_id}", note="no events")
        from plotly.subplots import make_subplots

        fig = make_subplots(
            rows=2,
            cols=1,
            shared_xaxes=True,
            subplot_titles=("Tool calls (timeline)", "Cumulative cost (USD)"),
            row_heights=[0.6, 0.4],
            vertical_spacing=0.08,
        )

        # Row 1 — tool call markers with latency as bar height.
        tool_rows = self.cost_breakdown(run_id).filter(
            pl.col("kind").is_in(["tool_start", "tool_end", "error"])
        )
        if tool_rows.height:
            unique_tools = sorted(
                {t for t in tool_rows["tool"].to_list() if t is not None}
            )
            for i, t in enumerate(unique_tools):
                sub = tool_rows.filter(pl.col("tool") == t)
                fig.add_trace(
                    go.Scatter(
                        x=sub["idx"].to_list(),
                        y=[i] * sub.height,
                        mode="markers+lines",
                        name=t,
                        marker=dict(
                            size=10, color=_plots.color_for(i), symbol="square"
                        ),
                        hovertext=sub["latency_ms"]
                        .map_elements(
                            lambda v: f"latency={v:.0f}ms", return_dtype=pl.Utf8
                        )
                        .to_list(),
                    ),
                    row=1,
                    col=1,
                )
            fig.update_yaxes(
                tickmode="array",
                tickvals=list(range(len(unique_tools))),
                ticktext=unique_tools,
                row=1,
                col=1,
            )

        # Row 2 — cumulative cost curve.
        cost_df = self.cost_breakdown(run_id)
        if cost_df.height:
            fig.add_trace(
                go.Scatter(
                    x=cost_df["idx"].to_list(),
                    y=cost_df["cum_cost_usd"].to_list(),
                    mode="lines",
                    line=dict(color=_plots.WARN, width=2),
                    name="cum cost",
                ),
                row=2,
                col=1,
            )

        # Loop markers (vertical bands) when stuck patterns are present.
        for finding in self._loops:
            if finding.run_id != run_id:
                continue
            fig.add_vrect(
                x0=finding.first_idx,
                x1=finding.last_idx,
                fillcolor=_plots.ACCENT,
                opacity=0.15,
                line_width=0,
                annotation_text=f"loop x{finding.repeats}",
                annotation_position="top left",
                row=1,
                col=1,
            )

        fig.update_layout(
            title=f"Agent MRI — {run_id}",
            template=_plots.TEMPLATE,
            height=620,
            showlegend=True,
        )
        return fig

    def plot_cost_across_runs(self) -> go.Figure:
        """Bar chart of total cost per captured run."""
        if not self._traces:
            return _plots.empty_figure("Agent cost across runs")
        rows = [
            {
                "run_id": rid,
                "cost_usd": trace.total_cost_usd(),
                "latency_ms": trace.total_latency_ms(),
                "events": len(trace),
            }
            for rid, trace in self._traces.items()
        ]
        df = pl.DataFrame(rows).sort("cost_usd", descending=True)
        fig = go.Figure(
            go.Bar(
                x=df["run_id"].to_list(),
                y=df["cost_usd"].to_list(),
                marker_color=_plots.PRIMARY,
            )
        )
        _plots.style(fig, "Agent cost per run", x="run_id", y="cost (USD)")
        return fig

    # ── Report ─────────────────────────────────────────────────────────

    def report(self, run_id: str | None = None) -> str:
        """Plain-text Prescription Pad — one run or all runs."""
        target_ids = [run_id] if run_id else list(self._traces.keys())
        if not target_ids:
            return "agent-lens: no runs captured yet."

        out: list[str] = []
        for rid in target_ids:
            if rid not in self._traces:
                continue
            trace = self._traces[rid]
            tool_df = self.tool_usage(rid)
            cost = trace.total_cost_usd()
            latency = trace.total_latency_ms()
            out.append(
                f"run={rid}: {len(trace)} events, cost=${cost:.4f}, "
                f"latency={latency:.0f}ms, tools_used={tool_df.height}"
            )
            if tool_df.height:
                top = tool_df.row(0, named=True)
                out.append(
                    f"  top_tool={top['tool']} calls={top['calls']} "
                    f"mean_latency={top['mean_latency_ms']:.0f}ms "
                    f"errors={top['errors']}"
                )
            errors = sum(1 for ev in trace.events if ev.kind == "error" or ev.error)
            if errors:
                out.append(f"  -> {errors} error event(s) — inspect trace")
            loops_here = [f for f in self._loops if f.run_id == rid]
            if loops_here:
                out.append(
                    f"  -> stuck loop detected: {loops_here[0].tool} x"
                    f"{loops_here[0].repeats} "
                    f"(idx {loops_here[0].first_idx}..{loops_here[0].last_idx})"
                )
        return "agent-lens:\n  " + "\n  ".join(out)

    # ── Langfuse export ────────────────────────────────────────────────

    def _export_to_langfuse(
        self, trace: AgentTrace, *, prompt: str, tags: tuple[str, ...]
    ) -> None:
        if not self._langfuse_host:
            return
        client = self._ensure_langfuse()
        if client is None:
            return
        try:
            lf_trace = client.trace(
                name="mlfp06.agent.run",
                id=trace.run_id,
                input=prompt,
                metadata={"n_events": len(trace)},
                tags=list(tags),
            )
            for ev in trace.events:
                if ev.kind in {"tool_start", "tool_end", "error"} and ev.tool:
                    lf_trace.span(
                        name=f"tool:{ev.tool}",
                        input=ev.args or {},
                        output=ev.result,
                        metadata={
                            "latency_ms": ev.latency_ms,
                            "cost_usd": ev.cost_usd,
                            "error": ev.error,
                        },
                    )
                elif ev.kind == "complete":
                    lf_trace.generation(
                        name="agent.complete",
                        output=ev.content,
                        usage={
                            "input": ev.tokens_in or 0,
                            "output": ev.tokens_out or 0,
                        },
                    )
            logger.info(
                "agent.langfuse.export.ok",
                extra={"run_id": trace.run_id, "host": self._langfuse_host},
            )
        except Exception as exc:  # pragma: no cover — network-dependent
            logger.warning(
                "agent.langfuse.export.error",
                extra={"run_id": trace.run_id, "error": str(exc)},
            )

    def _ensure_langfuse(self) -> Any:
        if self._langfuse_client is not None:
            return self._langfuse_client
        try:
            # langfuse 2.x exposes the client as ``Langfuse``.
            from langfuse import Langfuse  # type: ignore[import-not-found]
        except ImportError:
            logger.warning(
                "agent.langfuse.unavailable",
                extra={
                    "reason": "langfuse package not installed",
                    "host": self._langfuse_host,
                },
            )
            return None
        try:
            self._langfuse_client = Langfuse(
                host=self._langfuse_host,
                public_key=self._langfuse_public,
                secret_key=self._langfuse_secret,
            )
        except Exception as exc:  # pragma: no cover — credential-dependent
            logger.warning(
                "agent.langfuse.construct_error",
                extra={"error": str(exc), "host": self._langfuse_host},
            )
            return None
        return self._langfuse_client


# ════════════════════════════════════════════════════════════════════════
# Helpers
# ════════════════════════════════════════════════════════════════════════


async def _collect_delegate_events(delegate: Any, prompt: str) -> list[Any]:
    """Drain ``delegate.run(prompt)`` into a list of StreamEvents.

    The Kaizen 0.9 API returns an async iterator. We support three shapes:

        1. ``async for event in delegate.run(prompt): ...`` (primary)
        2. ``for event in delegate.run(prompt): ...`` (sync iterator — used
           by tests that want zero async boilerplate)
        3. a pre-collected list (returned directly by mocks)
    """
    call = delegate.run(prompt)
    # Case 3 — already a list / tuple.
    if isinstance(call, (list, tuple)):
        return list(call)
    # Case 1 — async iterator.
    if hasattr(call, "__aiter__"):
        events: list[Any] = []
        async for event in call:
            events.append(event)
        return events
    # Coroutine returning iterable.
    if inspect.iscoroutine(call):
        result = await call
        if isinstance(result, (list, tuple)):
            return list(result)
        if hasattr(result, "__aiter__"):
            events = []
            async for event in result:
                events.append(event)
            return events
        if hasattr(result, "__iter__"):
            return list(result)
        return [result]
    # Case 2 — sync iterator.
    if hasattr(call, "__iter__"):
        return list(call)
    return [call]


def _arg_signature(args: Any) -> str:
    """Stable short string for ``(tool, args)`` equality checks."""
    if args is None:
        return "()"
    try:
        import json

        return json.dumps(args, sort_keys=True, default=str)[:200]
    except Exception:
        return repr(args)[:200]


def _detect_tool_loops(trace: AgentTrace) -> list[_LoopFinding]:
    """Identify consecutive runs of the same (tool, args) signature."""
    findings: list[_LoopFinding] = []
    run_id = trace.run_id
    prev_sig: str | None = None
    prev_tool: str | None = None
    count = 0
    first_idx = 0
    last_idx = 0

    for idx, ev in enumerate(trace.events):
        if ev.kind != "tool_start":
            continue
        sig = f"{ev.tool}|{_arg_signature(ev.args)}"
        if sig == prev_sig:
            count += 1
            last_idx = idx
        else:
            if prev_sig is not None and count >= _LOOP_MIN_REPEATS:
                findings.append(
                    _LoopFinding(
                        run_id=run_id,
                        tool=prev_tool or "<unknown>",
                        signature=prev_sig,
                        repeats=count,
                        first_idx=first_idx,
                        last_idx=last_idx,
                    )
                )
            prev_sig = sig
            prev_tool = ev.tool
            count = 1
            first_idx = idx
            last_idx = idx
    if prev_sig is not None and count >= _LOOP_MIN_REPEATS:
        findings.append(
            _LoopFinding(
                run_id=run_id,
                tool=prev_tool or "<unknown>",
                signature=prev_sig,
                repeats=count,
                first_idx=first_idx,
                last_idx=last_idx,
            )
        )
    return findings
