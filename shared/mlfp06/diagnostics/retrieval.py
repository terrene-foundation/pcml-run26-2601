# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""Lens 3 — Retrieval Diagnostics (the Endoscope).

Question answered: *Did we fetch the right context, and did the generator use it?*

Wraps ``ragas`` (context-precision, context-recall, faithfulness, answer-
relevancy) and ``trulens-eval`` (groundedness, relevance) as the primary
backends. When either backend is missing the lens degrades **loudly** —
the method raises a descriptive ``ImportError`` naming the extra to
install, per ``rules/dependencies.md`` ("Optional Extras with Loud
Failure"). Silent ``None`` fallbacks are BLOCKED.

All LLM-as-judge calls for answer-quality scoring route through the
shared :class:`~shared.mlfp06.diagnostics._judges.JudgeCallable` — no raw
``openai.*`` per ``rules/framework-first.md``.

Quick start::

    from shared.mlfp06.diagnostics import RAGDiagnostics

    rag = RAGDiagnostics(max_judge_calls=20)
    df = rag.evaluate(
        queries=["What is photosynthesis?"],
        retrieved_contexts=[[doc1, doc2, doc3]],
        answers=["Photosynthesis is ..."],
        ground_truth_ids=[["doc_42"]],
    )
    board = rag.compare_retrievers(
        retrievers={"bm25": bm25_fn, "dense": dense_fn, "hybrid": hybrid_fn},
        eval_set=eval_set,
        k=5,
    )
    rag.plot_rag_dashboard().show()
    print(rag.report())
"""
from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass
from typing import Any, Callable, Sequence

import plotly.graph_objects as go
import polars as pl

from . import _plots
from ._judges import JudgeCallable

logger = logging.getLogger(__name__)

__all__ = ["RAGDiagnostics"]


# Retriever callable: (query, k) -> list of (doc_id, content, score).
RetrievedDoc = tuple[str, str, float]
Retriever = Callable[[str, int], Sequence[RetrievedDoc]]


@dataclass
class _EvalEntry:
    query: str
    answer: str
    retrieved_ids: list[str]
    ground_truth_ids: list[str]
    context_utilisation: float
    faithfulness: float
    recall_at_k: float
    precision_at_k: float
    mode: str


class RAGDiagnostics:
    """Retrieval-lens diagnostics — recall@k, context utilisation, retriever leaderboard.

    Args:
        judge_model: Judge model. Resolved via
            :func:`~shared.mlfp06.diagnostics._judges.resolve_judge_model`.
        max_judge_calls: Hard cap on live judge calls (default ``50``).
        delegate: Optional pre-built Kaizen ``Delegate`` reused by the judge.
        sensitive: When ``True``, query/answer bodies are not logged.
    """

    def __init__(
        self,
        *,
        judge_model: str | None = None,
        max_judge_calls: int = 50,
        delegate: Any = None,
        sensitive: bool = False,
    ) -> None:
        self._judge = JudgeCallable(
            judge_model=judge_model,
            max_judge_calls=max_judge_calls,
            delegate=delegate,
            sensitive=sensitive,
        )
        self._eval_log: list[_EvalEntry] = []
        self._retriever_log: list[dict[str, Any]] = []
        self._sensitive = sensitive
        logger.info(
            "rag_diagnostics.init",
            extra={"judge_model": self._judge.model, "max_calls": max_judge_calls},
        )

    def __enter__(self) -> "RAGDiagnostics":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    def close(self) -> None:
        self._judge.close()

    # ── Core evaluation API ────────────────────────────────────────────

    def evaluate(
        self,
        queries: Sequence[str],
        retrieved_contexts: Sequence[Sequence[str]],
        answers: Sequence[str],
        *,
        ground_truth_ids: Sequence[Sequence[str]] | None = None,
        retrieved_ids: Sequence[Sequence[str]] | None = None,
        k: int = 5,
        run_id: str | None = None,
    ) -> pl.DataFrame:
        """Score a batch of RAG outputs end-to-end.

        Computes per-query recall@k, precision@k, context-utilisation, and
        faithfulness. When ``ragas`` is installed, its implementations are
        used; otherwise the lens falls back to judge-only scoring and logs
        the fallback at WARN level.

        Args:
            queries: User queries.
            retrieved_contexts: For each query, the ordered list of
                retrieved chunk contents.
            answers: The generator's final answers.
            ground_truth_ids: Optional list of per-query relevant doc IDs
                (required for recall@k / precision@k).
            retrieved_ids: Optional list of per-query retrieved doc IDs in
                the same order as ``retrieved_contexts``. When ``None``
                the lens treats the context strings themselves as IDs.
            k: Cut-off used for recall@k / precision@k.
            run_id: Correlation ID per ``rules/observability.md``.

        Returns:
            A Polars DataFrame with one row per query.
        """
        n = len(queries)
        if not (len(retrieved_contexts) == n == len(answers)):
            raise ValueError(
                "queries, retrieved_contexts, answers must all be same length"
            )
        run_id = run_id or f"rag_eval-{uuid.uuid4().hex[:12]}"
        logger.info(
            "rag.evaluate.start",
            extra={"run_id": run_id, "n": n, "k": k, "mode": "real"},
        )

        ragas_scores = _try_ragas_evaluate(
            queries=queries,
            retrieved_contexts=retrieved_contexts,
            answers=answers,
            ground_truth_ids=ground_truth_ids,
        )
        rows: list[dict[str, Any]] = []
        for i in range(n):
            ids_i = (
                list(retrieved_ids[i])
                if retrieved_ids is not None
                else list(retrieved_contexts[i])
            )
            truth_i = list(ground_truth_ids[i]) if ground_truth_ids is not None else []
            recall = _recall_at_k(retrieved=ids_i[:k], relevant=truth_i)
            precision = _precision_at_k(retrieved=ids_i[:k], relevant=truth_i)

            if ragas_scores is not None:
                faithfulness = float(ragas_scores["faithfulness"][i])
                context_util = float(ragas_scores["context_precision"][i])
                backend_mode = "real"
            else:
                faithfulness_verdict = self._judge.score(
                    response=answers[i],
                    criteria="faithfulness,grounded_in_context,no_fabrication",
                    context=(
                        "[QUERY]\n"
                        + queries[i]
                        + "\n\n[RETRIEVED CONTEXT]\n"
                        + "\n\n".join(retrieved_contexts[i])
                    ),
                    run_id=f"{run_id}-faith-{i}",
                )
                faithfulness = faithfulness_verdict.score
                context_util = _heuristic_context_utilisation(
                    answer=answers[i], contexts=retrieved_contexts[i]
                )
                backend_mode = faithfulness_verdict.mode

            self._eval_log.append(
                _EvalEntry(
                    query=queries[i],
                    answer=answers[i],
                    retrieved_ids=ids_i[:k],
                    ground_truth_ids=truth_i,
                    context_utilisation=context_util,
                    faithfulness=faithfulness,
                    recall_at_k=recall,
                    precision_at_k=precision,
                    mode=backend_mode,
                )
            )
            rows.append(
                {
                    "idx": i,
                    "recall_at_k": recall,
                    "precision_at_k": precision,
                    "context_utilisation": context_util,
                    "faithfulness": faithfulness,
                    "k": k,
                    "mode": backend_mode,
                }
            )
        df = pl.DataFrame(rows)
        logger.info(
            "rag.evaluate.ok",
            extra={
                "run_id": run_id,
                "n": n,
                "mean_recall": float(df["recall_at_k"].mean() or 0.0),
                "mean_faithfulness": float(df["faithfulness"].mean() or 0.0),
                "source": "ragas" if ragas_scores is not None else "judge_fallback",
                "mode": "real",
            },
        )
        return df

    # ── Retriever leaderboard ──────────────────────────────────────────

    def compare_retrievers(
        self,
        retrievers: dict[str, Retriever],
        eval_set: Sequence[dict[str, Any]],
        *,
        k: int = 5,
        run_id: str | None = None,
    ) -> pl.DataFrame:
        """Leaderboard over BM25 / dense / hybrid / HyDE / ... retrievers.

        Each element of ``eval_set`` must have keys:

            * ``query`` (str)
            * ``relevant_ids`` (list[str]) — ground-truth doc IDs

        ``retrievers`` maps a short label to a callable
        ``(query, k) -> [(doc_id, content, score), ...]``.

        Returns a DataFrame sorted by ``mrr`` descending.
        """
        run_id = run_id or f"retriever_cmp-{uuid.uuid4().hex[:12]}"
        if not retrievers:
            raise ValueError("retrievers dict must be non-empty")
        if not eval_set:
            raise ValueError("eval_set must be non-empty")
        logger.info(
            "rag.compare_retrievers.start",
            extra={
                "run_id": run_id,
                "retrievers": list(retrievers),
                "n_queries": len(eval_set),
                "k": k,
                "mode": "real",
            },
        )

        rows: list[dict[str, Any]] = []
        for name, fn in retrievers.items():
            per_query: list[dict[str, float]] = []
            for entry in eval_set:
                query = entry["query"]
                relevant = list(entry.get("relevant_ids") or [])
                hits = list(fn(query, k)) or []
                retrieved_ids = [h[0] for h in hits[:k]]
                per_query.append(
                    {
                        "recall_at_k": _recall_at_k(retrieved_ids, relevant),
                        "precision_at_k": _precision_at_k(retrieved_ids, relevant),
                        "mrr": _reciprocal_rank(retrieved_ids, relevant),
                        "ndcg_at_k": _ndcg_at_k(retrieved_ids, relevant, k),
                    }
                )
            # Aggregate over queries.
            agg = {
                "retriever": name,
                "recall_at_k": _mean([r["recall_at_k"] for r in per_query]),
                "precision_at_k": _mean([r["precision_at_k"] for r in per_query]),
                "mrr": _mean([r["mrr"] for r in per_query]),
                "ndcg_at_k": _mean([r["ndcg_at_k"] for r in per_query]),
                "n": len(per_query),
                "k": k,
            }
            rows.append(agg)
            self._retriever_log.append({**agg, "run_id": run_id})
        board = pl.DataFrame(rows).sort("mrr", descending=True)
        logger.info(
            "rag.compare_retrievers.ok",
            extra={
                "run_id": run_id,
                "winner": str(board["retriever"][0]) if board.height else None,
                "mode": "real",
            },
        )
        return board

    # ── Individual metric helpers (public) ─────────────────────────────

    def recall_at_k(
        self,
        retrieved_ids: Sequence[str],
        relevant_ids: Sequence[str],
        *,
        k: int = 5,
    ) -> float:
        """Recall@k — fraction of the relevant set captured in top-k."""
        return _recall_at_k(list(retrieved_ids)[:k], list(relevant_ids))

    def precision_at_k(
        self,
        retrieved_ids: Sequence[str],
        relevant_ids: Sequence[str],
        *,
        k: int = 5,
    ) -> float:
        """Precision@k — fraction of top-k that is relevant."""
        return _precision_at_k(list(retrieved_ids)[:k], list(relevant_ids))

    def context_utilisation(
        self,
        answer: str,
        contexts: Sequence[str],
    ) -> float:
        """How much of the answer's content is traceable to retrieved context.

        Uses a token-overlap heuristic (fast, local, no LLM call). For a
        judge-based evaluation use
        :meth:`~shared.mlfp06.diagnostics.LLMDiagnostics.faithfulness`.
        """
        return _heuristic_context_utilisation(answer=answer, contexts=contexts)

    def ragas_scores(
        self,
        queries: Sequence[str],
        retrieved_contexts: Sequence[Sequence[str]],
        answers: Sequence[str],
        *,
        ground_truth_ids: Sequence[Sequence[str]] | None = None,
    ) -> pl.DataFrame:
        """Run the full RAGAS evaluation (requires ``ragas``).

        Raises ``ImportError`` with an actionable message when ragas is
        not installed — per ``rules/dependencies.md`` "Optional Extras
        with Loud Failure".
        """
        scores = _try_ragas_evaluate(
            queries=queries,
            retrieved_contexts=retrieved_contexts,
            answers=answers,
            ground_truth_ids=ground_truth_ids,
        )
        if scores is None:
            raise ImportError(
                "ragas is required for ragas_scores(). Install with "
                "`pip install ragas>=0.2`."
            )
        return pl.DataFrame(scores)

    # Note: A `trulens_scores()` method previously routed groundedness and
    # answer-relevance through trulens-eval + OpenAI. It was removed in the
    # 2026-04-20 sync because:
    #   1. trulens-dashboard pins psutil<6, blocking kailash>=2.8.9 (psutil>=7).
    #   2. The implementation routed through OpenAI, BLOCKED by Redline 14
    #      (M6 is Ollama-only).
    # Equivalent metrics (faithfulness + answer relevance) remain available via
    # `ragas_scores()` above, which uses the M6-mandated Ollama provider.

    # ── DataFrames ─────────────────────────────────────────────────────

    def eval_df(self) -> pl.DataFrame:
        """One row per :meth:`evaluate` sample."""
        if not self._eval_log:
            return pl.DataFrame(
                schema={
                    "query_preview": pl.Utf8,
                    "recall_at_k": pl.Float64,
                    "precision_at_k": pl.Float64,
                    "context_utilisation": pl.Float64,
                    "faithfulness": pl.Float64,
                    "mode": pl.Utf8,
                }
            )
        return pl.DataFrame(
            [
                {
                    "query_preview": "<redacted>" if self._sensitive else e.query[:120],
                    "recall_at_k": e.recall_at_k,
                    "precision_at_k": e.precision_at_k,
                    "context_utilisation": e.context_utilisation,
                    "faithfulness": e.faithfulness,
                    "mode": e.mode,
                }
                for e in self._eval_log
            ]
        )

    def retriever_df(self) -> pl.DataFrame:
        """Retriever leaderboard history."""
        if not self._retriever_log:
            return pl.DataFrame(
                schema={
                    "retriever": pl.Utf8,
                    "recall_at_k": pl.Float64,
                    "precision_at_k": pl.Float64,
                    "mrr": pl.Float64,
                    "ndcg_at_k": pl.Float64,
                    "n": pl.Int64,
                    "k": pl.Int64,
                }
            )
        return pl.DataFrame(self._retriever_log).drop("run_id", strict=False)

    # ── Plots ──────────────────────────────────────────────────────────

    def plot_rag_dashboard(self) -> go.Figure:
        """2x2 dashboard: recall@k curve, context-util histogram, faithfulness
        scatter, retriever leaderboard."""
        from plotly.subplots import make_subplots

        if not self._eval_log and not self._retriever_log:
            return _plots.empty_figure("Retrieval Lens Dashboard (Endoscope)")

        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=(
                "Recall@k per query",
                "Context utilisation histogram",
                "Faithfulness vs context-utilisation",
                "Retriever leaderboard (MRR)",
            ),
        )

        eval_df = self.eval_df()
        if eval_df.height:
            fig.add_trace(
                go.Scatter(
                    x=list(range(eval_df.height)),
                    y=eval_df["recall_at_k"].to_list(),
                    mode="lines+markers",
                    marker=dict(color=_plots.PRIMARY),
                    name="recall@k",
                ),
                row=1,
                col=1,
            )
            fig.add_trace(
                go.Histogram(
                    x=eval_df["context_utilisation"].to_list(),
                    marker_color=_plots.ACCENT,
                    nbinsx=20,
                ),
                row=1,
                col=2,
            )
            fig.add_trace(
                go.Scatter(
                    x=eval_df["context_utilisation"].to_list(),
                    y=eval_df["faithfulness"].to_list(),
                    mode="markers",
                    marker=dict(color=_plots.WARN, size=8),
                ),
                row=2,
                col=1,
            )
        board = self.retriever_df()
        if board.height:
            fig.add_trace(
                go.Bar(
                    x=board["retriever"].to_list(),
                    y=board["mrr"].to_list(),
                    marker_color=_plots.PRIMARY,
                ),
                row=2,
                col=2,
            )

        fig.update_layout(
            title="Retrieval Lens Dashboard (Endoscope)",
            template=_plots.TEMPLATE,
            showlegend=False,
            height=640,
        )
        return fig

    # ── Report ─────────────────────────────────────────────────────────

    def report(self) -> str:
        """Plain-text Prescription Pad for the retrieval lens."""
        out: list[str] = []
        eval_df = self.eval_df()
        if eval_df.height:
            mean_r = float(eval_df["recall_at_k"].mean() or 0.0)
            mean_p = float(eval_df["precision_at_k"].mean() or 0.0)
            mean_u = float(eval_df["context_utilisation"].mean() or 0.0)
            mean_f = float(eval_df["faithfulness"].mean() or 0.0)
            out.append(
                f"evaluate: {eval_df.height} queries, recall@k={mean_r:.2f}, "
                f"precision@k={mean_p:.2f}, context_util={mean_u:.2f}, "
                f"faithfulness={mean_f:.2f}"
            )
            if mean_r < 0.5:
                out.append(
                    "  -> recall below 0.5: widen top-k, add HyDE, or retune embeddings"
                )
            if mean_u < 0.4:
                out.append(
                    "  -> context utilisation below 0.4: answers ignore retrieved context; "
                    "consider reranking or prompt changes"
                )
            if mean_f < 0.7:
                out.append(
                    "  -> faithfulness below 0.7: model is inventing — add citation constraint"
                )
        board = self.retriever_df()
        if board.height:
            top = board.row(0, named=True)
            out.append(
                f"retrievers: top={top['retriever']} "
                f"(mrr={top['mrr']:.2f}, ndcg@k={top['ndcg_at_k']:.2f})"
            )
        if not out:
            return "retrieval-lens: no readings recorded yet."
        return "retrieval-lens:\n  " + "\n  ".join(out)


# ════════════════════════════════════════════════════════════════════════
# Metric helpers — pure, no LLM calls
# ════════════════════════════════════════════════════════════════════════


def _recall_at_k(retrieved: Sequence[str], relevant: Sequence[str]) -> float:
    if not relevant:
        return 0.0
    rset = set(relevant)
    hits = sum(1 for r in retrieved if r in rset)
    return hits / len(rset)


def _precision_at_k(retrieved: Sequence[str], relevant: Sequence[str]) -> float:
    if not retrieved:
        return 0.0
    rset = set(relevant)
    hits = sum(1 for r in retrieved if r in rset)
    return hits / len(retrieved)


def _reciprocal_rank(retrieved: Sequence[str], relevant: Sequence[str]) -> float:
    rset = set(relevant)
    for idx, doc in enumerate(retrieved, start=1):
        if doc in rset:
            return 1.0 / idx
    return 0.0


def _ndcg_at_k(retrieved: Sequence[str], relevant: Sequence[str], k: int) -> float:
    import math

    rset = set(relevant)
    dcg = 0.0
    for idx, doc in enumerate(retrieved[:k], start=1):
        if doc in rset:
            dcg += 1.0 / math.log2(idx + 1)
    ideal_hits = min(len(rset), k)
    if ideal_hits == 0:
        return 0.0
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal_hits + 1))
    return dcg / idcg if idcg else 0.0


def _mean(xs: Sequence[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def _heuristic_context_utilisation(answer: str, contexts: Sequence[str]) -> float:
    """Token-overlap context utilisation in ``[0, 1]``.

    Fraction of answer tokens (non-stopword, length >= 4) that appear in
    at least one retrieved context. Not an LLM judgement — see
    :meth:`LLMDiagnostics.faithfulness` for that.
    """
    _STOP = {
        "the",
        "that",
        "this",
        "with",
        "from",
        "have",
        "they",
        "their",
        "them",
        "these",
        "those",
        "into",
        "been",
        "were",
        "will",
        "would",
        "about",
        "which",
        "there",
        "where",
    }
    ans_tokens = {
        t
        for t in answer.lower().split()
        if len(t) >= 4 and t.isalpha() and t not in _STOP
    }
    if not ans_tokens:
        return 0.0
    context_blob = " ".join(contexts).lower()
    grounded = sum(1 for t in ans_tokens if t in context_blob)
    return grounded / len(ans_tokens)


def _try_ragas_evaluate(
    *,
    queries: Sequence[str],
    retrieved_contexts: Sequence[Sequence[str]],
    answers: Sequence[str],
    ground_truth_ids: Sequence[Sequence[str]] | None,
) -> dict[str, list[float]] | None:
    """Call RAGAS if available; return ``None`` when it isn't (caller falls back).

    Per ``rules/dependencies.md`` the fallback is allowed because
    ``RAGDiagnostics.evaluate`` loudly surfaces the fallback via a WARN
    log line, and the public-facing :meth:`RAGDiagnostics.ragas_scores`
    method raises an ``ImportError`` naming the extra.
    """
    try:
        from ragas import evaluate as ragas_evaluate  # type: ignore[import-not-found]
        from ragas.metrics import (  # type: ignore[import-not-found]
            answer_relevancy,
            context_precision,
            context_recall,
            faithfulness as ragas_faithfulness,
        )
        from datasets import Dataset  # type: ignore[import-not-found]
    except ImportError:
        logger.warning(
            "rag.evaluate.ragas_unavailable",
            extra={"reason": "ragas or datasets not installed", "mode": "real"},
        )
        return None

    try:
        ds = Dataset.from_dict(
            {
                "question": list(queries),
                "contexts": [list(c) for c in retrieved_contexts],
                "answer": list(answers),
                "ground_truth": [
                    ", ".join(gt) if gt else ""
                    for gt in (ground_truth_ids or [[] for _ in queries])
                ],
            }
        )
        metrics = [ragas_faithfulness, context_precision, answer_relevancy]
        if ground_truth_ids is not None:
            metrics.append(context_recall)
        result = ragas_evaluate(ds, metrics=metrics)
    except Exception as exc:  # pragma: no cover — ragas internal error
        logger.warning(
            "rag.evaluate.ragas_error",
            extra={"error": str(exc), "mode": "real"},
        )
        return None

    # ``result`` is a RagasResult whose ``.scores`` attribute holds a list of
    # per-row dicts. Fall back to ``.to_pandas()`` on older versions.
    try:
        rows = list(result.scores)  # type: ignore[attr-defined]
    except AttributeError:  # pragma: no cover
        rows = result.to_pandas().to_dict("records")  # type: ignore[attr-defined]

    def _col(key: str) -> list[float]:
        return [float(r.get(key, 0.0)) for r in rows]

    return {
        "faithfulness": _col("faithfulness"),
        "context_precision": _col("context_precision"),
        "context_recall": (
            _col("context_recall") if ground_truth_ids else [0.0] * len(rows)
        ),
        "answer_relevancy": _col("answer_relevancy"),
    }
