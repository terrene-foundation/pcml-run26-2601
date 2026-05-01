# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
Shared infrastructure for MLFP06 Exercise 4 — RAG Systems.

Contains: corpus loading, LLM/Delegate setup, embedding helpers, evaluation
utilities, and plot output directory. Technique-specific logic (chunking
algorithms, BM25 internals, RRF, reranking prompts, RAGAS judge prompts,
HyDE generation) lives in the per-technique files under
``modules/mlfp06/solutions/ex_4/``.
"""
from __future__ import annotations

import asyncio
import math
from pathlib import Path

import matplotlib.pyplot as plt
import polars as pl

from shared.kailash_helpers import setup_environment
from shared.mlfp06._ollama_bootstrap import (
    DEFAULT_CHAT_MODEL,
    DEFAULT_EMBED_MODEL,
    make_delegate as _make_chat_delegate,
    make_embedder,
    run_delegate_text,
)

# ════════════════════════════════════════════════════════════════════════
# ENVIRONMENT SETUP
# ════════════════════════════════════════════════════════════════════════

setup_environment()

MODEL = DEFAULT_CHAT_MODEL
EMBED_MODEL = DEFAULT_EMBED_MODEL

# Real embedding dimensionality — nomic-embed-text returns 768-dim vectors.
# The previous pedagogical 8-dim "LLM-as-projector" trick was a workaround
# for the OpenAI-cost-per-call problem; with a free local embedder we use
# the real thing so retrieval behaves like a production RAG system.
EMBED_DIM = 768

OUTPUT_DIR = Path("outputs") / "ex4_rag"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CACHE_DIR = Path("data/mlfp06/rag")
CACHE_DIR.mkdir(parents=True, exist_ok=True)
CACHE_FILE = CACHE_DIR / "rag_corpus_1k.parquet"


# ════════════════════════════════════════════════════════════════════════
# CORPUS LOADING
# ════════════════════════════════════════════════════════════════════════


def load_rag_corpus(sample_size: int = 1000) -> pl.DataFrame:
    """Load the neural-bridge/rag-dataset-12000 corpus (cached locally).

    Returns a polars DataFrame with columns: section, text, question, answer.
    The corpus is both the retrieval corpus (via ``text``) and the evaluation
    set (via ``question`` / ``answer``).
    """
    if CACHE_FILE.exists():
        print(f"Loading cached RAG corpus from {CACHE_FILE}")
        return pl.read_parquet(CACHE_FILE)

    print("Downloading neural-bridge/rag-dataset-12000 from HuggingFace...")
    from datasets import load_dataset

    ds = load_dataset("neural-bridge/rag-dataset-12000", split="train")
    ds = ds.shuffle(seed=42).select(range(min(sample_size, len(ds))))
    rows = [
        {
            "section": f"doc_{i:04d}",
            "text": row["context"],
            "question": row["question"],
            "answer": row["answer"],
        }
        for i, row in enumerate(ds)
    ]
    corpus = pl.DataFrame(rows)
    corpus.write_parquet(CACHE_FILE)
    print(f"Cached {corpus.height} documents to {CACHE_FILE}")
    return corpus


def split_corpus(
    corpus: pl.DataFrame, n_eval: int = 20
) -> tuple[list[str], list[str], list[str]]:
    """Split corpus into retrieval texts and evaluation (question, answer) pairs."""
    doc_texts = corpus["text"].to_list()
    eval_questions = corpus["question"].to_list()[:n_eval]
    eval_answers = corpus["answer"].to_list()[:n_eval]
    return doc_texts, eval_questions, eval_answers


# ════════════════════════════════════════════════════════════════════════
# DELEGATE / LLM HELPERS
# ════════════════════════════════════════════════════════════════════════


def make_delegate(budget_usd: float = 1.0) -> object:
    """Construct an Ollama-backed Kaizen Delegate for RAG generation.

    The ``budget_usd`` parameter is retained for API compatibility with
    older callers and IGNORED — Ollama is free, so cost budgets are
    meaningless. The Delegate raises :class:`OllamaUnreachableError`
    transparently if the daemon is not running (no silent fallback).
    """
    del budget_usd  # API compat — see docstring
    return _make_chat_delegate(model=MODEL)


async def delegate_text(delegate: object, prompt: str) -> str:
    """Run an Ollama Delegate, return the streamed text.

    Hard-fails (via the underlying adapter) if the daemon is unreachable.
    """
    text, *_ = await run_delegate_text(delegate, prompt)
    return text.strip()


# ════════════════════════════════════════════════════════════════════════
# EMBEDDING HELPERS (pedagogical — uses an LLM as a low-dim projector)
# ════════════════════════════════════════════════════════════════════════


async def generate_embedding(text: str, delegate: object | None = None) -> list[float]:
    """Embed a single string with the Ollama embedding model.

    Returns a 768-dimensional dense vector from ``nomic-embed-text``
    (overridable via ``OLLAMA_EMBED_MODEL``). The ``delegate`` argument
    is retained for API compatibility with the previous OpenAI-era
    signature and is ignored — embedding goes through a dedicated
    embedding adapter, not a chat Delegate.
    """
    del delegate  # API compat — see docstring
    embedder = make_embedder(model=EMBED_MODEL)
    vectors = await embedder.embed([text])
    return list(vectors[0])


async def embed_many(texts: list[str], budget_usd: float = 3.0) -> list[list[float]]:
    """Embed a batch of texts via a single Ollama embedding call.

    The embedding adapter batches internally, which is materially faster
    than the previous one-call-per-text loop (that was a workaround for
    OpenAI per-call rate limits, not a real constraint with local Ollama).
    The ``budget_usd`` argument is retained for API compatibility and
    ignored.
    """
    del budget_usd  # API compat — see docstring
    embedder = make_embedder(model=EMBED_MODEL)
    vectors = await embedder.embed(texts)
    return [list(v) for v in vectors]


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Cosine similarity between two vectors — direction only, not magnitude."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


class DenseVectorStore:
    """Vector store using cosine similarity for dense retrieval."""

    def __init__(self) -> None:
        self.documents: list[str] = []
        self.embeddings: list[list[float]] = []
        self.metadata: list[dict] = []

    def add(self, text: str, embedding: list[float], meta: dict | None = None) -> None:
        self.documents.append(text)
        self.embeddings.append(embedding)
        self.metadata.append(meta or {})

    def search(self, query_embedding: list[float], top_k: int = 5) -> list[dict]:
        scores = [
            (i, cosine_similarity(query_embedding, emb))
            for i, emb in enumerate(self.embeddings)
        ]
        scores.sort(key=lambda x: x[1], reverse=True)
        return [
            {
                "text": self.documents[idx],
                "score": score,
                "metadata": self.metadata[idx],
            }
            for idx, score in scores[:top_k]
        ]


# ════════════════════════════════════════════════════════════════════════
# SIMPLE GENERATION (grounded QA) — shared between rerank, RAGAS, HyDE
# ════════════════════════════════════════════════════════════════════════


async def rag_answer(query: str, context: str, budget_usd: float = 0.5) -> str:
    """Generate an answer grounded in retrieved context.

    The ``budget_usd`` argument is retained for API compatibility and ignored
    (Ollama is free).
    """
    delegate = make_delegate(budget_usd=budget_usd)
    prompt = (
        "Answer the question using ONLY the provided context. "
        "If the context doesn't contain enough information, say so.\n\n"
        f"Context:\n{context[:2000]}\n\n"
        f"Question: {query}\n\nAnswer:"
    )
    return await delegate_text(delegate, prompt)


# ════════════════════════════════════════════════════════════════════════
# VISUALISATION — "seeing is believing" for RAG
# ════════════════════════════════════════════════════════════════════════


def plot_chunking_comparison(
    strategies: dict[str, list[str]], title: str, filename: str
) -> None:
    """Bar chart of chunks-per-strategy + avg length, side by side."""
    names = list(strategies.keys())
    counts = [len(v) for v in strategies.values()]
    avg_lens = [
        sum(len(c) for c in v) / len(v) if v else 0 for v in strategies.values()
    ]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(title, fontsize=13, fontweight="bold")

    axes[0].bar(names, counts, color="steelblue", edgecolor="white")
    axes[0].set_ylabel("Number of chunks")
    axes[0].set_title("Chunks produced")
    axes[0].tick_params(axis="x", rotation=30)

    axes[1].bar(names, avg_lens, color="darkorange", edgecolor="white")
    axes[1].set_ylabel("Average chars/chunk")
    axes[1].set_title("Chunk size")
    axes[1].tick_params(axis="x", rotation=30)

    plt.tight_layout()
    fname = OUTPUT_DIR / filename
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"  Saved: {fname}")


def plot_score_distribution(
    scores: list[float], title: str, xlabel: str, filename: str
) -> None:
    """Histogram of retrieval scores — shows how sharp the ranking is."""
    _, ax = plt.subplots(1, 1, figsize=(8, 4))
    ax.hist(scores, bins=20, color="seagreen", edgecolor="white", alpha=0.85)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Document count")
    ax.axvline(
        sum(scores) / len(scores) if scores else 0,
        color="crimson",
        linestyle="--",
        label=f"mean={sum(scores)/len(scores):.3f}" if scores else "mean=0",
    )
    ax.legend()
    plt.tight_layout()
    fname = OUTPUT_DIR / filename
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"  Saved: {fname}")


def plot_strategy_comparison(
    strategy_scores: dict[str, list[float]], title: str, filename: str
) -> None:
    """Overlayed histograms comparing top-k scores across retrieval strategies."""
    _, ax = plt.subplots(1, 1, figsize=(9, 5))
    colors = ["steelblue", "darkorange", "seagreen", "purple", "crimson"]
    for (name, scores), color in zip(strategy_scores.items(), colors):
        ax.hist(scores, bins=15, alpha=0.55, label=name, color=color, edgecolor="white")
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel("Top-k similarity / score")
    ax.set_ylabel("Retrieved chunks")
    ax.legend()
    plt.tight_layout()
    fname = OUTPUT_DIR / filename
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"  Saved: {fname}")


def plot_ragas_metrics(metrics: dict[str, float], title: str, filename: str) -> None:
    """Horizontal bar chart of RAGAS metrics, coloured by pass/fail."""
    names = list(metrics.keys())
    values = [metrics[n] for n in names]
    colors = [
        "seagreen" if v >= 0.7 else "darkorange" if v >= 0.4 else "crimson"
        for v in values
    ]

    _, ax = plt.subplots(1, 1, figsize=(8, 4))
    ax.barh(names, values, color=colors, edgecolor="white")
    ax.set_xlim(0, 1)
    ax.set_xlabel("Score (0.0 – 1.0)")
    ax.set_title(title, fontsize=13, fontweight="bold")
    for i, v in enumerate(values):
        ax.text(v + 0.02, i, f"{v:.2f}", va="center", fontsize=10)
    ax.axvline(0.7, color="grey", linestyle="--", alpha=0.5, label="target=0.70")
    ax.legend(loc="lower right")
    plt.tight_layout()
    fname = OUTPUT_DIR / filename
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"  Saved: {fname}")


def run_async(coro):
    """Uniform sync-wrapper so technique files stay readable."""
    return asyncio.run(coro)
