# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
Shared infrastructure for MLFP06 Exercise 5 — AI Agents (ReAct, Structured,
Critic, Cost-Bounded).

Contains:
  - HotpotQA multi-hop QA dataset loading (cached parquet)
  - Agent tools: data_summary, search_documents, run_query, answer_question
  - Model resolution from environment
  - Output directory setup

Technique-specific agent classes and signatures live in the per-technique
files under modules/mlfp06/solutions/ex_5/.
"""
from __future__ import annotations

import inspect
import json
from pathlib import Path

import polars as pl

from shared.kailash_helpers import setup_environment

# ════════════════════════════════════════════════════════════════════════
# ENVIRONMENT SETUP
# ════════════════════════════════════════════════════════════════════════

setup_environment()

from shared.mlfp06._ollama_bootstrap import DEFAULT_CHAT_MODEL

MODEL = DEFAULT_CHAT_MODEL

OUTPUT_DIR = Path("outputs") / "ex5_agents"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ════════════════════════════════════════════════════════════════════════
# DATASET — HotpotQA (multi-hop QA, 500 cached examples)
# ════════════════════════════════════════════════════════════════════════

CACHE_DIR = Path("data") / "mlfp06" / "hotpotqa"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
CACHE_FILE = CACHE_DIR / "hotpotqa_500.parquet"


def load_hotpotqa() -> pl.DataFrame:
    """Load HotpotQA distractor split (500 shuffled examples, cached).

    Returns:
        Polars DataFrame with columns: text, question, answer, level, type.
    """
    if CACHE_FILE.exists():
        print(f"Loading cached HotpotQA from {CACHE_FILE}")
        return pl.read_parquet(CACHE_FILE)

    print("Downloading hotpotqa/hotpot_qa from HuggingFace...")
    from datasets import load_dataset

    ds = load_dataset(
        "hotpotqa/hotpot_qa",
        "distractor",
        split="validation",
        trust_remote_code=True,
    )
    ds = ds.shuffle(seed=42).select(range(min(500, len(ds))))
    rows = []
    for row in ds:
        context = row["context"]
        titles = context["title"]
        sentences = context["sentences"]
        joined = "\n".join(f"[{t}] " + " ".join(s) for t, s in zip(titles, sentences))
        rows.append(
            {
                "text": joined[:4000],
                "question": row["question"],
                "answer": row["answer"],
                "level": row["level"],
                "type": row["type"],
            }
        )
    qa_data = pl.DataFrame(rows)
    qa_data.write_parquet(CACHE_FILE)
    print(f"Cached {qa_data.height} HotpotQA examples at {CACHE_FILE}")
    return qa_data


# ════════════════════════════════════════════════════════════════════════
# AGENT TOOLS — docstrings ARE the agent's API documentation
# ════════════════════════════════════════════════════════════════════════
#
# Tool docstrings are what the LLM reads to choose WHICH tool to call and
# WHAT arguments to pass. Precise docstrings -> accurate tool selection.
# The tools close over a module-level `_qa_data` DataFrame populated by
# `make_tools()`.

_qa_data: pl.DataFrame | None = None


def _require_data() -> pl.DataFrame:
    if _qa_data is None:
        raise RuntimeError(
            "Agent tools used before make_tools() was called — "
            "call shared.mlfp06.ex_5.make_tools(qa_data) first."
        )
    return _qa_data


def data_summary(dataset_name: str = "qa_data") -> str:
    """Get a statistical summary of the QA dataset.

    Args:
        dataset_name: Which dataset to summarise.  Currently 'qa_data'.

    Returns:
        Text summary including shape, columns, type distribution, and
        average text lengths.
    """
    df = _require_data()
    parts = [
        f"Dataset: {dataset_name}",
        f"Shape: {df.height} rows x {df.width} columns",
        f"Columns: {', '.join(df.columns)}",
    ]
    for col in df.columns:
        dtype = str(df.schema[col])
        if "Utf8" in dtype or "String" in dtype:
            n_unique = df.select(pl.col(col).n_unique()).item()
            avg_len = df.select(pl.col(col).str.len_chars().mean()).item()
            parts.append(f"  {col} ({dtype}): {n_unique} unique, avg_len={avg_len:.0f}")
        elif "Int" in dtype or "Float" in dtype:
            stats = df.select(
                pl.col(col).mean().alias("mean"),
                pl.col(col).min().alias("min"),
                pl.col(col).max().alias("max"),
            ).row(0)
            parts.append(
                f"  {col} ({dtype}): mean={stats[0]}, range=[{stats[1]}, {stats[2]}]"
            )
    return "\n".join(parts)


def search_documents(query: str, top_k: int = 3) -> str:
    """Search the QA corpus for documents matching a keyword query.

    Args:
        query: Keywords to search for in the document texts.
        top_k:  Maximum number of matching documents to return.

    Returns:
        Matching document excerpts with their questions and answers.
    """
    df = _require_data()
    query_lower = query.lower()
    scored = []
    for i, row in enumerate(df.iter_rows(named=True)):
        text = row["text"].lower()
        score = sum(1 for word in query_lower.split() if word in text)
        if score > 0:
            scored.append((score, i, row))
    scored.sort(key=lambda x: x[0], reverse=True)

    results = []
    for score, idx, row in scored[:top_k]:
        results.append(
            f"[Doc {idx}] Q: {row['question']}\n"
            f"  A: {row['answer']}\n"
            f"  Context (excerpt): {row['text'][:300]}..."
        )
    return "\n\n".join(results) if results else f"No documents matching '{query}'"


def run_query(query_description: str) -> str:
    """Run a descriptive query against the QA dataset.

    Args:
        query_description: Natural language description of the query
            (e.g., 'count comparison questions', 'find bridge-type questions').

    Returns:
        Query results as formatted text.
    """
    df = _require_data()
    desc = query_description.lower()

    if "count" in desc and "type" in desc:
        counts = df.group_by("type").len().sort("len", descending=True)
        return f"Question types:\n{counts}"
    elif "count" in desc and "level" in desc:
        counts = df.group_by("level").len().sort("len", descending=True)
        return f"Difficulty levels:\n{counts}"
    elif "comparison" in desc:
        comparison = df.filter(pl.col("type") == "comparison")
        return (
            f"Comparison questions: {comparison.height}\n"
            f"Sample: {comparison['question'][0]}"
        )
    elif "bridge" in desc:
        bridge = df.filter(pl.col("type") == "bridge")
        return f"Bridge questions: {bridge.height}\nSample: {bridge['question'][0]}"
    elif "top" in desc or "longest" in desc:
        df_with_len = df.with_columns(pl.col("text").str.len_chars().alias("text_len"))
        top = df_with_len.sort("text_len", descending=True).head(5)
        return f"Top 5 by text length:\n{top.select('question', 'text_len')}"
    else:
        return f"Dataset has {df.height} rows. Columns: {df.columns}"


def answer_question(question: str) -> str:
    """Look up the answer to a specific HotpotQA question.

    Args:
        question: The exact question text to look up.

    Returns:
        The ground-truth answer if found, or 'not found'.
    """
    df = _require_data()
    for row in df.iter_rows(named=True):
        if question.lower().strip() in row["question"].lower():
            return (
                f"Answer: {row['answer']}\n"
                f"Type: {row['type']}, Level: {row['level']}"
            )
    return "Question not found in dataset."


def make_tools(qa_data: pl.DataFrame) -> list:
    """Bind the qa_data DataFrame to the tool closures and return the list.

    Args:
        qa_data: The HotpotQA DataFrame from load_hotpotqa().

    Returns:
        List of 4 tool callables ready to hand to a ReActAgent.
    """
    global _qa_data
    _qa_data = qa_data
    return [data_summary, search_documents, run_query, answer_question]


def tool_schemas(tools: list) -> list[dict]:
    """Build JSON Schema descriptors for a list of tool callables.

    Mirrors what OpenAI / Anthropic function-calling protocols expect
    (name, description, parameters.properties). Used in the ReAct
    technique file to illustrate function calling.
    """
    schemas = []
    for tool in tools:
        sig = inspect.signature(tool)
        params = {}
        for name, param in sig.parameters.items():
            annotation = param.annotation
            param_type = "string"
            if annotation is int:
                param_type = "integer"
            elif annotation is float:
                param_type = "number"
            params[name] = {
                "type": param_type,
                "description": f"Parameter: {name}",
            }
        schemas.append(
            {
                "name": tool.__name__,
                "description": (tool.__doc__ or "").strip().split("\n")[0],
                "parameters": {"type": "object", "properties": params},
            }
        )
    return schemas


def print_tool_registry(tools: list) -> None:
    """Human-readable dump of tool names, schemas, and first-line docs."""
    print("Registered tools:")
    for tool in tools:
        doc_first = (tool.__doc__ or "").strip().split("\n")[0]
        print(f"  {tool.__name__}: {doc_first}")
    schemas = tool_schemas(tools)
    print(f"\nGenerated {len(schemas)} JSON Schema descriptors " f"(first shown):")
    print(json.dumps(schemas[0], indent=2)[:400])


__all__ = [
    "MODEL",
    "OUTPUT_DIR",
    "load_hotpotqa",
    "make_tools",
    "tool_schemas",
    "print_tool_registry",
    "data_summary",
    "search_documents",
    "run_query",
    "answer_question",
]
