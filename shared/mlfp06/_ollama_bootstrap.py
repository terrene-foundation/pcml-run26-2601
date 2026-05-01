# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""Ollama bootstrap for MLFP06 — local LLMs, no API keys, real calls only.

Why this module exists
----------------------
M6 trains, prompts, evaluates, and deploys real LLMs. Routing every call
through OpenAI:

  * forces every learner to hold an API key
  * burns budget every time a notebook reruns a cell
  * makes "did the LLM actually fire?" invisible (a stubbed `cost=$0`
    looks identical to a successful call)

This module switches every M6 LLM call to a locally-running Ollama daemon
that the student installs once and re-uses across every lesson.

Contract
--------
* :func:`make_delegate` — the only sanctioned way to construct a Kaizen
  ``Delegate`` for M6. Wraps the Ollama adapter with sensible defaults and
  forces ``budget_usd=None`` (Kaizen mis-prices Ollama as $3/$15-per-Mtok;
  budget tracking against a free local provider is meaningless).
* :func:`make_embedder` — the only sanctioned way to construct an embedding
  adapter for M6 RAG (lesson 6.4).
* :func:`preflight_ollama` — pre-flight reachability + model-presence check.
  Raises :class:`OllamaUnreachableError` with an actionable message. Call
  this at the top of every notebook / script before constructing a
  Delegate.
* :data:`LESSON_MODELS` — manifest of which models each lesson needs.
  Consumed by ``scripts/generate_selfcontained_notebook.py`` to inject the
  Colab bootstrap cell.

Per ``rules/zero-tolerance.md`` Rule 2: there is no silent stub fallback
in this module. If Ollama is not reachable, we raise. The previous
``run_delegate(...) -> ("unknown", 0.0, latency)`` fallback is BLOCKED.

Per ``rules/framework-first.md``: every LLM call goes through Kaizen
``Delegate``. Raw ``httpx`` calls to ``/api/generate`` are BLOCKED for
generation paths; ``httpx`` is used here only for the daemon-reachability
preflight.
"""
from __future__ import annotations

import logging
import os
import shutil
import subprocess
import sys
import time
from typing import Any

import httpx
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

# ════════════════════════════════════════════════════════════════════════
# Configuration — env-driven per rules/env-models.md
# ════════════════════════════════════════════════════════════════════════

load_dotenv()


def _env(key: str, default: str) -> str:
    val = os.environ.get(key)
    return val if val else default


OLLAMA_BASE_URL: str = _env("OLLAMA_BASE_URL", "http://localhost:11434")
"""Where the Ollama daemon listens. Local default; Colab uses the same."""

DEFAULT_CHAT_MODEL: str = _env("OLLAMA_CHAT_MODEL", "llama3.2:3b")
"""Tool-capable instruct model. Allowlisted by Kaizen for function calling."""

DEFAULT_EMBED_MODEL: str = _env("OLLAMA_EMBED_MODEL", "nomic-embed-text")
"""Compact embedding model — ~270MB, 768-dim. Used by M6 RAG (lesson 6.4)."""

DEFAULT_FT_BASE_MODEL: str = _env("OLLAMA_FT_BASE_MODEL", "qwen2.5:0.5b")
"""Tiny base model for fine-tuning demos (lessons 6.2, 6.3). Fits T4."""


# ════════════════════════════════════════════════════════════════════════
# Lesson → models manifest (used by generator + preflight)
# ════════════════════════════════════════════════════════════════════════


LESSON_MODELS: dict[str, list[str]] = {
    # 6.1 Prompting — chat only
    "ex_1": [DEFAULT_CHAT_MODEL],
    # 6.2 Fine-tuning — base for LoRA + chat for evaluation
    "ex_2": [DEFAULT_FT_BASE_MODEL, DEFAULT_CHAT_MODEL],
    # 6.3 Alignment (DPO/GRPO) — base + chat for judge
    "ex_3": [DEFAULT_FT_BASE_MODEL, DEFAULT_CHAT_MODEL],
    # 6.4 RAG — chat + embeddings
    "ex_4": [DEFAULT_CHAT_MODEL, DEFAULT_EMBED_MODEL],
    # 6.5 Agents (ReAct) — tool-capable chat
    "ex_5": [DEFAULT_CHAT_MODEL],
    # 6.6 Multi-agent + MCP — tool-capable chat
    "ex_6": [DEFAULT_CHAT_MODEL],
    # 6.7 PACT governance — chat for governed Delegate
    "ex_7": [DEFAULT_CHAT_MODEL],
    # 6.8 Nexus deployment — chat behind the API
    "ex_8": [DEFAULT_CHAT_MODEL],
}


# ════════════════════════════════════════════════════════════════════════
# Errors
# ════════════════════════════════════════════════════════════════════════


class OllamaUnreachableError(RuntimeError):
    """Daemon not running, daemon unreachable, or required model not pulled.

    The error message MUST be actionable: it tells the user the exact
    command to run to fix the situation. We never wrap a low-level
    ``httpx.ConnectError`` and re-raise without translating it.
    """


# ════════════════════════════════════════════════════════════════════════
# Preflight — reachability + model presence
# ════════════════════════════════════════════════════════════════════════


def _running_in_colab() -> bool:
    return "google.colab" in sys.modules


def _list_pulled_families(*, timeout_s: float = 2.0) -> set[str]:
    """Hit /api/tags and return the set of pulled model families (no tag).

    Returns ``{"llama3.2", "qwen2.5"}`` for a daemon with
    ``llama3.2:3b`` and ``qwen2.5:0.5b`` pulled. The model family is
    everything before the first colon — we match against family rather
    than the full ``family:tag`` because pulling ``llama3.2:3b-instruct``
    or ``llama3.2:3b-instruct-q4_K_M`` should still satisfy a request
    for the ``llama3.2`` family.
    """
    url = f"{OLLAMA_BASE_URL.rstrip('/')}/api/tags"
    try:
        r = httpx.get(url, timeout=timeout_s)
        r.raise_for_status()
    except (httpx.HTTPError, httpx.ConnectError) as exc:
        raise OllamaUnreachableError(
            f"ollama daemon not reachable at {OLLAMA_BASE_URL}. "
            f"Fix: open a terminal and run `ollama serve`. "
            f"On macOS the desktop app starts the daemon automatically. "
            f"Underlying: {type(exc).__name__}: {exc}"
        ) from exc
    payload = r.json() or {}
    families: set[str] = set()
    for entry in payload.get("models", []):
        name = entry.get("name") or entry.get("model") or ""
        if not name:
            continue
        families.add(name.split(":", 1)[0])
    return families


def preflight_ollama(
    *,
    required_models: list[str] | None = None,
    auto_pull: bool | None = None,
    timeout_s: float = 2.0,
) -> None:
    """Verify the daemon is reachable and the required models are pulled.

    Args:
        required_models: Models that must be present, e.g.
            ``["llama3.2:3b", "nomic-embed-text"]``. Family-matched (the
            ``:tag`` portion is informational; any pulled tag for the
            same family satisfies the requirement).
        auto_pull: If ``True``, attempt ``ollama pull <model>`` for any
            missing model via subprocess. Defaults to ``True`` on Colab
            (where the bootstrap cell has just installed the binary)
            and ``False`` locally (where pulling is the student's job
            and surprise downloads are unwelcome).
        timeout_s: HTTP timeout for the reachability check. The default
            is intentionally short — if the daemon is alive it answers
            within 100ms.

    Raises:
        OllamaUnreachableError: Daemon unreachable, or a required model
            is missing after pull (or auto_pull was disabled).
    """
    if auto_pull is None:
        auto_pull = _running_in_colab()

    have = _list_pulled_families(timeout_s=timeout_s)
    needed = required_models or []
    missing = [m for m in needed if m.split(":", 1)[0] not in have]

    if missing and auto_pull:
        for model in missing:
            _pull_model(model)
        # Re-check
        have = _list_pulled_families(timeout_s=timeout_s)
        missing = [m for m in needed if m.split(":", 1)[0] not in have]

    if missing:
        cmds = "\n  ".join(f"ollama pull {m}" for m in missing)
        raise OllamaUnreachableError(
            f"Required model(s) not pulled: {missing}.\n"
            f"Fix — run in a terminal:\n  {cmds}"
        )

    logger.info(
        "ollama.preflight.ok",
        extra={
            "base_url": OLLAMA_BASE_URL,
            "models_present": sorted(have),
            "required": needed,
        },
    )


def _pull_model(model: str) -> None:
    """Run ``ollama pull <model>`` as a subprocess. Streams to stdout."""
    if shutil.which("ollama") is None:
        raise OllamaUnreachableError(
            f"`ollama` binary not on PATH — cannot pull {model}. "
            f"Install: https://ollama.com/download"
        )
    print(f"[ollama] pulling {model} (one-time download) …", flush=True)
    t0 = time.monotonic()
    proc = subprocess.run(
        ["ollama", "pull", model],
        capture_output=False,
        text=True,
    )
    elapsed = time.monotonic() - t0
    if proc.returncode != 0:
        raise OllamaUnreachableError(
            f"`ollama pull {model}` failed with exit {proc.returncode} "
            f"after {elapsed:.1f}s. Check network and disk space."
        )
    print(f"[ollama] {model} pulled in {elapsed:.1f}s", flush=True)


# ════════════════════════════════════════════════════════════════════════
# Delegate factory — the only sanctioned constructor for M6
# ════════════════════════════════════════════════════════════════════════


def make_delegate(
    *,
    model: str | None = None,
    tools: Any = None,
    system_prompt: str | None = None,
    temperature: float = 0.4,
    max_tokens: int = 4096,
    **kwargs: Any,
) -> Any:
    """Construct a Kaizen ``Delegate`` backed by the local Ollama daemon.

    This is the ONLY way M6 helpers and exercises construct a Delegate.
    Direct ``Delegate(...)`` construction in M6 code is BLOCKED — the
    cost-mispricing trap (``budget_usd`` against a free provider) and
    the silent OpenAI fallback that comes from omitting ``adapter=`` are
    the reasons.

    Args:
        model: Ollama model tag (e.g. ``"llama3.2:3b"``). Defaults to
            :data:`DEFAULT_CHAT_MODEL`.
        tools: Tool registry or list — passed through to Delegate.
            Auto-disables streaming on the Ollama adapter.
        system_prompt: Optional system prompt.
        temperature: Sampling temperature. Default 0.4.
        max_tokens: Hard cap on output length. Default 4096.
        **kwargs: Forwarded to ``Delegate(...)``. ``budget_usd`` is
            forced to ``None`` and ``api_key`` / ``base_url`` are
            BLOCKED (the adapter owns those).

    Returns:
        A configured Kaizen ``Delegate`` instance.
    """
    for blocked in ("api_key", "base_url", "budget_usd", "adapter"):
        if blocked in kwargs:
            raise ValueError(
                f"make_delegate(): {blocked!r} is owned by the bootstrap "
                f"and cannot be overridden by callers."
            )

    # Lazy imports — keep this module importable in environments that have
    # not yet installed kaizen-agents (e.g. early bootstrap cells).
    from kaizen_agents import Delegate
    from kaizen_agents.delegate.adapters.registry import get_adapter

    resolved = model or DEFAULT_CHAT_MODEL
    adapter = get_adapter(
        "ollama",
        model=resolved,
        base_url=OLLAMA_BASE_URL,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return Delegate(
        model=resolved,
        adapter=adapter,
        tools=tools,
        system_prompt=system_prompt,
        budget_usd=None,
        **kwargs,
    )


# ════════════════════════════════════════════════════════════════════════
# Embedding factory — the only sanctioned constructor for M6 RAG
# ════════════════════════════════════════════════════════════════════════


def make_embedder(*, model: str | None = None) -> Any:
    """Construct a Kaizen embedding adapter backed by Ollama.

    Args:
        model: Ollama embedding model tag. Defaults to
            :data:`DEFAULT_EMBED_MODEL`.

    Returns:
        An adapter with ``async def embed(texts: list[str]) -> list[list[float]]``.
    """
    from kaizen_agents.delegate.adapters.registry import get_embedding_adapter

    resolved = model or DEFAULT_EMBED_MODEL
    return get_embedding_adapter(
        "ollama",
        model=resolved,
        base_url=OLLAMA_BASE_URL,
    )


# ════════════════════════════════════════════════════════════════════════
# Streaming helpers — same call shape as the old `run_delegate`
# ════════════════════════════════════════════════════════════════════════


async def run_delegate_text(
    delegate: Any, prompt: str
) -> tuple[str, dict[str, int], float]:
    """Run a Delegate to completion, return ``(text, usage, latency_s)``.

    ``usage`` is a dict with ``prompt_tokens``, ``completion_tokens``,
    ``total_tokens`` (zero for events that don't carry usage). ``cost``
    is intentionally absent — Ollama is free, and Kaizen's ``consumed_usd``
    is mis-priced for local models.

    Event handling: the Kaizen ``TextDelta`` events expose the streamed
    chunk on ``.text`` (not ``.delta_text`` — that name is reserved for a
    different StreamEvent shape). The terminal ``TurnComplete`` event
    carries the final ``.text`` (entire response, not a delta) plus the
    ``.usage`` dict. We accumulate from ``TextDelta.text`` for streaming
    UX and fall back to ``TurnComplete.text`` if no deltas arrived.
    """
    t0 = time.perf_counter()
    response = ""
    usage: dict[str, int] = {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
    }
    final_text: str | None = None
    async for event in delegate.run(prompt):
        ev_type = getattr(event, "event_type", None)
        text = getattr(event, "text", None)
        if ev_type == "text_delta" and text:
            response += text
        elif ev_type == "turn_complete":
            final_text = text or final_text
            ev_usage = getattr(event, "usage", None) or {}
            for k in usage:
                if k in ev_usage:
                    usage[k] = int(ev_usage[k])
    if not response and final_text:
        response = final_text
    elapsed = time.perf_counter() - t0
    return response, usage, elapsed


# ════════════════════════════════════════════════════════════════════════
# Colab installer — used by the generator's Cell 0
# ════════════════════════════════════════════════════════════════════════


def colab_install_and_start_ollama(*, log_path: str = "/tmp/ollama.log") -> None:
    """Install the Ollama binary and start the daemon (Colab only).

    Idempotent: skips install if the binary is already on PATH and
    skips daemon start if a process is already listening on
    :data:`OLLAMA_BASE_URL`. Raises :class:`OllamaUnreachableError`
    if either step fails.

    Not for local use — locally, students run ``ollama serve`` themselves.
    """
    if not _running_in_colab():
        raise RuntimeError(
            "colab_install_and_start_ollama() is Colab-only. "
            "On a local machine, run `ollama serve` in a terminal."
        )

    # Step 1 — install if missing
    if shutil.which("ollama") is None:
        print("[colab] installing ollama …", flush=True)
        proc = subprocess.run(
            "curl -fsSL https://ollama.com/install.sh | sh",
            shell=True,
            check=False,
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0:
            raise OllamaUnreachableError(
                f"ollama install failed (exit {proc.returncode}):\n"
                f"stdout: {proc.stdout[-500:]}\nstderr: {proc.stderr[-500:]}"
            )

    # Step 2 — start daemon if not already listening
    try:
        httpx.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=1.0).raise_for_status()
        print("[colab] ollama daemon already running", flush=True)
        return
    except (httpx.HTTPError, httpx.ConnectError):
        pass

    print(f"[colab] starting ollama daemon (logs → {log_path}) …", flush=True)
    subprocess.Popen(
        f"nohup ollama serve > {log_path} 2>&1 &",
        shell=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    # Step 3 — wait up to 30s for the daemon to answer /api/tags
    deadline = time.monotonic() + 30.0
    last_err: Exception | None = None
    while time.monotonic() < deadline:
        try:
            httpx.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=1.0).raise_for_status()
            print("[colab] ollama daemon ready", flush=True)
            return
        except (httpx.HTTPError, httpx.ConnectError) as exc:
            last_err = exc
            time.sleep(0.5)

    raise OllamaUnreachableError(
        f"ollama daemon failed to come up within 30s. "
        f"Last error: {type(last_err).__name__}: {last_err}. "
        f"Inspect {log_path} for details."
    )
