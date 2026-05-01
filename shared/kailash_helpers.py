# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""Common Kailash SDK setup patterns for MLFP exercises."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv


def setup_environment() -> None:
    """Load .env and validate common configuration.

    Call this at the top of every exercise that needs API keys or DB connections.
    """
    # Find .env by walking up from the exercise file
    env_path = Path.cwd() / ".env"
    if not env_path.exists():
        # Try repo root
        for parent in Path.cwd().parents:
            candidate = parent / ".env"
            if candidate.exists():
                env_path = candidate
                break

    load_dotenv(env_path)


def get_connection_manager(db_url: str | None = None):
    """Create a ConnectionManager for kailash-ml engines.

    Args:
        db_url: Database URL. Defaults to SQLite at ./mlfp.db
    """
    from kailash.db import ConnectionManager

    url = db_url or os.environ.get("DATABASE_URL", "sqlite:///mlfp.db")
    return ConnectionManager(url)


def get_device() -> "torch.device":
    """Return the best available compute device as a ``torch.device``.

    Routes through ``kailash_ml.device()`` (the canonical kailash-ml 0.12+
    accelerator detector) so MPS / CUDA / ROCm / Intel XPU / CPU are all
    selected by the same logic the rest of the platform uses. The previous
    hand-rolled ``mps→cuda→cpu`` cascade is replaced because:

      * kailash-ml's detector knows about ROCm, Intel XPU, and fp16/bf16
        capability flags — the cascade in this helper did not.
      * Apple-Silicon students get the Metal Performance Shaders backend
        with mixed-precision (fp16) without any opt-in.
      * One detection point means lessons that print "Using device: …"
        agree with what kailash-ml's MLEngine() actually picks.
    """
    import kailash_ml as km
    import torch

    backend = km.device()  # BackendInfo (auto MPS on Mac, CUDA on Linux+NVIDIA, …)
    return torch.device(backend.device_string)


def get_llm_model() -> str:
    """Get the configured LLM model name from environment."""
    setup_environment()
    model = os.environ.get("DEFAULT_LLM_MODEL", os.environ.get("OPENAI_PROD_MODEL"))
    if not model:
        raise EnvironmentError(
            "No LLM model configured. Set DEFAULT_LLM_MODEL or OPENAI_PROD_MODEL in .env"
        )
    return model
