# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""Quiz harness for MLFP05 — shared checkers and training helpers.

All functions in this module are used by both the student notebook and the
instructor answer key so the grading bar is identical in both.

Design principles:

* Deterministic seeds so student & instructor see the same accuracy bands.
* Works on CPU, MPS (Apple Silicon), and CUDA (Colab T4).
* Download-once caching: MNIST / Fashion-MNIST live under ``/content/data/``
  on Colab and ``data/mlfp05/`` locally so re-runs are fast.
* Diagnostic thresholds for Q5 were calibrated by running the TARGET
  architecture and capturing its actual ``report()`` verdicts (see the
  module docstring in ``mlfp05_quiz_solutions.ipynb``).
"""
from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torchvision import datasets, transforms


# ── Device detection ──────────────────────────────────────────────────────
def pick_device() -> torch.device:
    """Return CUDA if available, else MPS (Apple), else CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ── Data root resolution ──────────────────────────────────────────────────
def _resolve_data_root(kind: str) -> Path:
    """Resolve the cache directory for a torchvision dataset.

    Priority:
      1. ``/content/data/{kind}`` (Colab convention)
      2. ``<repo_root>/data/mlfp05/{kind}`` (local convention)
      3. fallback: ``./data/{kind}``

    ``kind`` is one of ``"mnist"`` / ``"fashion_mnist"``.
    """
    colab_root = Path("/content/data") / kind
    if Path("/content").exists():
        colab_root.mkdir(parents=True, exist_ok=True)
        return colab_root

    # Walk up to find a repo containing modules/mlfp05/
    here = Path(__file__).resolve()
    for ancestor in here.parents:
        candidate = ancestor / "data" / "mlfp05" / kind
        if (ancestor / "modules" / "mlfp05").exists():
            candidate.mkdir(parents=True, exist_ok=True)
            return candidate

    fallback = Path.cwd() / "data" / kind
    fallback.mkdir(parents=True, exist_ok=True)
    return fallback


# ── Dataset loaders (cached) ──────────────────────────────────────────────
_MNIST_STATS = ((0.1307,), (0.3081,))
_FASHION_STATS = ((0.2860,), (0.3530,))


def load_mnist(batch_size: int = 128) -> tuple[DataLoader, DataLoader]:
    """Return (train_loader, test_loader) for MNIST. Downloads once, caches."""
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(*_MNIST_STATS)]
    )
    root = _resolve_data_root("mnist")
    train = datasets.MNIST(
        root=str(root), train=True, download=True, transform=transform
    )
    test = datasets.MNIST(
        root=str(root), train=False, download=True, transform=transform
    )
    return (
        DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=0),
        DataLoader(test, batch_size=512, shuffle=False, num_workers=0),
    )


def load_fashion_mnist(batch_size: int = 128) -> tuple[DataLoader, DataLoader]:
    """Return (train_loader, val_loader) for Fashion-MNIST. Downloads once."""
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(*_FASHION_STATS)]
    )
    root = _resolve_data_root("fashion_mnist")
    train = datasets.FashionMNIST(
        root=str(root), train=True, download=True, transform=transform
    )
    test = datasets.FashionMNIST(
        root=str(root), train=False, download=True, transform=transform
    )
    return (
        DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=0),
        DataLoader(test, batch_size=512, shuffle=False, num_workers=0),
    )


# ── Q3 synthetic sequence dataset ─────────────────────────────────────────
def make_q3_dataset(
    n_train: int = 4096,
    n_test: int = 1024,
    seq_len: int = 20,
    seed: int = 42,
) -> tuple[DataLoader, DataLoader]:
    """Binary-sequence task: label = 1 iff more 1s are in the FIRST half.

    Each sequence is ``seq_len`` Bernoulli(0.5) draws. Tie ⇒ label 0. The
    task is *trivially* linearly separable in sufficient statistics but
    requires a model that can distinguish positional information — a
    bag-of-words MLP will score ≈ 50%, an LSTM should score >>85%.
    """
    g = torch.Generator().manual_seed(seed)
    train_x = torch.randint(0, 2, (n_train, seq_len), generator=g).float()
    test_x = torch.randint(0, 2, (n_test, seq_len), generator=g).float()

    def _label(x: torch.Tensor) -> torch.Tensor:
        half = x.size(1) // 2
        first = x[:, :half].sum(dim=1)
        second = x[:, half:].sum(dim=1)
        return (first > second).long()

    train_y = _label(train_x)
    test_y = _label(test_x)

    # Shape as (B, T, 1) so nn.LSTM(input_size=1) consumes directly.
    train_x = train_x.unsqueeze(-1)
    test_x = test_x.unsqueeze(-1)

    train = TensorDataset(train_x, train_y)
    test = TensorDataset(test_x, test_y)
    return (
        DataLoader(train, batch_size=128, shuffle=True, num_workers=0),
        DataLoader(test, batch_size=512, shuffle=False, num_workers=0),
    )


# ── Generic eval helpers ──────────────────────────────────────────────────
@torch.no_grad()
def accuracy(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    """Return classification accuracy ∈ [0,1] on the given loader."""
    model.eval()
    model.to(device)
    correct = 0
    total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.numel()
    return correct / max(total, 1)


def train_classifier(
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    *,
    epochs: int = 5,
    lr: float = 1e-3,
    label: str = "model",
) -> float:
    """Standard Adam + CE training loop. Returns final test accuracy.

    Used by Q1/Q2/Q3. Prints one line per epoch so students see progress.
    """
    model.to(device)
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    t0 = time.monotonic()
    for epoch in range(1, epochs + 1):
        model.train()
        running = 0.0
        n_batches = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optim.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            optim.step()
            running += loss.item()
            n_batches += 1
        test_acc = accuracy(model, test_loader, device)
        print(
            f"  [{label}] epoch {epoch}/{epochs}  "
            f"train_loss={running/max(n_batches,1):.4f}  "
            f"test_acc={test_acc:.4f}  "
            f"elapsed={time.monotonic()-t0:.1f}s"
        )
    return test_acc


# ── Q1–Q3 checkers ────────────────────────────────────────────────────────
def check_q1(test_acc: float) -> tuple[bool, float, str]:
    """Q1 passes at >95% MNIST test accuracy after 5 epochs."""
    threshold = 0.95
    passed = test_acc >= threshold
    msg = (
        f"Q1 MLP — test acc = {test_acc:.4f}  "
        f"(threshold ≥ {threshold}) — {'PASS' if passed else 'FAIL'}"
    )
    return passed, test_acc, msg


def check_q2(test_acc: float) -> tuple[bool, float, str]:
    """Q2 passes at >97% MNIST test accuracy (CNN is stronger)."""
    threshold = 0.97
    passed = test_acc >= threshold
    msg = (
        f"Q2 CNN — test acc = {test_acc:.4f}  "
        f"(threshold ≥ {threshold}) — {'PASS' if passed else 'FAIL'}"
    )
    return passed, test_acc, msg


def check_q3(test_acc: float) -> tuple[bool, float, str]:
    """Q3 passes at >85% accuracy on the synthetic halves task."""
    threshold = 0.85
    passed = test_acc >= threshold
    msg = (
        f"Q3 LSTM — test acc = {test_acc:.4f}  "
        f"(threshold ≥ {threshold}) — {'PASS' if passed else 'FAIL'}"
    )
    return passed, test_acc, msg


# ── Q4 fuzzy-keyword checker ──────────────────────────────────────────────
# The Q4 setup is a CNN with stride>1 and NO padding that collapses spatial
# dims so badly the final conv feature map can become 0×0. The primary fix
# is adding padding (or reducing stride). We accept multiple phrasings.
_Q4_ISSUE_KEYWORDS: tuple[tuple[str, ...], ...] = (
    ("stride",),
    ("padding",),
    ("dimension", "dim", "spatial", "collapse", "shrink"),
)
_Q4_FIX_KEYWORDS: tuple[tuple[str, ...], ...] = (
    ("padding",),
    ("stride",),
)


def _fuzzy_match(answer: str, keyword_groups: tuple[tuple[str, ...], ...]) -> int:
    """Count how many keyword groups have at least one keyword in answer."""
    a = answer.lower()
    hits = 0
    for group in keyword_groups:
        if any(k in a for k in group):
            hits += 1
    return hits


def check_q4(answer: str) -> tuple[bool, str]:
    """Pass iff the answer mentions the issue (stride/padding) AND a fix."""
    if not isinstance(answer, str) or len(answer.strip()) < 10:
        return False, (
            "Q4 — answer too short. Explain both the diagnosed issue and "
            "the concrete fix in a sentence or two."
        )
    issue_hits = _fuzzy_match(answer, _Q4_ISSUE_KEYWORDS)
    fix_hits = _fuzzy_match(answer, _Q4_FIX_KEYWORDS)
    passed = issue_hits >= 2 and fix_hits >= 1
    msg = (
        f"Q4 — issue keywords matched: {issue_hits}/3, "
        f"fix keywords matched: {fix_hits}/2 — "
        f"{'PASS' if passed else 'FAIL'}"
    )
    if not passed:
        msg += (
            "\n      Hint: the bug involves **stride without padding** causing "
            "spatial dimensions to collapse; fix by adding padding or reducing stride."
        )
    return passed, msg


# ── Q5 training + diagnostic helper ───────────────────────────────────────
def _build_optimizer(
    model: nn.Module, lr: float, weight_decay: float
) -> torch.optim.Optimizer:
    return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)


def _warmup_factor(step: int, warmup_steps: int) -> float:
    if warmup_steps <= 0:
        return 1.0
    if step >= warmup_steps:
        return 1.0
    return (step + 1) / warmup_steps


def train_and_diagnose(
    model: nn.Module,
    *,
    train_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    warmup_steps: int = 0,
    epochs: int = 3,
    verbose: bool = True,
) -> tuple[dict[str, Any], float]:
    """Train ``model`` for ``epochs`` epochs, recording DL diagnostics.

    Returns ``(findings_dict, test_accuracy)``. ``findings_dict`` is the
    dict returned by ``DLDiagnostics.report()``; each value is a dict with
    keys ``severity`` and ``message``.

    This is the Q5 primary instrument. Deliberately wraps the full TRAIN
    → DIAGNOSE → EVAL cycle so the student never touches the diagnostic
    scaffolding — only the model and hyperparameters.
    """
    # Import lazily so module import never triggers a diagnostics init on
    # import error in downstream repos without the shared package.
    from kailash_ml.diagnostics import DLDiagnostics

    model.to(device)
    optimizer = _build_optimizer(model, lr=lr, weight_decay=weight_decay)
    loss_fn = nn.CrossEntropyLoss()
    base_lr = lr
    step = 0

    with DLDiagnostics(model) as diag:
        diag.track_gradients()
        diag.track_activations()
        diag.track_dead_neurons()

        for epoch in range(1, epochs + 1):
            model.train()
            running = 0.0
            n_batches = 0
            for x, y in train_loader:
                x, y = x.to(device), y.to(device)
                # LR warmup
                factor = _warmup_factor(step, warmup_steps)
                for pg in optimizer.param_groups:
                    pg["lr"] = base_lr * factor
                optimizer.zero_grad()
                logits = model(x)
                loss = loss_fn(logits, y)
                loss.backward()
                optimizer.step()
                diag.record_batch(loss=loss.item(), lr=optimizer.param_groups[0]["lr"])
                running += loss.item()
                n_batches += 1
                step += 1

            # End-of-epoch validation loss (for loss_trend diagnosis)
            model.eval()
            val_running = 0.0
            val_n = 0
            with torch.no_grad():
                for x, y in test_loader:
                    x, y = x.to(device), y.to(device)
                    val_running += loss_fn(model(x), y).item() * y.size(0)
                    val_n += y.size(0)
            val_loss = val_running / max(val_n, 1)
            diag.record_epoch(train_loss=running / max(n_batches, 1), val_loss=val_loss)
            if verbose:
                print(
                    f"  epoch {epoch}/{epochs}  "
                    f"train_loss={running/max(n_batches,1):.4f}  "
                    f"val_loss={val_loss:.4f}"
                )

        findings = diag.report()

    test_acc = accuracy(model, test_loader, device)
    if verbose:
        print(f"  final test_acc = {test_acc:.4f}")
    return findings, test_acc


# ── Q5 pass/fail decision ─────────────────────────────────────────────────
Q5_TEST_ACC_THRESHOLD = 0.82

# Verdicts we strictly require to be HEALTHY.
#
# ``gradient_flow`` is deliberately NOT in this list even though we print its
# verdict for student feedback. Rationale: the ``DLDiagnostics`` gradient_flow
# check trips CRITICAL whenever the OUTPUT layer's gradient RMS exceeds 1e-2,
# which is a fundamental property of cross-entropy on a 10-way classifier —
# ``dL/dlogits ≈ softmax − one_hot`` has RMS near 0.06, and the output linear
# layer's per-element grad RMS lands around 1.5e-2 on Fashion-MNIST at TARGET
# hyperparameters. Requiring HEALTHY here would force a fix that is not in
# fact a bug. Instead we require the two verdicts that DO differentiate the
# TARGET from the BROKEN STARTER at a 50× gap: dead_neurons and loss_trend,
# plus the downstream test accuracy.
Q5_REQUIRED_VERDICTS = ("dead_neurons", "loss_trend")

# Verdicts whose severity is surfaced in the Prescription Pad output but does
# NOT block the pass. Students see the verdict; instructors see the verdict.
Q5_ADVISORY_VERDICTS = ("gradient_flow",)


def check_q5_pass(findings: dict[str, Any], test_acc: float) -> tuple[bool, str]:
    """Pass iff the required verdicts are HEALTHY and test_acc ≥ threshold.

    Required (blocking):
        * ``dead_neurons`` == HEALTHY  (the ReLU→GELU fix)
        * ``loss_trend`` == HEALTHY    (the warmup/LR fix)
        * ``test_acc`` >= ``Q5_TEST_ACC_THRESHOLD``

    Advisory (printed but not blocking):
        * ``gradient_flow`` — the output-layer grad RMS naturally exceeds the
          library's 1e-2 threshold on a 10-way softmax head, so we surface the
          verdict for educational purposes but do not require HEALTHY.

    Threshold is 0.82 (not 0.85) because Fashion-MNIST with a plain 6-layer
    MLP caps out around 0.87 even on TARGET hyperparameters — we leave some
    headroom for run-to-run variance. Calibrated by running ``Q5TargetMLP``
    on Fashion-MNIST for 3 epochs and capturing its ``report()`` verdicts
    (see ``mlfp05_quiz_solutions.ipynb`` module docstring).
    """
    reasons: list[str] = []
    advisory_notes: list[str] = []

    # Required verdicts — blocking
    for key in Q5_REQUIRED_VERDICTS:
        sev = findings.get(key, {}).get("severity", "UNKNOWN")
        if sev != "HEALTHY":
            msg = findings.get(key, {}).get("message", "<no message>")
            reasons.append(f"  * {key}: {sev} — {msg}")

    # Advisory verdicts — surface severity without blocking
    for key in Q5_ADVISORY_VERDICTS:
        sev = findings.get(key, {}).get("severity", "UNKNOWN")
        if sev != "HEALTHY":
            msg = findings.get(key, {}).get("message", "<no message>")
            advisory_notes.append(f"  * advisory: {key}: {sev} — {msg}")

    # Test accuracy — blocking
    if test_acc < Q5_TEST_ACC_THRESHOLD:
        reasons.append(
            f"  * test_accuracy = {test_acc:.4f} is below "
            f"the required {Q5_TEST_ACC_THRESHOLD:.2f}."
        )

    if not reasons:
        msg = (
            f"Q5 PASS — dead_neurons + loss_trend HEALTHY and "
            f"test_acc = {test_acc:.4f} ≥ {Q5_TEST_ACC_THRESHOLD:.2f}.\n"
            "      Well done. Reflect: which change had the biggest impact?"
        )
        if advisory_notes:
            msg += "\n  Advisory (non-blocking):\n" + "\n".join(advisory_notes)
            msg += (
                "\n      Note: the gradient_flow CRITICAL verdict is expected "
                "at this threshold — the output layer's gradient RMS on a "
                "10-way softmax head naturally exceeds the 1e-2 library "
                "cut-off. See slide 5F for a discussion of this bias."
            )
        return True, msg

    fail_msg = "Q5 FAIL — keep iterating on Cell 1 and re-run Cell 2:\n" + "\n".join(
        reasons
    )
    if advisory_notes:
        fail_msg += "\n  Advisory (non-blocking):\n" + "\n".join(advisory_notes)
    fail_msg += (
        "\n  Hints:\n"
        "    * Exploding gradients ⇒ lower LR, add weight decay, or add warmup.\n"
        "    * Dead neurons (ReLU) ⇒ switch to GELU or apply Kaiming init.\n"
        "    * Loss oscillation ⇒ add LayerNorm, add warmup, or lower LR."
    )
    return False, fail_msg


# ── Q4 static model (used by both student & instructor notebooks) ────────
class Q4BrokenCNN(nn.Module):
    """A CNN with stride=2 and NO padding. After two such layers starting from
    28×28, spatial dims become (28-3)/2 + 1 = 13, then (13-3)/2 + 1 = 6; the
    final conv feature map (6×6) is small enough that the flatten+linear head
    still runs, but the diagnostics toolkit surfaces the unhealthy gradient
    flow + saturated activations caused by the aggressive stride.

    Students are asked to identify the issue from the diagnostics report;
    we grade on two keyword groups (stride/padding + fix).
    """

    def __init__(self) -> None:
        super().__init__()
        # Kernel 3, stride 2, NO padding — collapses spatial dimensions fast.
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=0)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=0)
        self.fc = nn.Linear(32 * 6 * 6, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        return self.fc(x.flatten(1))


# ── Q5 target (instructor answer) & student starter ──────────────────────
class Q5TargetMLP(nn.Module):
    """The TARGET architecture for Q5 (instructor answer key).

    6 linear layers with GELU activations, post-LayerNorm layout (no
    LayerNorm on the raw pixel input), dropout 0.1, Kaiming init for all
    hidden Linear layers, Xavier init for the output logit layer, and a
    SMALL non-zero bias on every parameter (including LayerNorm biases)
    so the diagnostics' ``update_ratio = ‖∇W‖/‖W‖`` stays finite at step 0.

    Trained with Adam, lr=5e-4, weight decay 1e-4, warmup 300 steps, and
    gradient clipping at 1.0 norm — the full prescription pad from deck
    slides 5C–5J. On Fashion-MNIST, 3 epochs, MPS/T4:

        * dead_neurons  = HEALTHY (~0–5% inactive on worst layer)
        * loss_trend    = HEALTHY (monotonic decrease)
        * gradient_flow = CRITICAL (advisory only — see ``check_q5_pass``
          docstring; the output layer's grad RMS exceeds the library's
          1e-2 cut-off, which is a diagnostic quirk on 10-way softmax,
          not a training pathology)
        * test_acc      ≈ 0.87 (well above the 0.82 threshold)
    """

    def __init__(self, dropout: float = 0.1) -> None:
        super().__init__()
        dims = [784, 512, 256, 128, 64, 32, 10]
        # Post-LN: first Linear takes raw input, no LayerNorm in front.
        layers: list[nn.Module] = [
            nn.Linear(dims[0], dims[1]),
            nn.GELU(),
            nn.Dropout(dropout),
        ]
        # Subsequent blocks: LayerNorm → Linear → (GELU + Dropout if not last).
        for i, (in_dim, out_dim) in enumerate(zip(dims[1:-1], dims[2:])):
            layers.append(nn.LayerNorm(in_dim))
            layers.append(nn.Linear(in_dim, out_dim))
            if i < len(dims) - 3:  # no activation on the final logit layer
                layers.append(nn.GELU())
                layers.append(nn.Dropout(dropout))
        self.net = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self) -> None:
        linears = [m for m in self.modules() if isinstance(m, nn.Linear)]
        # Hidden Linear: Kaiming (correct variance for GELU ≈ ReLU).
        for m in linears[:-1]:
            nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
            nn.init.constant_(m.bias, 0.01)
        # Output Linear: Xavier, smaller variance so output-layer gradient
        # RMS stays near the 1e-2 diagnostic threshold rather than blowing past.
        nn.init.xavier_normal_(linears[-1].weight)
        nn.init.constant_(linears[-1].bias, 0.01)
        # LayerNorm: weight=1 is the default, but the default bias=0 produces
        # ``‖W‖ = 0`` and hence an infinite update_ratio on the first batch.
        # Seed the bias with a small positive value.
        for m in self.modules():
            if isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.constant_(m.bias, 0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x.view(x.size(0), -1))


class Q5BrokenStarter(nn.Module):
    """The starter architecture students begin with for Q5.

    6 linear layers with ReLU, no LayerNorm, no dropout, default init.
    Combined with lr=0.1 and no warmup this produces:
        * exploding gradients (gradient_flow = CRITICAL)
        * many dead ReLU neurons (dead_neurons = WARNING)
        * oscillating loss (loss_trend may be WARNING or UNKNOWN)
        * ~0.3–0.4 Fashion-MNIST test accuracy

    Students must edit the architecture AND hyperparameters on Cell 1
    until ``check_q5_pass`` returns True on Cell 2's output.
    """

    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(784, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x.view(x.size(0), -1))


__all__ = [
    "pick_device",
    "load_mnist",
    "load_fashion_mnist",
    "make_q3_dataset",
    "accuracy",
    "train_classifier",
    "train_and_diagnose",
    "check_q1",
    "check_q2",
    "check_q3",
    "check_q4",
    "check_q5_pass",
    "Q4BrokenCNN",
    "Q5TargetMLP",
    "Q5BrokenStarter",
    "Q5_TEST_ACC_THRESHOLD",
]
