# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# MLFP05 — Exercise 3.5: Architecture Comparison — RNN vs LSTM vs GRU vs Attention
# ════════════════════════════════════════════════════════════════════════
#
# WHAT YOU'LL LEARN:
#   After completing this section, you will be able to:
#   - Train all four architectures under identical conditions for fair comparison
#   - Compare accuracy, parameter efficiency, gradient health, and latency
#   - Run multi-stock generalisation tests (does the best model generalise?)
#   - Generate a comprehensive comparison dashboard with visual evidence
#   - Make architecture selection decisions for different business scenarios
#   - Register the overall best model in ModelRegistry
#
# PREREQUISITES: 01-04 (all four architecture files).
# ESTIMATED TIME: ~25-30 min
#
# DATASET: STI + APAC/global stocks via yfinance (2010-2024).
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations
import time
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from shared.mlfp05.ex_3 import (
    BATCH_SIZE,
    CLIP,
    EPOCHS,
    FEATURES,
    FORECAST_HORIZON,
    HIDDEN_DIM,
    LR,
    OUTPUT_DIR,
    SEQ_LEN,
    TICKERS,
    build_dataset,
    init_environment,
    load_stock_data,
    prepare_dataloaders,
    setup_engines,
    train_model,
    register_best_model,
    get_visualizer,
)
# ════════════════════════════════════════════════════════════════════════
# THEORY — Choosing the Right Sequence Architecture
# ════════════════════════════════════════════════════════════════════════
#
# By now you have built four sequence architectures. Here is when to
# use each one in production:
#
# ┌──────────────────┬─────────────────────────────────────────────────┐
# │ Architecture     │ Best For                                       │
# ├──────────────────┼─────────────────────────────────────────────────┤
# │ Vanilla RNN      │ Short sequences (<10 steps), simple patterns,  │
# │                  │ tiny models for edge/mobile deployment          │
# ├──────────────────┼─────────────────────────────────────────────────┤
# │ LSTM             │ Long sequences (20-200 steps), complex          │
# │                  │ dependencies, when you need fine-grained        │
# │                  │ memory control via forget/input/output gates    │
# ├──────────────────┼─────────────────────────────────────────────────┤
# │ GRU              │ Same tasks as LSTM but when latency or model    │
# │                  │ size matters. ~75% of LSTM parameters,          │
# │                  │ measurably faster inference.                    │
# ├──────────────────┼─────────────────────────────────────────────────┤
# │ LSTM+Attention   │ When you need explainability (which timesteps   │
# │                  │ matter?) or when different parts of the         │
# │                  │ sequence are unequally important. Healthcare,   │
# │                  │ finance, any domain with "trigger events."      │
# └──────────────────┴─────────────────────────────────────────────────┘
#
# For sequences longer than ~200 steps, Transformers (Exercise 4)
# replace all of the above with parallel self-attention.
# ════════════════════════════════════════════════════════════════════════
device = init_environment()
# ════════════════════════════════════════════════════════════════════════
# TASK 1 — Load data and set up experiment tracking
# ════════════════════════════════════════════════════════════════════════
stock_data, PRIMARY, primary_df = load_stock_data()
(
    train_loader,
    val_loader,
    X_train_t,
    y_train_t,
    X_val_t,
    y_val_t,
    norm_mean,
    norm_std,
    n_train_w,
    N_FEATURES,
) = prepare_dataloaders(primary_df, device)
conn, tracker, exp_name, registry, has_registry = setup_engines(
    PRIMARY, experiment_suffix="comparison"
)
# ── Checkpoint 1 ─────────────────────────────────────────────────────
assert X_train_t.shape[1] == SEQ_LEN
assert tracker is not None
print("--- Checkpoint 1 passed --- data and tracking ready\n")
# ════════════════════════════════════════════════════════════════════════
# TASK 2 — Define all four architectures
# ════════════════════════════════════════════════════════════════════════
class VanillaRNN(nn.Module):
    def __init__(
        self, input_dim: int, hidden_dim: int, horizon: int = FORECAST_HORIZON
    ):
        super().__init__()
        self.rnn = nn.RNN(input_dim, hidden_dim, batch_first=True, nonlinearity="tanh")
        self.head = nn.Linear(hidden_dim, horizon)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.rnn(x)
        return self.head(out[:, -1])
class LSTMRegressor(nn.Module):
    def __init__(
        self, input_dim: int, hidden_dim: int, horizon: int = FORECAST_HORIZON
    ):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.head = nn.Linear(hidden_dim, horizon)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return self.head(out[:, -1])
class GRURegressor(nn.Module):
    def __init__(
        self, input_dim: int, hidden_dim: int, horizon: int = FORECAST_HORIZON
    ):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.head = nn.Linear(hidden_dim, horizon)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.gru(x)
        return self.head(out[:, -1])
class TemporalAttention(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.W = nn.Linear(hidden_dim, hidden_dim)
        self.v = nn.Linear(hidden_dim, 1, bias=False)
    def forward(self, lstm_outputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        energy = torch.tanh(self.W(lstm_outputs))
        scores = self.v(energy).squeeze(-1)
        weights = torch.softmax(scores, dim=-1)
        context = torch.bmm(weights.unsqueeze(1), lstm_outputs).squeeze(1)
        return context, weights
class LSTMWithAttention(nn.Module):
    def __init__(
        self, input_dim: int, hidden_dim: int, horizon: int = FORECAST_HORIZON
    ):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.attention = TemporalAttention(hidden_dim)
        self.head = nn.Linear(hidden_dim, horizon)
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        lstm_out, _ = self.lstm(x)
        context, attn_weights = self.attention(lstm_out)
        pred = self.head(context)
        return pred, attn_weights
# Instantiate all four
models = {
    "VanillaRNN": VanillaRNN(input_dim=N_FEATURES, hidden_dim=HIDDEN_DIM),
    "LSTM": LSTMRegressor(input_dim=N_FEATURES, hidden_dim=HIDDEN_DIM),
    "GRU": GRURegressor(input_dim=N_FEATURES, hidden_dim=HIDDEN_DIM),
    "LSTM+Attention": LSTMWithAttention(input_dim=N_FEATURES, hidden_dim=HIDDEN_DIM),
}
is_attn = {"VanillaRNN": False, "LSTM": False, "GRU": False, "LSTM+Attention": True}
param_counts = {}
for name, model in models.items():
    n_params = sum(p.numel() for p in model.parameters())
    param_counts[name] = n_params
    print(f"  {name}: {n_params:,} parameters")
# ── Checkpoint 2 ─────────────────────────────────────────────────────
assert len(models) == 4
assert param_counts["GRU"] < param_counts["LSTM"]
print("--- Checkpoint 2 passed --- all four architectures defined\n")
# ════════════════════════════════════════════════════════════════════════
# TASK 3 — Train all four architectures under identical conditions
# ════════════════════════════════════════════════════════════════════════
print(f"\n== Training all four on {PRIMARY} (identical conditions) ==")
all_results = {}
for name, model in models.items():
    print(f"\n--- {name} ---")
    results = train_model(
        model,
        name,
        tracker,
        exp_name,
        train_loader,
        val_loader,
        device,
        attn=is_attn[name],
    )
    all_results[name] = results
    # Per-architecture diagnostic — the comparison is the teaching
    # moment: VanillaRNN should light up CRITICAL while LSTM/GRU/
    # Attention stay HEALTHY on the same task and identical data.
    from kailash_ml import diagnose
    print(f"  ── Diagnostic Report ({name}) ──")
    report = diagnose(
        model,
        kind="dl",
        data=val_loader,
        show=False,
    )
    # ══════ EXPECTED OUTPUT (synthesized reference — side-by-side across 4 architectures) ══════
    # ┌────────────────┬──────────────┬──────────────┬──────────────┬──────────────┐
    # │ Architecture   │ VanillaRNN   │ GRU          │ LSTM         │ Attention    │
    # ├────────────────┼──────────────┼──────────────┼──────────────┼──────────────┤
    # │ Blood Test     │ [X] CRITICAL │ [✓] HEALTHY  │ [✓] HEALTHY  │ [✓] HEALTHY  │
    # │   min RMS      │ 8.2e-07      │ 2.7e-04      │ 2.4e-04      │ 4.1e-04      │
    # ├────────────────┼──────────────┼──────────────┼──────────────┼──────────────┤
    # │ Saturation     │ [!] tanh 0.99│ [✓]          │ [✓]          │ [✓]          │
    # ├────────────────┼──────────────┼──────────────┼──────────────┼──────────────┤
    # │ Stethoscope    │ [!] slow/    │ [✓] -3.1e-03 │ [✓] -2.8e-03 │ [✓] -3.6e-03 │
    # │   slope        │   oscillates │   /epoch     │   /epoch     │   /epoch     │
    # ├────────────────┼──────────────┼──────────────┼──────────────┼──────────────┤
    # │ Final val loss │ ~3.8         │ ~1.3         │ ~1.4         │ ~0.95        │
    # │ Final val RMSE │ ~28 μg/m³    │ ~10 μg/m³    │ ~10 μg/m³    │ ~8 μg/m³     │
    # └────────────────┴──────────────┴──────────────┴──────────────┴──────────────┘
    #
    # STUDENT INTERPRETATION GUIDE — reading the comparison:
    #
    #  [BLOOD TEST — THE HISTORICAL ARC] The min RMS column is the
    #     single most important comparison in the table. 8.2e-07 →
    #     2.7e-04 is a THOUSAND-FOLD gradient preservation improvement
    #     between VanillaRNN and GRU. This IS the vanishing-gradient
    #     fix that Hochreiter posed in 1991 and that Cho (GRU) and
    #     Bengio (LSTM) solved in 2014. Slide 5T frames it: "1991-2014
    #     = 23 years between problem identification and scalable
    #     solution. The diagnostic instruments let you SEE the fix
    #     happen across architectures in a single afternoon."
    #     >> Prescription: Any time-series task with sequences >30
    #        steps MUST start with a gated architecture. VanillaRNN is
    #        a pedagogical tool, not a production option.
    #
    #  [SATURATION — TANH COLLAPSE] Only VanillaRNN shows saturation.
    #     Gated architectures use sigmoid gates to MODULATE tanh
    #     rather than to produce final output, so saturation occurs
    #     only at gate extremes (which is informative, not
    #     pathological). In contrast, VanillaRNN's output tanh
    #     saturates because there's no gating — all the hidden state
    #     flows through one tanh, which drives to ±1 on any strong
    #     signal.
    #     >> Prescription: No architectural fix — VanillaRNN
    #        fundamentally cannot be rescued on PM2.5 sequences.
    #        Choose GRU or LSTM.
    #
    #  [STETHOSCOPE — SLOPE COMPARISON] All three gated architectures
    #     converge at similar rates (~-3e-3/epoch). Attention is
    #     fastest (-3.6e-3) because the decoder gets richer per-step
    #     signal from attending to all timesteps. VanillaRNN's
    #     OSCILLATING loss (not just slow) is the pattern diagnostic:
    #     gradient explosions alternating with vanishing means
    #     gradient clipping is masking, not fixing, the underlying
    #     issue.
    #     >> Prescription: An oscillating training curve on a
    #        sequence task is never a tuning problem — it is an
    #        architecture problem.
    #
    #  [RMSE — BUSINESS IMPACT] For Singapore PM2.5 forecasting:
    #     28 μg/m³ error (VanillaRNN) means the model predicts
    #     "moderate air quality" when reality is "unhealthy" and
    #     vice versa — CLASSIFICATION FAILURES that misinform
    #     public-health messaging. 8 μg/m³ (Attention) stays WITHIN
    #     one air-quality band: predictions are actionable.
    #     >> Prescription: On public-health-adjacent tasks, accuracy
    #        is a population-health KPI. A 3x RMSE reduction (28 →
    #        8) means 3x fewer misleading alerts per year.
    #
    #  FIVE-INSTRUMENT TAKEAWAY: the comparison table IS the
    #  pedagogical payload. Four architectures, same data, same
    #  instruments — the Prescription Pad reveals WHY each wins or
    #  loses. Use this pattern throughout M5: any time you compare
    #  architectures, line them up on the five instruments, not just
    #  on final accuracy.
    # ═════════════════════════════════════════════════════════════════════
# ── Checkpoint 3 ─────────────────────────────────────────────────────
for name, res in all_results.items():
    assert len(res["train_losses"]) == EPOCHS, f"{name} should have {EPOCHS} epochs"
    assert res["final_val_loss"] < 5.0, f"{name} val loss suspiciously high"
print("\n--- Checkpoint 3 passed --- all four architectures trained\n")
# ════════════════════════════════════════════════════════════════════════
# TASK 4 — Comprehensive comparison table
# ════════════════════════════════════════════════════════════════════════
print("=" * 80)
print(f"  {'Model':<18s} {'Params':>8s} {'Train':>8s} {'Val':>8s} {'GradNorm':>10s}")
print("-" * 80)
for name, res in all_results.items():
    print(
        f"  {name:<18s} {param_counts[name]:>8,d} "
        f"{res['train_losses'][-1]:>8.4f} {res['final_val_loss']:>8.4f} "
        f"{np.mean(res['gradient_norms']):>10.4f}"
    )
print("=" * 80)
best_name = min(all_results, key=lambda k: all_results[k]["final_val_loss"])
best_val = all_results[best_name]["final_val_loss"]
print(f"\n  Best model: {best_name} (val_loss={best_val:.4f})")
# ════════════════════════════════════════════════════════════════════════
# TASK 5 — Inference latency benchmark
# ════════════════════════════════════════════════════════════════════════
def _predict_for_bench(model, x, attn=False):
    out = model(x)
    return out[0] if attn else out
def benchmark_inference(
    model: nn.Module, name: str, attn: bool, n_runs: int = 200
) -> float:
    model.eval()
    test_input = torch.randn(1, SEQ_LEN, N_FEATURES, device=device)
    with torch.no_grad():
        for _ in range(20):
            _predict_for_bench(model, test_input, attn)
    if device.type == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    with torch.no_grad():
        for _ in range(n_runs):
            _predict_for_bench(model, test_input, attn)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed_ms = (time.perf_counter() - start) / n_runs * 1000
    return elapsed_ms
print("\n== Inference Latency ==")
latencies = {}
for name, model in models.items():
    lat = benchmark_inference(model, name, is_attn[name])
    latencies[name] = lat
    print(f"  {name}: {lat:.3f} ms/inference")
fastest = min(latencies, key=latencies.get)
print(f"  Fastest: {fastest} ({latencies[fastest]:.3f} ms)")
# ════════════════════════════════════════════════════════════════════════
# TASK 6 — Multi-stock generalisation test
# ════════════════════════════════════════════════════════════════════════
print(f"\n== Multi-Stock Generalisation ({best_name}) ==")
multi_stock_results: dict[str, float] = {PRIMARY: best_val}
for symbol, sdf in stock_data.items():
    if symbol == PRIMARY or len(sdf) < SEQ_LEN + FORECAST_HORIZON + 50:
        continue
    X_s, y_s, _, _, sp = build_dataset(sdf, SEQ_LEN, FORECAST_HORIZON)
    ldr = DataLoader(
        TensorDataset(
            torch.from_numpy(X_s[:sp]).to(device),
            torch.from_numpy(y_s[:sp]).to(device),
        ),
        batch_size=BATCH_SIZE,
        shuffle=True,
    )
    ldr_v = DataLoader(
        TensorDataset(
            torch.from_numpy(X_s[sp:]).to(device),
            torch.from_numpy(y_s[sp:]).to(device),
        ),
        batch_size=BATCH_SIZE,
    )
    # Train the best architecture on this stock
    if best_name == "LSTM+Attention":
        m = LSTMWithAttention(input_dim=N_FEATURES, hidden_dim=HIDDEN_DIM).to(device)
        attn_flag = True
    elif best_name == "GRU":
        m = GRURegressor(input_dim=N_FEATURES, hidden_dim=HIDDEN_DIM).to(device)
        attn_flag = False
    elif best_name == "LSTM":
        m = LSTMRegressor(input_dim=N_FEATURES, hidden_dim=HIDDEN_DIM).to(device)
        attn_flag = False
    else:
        m = VanillaRNN(input_dim=N_FEATURES, hidden_dim=HIDDEN_DIM).to(device)
        attn_flag = False
    opt = torch.optim.Adam(m.parameters(), lr=LR)
    for _ in range(8):
        m.train()
        for xb, yb in ldr:
            opt.zero_grad()
            pred = _predict_for_bench(m, xb, attn_flag)
            F.mse_loss(pred, yb).backward()
            nn.utils.clip_grad_norm_(m.parameters(), max_norm=CLIP)
            opt.step()
    m.eval()
    with torch.no_grad():
        vl = float(
            np.mean(
                [
                    F.mse_loss(_predict_for_bench(m, xb, attn_flag), yb).item()
                    for xb, yb in ldr_v
                ]
            )
        )
    multi_stock_results[symbol] = vl
    print(f"  {symbol} ({TICKERS[symbol]}): val_loss={vl:.4f}")
avg_cross_stock = np.mean(list(multi_stock_results.values()))
print(f"\n  Average cross-stock val loss: {avg_cross_stock:.4f}")
# ── Checkpoint 4 ─────────────────────────────────────────────────────
assert len(multi_stock_results) >= 2, "Need multi-stock results"
print("--- Checkpoint 4 passed --- multi-stock generalisation complete\n")
# ════════════════════════════════════════════════════════════════════════
# TASK 7 — Comprehensive comparison dashboard
# ════════════════════════════════════════════════════════════════════════
viz = get_visualizer()
# 7A: Training curves (all models overlaid)
train_metrics = {}
for label, res in all_results.items():
    train_metrics[f"{label} train"] = res["train_losses"]
    train_metrics[f"{label} val"] = res["val_losses"]
viz.training_history(
    metrics=train_metrics, x_label="Epoch", y_label="MSE Loss"
).write_html(str(OUTPUT_DIR / "05_comparison_training_curves.html"))
# 7B: Gradient norms (all models overlaid)
grad_metrics = {k: v["gradient_norms"] for k, v in all_results.items()}
viz.training_history(
    metrics=grad_metrics, x_label="Epoch", y_label="Gradient L2 Norm"
).write_html(str(OUTPUT_DIR / "05_comparison_gradient_norms.html"))
# 7C: Prediction vs actual for best model
best_model = models[best_name]
best_model.eval()
with torch.no_grad():
    if is_attn[best_name]:
        val_preds, val_attn_weights = best_model(X_val_t)
    else:
        val_preds = best_model(X_val_t)
        val_attn_weights = None
close_mean, close_std = norm_mean[0, 0], norm_std[0, 0]
preds_denorm = val_preds.cpu().numpy() * close_std + close_mean
actual_denorm = y_val_t.cpu().numpy() * close_std + close_mean
pred_df = pl.DataFrame(
    {"actual": actual_denorm[:, 0].tolist(), "predicted": preds_denorm[:, 0].tolist()}
)
viz.scatter(pred_df, x="actual", y="predicted").write_html(
    str(OUTPUT_DIR / "05_comparison_pred_vs_actual.html")
)
# 7D: Comprehensive comparison figure (matplotlib)
fig, axes = plt.subplots(2, 3, figsize=(20, 12))
model_names = list(all_results.keys())
colors = {
    "VanillaRNN": "#F44336",
    "LSTM": "#2196F3",
    "GRU": "#4CAF50",
    "LSTM+Attention": "#FF9800",
}
# Panel 1: Val loss over epochs
ax = axes[0, 0]
for name in model_names:
    ax.plot(
        range(1, EPOCHS + 1),
        all_results[name]["val_losses"],
        color=colors[name],
        linewidth=2,
        label=name,
    )
ax.set_xlabel("Epoch")
ax.set_ylabel("Validation MSE Loss")
ax.set_title("Learning Curves")
ax.legend()
ax.grid(True, alpha=0.3)
# Panel 2: Gradient norms over epochs
ax = axes[0, 1]
for name in model_names:
    ax.plot(
        range(1, EPOCHS + 1),
        all_results[name]["gradient_norms"],
        color=colors[name],
        linewidth=2,
        label=name,
    )
ax.set_xlabel("Epoch")
ax.set_ylabel("Gradient L2 Norm")
ax.set_title("Gradient Health")
ax.legend()
ax.grid(True, alpha=0.3)
# Panel 3: Parameter count vs val loss (efficiency frontier)
ax = axes[0, 2]
for name in model_names:
    ax.scatter(
        param_counts[name],
        all_results[name]["final_val_loss"],
        color=colors[name],
        s=200,
        zorder=5,
        edgecolors="white",
        linewidth=2,
    )
    ax.annotate(
        name,
        (param_counts[name], all_results[name]["final_val_loss"]),
        textcoords="offset points",
        xytext=(10, 5),
        fontsize=9,
    )
ax.set_xlabel("Parameter Count")
ax.set_ylabel("Final Val Loss")
ax.set_title("Efficiency: Parameters vs Accuracy")
ax.grid(True, alpha=0.3)
# Panel 4: Latency comparison
ax = axes[1, 0]
bars = ax.bar(
    model_names,
    [latencies[n] for n in model_names],
    color=[colors[n] for n in model_names],
    edgecolor="white",
)
ax.set_ylabel("Inference Latency (ms)")
ax.set_title("Inference Speed")
for bar, name in zip(bars, model_names):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.005,
        f"{latencies[name]:.3f}ms",
        ha="center",
        va="bottom",
        fontsize=9,
        fontweight="bold",
    )
ax.grid(True, alpha=0.3, axis="y")
# Panel 5: Predicted vs actual time series (best model)
ax = axes[1, 1]
n_show = 150
ax.plot(
    range(n_show),
    actual_denorm[:n_show, 0],
    label="Actual",
    color="#2196F3",
    linewidth=1.5,
)
ax.plot(
    range(n_show),
    preds_denorm[:n_show, 0],
    label=f"{best_name} Predicted",
    color=colors[best_name],
    linewidth=1.5,
    linestyle="--",
    alpha=0.85,
)
ax.set_xlabel("Validation Window Index")
ax.set_ylabel("Close Price")
ax.set_title(f"Best Model ({best_name}): Predicted vs Actual")
ax.legend()
ax.grid(True, alpha=0.3)
# Panel 6: Multi-stock generalisation
ax = axes[1, 2]
stock_names = list(multi_stock_results.keys())
stock_losses = [multi_stock_results[s] for s in stock_names]
short_names = [
    s.replace("^", "")
    .replace(".SI", "")
    .replace(".HK", "")
    .replace(".KS", "")
    .replace(".T", "")[:6]
    for s in stock_names
]
bar_colors = [colors[best_name] if s == PRIMARY else "#78909C" for s in stock_names]
bars = ax.bar(short_names, stock_losses, color=bar_colors, edgecolor="white")
ax.axhline(
    y=avg_cross_stock,
    color="#333",
    linestyle="--",
    linewidth=1,
    label=f"Avg={avg_cross_stock:.4f}",
)
ax.set_ylabel("Validation Loss")
ax.set_title(f"Multi-Stock Generalisation ({best_name})")
ax.legend()
ax.grid(True, alpha=0.3, axis="y")
fig.suptitle(
    f"Architecture Comparison: RNN vs LSTM vs GRU vs Attention on {PRIMARY}",
    fontsize=15,
    fontweight="bold",
)
fig.tight_layout()
fig.savefig(str(OUTPUT_DIR / "05_comparison_dashboard.png"), dpi=150)
plt.close(fig)
print("  Saved: 05_comparison_dashboard.png")
# 7E: Horizon error by day for all models
print("\n== Forecast Error by Horizon Day (all models) ==")
print(f"  {'Day':<6s}", end="")
for name in model_names:
    print(f"  {name:>16s}", end="")
print()
all_horizon_rmses = {}
for name in model_names:
    model = models[name]
    model.eval()
    with torch.no_grad():
        if is_attn[name]:
            vp, _ = model(X_val_t)
        else:
            vp = model(X_val_t)
    vp_denorm = vp.cpu().numpy() * close_std + close_mean
    rmses = []
    for day in range(FORECAST_HORIZON):
        rmse = float(np.mean((vp_denorm[:, day] - actual_denorm[:, day]) ** 2)) ** 0.5
        rmses.append(rmse)
    all_horizon_rmses[name] = rmses
for day in range(FORECAST_HORIZON):
    print(f"  Day {day+1:<3d}", end="")
    for name in model_names:
        print(f"  {all_horizon_rmses[name][day]:>16.2f}", end="")
    print()
# ── Checkpoint 5 ─────────────────────────────────────────────────────
assert (OUTPUT_DIR / "05_comparison_dashboard.png").exists()
assert (OUTPUT_DIR / "05_comparison_training_curves.html").exists()
assert (OUTPUT_DIR / "05_comparison_pred_vs_actual.html").exists()
print("\n--- Checkpoint 5 passed --- comparison dashboard generated\n")
# ════════════════════════════════════════════════════════════════════════
# TASK 8 — Register overall best model
# ════════════════════════════════════════════════════════════════════════
register_best_model(
    models[best_name],
    best_name,
    best_val,
    PRIMARY,
    registry,
    has_registry,
)
# ── Checkpoint 6 ─────────────────────────────────────────────────────
assert best_val < 5.0, "Best model val loss should be reasonable"
print("--- Checkpoint 6 passed --- best model registered\n")
# ════════════════════════════════════════════════════════════════════════
# TASK 9 — Architecture selection guide (decision framework)
# ════════════════════════════════════════════════════════════════════════
print("=" * 80)
print("  ARCHITECTURE SELECTION GUIDE")
print("=" * 80)
print(
    f"""
  Based on the experiments above, here is a decision framework:
  QUESTION 1: How long are your sequences?
    < 10 steps  -> VanillaRNN (simplest, fastest, adequate)
    10-200 steps -> LSTM or GRU (gates preserve long-range dependencies)
    > 200 steps  -> Transformer (Exercise 4 — parallel self-attention)
  QUESTION 2: Does latency matter?
    Yes (real-time, sensor, trading) -> GRU ({latencies['GRU']:.3f}ms vs LSTM {latencies['LSTM']:.3f}ms)
    No (batch, offline, training)    -> LSTM or LSTM+Attention
  QUESTION 3: Do you need explainability?
    Yes (healthcare, finance, regulated) -> LSTM+Attention (attention heatmaps)
    No (internal tool, non-regulated)    -> LSTM or GRU (simpler)
  QUESTION 4: How much data do you have?
    Small dataset (<1K sequences) -> GRU (fewer parameters, less overfitting)
    Large dataset (>10K sequences) -> LSTM+Attention (can leverage capacity)
  Results from THIS experiment on {PRIMARY}:
    Best accuracy:     {best_name} (val={best_val:.4f})
    Most efficient:    GRU ({param_counts['GRU']:,} params)
    Fastest inference: {fastest} ({latencies[fastest]:.3f}ms)
    Best gradient flow: LSTM or LSTM+Attention (gated architectures)
"""
)
# ════════════════════════════════════════════════════════════════════════
# DESTINATION-FIRST CLOSE — km.diagnose
# ════════════════════════════════════════════════════════════════════════
# This lesson walked the journey of recurrent architectures — VanillaRNN,
# LSTM, GRU, LSTM+Attention — each with its own training loop, gradient
# norm tracking, and benchmark grid. The kailash-ml SDK ships a
# single-call diagnostic primitive that closes the production loop:
# km.diagnose inspects a trained model and emits an auto-dashboard
# (loss curves, gradient flow, dead neurons, activation stats, weight
# distributions). One cell. Every diagnostic students would otherwise
# hand-roll, ready to surface in a Plotly dashboard.
from kailash_ml import diagnose
# `kind='auto'` dispatches by model type — DLDiagnostics for torch.nn.Module.
# `data=` accepts any iterable yielding tensors; we reuse val_loader.
report = diagnose(best_model, kind="auto", data=val_loader, show=False)
report.plot_training_dashboard()
print()
print("km.diagnose: 1 line of code -> the same observability the lesson")
print("body hand-rolled in 200+ lines. This is what 'destination-first'")
print("means — when the journey is internalised, the SDK is one call.")
# ══════════════════════════════════════════════════════════════════════
# REFLECTION
# ══════════════════════════════════════════════════════════════════════
print("=" * 70)
print("  WHAT YOU'VE MASTERED — EXERCISE 3 COMPLETE")
print("=" * 70)
print(
    f"""
  [x] Loaded {sum(len(df) for df in stock_data.values()):,} days across {len(stock_data)} tickers
  [x] Built VanillaRNN, LSTM, GRU, and LSTM+Attention in torch.nn
  [x] Wrote LSTM gate equations as vectorised torch operations
  [x] Multi-step forecasting: {SEQ_LEN}-day window -> next {FORECAST_HORIZON} days
  [x] Tracked every variant with ExperimentTracker (per-epoch loss + grad norms)
  [x] Vanishing gradients: demonstrated and explained with visual evidence
  [x] Temporal attention: learnable focus over past timesteps (preview of M5.4)
  [x] Best model ({best_name}) registered in ModelRegistry
  [x] Multi-stock generalisation across {len(multi_stock_results)} tickers
  [x] Architecture selection guide for real-world decision making
  [x] Applied to Singapore business scenarios:
      - Ya Kun Kaya Toast: F&B demand forecasting (RNN)
      - SGX/DBS: Equity forecasting with prediction intervals (LSTM)
      - SMRT: Predictive maintenance for trains (GRU)
      - SGH: Clinical deterioration prediction with explainability (Attention)
  Key insight: There is no single "best" architecture. The right choice
  depends on sequence length, latency requirements, explainability needs,
  and data volume. RNNs fail on long sequences. LSTMs fix this with
  additive cell-state updates. GRUs match with fewer parameters. Attention
  lets the model choose which past steps matter. Error compounds across
  the forecast horizon.
  This exercise teaches architectures, not market timing.
  Next: Exercise 4 — Transformers replace recurrence with pure attention.
"""
)
