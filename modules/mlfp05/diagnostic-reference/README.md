# M5 Diagnostic Reference Outputs

Captured outputs from running the M5 exercise solutions with the
`DLDiagnostics` toolkit. Use these as **reference** for what a healthy
(or unhealthy) training run looks like — then compare to your own
output when you run the exercise.

## What's here

Each exercise directory contains:

- **`NN_technique_report.txt`** — the full stdout from running the
  solution, including:
  - Training loss curves printed per epoch
  - The Prescription Pad output from `diag.report()` (gradient flow,
    dead neurons, loss trend verdicts)
  - The CAPTURED OUTPUT block and STUDENT INTERPRETATION GUIDE that
    the solution file embeds as comments (these are visible in your
    terminal when you run the file)

- **`NN_technique_dashboard.html`** — interactive Plotly dashboard
  with four panels:
  1. Train / val loss curves (Stethoscope)
  2. Per-layer gradient norms (Blood Test)
  3. Activation statistics heatmap (X-Ray)
  4. Learning rate schedule (Vital Signs)

  Open in any browser — zoom, hover, compare panels.

## Which exercises are captured

Not every exercise's full output is here — some training runs exceed
10 minutes on MPS/CPU. The ones captured are the **fast-running
references**:

| Exercise            | Report  | Dashboard | Notes                     |
| ------------------- | ------- | --------- | ------------------------- |
| ex_2/01 Simple CNN  | ✓       | ✓         | CIFAR-10, full training   |
| ex_3/01 Vanilla RNN | ✓       | ✓         | Small sequence task       |
| ex_3/02 LSTM        | partial | —         | Timed out during training |
| ex_3/03 GRU         | partial | —         | Timed out during training |
| ex_6/01 GCN         | partial | —         | Cora dataset              |
| ex_6/02 GAT         | partial | —         | Cora dataset              |
| ex_6/03 GraphSAGE   | partial | —         | Cora dataset              |

For exercises not captured here (all of ex_1 autoencoders, ex_4
transformers, ex_5 GANs, ex_7 transfer learning, ex_8 RL): open the
corresponding `modules/mlfp05/solutions/ex_N/NN_technique.py` file
and scroll to the `# ══════ EXPECTED OUTPUT ══════` block — that
shows what the Prescription Pad should look like when you run it.

## How to use these

**Before running your own version** (student exercise under `local/`):

1. Open the dashboard HTML in the matching `ex_N/` folder — see what
   a healthy training run looks like for this technique
2. Read the `_report.txt` — note the Prescription Pad verdicts and
   the interpretation guide

**While running your own version:**

1. Run `.venv/bin/python modules/mlfp05/local/ex_N/NN_technique.py`
2. Compare your `diag.report()` output to the reference `_report.txt`
3. If your verdicts differ, consult the interpretation guide in the
   solution file — it explains what each finding means and how to
   fix it

**If your results differ from the reference:**

That is expected — random seeds, hardware (MPS vs CUDA vs CPU), and
per-run stochasticity produce different exact numbers. What should
match is the **shape** of the diagnosis:

- Standard AE (ex_1/01): you should see vanishing gradients at
  `decoder.2.weight` and 50%+ dead neurons at `decoder.1`
- Vanilla RNN (ex_3/01): you should see vanishing gradients through
  time, a clear signal in the Blood Test panel
- GAT (ex_6/02): you should see attention head entropy low enough
  to trigger the warning

The specific numbers differ; the pathology pattern should be the same.

## Regenerating these outputs

```bash
# Run the diagnostic capture script (takes ~30-60 min for the quick set)
.venv/bin/python /tmp/save_diag_outputs.py
```

(Script lives in the source repo; not part of student materials.)
