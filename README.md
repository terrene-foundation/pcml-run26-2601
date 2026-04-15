# PCML Run 26 — Student Repository

This is your **personal fork** of the PCML Run 26 course materials. Everything
below applies to your fork; the instructor's repo at
`https://github.com/pcml-run26/pcml-run26-2601` is the "upstream" you pull
updates from.

---

## What's in this repo

| Path                                   | What it is                                                                       |
| -------------------------------------- | -------------------------------------------------------------------------------- |
| `modules/mlfpNN/readings/deck.pdf`     | Full slide deck (read this first for each module)                                |
| `modules/mlfpNN/readings/textbook.pdf` | Long-form textbook (deep reading)                                                |
| `modules/mlfpNN/readings/notes.pdf`    | Condensed notes (revision)                                                       |
| `modules/mlfpNN/local/ex_N/`           | Your exercise files (edit these)                                                 |
| `modules/mlfpNN/colab/ex_N/`           | Colab notebook versions of the exercises                                         |
| `modules/mlfpNN/solutions/ex_N/`       | **Reference solutions** — consult AFTER attempting your own                      |
| `modules/mlfpNN/diagnostic-reference/` | Captured diagnostic outputs (plots + reports) for reference                      |
| `shared/`                              | Helper code the exercises import (diagnostics, data loaders, etc.) — do not edit |
| `data/`                                | Small datasets (large ones auto-download on first run)                           |

---

## One-time setup (VSCode terminal)

Open the VSCode integrated terminal (`` Ctrl+` `` or `` Cmd+` ``) and paste:

```bash
# 1. Confirm you're inside YOUR fork
git remote -v

# 2. Add the instructor's repo as "upstream" — copy this ONE line
git remote add upstream https://github.com/pcml-run26/pcml-run26-2601.git

# 3. Verify both remotes
git remote -v
# origin    → your fork
# upstream  → pcml-run26/pcml-run26-2601
```

That's it — setup done once per machine.

---

## Pull the latest from your instructor (do every class)

In the VSCode terminal:

```bash
# Save your own work first
git add -A && git commit -m "wip" || true

# Fetch + merge from instructor
git fetch upstream && git merge upstream/main

# Push the merged result back to your fork
git push origin main
```

If you see merge conflicts: VSCode highlights them with **Accept Incoming /
Current / Both** buttons in each conflicted file. Choose:

- **Accept Incoming** for any files in `shared/**`, `pyproject.toml`,
  `uv.lock`, `data/**` (these are instructor-managed)
- **Accept Current** or resolve by hand for files YOU edited (usually your
  exercise solutions under `modules/mlfpNN/local/`)

---

## Running exercises locally

```bash
# One-time — install dependencies
uv sync

# Run an exercise
.venv/bin/python modules/mlfp05/local/ex_1/01_standard_ae.py
```

---

## Running exercises in Google Colab

Each exercise has a matching `.ipynb` notebook under
`modules/mlfpNN/colab/ex_N/`. Open it in
[Google Colab](https://colab.research.google.com):

### The Colab setup cell — edit this ONCE per notebook

The first cell of every Colab notebook looks like this:

```python
# ════════════════════════════════════════════════════════════════
# Google Colab setup — clones YOUR GitHub Classroom fork
# ════════════════════════════════════════════════════════════════
import os, sys

# ① EDIT THIS to point at YOUR fork of the Classroom repo.
FORK_URL = "https://github.com/<your-github-username>/<your-fork>.git"
REPO_DIR = "/content/pcml-run26"

if not os.path.exists(REPO_DIR):
    !git clone {FORK_URL} {REPO_DIR}

# ② cd into the repo so relative data paths resolve
%cd {REPO_DIR}

# ③ Install deps
!pip install -q kailash kailash-ml polars plotly python-dotenv

# ④ Make the `shared` package importable
if REPO_DIR not in sys.path:
    sys.path.insert(0, REPO_DIR)

print("✓ Colab setup complete — shared.mlfp05 is importable")
```

**What to change**: the line `FORK_URL = "https://github.com/<your-github-username>/<your-fork>.git"`.

Replace it with your actual fork URL. Example if your username is `janedoe`:

```python
FORK_URL = "https://github.com/janedoe/pcml-run26-2601-janedoe.git"
```

After editing, run the cell. It will:

1. Clone your fork to `/content/pcml-run26`
2. `cd` into the repo
3. Install the required packages
4. Add the repo to `sys.path` so `from shared.mlfp05.diagnostics import DLDiagnostics` works
5. Print `✓ Colab setup complete`

Then run the rest of the notebook normally.

### Pull updates from instructor inside Colab

When you open Colab later after an instructor update, cell 0 clones from your
fork. To pull from upstream inside Colab, run this in a new cell:

```python
!cd /content/pcml-run26 && git fetch upstream && git merge upstream/main
```

(This requires you to have set up `upstream` in your fork first — see the
one-time setup above, then push that change to your fork with
`git push origin main`.)

---

## Where to find the diagnostic outputs

Module 5 exercises use the `DLDiagnostics` toolkit to audit your trained
models. You'll see its output in three places:

1. **Inline in each exercise file** — search for
   `# ══════ EXPECTED OUTPUT ══════` to see what a healthy (or unhealthy)
   diagnostic report looks like
2. **In your terminal/notebook output** — when you run the exercise, the
   `diag.report()` call prints the live Prescription Pad
3. **In `modules/mlfp05/diagnostic-reference/`** — captured reports and
   interactive Plotly dashboards from reference runs. Open the `.html` files
   in any browser to see a healthy training run's 4-panel dashboard.

To see the live dashboard for **your own** run, change `show=False` to
`show=True` in the diagnostic call, or use:

```python
diag.plot_training_dashboard().show()
```

---

## Help / Troubleshooting

| Problem                              | Fix                                                                                                |
| ------------------------------------ | -------------------------------------------------------------------------------------------------- |
| `ModuleNotFoundError: shared.mlfp05` | Locally: run `uv sync`. In Colab: check you edited `FORK_URL` correctly in cell 0                  |
| Merge conflicts on `git pull`        | Keep your answers; take instructor's version for files in `shared/**`, `pyproject.toml`, `uv.lock` |
| "Authentication failed" on push      | Use a GitHub Personal Access Token, or set up `gh auth login`                                      |
| Colab says "FORK_URL invalid"        | You haven't edited cell 0 — replace `<your-github-username>/<your-fork>` with your actual values   |
| PDF looks stale after `git pull`     | PDFs are binary — git pulls them correctly. If you opened one before pulling, close & reopen       |

---

## Support

- Questions about exercises: post in the class Slack / LMS
- Bug in course materials: open an issue on the instructor repo
- Can't resolve a merge conflict: ask your TA or paste the conflict into Slack
