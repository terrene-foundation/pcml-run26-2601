# MLFP05 Quiz — Deep Learning Mastery

The MLFP05 quiz certifies that a learner can **build**, **debug**, and **prescribe fixes** for deep networks. It is designed for a free Colab T4 session and a single 90-minute sitting.

## Structure

Five questions, each scored pass / fail. Four of them test **build** (you write the model); one of them tests **prescribe** (you fix a broken model using the DL Diagnostics toolkit taught in Lesson 5 and summarised on deck slides 5C–5J).

| #   | Skill                         | Dataset          | Threshold                         | Time      |
| --- | ----------------------------- | ---------------- | --------------------------------- | --------- |
| 1   | MLP from scratch              | MNIST            | test accuracy ≥ 0.95              | 10 min    |
| 2   | Convolutional classifier      | MNIST            | test accuracy ≥ 0.97              | 15 min    |
| 3   | LSTM on sequential signal     | synthetic halves | test accuracy ≥ 0.85              | 10 min    |
| 4   | Diagnose a broken CNN         | MNIST            | identify **stride + padding** bug | 10 min    |
| 5   | Train-and-prescribe — iterate | Fashion-MNIST    | 3 HEALTHY verdicts + test ≥ 0.82  | 30–45 min |

Total: **5 / 5** required for a course pass; 4 / 5 earns a conditional pass with instructor review.

## How Grading Works

Every question ends with a grading cell that calls a `check_qN(...)` function from
`quiz_harness.py`. Each checker returns a `(passed, ...)` tuple and prints a message
explaining why. The final cell in the notebook aggregates the five checkers into a
submission summary so both learner and instructor see the same score.

Q1, Q2, Q3 grade on an accuracy threshold after a capped number of epochs (3 for Q2,
5 for Q1 and Q3). Q4 grades on a keyword match against the learner's free-text
answer — it accepts multiple phrasings of "stride without padding collapses the
spatial dimensions" and a concrete fix. Q5 grades on the output of
`DLDiagnostics.report()` — all three verdicts (`gradient_flow`, `dead_neurons`,
`loss_trend`) must be **HEALTHY** and test accuracy must be **≥ 0.82** on
Fashion-MNIST after 3 epochs.

The Q5 threshold was calibrated by running the TARGET architecture (6-layer
LayerNorm + GELU + Kaiming MLP, Adam 1e-3, weight decay 1e-4, 200-step warmup) and
capturing its actual `report()` verdicts. See the module docstring of
`mlfp05_quiz_solutions.ipynb` for the captured run.

## How to Run

Learners open `mlfp05_quiz.ipynb` in Google Colab:

1. **Runtime → Change runtime type → T4 GPU** (free tier is sufficient).
2. Edit the `FORK_URL` variable in cell 1 to point at the learner's own fork of
   the Classroom repo. All subsequent cells auto-install `kailash-ml`,
   `kailash-nexus`, `kailash-kaizen`, PyTorch, and polars.
3. Work through Q1 → Q5 top to bottom. Re-run Q5 Cell 2 as many times as needed
   until `check_q5_pass` reports PASS.
4. The final submission cell prints the aggregate score; learners commit and
   push the completed notebook to their fork.

Instructors open `mlfp05_quiz_solutions.ipynb`, which is identical in structure
but has all blanks filled in and the target Q5 architecture pre-wired.

## How to Grade a Submission

1. Clone the learner's fork and open the submitted `mlfp05_quiz.ipynb` in a
   Colab T4 session (or `.venv/bin/jupyter lab` locally).
2. Run all cells. Every grading cell prints `PASS` or `FAIL` with the measured
   accuracy or verdict severity. Total time on T4: ~8 minutes.
3. The submission summary cell at the end prints `score: N/5`.
4. Spot-check Q5 iteration count: the DL Diagnostics dashboard (printed
   inline) reveals whether the learner iterated systematically or guessed
   at random. Iteration history lives in notebook output so it survives
   the submission intact.
5. If any checker reports FAIL, read the inline hint. The checkers each point
   at the specific deck slide (5C–5J) that explains the prescription.

## Files

| File                          | Audience   | What it does                                                                 |
| ----------------------------- | ---------- | ---------------------------------------------------------------------------- |
| `quiz_harness.py`             | both       | Reusable data loaders, training helpers, and `check_qN` grading functions.   |
| `mlfp05_quiz.ipynb`           | learner    | Student notebook with blanks. FORK_URL is a template the learner edits.      |
| `mlfp05_quiz_solutions.ipynb` | instructor | Identical structure, all blanks filled, FORK_URL pre-set to the public fork. |
| `README.md`                   | instructor | This file.                                                                   |

## Thresholds Source of Truth

| Checker         | Source constant                | Rationale                                                                                  |
| --------------- | ------------------------------ | ------------------------------------------------------------------------------------------ |
| `check_q1`      | 0.95 (MNIST MLP)               | Textbook 2-layer MLP hits 0.97 in 5 epochs; 0.95 leaves room for off-by-one mistakes.      |
| `check_q2`      | 0.97 (MNIST CNN)               | A 2-conv CNN hits 0.99 after 3 epochs; 0.97 rejects architectures that collapse dims.      |
| `check_q3`      | 0.85 (halves task)             | A BoW MLP scores 0.50 on this task; a correctly-wired LSTM scores 0.95+.                   |
| `check_q4`      | keyword match                  | The fix is "add padding or reduce stride"; we accept either phrasing.                      |
| `check_q5_pass` | `Q5_TEST_ACC_THRESHOLD = 0.82` | TARGET architecture reliably hits 0.87 in 3 epochs on Fashion-MNIST; 0.82 leaves headroom. |

All thresholds live in `quiz_harness.py` as named constants so they can be
audited and updated in one place.
