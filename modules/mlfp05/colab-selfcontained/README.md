# M5 Self-Contained Student Notebooks

Open any notebook in Google Colab. **No git clone, no FORK_URL, no sys.path setup needed.**

## How to use

1. Download a notebook (`modules/mlfp05/colab-selfcontained/ex_N/NN_technique.ipynb`)
2. Open in [Google Colab](https://colab.research.google.com) — File → Upload notebook
3. **Before running**: Runtime → Change runtime type → T4 GPU (free tier)
4. Run **Cell 0** (pip install + GPU check)
5. Run **Cell 1** (helpers — auto-generated, collapse if you want)
6. Fill in the `____` blanks in subsequent cells
7. Run the exercise end-to-end

## What's different from `modules/mlfp05/colab/`

| Feature | `colab/` | `colab-selfcontained/` |
|---|---|---|
| Needs git fork? | Yes | No |
| Needs FORK_URL edit? | Yes | No |
| Size | ~30KB | ~130KB |
| Best for | Persistent work across sessions | Quick single-session exercises |

## Blanks to fill in

Student version has `____` placeholders in model definitions, training loops, and visualisations. Fill them in and run the cell to see your result.
