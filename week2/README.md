# Week 2 — Classification & Risk Management

Credit default, fraud detection, and why accuracy is the wrong metric for both.

| File | What it is |
|---|---|
| `homework_week2.ipynb` | **Start here.** The scaffold — plumbing done, each step a `TODO`, answers to fill in. |
| `homework_week2_solution.ipynb` | The worked version. All four parts, code and written answers, outputs executed. Use it to check yourself. |
| `homework_week2.py` / `homework_week2_solution.py` | The same two, as `# %%` cell scripts. |
| `lecture_week2.py` | The course lecture notebook converted to a cell script. |
| `P1_Classification_problem_Week_2_2026.ipynb` | The original course file. The homework brief is its last markdown cell. |
| `Intro_ML_P1_slides_week2.pdf` | The week 2 slides. |

## Running it

The environment from Week 1 covers this, with one addition — reading `.xlsx` needs
`openpyxl`:

```powershell
conda install openpyxl -y
```

Then open `homework_week2.ipynb`, pick the `base` kernel, and work down it. The dataset
is pulled from GitHub, so you need a connection the first time.

`homework_week2_solution.ipynb` already has every output and figure stored, so you can
read it without running anything.

## What the answers turn on

Four results carry the whole assignment:

1. **The base rate is 3.33%**, so a model predicting "nobody defaults" scores 96.67%
   accuracy and is worthless. Every accuracy figure in the notebook has to be read against
   that number.
2. **`income` varies 27.6x more widely than `balance`.** That single ratio explains why
   unscaled KNN collapses from 0.41 recall to 0.06 — distance is measured almost entirely
   along the feature that carries no signal.
3. **Balance drives default; income and student status barely move it.** Three independent
   routes agree: the boxplots, the standardised coefficients, and L1 zeroing the other two
   at `C=0.01`.
4. **Moving the decision threshold from 0.50 to 0.06 saves EUR 623,000** on 3,000
   customers, without changing the model at all. And KNN with k=9 cannot implement that
   threshold — it has only ten possible scores.

## Two caveats stated in the notebook

- The threshold in Part 4 is swept on the **test** set for speed. Properly you choose it
  on a validation split and report the cost on untouched test data — the same mistake as
  tuning `alpha` on the test set in Week 1. The direction holds; the exact figure would
  move.
- The model comparison is one split (`random_state=1`) with k fixed at 9 as the brief
  specifies, not tuned. A fairer contest would cross-validate k.
