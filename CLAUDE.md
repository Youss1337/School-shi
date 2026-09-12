# School-shi — project context

Fintech P1 coursework. Week 1 is linear models, regularization and overfitting.

## This is graded coursework

`week1/homework_week1.ipynb` is a **deliberately unfilled scaffold**, and it is
the file the student submits. The plumbing is done — imports, data load,
train/test split, a scoring helper — and each homework step is a `TODO` pointing
at the lecture cell with the matching pattern. Written answers go in the `ANSWER`
cells.

**Help the student reason to their own answers.** Explain concepts, review what
they write, point at the relevant lecture pattern, debug their errors. Do not
fill in the `TODO`s or write the `ANSWER` blocks for them unless they explicitly
ask for a worked solution.

## Environment

Anaconda **`base`** (Python 3.14.6) on Windows, chosen deliberately over a
dedicated env — base already ships every package the coursework needs.

- Use `conda install`, **never** `pip install`, in base. Mixing pip into conda's
  base is the one thing that can break it, and base is shared with Spyder and
  Navigator.
- The command is `python`, not `python3` — Windows.
- `week1/check_setup.py` verifies the environment end to end.

## Files

| Path | What it is |
|---|---|
| `week1/homework_week1.ipynb` | **The student's working file and submission.** The scaffold. |
| `week1/homework_week1.py` | Same scaffold as a `# %%` cell script. Kept in sync manually; the notebook is the live one. |
| `week1/lecture_week1.py` | The course lecture notebook converted to a cell script. The reference for every pattern the homework asks for. |
| `week1/P1_Regression_problem_Week_1_2026.ipynb` | The original course file. Its last markdown cell is the full homework brief. |
| `week1/check_setup.py` | Environment check. |
| `VSCODE_SETUP.md` | Environment setup, and the Spyder/VS Code trade-offs. |
| `week1/README.md` | Spyder-specific notes and `.py` ↔ `.ipynb` conversion. |

## Domain gotchas

- `pd.read_csv(url, index_col=0)` on the Advertising dataset. Without
  `index_col=0` the unnamed row-number column silently trains as a fourth
  feature.
- `PolynomialFeatures` and `StandardScaler`: `fit_transform` on train,
  `transform` only on test. Fitting either on the test set leaks and inflates
  every number reported afterwards.
- `Lasso(max_iter=100000)`. The polynomial design matrix here is badly
  conditioned (condition number ~2e11) and the default 1000 iterations stops
  early with a `ConvergenceWarning`. Scaling does not fix it — the collinearity
  is real.
- Cross-validation for `alpha` runs on the **training set only**. The test set
  stays sealed until the final report.

## Working agreement

`homework_week1.ipynb` belongs to the student — avoid pushing changes to it
without asking, since a notebook merge conflict is painful to resolve and
would land on their answers.
