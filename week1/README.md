# Week 1 — Linear Models & Overfitting (Spyder setup)

## Short answer: yes, but with one wrinkle

You can absolutely do this homework in Spyder. The wrinkle is that **Spyder does
not open `.ipynb` notebook files** out of the box — it is a script editor, not a
notebook editor. What it *does* have is **cell mode**: any line starting with
`# %%` splits a `.py` file into blocks you can run one at a time with
`Ctrl+Enter`, exactly like notebook cells. Same workflow, plain `.py` file.

So the two files in this folder are the notebook, converted:

| File | What it is |
|---|---|
| `lecture_week1.py` | The whole lecture notebook as a Spyder cell script. Run it top to bottom to follow the class. |
| `homework_week1.py` | A scaffold for the homework. Plumbing done, each step marked `TODO`, answer blocks to fill in. |

Both are verified to run end to end.

---

## 1. Install

### Option A — Anaconda (recommended if you don't already have Python set up)

Download from <https://www.anaconda.com/download>. It ships Spyder, numpy,
pandas, scikit-learn and matplotlib in one installer, already working together.
Launch Spyder from Anaconda Navigator or by running `spyder` in a terminal.

### Option B — you already have Python

```bash
pip install spyder numpy pandas scikit-learn matplotlib
spyder
```

(Or `pip install -r requirements.txt` from this folder.)

### Check it worked

Paste this into the Spyder console (bottom-right pane) and press Enter:

```python
import numpy, pandas, sklearn, matplotlib
print(sklearn.__version__)
```

If that prints a version number you are ready.

---

## 2. Point Spyder at this folder

1. Download / clone this folder to somewhere sensible on your machine.
2. In Spyder, set the **working directory** (the folder path box in the top-right
   toolbar) to this folder. This matters if you ever save a CSV or a figure —
   otherwise it lands somewhere surprising.
3. `File → Open` → `lecture_week1.py`.

---

## 3. The five Spyder things you actually need

| Action | Key |
|---|---|
| Run the current cell, stay put | `Ctrl+Enter` |
| Run the current cell, jump to the next | `Shift+Enter` |
| Run just the selected lines (or the current line) | `F9` |
| Run the whole file | `F5` |
| Re-run the last cell | `Alt+Shift+Enter` |

And two panes worth knowing:

- **Variable Explorer** (top right) — every variable currently alive, with its
  type and shape. Double-click a DataFrame and it opens in a spreadsheet view.
  This is the one thing Spyder does better than Jupyter, and it is genuinely
  useful for this homework: after the train/test split, click `X_train` and
  confirm it really has 140 rows and three columns.
- **Plots** (top right, next to Variable Explorer) — every figure you have drawn,
  scrollable. If your plots open in separate pop-up windows instead and you would
  rather have them here: `Tools → Preferences → IPython console → Graphics →
  Graphics backend → Inline`. (The reverse — `Automatic` — gives you zoomable
  pop-up windows. Either is fine, it's taste.)

---

## 4. Doing the work

1. Open `lecture_week1.py`, run it cell by cell with `Shift+Enter`. Don't skip —
   later cells use variables from earlier ones. This is your reference for every
   pattern the homework asks for.
2. Open `homework_week1.py`. Work down it. Each `# TODO: your code here` cell
   tells you what to write and which lecture cell shows the pattern.
3. Write your written answers into the `"""ANSWER ..."""` blocks in the file
   itself. Those blocks are plain strings — Python ignores them, so the file
   keeps running while you fill them in.

Work in order. Part 2 needs the split from Part 1, Part 3 needs the polynomial
features from Part 2.

---

## 5. Gotchas that will bite you

**Bare expressions don't print.** In Jupyter, a cell ending in
`pd.DataFrame(scores.values())` displays the table. In a `.py` script, don't rely
on that — wrap it: `print(pd.DataFrame(scores.values()))`. Both files here
already do. (A DataFrame printed in the console is wide and ugly; the nicer move
in Spyder is to let the last line assign to a variable and then double-click it in
the Variable Explorer for the spreadsheet view.)

**Stale variables.** Cell-by-cell running means variables from an old run stick
around, so a cell can appear to work because of a leftover variable from a
version of the code you already deleted. When something behaves inexplicably:
click the console and press `Ctrl+.` (restart kernel), then run from the top.
This is the single most common source of "but it worked a minute ago".

**`F5` vs cells.** `F5` runs the whole file from scratch. That is the honest test
that your homework actually works start to finish — do it once before you submit.

**The `index_col=0`.** `pd.read_csv(url, index_col=0)`. The Advertising CSV has an
unnamed first column that is just the row number. Forget the argument and you
silently train on row numbers as a fourth feature, and your coefficients will be
quietly wrong.

**Lasso's ConvergenceWarning.** Pass `max_iter=100000`. The polynomial design
matrix is badly conditioned and the default 1000 iterations stops early.

---

## 6. If your lecturer wants an `.ipynb` back

Do the work in `.py` in Spyder, then convert at the end:

```bash
pip install jupytext
jupytext --to notebook homework_week1.py      # -> homework_week1.ipynb
```

`jupytext` reads the `# %%` markers as cell boundaries, so the structure survives
the round trip. Open the result once in Jupyter or Colab, `Run All`, save so the
outputs are stored, and submit that.

Alternatives, if you'd rather stay in notebooks:
- **The `spyder-notebook` plugin** (`pip install spyder-notebook`) embeds real
  notebooks inside Spyder. It works, but it is a third-party plugin that lags
  Spyder releases and can be fiddly to install — I would not make it the thing
  your homework depends on.
- **VS Code** edits `.ipynb` natively with a Jupyter extension, and has a
  Variable Explorer too.
- **Google Colab** — zero install, and it is where the original notebook came
  from (the metadata says so).

---

## 7. "Can you be linked to Spyder?"

Not directly — there is no Claude plugin for Spyder. The official editor
integrations are VS Code and JetBrains only.

What does work:

- **Claude Code in a terminal, alongside Spyder.** Install it
  (<https://claude.ai/code>), `cd` into this folder, run `claude`. It reads and
  edits the same files on disk. Spyder notices external changes and reloads the
  file when you click back into it. That is effectively the same experience as an
  IDE plugin, just in a second window.
- **This session.** I can work on the files in this repo directly — push changes
  here, you `git pull` and they appear in Spyder.
- **Copy-paste.** Unglamorous, works fine: paste an error traceback or a cell of
  code and ask.

If you want to go the VS Code route instead, it edits `.ipynb` natively *and*
takes the Claude Code extension — that is the setup with the fewest moving parts.
