# VS Code setup for the coursework

Do these once. Steps 4 and 5 are the ones that actually matter — nearly every
"but numpy IS installed" problem comes from skipping them.

**Windows users: the command is `python`, not `python3`.** `python3` only exists
on macOS and Linux. On Windows it hits a Microsoft Store stub that prints
*"Python was not found; run without arguments to install from the Microsoft
Store"* — which is misleading, because it says nothing about whether Python is
actually installed.

Throughout this guide, run `python` on Windows and `python3` on macOS/Linux.

---

## 1. Get the repo open in VS Code

`Ctrl+Shift+P` (`Cmd+Shift+P` on Mac) opens the Command Palette — you'll use it
for everything below. Type **`Git: Clone`**, paste:

```
https://github.com/Youss1337/School-shi
```

Pick a folder, and say yes when it offers to open it.

The repo has one branch and it's the default, so the files are there immediately —
no checkout step.

---

## 2. Two extensions

Open the Extensions pane (`Ctrl+Shift+X`) and install:

| Extension | Publisher | Why |
|---|---|---|
| **Python** | Microsoft | Interpreter handling, IntelliSense, debugging |
| **Jupyter** | Microsoft | Runs `.ipynb` notebooks inside VS Code |

Search by name and check the publisher says **Microsoft** — there are lookalikes.

Optional but worth it: **Data Wrangler** (Microsoft). It's the closest thing VS
Code has to Spyder's Variable Explorer — click a DataFrame, get a spreadsheet
view with column stats.

---

## 3. Which Python do you already have?

Open the built-in terminal with ``Ctrl+` ``. **Look at the start of the prompt
before you type anything.**

If it begins with **`(base)`**, like this:

```
(base) PS C:\Users\You\Projects\School-shi>
```

then **Anaconda is installed and active**, and you already have Python plus
numpy, pandas, scikit-learn and matplotlib. Skip to step 4A.

Otherwise, check for Python:

```bash
python --version        # Windows
python3 --version       # macOS / Linux
```

A version number means you're set — go to step 4B. "Command not found" (or the
Microsoft Store message on Windows) means you need Python: get it from
<https://www.python.org/downloads/> and, on Windows, **tick "Add Python to
PATH"** during install. It's easy to miss and causes exactly this problem later.

<details>
<summary>Optional: silence the Windows <code>python3</code> stub</summary>

Settings → Apps → Advanced app settings → **App execution aliases** → turn off
`python.exe` and `python3.exe`. Then an unknown `python3` gives you an honest
"not recognized" instead of a Store advert. Purely cosmetic.
</details>

---

## 4. Create a project environment

A project environment is a private Python install for this course. It's not
bureaucracy — it's what stops one project's package versions from breaking
another's, and it means a bad `pip install` can never take down anything but
this one folder.

Follow **4A** if you have Anaconda (the `(base)` prompt), **4B** if you don't.

### 4A — You have Anaconda

In the terminal:

```powershell
conda create -n fintech python=3.12 -y
conda activate fintech
pip install -r requirements.txt
```

Your prompt changes from `(base)` to `(fintech)`. That's how you know it worked.

> You *can* just work in `base` — it already has everything. But base is shared
> with every other project on your machine, Spyder included, so breaking it
> breaks all of them at once. One command to avoid that is worth it.

> Don't use `Python: Create Environment` → `Venv` when you have Anaconda.
> Mixing venv and conda works, but it produces genuinely confusing failures.
> Conda manages conda.

### 4B — You don't have Anaconda

`Ctrl+Shift+P` → **`Python: Create Environment`** → **`Venv`** → pick your Python
→ when it asks about dependencies, tick **`requirements.txt`**.

It creates `.venv/`, installs everything, and selects the interpreter for you.

<details>
<summary>Same thing in the terminal</summary>

```bash
python -m venv .venv                 # Windows   (python3 on macOS/Linux)

.venv\Scripts\Activate.ps1           # Windows (PowerShell)
source .venv/bin/activate            # macOS / Linux

pip install -r requirements.txt
```
</details>

---

## 5. Point VS Code at that environment

**This is the step that causes the most confusion when skipped.**

`Ctrl+Shift+P` → **`Python: Select Interpreter`** → pick:

- **4A (conda):** the entry labelled `fintech`
- **4B (venv):** the entry with `('.venv')` in it, usually marked *Recommended*

The bottom-right status bar shows the active one. Check it says what you expect.

The symptom of getting this wrong: `ModuleNotFoundError: No module named 'numpy'`
on a machine where you just watched pip install numpy. It *is* installed — into a
*different* Python than the one running your code.

If the terminal was already open, close it and open a new one (the trash-can icon,
then ``Ctrl+` ``) so it picks up the new environment.

---

## 6. Verify

```bash
python week1/check_setup.py         # Windows
python3 week1/check_setup.py        # macOS / Linux
```

It checks which environment you're in, the four packages, that the homework dataset
downloads, and that matplotlib can draw. If every line says `OK` you're done
setting up.

---

## 7. Open the notebook and pick the kernel

Open `week1/homework_week1.ipynb`. Top right of the notebook, click
**Select Kernel** → **Python Environments** → the same environment you picked in
step 5 (`fintech`, or `.venv`).

If it isn't listed, run `pip install ipykernel` inside the activated environment
and reopen the notebook.

(The kernel is chosen *per notebook*, separately from step 5. Same idea, second
place to set it — this catches people out.)

Then: `Shift+Enter` runs a cell and moves on, `Ctrl+Enter` runs it and stays.
Same keys as Jupyter and as Spyder.

---

## 8. Claude Code extension

Extensions pane → search **Claude Code** → install → sign in. It reads the file
you're in and shows edits as inline diffs you accept or reject, instead of you
copying code back and forth.

---

## 9. Saving your work back to GitHub

Source Control pane (`Ctrl+Shift+G`): type a message, click **Commit**, then
**Sync Changes**. That's `git add` + `commit` + `push` in one place.

Commit whenever you finish a part of the homework. It's a free undo history, and
it means your work isn't only on one laptop.

---

## What's in the repo

| File | What it's for |
|---|---|
| `week1/homework_week1.ipynb` | **Start here.** The homework scaffold as a notebook. |
| `week1/homework_week1.py` | The same scaffold as a cell script, if you prefer scripts. |
| `week1/lecture_week1.py` | The lecture notebook, converted. Your reference for every pattern. |
| `week1/P1_Regression_problem_Week_1_2026.ipynb` | The original file from the course. |
| `week1/check_setup.py` | The environment check from step 6. |
| `week1/README.md` | Spyder-specific notes, and how to convert between `.py` and `.ipynb`. |

The `.py` and `.ipynb` versions of the homework are the same content — edit
whichever you prefer, just don't work in both at once.

---

## A note on your folder path

Avoid `&` and, where you can, spaces in the folders above your project — a path
like `Projects\ML & AI\School-shi` is legal, and VS Code and conda handle it, but
`&` is a command separator in most shells and some tools mishandle it. If you hit
a strange path-related error later, this is a prime suspect. Renaming the folder
to `ML_and_AI` now is cheaper than debugging it in week 6.
