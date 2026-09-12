# VS Code setup for the coursework

Do these once. Steps 4 and 5 are the ones that actually matter — nearly every
"but numpy IS installed" problem comes from skipping them.

On Windows, type `python` where this guide says `python3`, and `py -3` if that
doesn't work either.

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

## 3. Make sure Python itself is installed

Open the built-in terminal with ``Ctrl+` `` and run:

```bash
python3 --version
```

A version number means you're set. "Command not found" means you need Python —
get it from <https://www.python.org/downloads/> (tick **"Add Python to PATH"** on
Windows, it's easy to miss and causes exactly this problem later) or install
Anaconda.

---

## 4. Create a virtual environment

A virtual environment is a private Python install for this project. It's not
bureaucracy — it's what stops one course's package versions from breaking
another's, and VS Code is built around the assumption you have one.

`Ctrl+Shift+P` → **`Python: Create Environment`** → **`Venv`** → pick your Python
→ when it asks about dependencies, tick **`requirements.txt`**.

It creates `.venv/`, installs everything, and selects the interpreter for you.
Takes a minute or two.

<details>
<summary>If you'd rather do it in the terminal</summary>

```bash
python3 -m venv .venv

# macOS / Linux
source .venv/bin/activate
# Windows (PowerShell)
.venv\Scripts\Activate.ps1

pip install -r requirements.txt
```
</details>

---

## 5. Check the interpreter is the venv one

**This is the step that causes the most confusion when skipped.**

Look at the bottom-right of the VS Code status bar — it shows the active Python.
It should mention `.venv`. If it doesn't:

`Ctrl+Shift+P` → **`Python: Select Interpreter`** → pick the entry with
**`('.venv')`** in it, usually marked *Recommended*.

The symptom of getting this wrong: `ModuleNotFoundError: No module named 'numpy'`
on a machine where you just watched pip install numpy. It's installed — into a
*different* Python than the one running your code.

---

## 6. Verify

```bash
python3 week1/check_setup.py
```

It checks the interpreter, the four packages, that the homework dataset
downloads, and that matplotlib can draw. If every line says `OK` you're done
setting up.

---

## 7. Open the notebook and pick the kernel

Open `week1/homework_week1.ipynb`. Top right of the notebook, click
**Select Kernel** → **Python Environments** → the `.venv` one.

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
