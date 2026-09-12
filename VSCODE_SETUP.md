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

## 4. Which environment to use

**If your prompt says `(base)` — you already have everything you need.**
Anaconda ships numpy, pandas, scikit-learn, matplotlib and ipykernel. Install
nothing, and go straight to step 5.

That is the documented path for this repo. It is the fastest way to start, and
for a single course it is perfectly normal.

<details>
<summary>Optional: a dedicated environment instead (and why you might want one)</summary>

Working in `base` has one real risk: `pip install` and conda both manage
packages and don't know about each other, so a pip install into base can
overwrite a conda-managed library and leave the install unrepairable. Base is
what Spyder, Anaconda Navigator and Jupyter all run on, so that breaks
everything at once and the fix is reinstalling Anaconda.

In a named environment the same accident costs you two minutes:

```powershell
conda create -n fintech python=3.12 -y
conda activate fintech
pip install -r requirements.txt
conda env remove -n fintech        # if it ever goes wrong - then recreate
```

So: **if you stick to `conda install` in base, base is fine.** The moment you
find yourself running `pip install` for this course, make an environment first.

You'll also want one eventually when two modules need different versions of the
same package — one environment can't hold both.

To use a named env from Spyder as well, Spyder needs a matching kernel in it:
`conda install -n fintech spyder-kernels -y`, then point Spyder at that
interpreter in Preferences.
</details>

<details>
<summary>If you don't have Anaconda at all</summary>

`Ctrl+Shift+P` → **`Python: Create Environment`** → **`Venv`** → pick your Python
→ tick **`requirements.txt`** when it asks. It creates `.venv/`, installs
everything, and selects the interpreter for you.
</details>

---

## 5. Point VS Code at that environment

**This is the step that causes the most confusion when skipped.**

`Ctrl+Shift+P` → **`Python: Select Interpreter`** → pick the entry labelled
**`base`** (or your named env / `.venv` if you made one in step 4).

The bottom-right status bar shows the active one. Check it says what you expect.

The symptom of getting this wrong: `ModuleNotFoundError: No module named 'numpy'`
on a machine where the package is definitely installed. It *is* installed — into
a *different* Python than the one running your code.

If the terminal was already open, close it (trash-can icon) and open a new one
with ``Ctrl+` `` so it picks up the environment.

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
step 5 (`base`, or your named env).

If it isn't listed, run `pip install ipykernel` inside the activated environment
and reopen the notebook.

(The kernel is chosen *per notebook*, separately from step 5. Same idea, second
place to set it — this catches people out.)

Then: `Shift+Enter` runs a cell and moves on, `Ctrl+Enter` runs it and stays.
Same keys as Jupyter and as Spyder.

---

## 8. Claude Code extension

**Prerequisites:** VS Code 1.94.0 or higher, and a paid Claude plan (Pro, Max,
Team or Enterprise) or a Claude Console account. No API key needed. There is no
free tier for this — if you're on the free plan, skip this step and paste code
into claude.ai instead.

You do **not** need to install Node.js or the CLI separately: the extension
bundles its own copy. (A separate CLI install is only needed if you want to type
`claude` in the integrated terminal.)

1. Extensions pane (`Ctrl+Shift+X`) → search **Claude Code** → publisher
   **Anthropic** → **Install**.
   If it doesn't show up afterwards: Command Palette →
   **`Developer: Reload Window`**.
2. Open the panel. The ✱ Spark icon is the marker:
   - **Editor toolbar**, top-right of the editor — quickest, but only appears
     when a file is open.
   - **Activity Bar**, left sidebar — always visible, opens the sessions list.
   - Command Palette → type **`Claude Code`**.
3. A sign-in screen appears the first time. Click **Sign in** and finish
   authorising in the browser.

Worth knowing once you're in:

| | |
|---|---|
| Selected text | Claude sees it automatically — no pasting |
| `Alt+K` | Insert an @-mention of the selection, e.g. `@homework_week1.ipynb#42-51` |
| `@` | Reference any file or folder by name (fuzzy matches) |
| `Shift+Enter` | New line without sending |
| Permission mode | Bottom of the prompt box. **Manual** shows a diff and asks before every edit — worth using while you're learning, so nothing changes without you reading it first |

For coursework, the useful move is selecting a cell you don't understand and
asking about it directly, rather than describing it.

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
