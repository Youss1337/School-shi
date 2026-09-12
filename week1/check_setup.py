"""
Environment check. Run this once after setting up VS Code.

    VS Code:  open this file, press the Run button (top right), or
              F5, or in the terminal:  python week1/check_setup.py

If every line below says OK, your environment is ready for the coursework.
"""

import sys

print(f"Python {sys.version.split()[0]}")
print(f"  interpreter: {sys.executable}")
if ".venv" in sys.executable or "venv" in sys.executable or "conda" in sys.executable:
    print("  OK  - running inside a virtual environment")
else:
    print("  WARNING - this does not look like a virtual environment.")
    print("            Run 'Python: Select Interpreter' and pick the .venv one.")
print()

REQUIRED = ["numpy", "pandas", "sklearn", "matplotlib"]
missing = []

for name in REQUIRED:
    try:
        module = __import__(name)
        version = getattr(module, "__version__", "?")
        print(f"  OK  {name:<12} {version}")
    except ImportError:
        missing.append(name)
        print(f"  MISSING  {name}")

if missing:
    print("\nInstall the missing packages with:")
    print("    pip install -r requirements.txt")
    sys.exit(1)

print()

# The homework dataset lives on GitHub, so the environment needs to reach it.
try:
    import pandas as pd

    url = ("https://raw.githubusercontent.com/JWarmenhoven/ISLR-python/"
           "master/Notebooks/Data/Advertising.csv")
    advertising = pd.read_csv(url, index_col=0)
    print(f"  OK  downloaded the homework dataset: {advertising.shape[0]} rows, "
          f"columns {list(advertising.columns)}")
except Exception as exc:                                    # noqa: BLE001
    print(f"  FAILED to download the dataset: {exc}")
    print("      Check your internet connection, or a proxy/firewall.")
    sys.exit(1)

# Matplotlib needs to be able to make a figure without a display attached.
try:
    import matplotlib
    import matplotlib.pyplot as plt

    plt.figure()
    plt.plot([0, 1], [0, 1])
    plt.close()
    print(f"  OK  matplotlib can draw (backend: {matplotlib.get_backend()})")
except Exception as exc:                                    # noqa: BLE001
    print(f"  FAILED to draw with matplotlib: {exc}")
    sys.exit(1)

print("\nEverything is ready. Open week1/homework_week1.ipynb and get started.")
