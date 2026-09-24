#!/usr/bin/env python3
"""Run the analysis pipeline and/or the dashboard.

Usage (from the project root):
    python main.py                  # execute nigeria_fuel_price.ipynb in place
    python main.py --test           # run the unit tests first, then the notebook
    python main.py --dashboard      # start the dashboard on http://127.0.0.1:8050
"""

import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
NOTEBOOK = PROJECT_ROOT / "nigeria_fuel_price.ipynb"


def run(cmd: list[str]) -> int:
    print("$", " ".join(cmd))
    return subprocess.run(cmd, cwd=PROJECT_ROOT).returncode


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--test", action="store_true", help="run the unit tests before the notebook")
    parser.add_argument("--dashboard", action="store_true", help="start the dashboard instead of the notebook")
    args = parser.parse_args()

    if args.dashboard:
        return run([sys.executable, "-m", "dashboard.app"])

    if args.test and run([sys.executable, "-m", "pytest", "-q"]) != 0:
        print("Tests failed; notebook not executed.")
        return 1

    code = run([
        sys.executable, "-m", "jupyter", "nbconvert",
        "--to", "notebook", "--execute", "--inplace",
        "--ExecutePreprocessor.timeout=1800",
        str(NOTEBOOK),
    ])
    if code == 0:
        print("Done. Figures, tables and model weights are in outputs/. Start the dashboard with: "
              "python main.py --dashboard")
    return code


if __name__ == "__main__":
    sys.exit(main())
