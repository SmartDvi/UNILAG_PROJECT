#!/usr/bin/env python3
"""
Nigeria Retail Fuel Price Intelligence System
Main entry point for pipeline execution.

Usage:
    python main.py --run-all          # Run full pipeline
    python main.py --forecast-only    # Generate 3-month forecast only
    python main.py --evaluate         # Evaluate model on test set
"""

import argparse
import subprocess
import sys
from pathlib import Path

def run_jupyter_notebook(notebook_path: str) -> int:
    """Execute a Jupyter notebook and return exit code."""
    cmd = [
        sys.executable, "-m", "jupyter", "nbconvert",
        "--to", "notebook",
        "--execute",
        "--inplace",
        notebook_path
    ]
    result = subprocess.run(cmd, capture_output=False)
    return result.returncode

def main():
    parser = argparse.ArgumentParser(
        description="Nigeria Fuel Price Intelligence System Pipeline"
    )
    parser.add_argument(
        "--run-all",
        action="store_true",
        help="Execute full analysis pipeline (K0-K10)"
    )
    parser.add_argument(
        "--forecast-only",
        action="store_true",
        help="Generate 3-month forecast from latest checkpoint"
    )
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Evaluate model on test set"
    )
    
    args = parser.parse_args()
    
    notebook_path = Path("nigeria_fuel_price.ipynb")
    
    if not notebook_path.exists():
        print(f"❌ ERROR: {notebook_path} not found in current directory")
        return 1
    
    if args.run_all or (not args.forecast_only and not args.evaluate):
        print("🚀 Running full Nigeria Fuel Price Intelligence pipeline...")
        print(f"📓 Executing: {notebook_path}")
        exit_code = run_jupyter_notebook(str(notebook_path))
        
        if exit_code == 0:
            print("\n✅ Pipeline execution complete!")
            print("📊 Check the following files for results:")
            print("   • nigeria_fuel_3m_forecast.csv")
            print("   • k10_evaluation.png")
            print("   • k10_loss_curve.png")
        else:
            print(f"\n❌ Pipeline failed with exit code {exit_code}")
        return exit_code
    
    elif args.forecast_only:
        print("⚠️  Forecast-only mode not yet implemented.")
        print("   Run with --run-all to execute full pipeline.")
        return 1
    
    elif args.evaluate:
        print("⚠️  Evaluate-only mode not yet implemented.")
        print("   Run with --run-all to execute full pipeline.")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
