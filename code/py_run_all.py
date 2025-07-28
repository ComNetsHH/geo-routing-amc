#!/usr/bin/env python3
"""
run_all.py

Sequentially executes the three analysis stages:
  1) Input preparation for the second-order amrkov model: 
     py_AnalyzeAdvance_markov_model_second_order_input_paper.py
  2) Monte Carlo simulation of geographic greedy routing:
     py_AnalyzeAdvance_simulation.py
  3) Second-order absorbing Markov chain analysis:
     py_AnalyzeAdvance_markov_model_second_order_paper.py

If any script fails (non-zero exit code), this launcher will stop immediately.
"""

import os
import sys
import subprocess

def run_script(path):
    """Run a Python script at `path` with the same interpreter."""
    print(f"\n>>> Running {path} ...")
    result = subprocess.run([sys.executable, path])
    if result.returncode != 0:
        print(f"✗ {path} exited with code {result.returncode}")
        sys.exit(result.returncode)
    print(f"✓ {path} completed successfully.")

if __name__ == "__main__":
    # List your scripts in the order they must be run:
    scripts = [
        "code\\py_AnalyzeAdvance_markov_model_second_order_input_paper.py",
        "code\\py_AnalyzeAdvance_simulation.py",
        "code\\py_AnalyzeAdvance_markov_model_second_order_paper.py",
    ]


    for script in scripts:
        run_script(script)

    print("\nAll stages finished without error.")
