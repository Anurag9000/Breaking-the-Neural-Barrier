#!/usr/bin/env python3
"""
run_all_models.py

Automatically finds and runs every `run_dnn_*.py` script under Dyn_DNN4OPF/examples/,
using their default configs.  Each pipeline will dump its .pth, CSV, plots, and Diagnostics folder.

"""

import subprocess
import sys
import os
import glob

def run_script(path_to_script: str) -> None:
    """
    Invoke a Python script using the same interpreter.
    If the script exits with a non-zero status, this will raise CalledProcessError.
    """
    print(f"\n=== Running: {path_to_script} ===\n")
    subprocess.run([sys.executable, path_to_script], check=True)
    print(f"\n=== Finished: {path_to_script} ===\n")

def main():
    # 1. Ensure we're in the repo root (where this file lives)
    project_root = os.path.dirname(os.path.abspath(__file__))
    os.chdir(project_root)

    # 2. Build a glob pattern pointing to Dyn_DNN4OPF/examples/run_dnn_*.py
    pattern = os.path.join(project_root, "Dyn_DNN4OPF", "examples", "run_dnn_*.py")
    run_files = sorted(glob.glob(pattern))

    if not run_files:
        print("WARNING: No run_dnn_*.py scripts found under:")
        print(f"    {pattern}")
        print("Make sure your pipeline scripts live in Dyn_DNN4OPF/examples and are named run_dnn_*.py")
        sys.exit(1)

    # 3. Execute each script in alphabetical order
    for abs_path in run_files:
        rel_path = os.path.relpath(abs_path, project_root)
        try:
            run_script(rel_path)
        except subprocess.CalledProcessError as e:
            print(f"ERROR: Script {rel_path} exited with status {e.returncode}.")
            print("Aborting the remaining pipelines.")
            sys.exit(e.returncode)

if __name__ == "__main__":
    main()
