"""
===============================================================================
Setup Script / Directory for Dyn_DNN4OPF
===============================================================================

This script/directory manages installation and environment setup tasks for the
Dyn_DNN4OPF project.

--------------------------------------------------------------------------------
Purpose:
    - Install required packages and dependencies.
    - Setup virtual environment if applicable.
    - Compile any extensions or perform code generation.
    - Automate environment configuration for reproducible experiments.

--------------------------------------------------------------------------------
Typical Contents (if directory):
    - setup.py or install.sh scripts.
    - Environment YAML or requirements files.
    - Post-install scripts to prepare data or caches.

--------------------------------------------------------------------------------
Usage:
    - Run via `python setup.py install` or shell scripts before training.
    - Ensure all dependencies (PyTorch, PyG, Optuna, etc.) are installed.
    - May include version checks and warnings.

--------------------------------------------------------------------------------
Notes:
    - Important for onboarding new users and deployment.
    - Keep updated as dependencies evolve.

--------------------------------------------------------------------------------
Example Commands:
    python setup.py install
    bash setup/install.sh
"""

from setuptools import setup, find_packages

setup(
    name="Dyn_DNN4OPF",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "torch",
        "torch-geometric",
        "pandas",
        "matplotlib",
        "numpy",
    ],
    include_package_data=True,
    zip_safe=False,
)
