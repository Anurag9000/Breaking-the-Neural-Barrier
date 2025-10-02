"""
CNN_STL_grid.py
----------------
Thin model module for the STL-style ConvNet used in grid sweeps.
Re-exports ConvNetSTL and stl_total_neurons from CNN_STL to keep a clean import
surface for grid experiments.
"""

# We intentionally *reuse* the user's canonical implementation to avoid divergence.
from CNN_STL import ConvNetSTL, stl_total_neurons  # noqa: F401
