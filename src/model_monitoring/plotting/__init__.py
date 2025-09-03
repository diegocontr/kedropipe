"""Plotting submodule for model monitoring.

This submodule provides visualization utilities for statistical analysis and monitoring,
including customizable panel-based plotting for segment statistics.
"""

from .core import plot_segment_statistics, set_plot_theme
from .global_plots import plot_global_statistics

__all__ = [
    "plot_global_statistics",
    "plot_segment_statistics",
    "set_plot_theme",
]
