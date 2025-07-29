"""Plotting submodule for model monitoring.

This submodule provides visualization utilities for statistical analysis and monitoring,
including customizable panel-based plotting for segment statistics.
"""

from .core import plot_segment_statistics, set_plot_theme

__all__ = [
    "plot_segment_statistics",
    "set_plot_theme",
]
