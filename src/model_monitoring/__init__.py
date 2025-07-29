"""Model Monitoring Package

This package provides tools for monitoring machine learning models, including:
- Data analysis and segmentation
- Statistical calculations with optional bootstrap confidence intervals
- Visualization utilities
- Pre-configured analysis classes for standardized reporting
"""

from .agg_stats import calculate_statistics
from .monitoring import AnalysisDataBuilder
from .plotting import plot_segment_statistics, set_plot_theme
from .pre_configured import BaseAnalysis, BasicModelAnalysis, InsuranceModelAnalysis
from .segmentation import SegmentCategorical, SegmentCustom

__all__ = [
    "AnalysisDataBuilder",
    "BaseAnalysis",
    "BasicModelAnalysis",
    "InsuranceModelAnalysis",
    "SegmentCategorical",
    "SegmentCustom",
    "calculate_statistics",
    "plot_segment_statistics",
    "set_plot_theme",
]
