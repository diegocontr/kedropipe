"""Analysis definitions for model validation pipeline."""

from .global_analysis import GlobalAnalysesRunner, build_and_run_global_analyses
from .segmented_analysis import SegmentedAnalysesRunner, build_and_run_segmented_analyses

__all__ = [
    "GlobalAnalysesRunner",
    "SegmentedAnalysesRunner",
    "build_and_run_global_analyses", 
    "build_and_run_segmented_analyses",
]