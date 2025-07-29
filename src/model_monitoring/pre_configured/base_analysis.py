"""Abstract base class for pre-configured analyses.

This module provides the foundation for creating reusable, customizable analysis classes
that wrap the model monitoring functionality with predictable outputs for reporting.
"""

import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd

from ..agg_stats import calculate_statistics
from ..monitoring import AnalysisDataBuilder
from ..plotting import plot_segment_statistics
from ..segmentation import SegmentCategorical, SegmentCustom


class BaseAnalysis(ABC):
    """Abstract base class for pre-configured analyses."""

    def __init__(
        self,
        data: Union[str, Path, pd.DataFrame],
        pred_dict: Optional[Dict] = None,
        segments: Optional[List[Union[SegmentCategorical, SegmentCustom]]] = None,
        func_dict: Optional[Dict] = None,
        report_panels: Optional[List[Dict]] = None,
        extra_cols: Optional[List[str]] = None,
        bootstrap: bool = False,
        n_resamples: int = 1000,
        ci_level: float = 0.95,
    ):
        """Initialize the analysis with data and configuration."""
        self.data = data
        self.extra_cols = extra_cols or []
        self.bootstrap = bootstrap
        self.n_resamples = n_resamples
        self.ci_level = ci_level

        # Use provided configurations or defaults
        self.pred_dict = (
            pred_dict if pred_dict is not None else self._get_default_pred_dict()
        )
        self.segments = (
            segments if segments is not None else self._get_default_segments()
        )
        self.func_dict = (
            func_dict if func_dict is not None else self._get_default_func_dict()
        )
        self.report_panels = (
            report_panels
            if report_panels is not None
            else self._get_default_report_panels()
        )

        # Initialize containers for results
        self.analysis_builder: Optional[AnalysisDataBuilder] = None
        self.dict_stats: Optional[Dict[str, pd.DataFrame]] = None
        self.agg_stats: Optional[pd.DataFrame] = None
        self.plots: Dict[str, Any] = {}

    @abstractmethod
    def _get_default_pred_dict(self) -> Dict:
        """Define default prediction dictionary for this analysis type."""
        pass

    @abstractmethod
    def _get_default_segments(self) -> List[Union[SegmentCategorical, SegmentCustom]]:
        """Define default segmentation strategies for this analysis type."""
        pass

    @abstractmethod
    def _get_default_func_dict(self) -> Dict:
        """Define default statistics dictionary for this analysis type."""
        pass

    @abstractmethod
    def _get_default_report_panels(self) -> List[Dict]:
        """Define default report panels for this analysis type."""
        pass

    def setup_analysis(self) -> None:
        """Set up the analysis by initializing the AnalysisDataBuilder."""
        # Initialize AnalysisDataBuilder
        self.analysis_builder = AnalysisDataBuilder(
            data=self.data, extra_cols=self.extra_cols
        )

        # Apply treatments (can be overridden in subclasses)
        self._apply_treatments()

        # Add segments
        for segment in self.segments:
            self.analysis_builder.add_segment(segment)

        # Load data and apply all transformations
        self.analysis_builder.load_data()
        self.analysis_builder.apply_treatments()
        self.analysis_builder.apply_segments()

    def _apply_treatments(self) -> None:
        """Apply data treatments. Default implementation does nothing."""
        # Default implementation - can be overridden in subclasses
        return

    def calculate_statistics(self) -> None:
        """Calculate statistics using the configured func_dict."""
        if self.analysis_builder is None:
            raise ValueError("Analysis must be set up first. Call setup_analysis().")

        self.dict_stats, self.agg_stats = calculate_statistics(
            self.analysis_builder,
            self.func_dict,
            bootstrap=self.bootstrap,
            n_resamples=self.n_resamples,
            ci_level=self.ci_level,
        )

    def generate_plots(self) -> Dict[str, Any]:
        """Generate all plots based on the report panels configuration."""
        if self.dict_stats is None or self.agg_stats is None:
            raise ValueError(
                "Statistics must be calculated first. Call calculate_statistics()."
            )

        self.plots = {}

        # Create report panels with bootstrap CI if enabled
        panels_config = self.report_panels.copy()
        if self.bootstrap:
            panels_config = [{**panel, "show_ci": True} for panel in panels_config]

        # Generate plots for each segment
        for segment_name, stats_df in self.dict_stats.items():
            try:
                plot_obj = plot_segment_statistics(
                    stats_df, panel_configs=panels_config, agg_stats=self.agg_stats
                )
                self.plots[segment_name] = plot_obj
            except Exception as e:
                print(
                    f"Warning: Could not generate plot for segment {segment_name}: {e}"
                )

        return self.plots

    def run_analysis(self) -> Dict[str, Any]:
        """Run the complete analysis pipeline."""
        self.setup_analysis()
        self.calculate_statistics()
        self.generate_plots()

        return self.get_output()

    def get_output(self) -> Dict[str, Any]:
        """Get standardized output for integration with larger reporting systems."""
        return {
            "plots": self.plots,
            "segment_statistics": self.dict_stats,
            "aggregate_statistics": self.agg_stats,
            "metadata": {
                "analysis_type": self.__class__.__name__,
                "bootstrap_enabled": self.bootstrap,
                "n_resamples": self.n_resamples if self.bootstrap else None,
                "ci_level": self.ci_level if self.bootstrap else None,
                "segments": [seg.seg_name for seg in self.segments],
                "coverages": list(self.pred_dict.keys()) if self.pred_dict else [],
            },
        }

    def save_data(self, filepath: Union[str, Path]) -> None:
        """Save the underlying data and statistics to a JSON file."""
        filepath = Path(filepath)

        # Helper function to safely convert DataFrame/Series to dict
        def safe_df_to_dict(df):
            if df is None:
                return None

            # Handle Series differently from DataFrame
            if isinstance(df, pd.Series):
                # Convert Series to dict, handling special index types
                series_copy = df.copy()
                # Convert any interval-type index to strings
                if (
                    hasattr(series_copy.index, "dtype")
                    and "interval" in str(series_copy.index.dtype).lower()
                ):
                    series_copy.index = series_copy.index.astype(str)
                return series_copy.to_dict()

            # Handle DataFrame
            df_copy = df.copy()
            # Convert any interval-type columns to strings
            for col in df_copy.columns:
                if (
                    hasattr(df_copy[col].dtype, "name")
                    and "interval" in str(df_copy[col].dtype).lower()
                ):
                    df_copy[col] = df_copy[col].astype(str)
            # Reset index to handle any special index types
            if df_copy.index.name:
                df_copy = df_copy.reset_index()
            return df_copy.to_dict()

        data_to_save = {
            "metadata": self.get_output()["metadata"],
            "aggregate_statistics": safe_df_to_dict(self.agg_stats),
            "segment_statistics": {
                name: safe_df_to_dict(df) for name, df in self.dict_stats.items()
            }
            if self.dict_stats
            else None,
        }

        with open(filepath, "w") as f:
            json.dump(data_to_save, f, indent=2, default=str)

        print(f"Analysis data saved to {filepath}")

    def get_segment_names(self) -> List[str]:
        """Get list of segment names used in the analysis."""
        return [seg.seg_name for seg in self.segments]

    def get_coverage_names(self) -> List[str]:
        """Get list of coverage names used in the analysis."""
        return list(self.pred_dict.keys()) if self.pred_dict else []
