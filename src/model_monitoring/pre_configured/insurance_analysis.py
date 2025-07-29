"""Insurance Model Performance Analysis.

This module provides a pre-configured analysis class specifically designed for
insurance model performance monitoring with KPI calculations and visualizations.
"""

from pathlib import Path
from typing import Dict, List, Optional, Union

import pandas as pd

from ..segmentation import SegmentCategorical, SegmentCustom
from .base_analysis import BaseAnalysis


class InsuranceModelAnalysis(BaseAnalysis):
    """Pre-configured analysis for insurance model performance monitoring.

    This class implements a comprehensive analysis workflow specifically designed
    for insurance models, including prediction vs target comparisons, loss ratios,
    Gini coefficients, and exposure analysis across different segments.
    """

    def _get_default_pred_dict(self) -> Dict:
        """Define default prediction dictionary for insurance analysis."""
        return {
            "A": {
                "sel_col": "weight",
                "pred_col": "prediction_A",
                "target_col": "target_A",
            },
            "B": {
                "sel_col": "weight",
                "pred_col": "prediction_B",
                "target_col": "target_B",
            },
            "C": {
                "sel_col": "weight",
                "pred_col": "prediction_C",
                "target_col": "target_C",
            },
        }

    def _get_default_segments(self) -> List[Union[SegmentCategorical, SegmentCustom]]:
        """Define default segmentation strategies for insurance analysis."""
        return [
            # Segment for Age with custom bins
            SegmentCustom(
                seg_col="age",
                seg_name="age_group",
                bins=[18, 30, 45, 60, 75],
                bin_labels=["18-29", "30-44", "45-59", "60+"],
            ),
            # Segment for Income with 5 equal-width bins
            SegmentCustom(seg_col="income", seg_name="income_level", bins=5),
            # Segment for Region (each category is a segment)
            SegmentCategorical(seg_col="region", seg_name="region_segment"),
        ]

    def _get_default_func_dict(self) -> Dict:
        """Define default statistics dictionary for insurance analysis."""
        return {
            "aggregations": {
                "prediction_total": (
                    "weighted_mean",
                    ["prediction_total", "weight", "weight"],
                ),
                "prediction_total_iso": (
                    "weighted_mean",
                    ["prediction_total_iso", "weight", "weight"],
                ),
                "target_total": (
                    "observed_charge",
                    ["target_total", "weight", "weight"],
                ),
                "ELR": (
                    "e_lr",
                    ["prediction_total_iso", "weight", "weight", "market_premium"],
                ),
                "LR": ("lr", ["target_total", "weight", "weight", "market_premium"]),
                "gini": (
                    "gini",
                    ["target_total", "prediction_total", "weight", "weight"],
                ),
                "exposure(k)": (lambda df, e: df[e].sum() / 1000, ["weight"]),
            },
            "post_aggregations": {
                "S/PP": ("division", ["target_total", "prediction_total_iso"]),
                "diffLR-ELR": ("subtraction", ["LR", "ELR"]),
            },
        }

    def _get_default_report_panels(self) -> List[Dict]:
        """Define default report panels for insurance analysis."""
        return [
            {
                "title": "Prediction vs. Target",
                "type": "pred_vs_target",
                "pred_col": ["prediction_total", "prediction_total_iso"],
                "target_col": "target_total",
                "plot_type": "line",
                "show_mean_line": "all",
            },
            {
                "title": "S/PP (Observed/Predicted Ratio)",
                "type": "spp",
                "spp_col": ["S/PP"],
                "plot_type": "line",
                "show_mean_line": False,
            },
            {
                "title": "Loss Ratios",
                "type": "loss_ratios",
                "elr_col": ["ELR"],
                "lr_col": "LR",
                "plot_type": "line",
            },
            {
                "title": "Prediction vs. Target",  # Reused title for twin plot
                "type": "exposure",
                "metric_col": "exposure(k)",
                "plot_type": "bar",
                "colors": ["#9F9DAA"],
            },
        ]

    def _apply_treatments(self) -> None:
        """Apply insurance-specific data treatments."""
        if self.analysis_builder is None:
            raise ValueError("Analysis builder not initialized")

        # Add treatments for insurance analysis
        self.analysis_builder.add_treatment("IsoResourceScaling", self.pred_dict)
        self.analysis_builder.add_treatment("AddTotalCoverage", self.pred_dict)
        self.analysis_builder.add_treatment(
            "AddTotalCoverage", self.pred_dict, suffix="_iso"
        )

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
        """Initialize insurance model analysis.

        Args:
            data: Path to data file or pandas DataFrame
            pred_dict: Dictionary mapping coverages to prediction/target columns
            segments: List of segmentation strategies
            func_dict: Dictionary defining statistics to calculate
            report_panels: Configuration for visualization panels
            extra_cols: Additional columns to include in analysis
            bootstrap: Whether to use bootstrap for confidence intervals
            n_resamples: Number of bootstrap resamples
            ci_level: Confidence interval level
        """
        # Ensure market_premium is included in extra_cols for insurance analysis
        extra_cols = extra_cols or []
        if "market_premium" not in extra_cols:
            extra_cols.append("market_premium")

        super().__init__(
            data=data,
            pred_dict=pred_dict,
            segments=segments,
            func_dict=func_dict,
            report_panels=report_panels,
            extra_cols=extra_cols,
            bootstrap=bootstrap,
            n_resamples=n_resamples,
            ci_level=ci_level,
        )
