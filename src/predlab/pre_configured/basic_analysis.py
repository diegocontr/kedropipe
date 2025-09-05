"""Basic Model Performance Analysis.

This module provides a simplified pre-configured analysis class for basic
model performance monitoring with essential KPIs and simple visualizations.
"""

from typing import Dict, List, Union

from ..segmentation import SegmentCategorical, SegmentCustom
from .base_analysis import BaseAnalysis


class BasicModelAnalysis(BaseAnalysis):
    """Pre-configured analysis for basic model performance monitoring.

    This class implements a simplified analysis workflow focusing on
    core model performance metrics like prediction accuracy, basic ratios,
    and simple segmentation analysis.
    """

    def _get_default_pred_dict(self) -> Dict:
        """Define default prediction dictionary for basic analysis."""
        return {
            "total": {
                "sel_col": "weight",
                "pred_col": "prediction_total",
                "target_col": "target_total",
            },
        }

    def _get_default_segments(self) -> List[Union[SegmentCategorical, SegmentCustom]]:
        """Define default segmentation strategies for basic analysis."""
        return [
            # Simple age segmentation
            SegmentCustom(
                seg_col="age",
                seg_name="age_group",
                bins=[18, 35, 50, 75],
                bin_labels=["Young", "Middle", "Senior"],
            ),
        ]

    def _get_default_func_dict(self) -> Dict:
        """Define default statistics dictionary for basic analysis."""
        return {
            "aggregations": {
                "prediction_total": (
                    "weighted_mean",
                    ["prediction_total", "weight", "weight"],
                ),
                "target_total": (
                    "observed_charge",
                    ["target_total", "weight", "weight"],
                ),
                "gini": (
                    "gini",
                    ["target_total", "prediction_total", "weight", "weight"],
                ),
                "exposure": (lambda df, e: df[e].sum(), ["weight"]),
            },
            "post_aggregations": {
                "ratio": ("division", ["target_total", "prediction_total"]),
            },
        }

    def _get_default_report_panels(self) -> List[Dict]:
        """Define default report panels for basic analysis."""
        return [
            {
                "title": "Prediction vs. Target",
                "type": "pred_vs_target",
                "pred_col": ["prediction_total"],
                "target_col": "target_total",
                "plot_type": "line",
                "show_mean_line": "all",
            },
            {
                "title": "Observed/Predicted Ratio",
                "type": "spp",
                "spp_col": ["ratio"],
                "plot_type": "line",
                "show_mean_line": False,
            },
            {
                "title": "Gini Coefficient",
                "type": "metric",
                "metric_col": "gini",
                "plot_type": "line",
                "colors": ["#3D27B9"],
            },
        ]

    def _apply_treatments(self) -> None:
        """Apply basic data treatments."""
        if self.analysis_builder is None:
            raise ValueError("Analysis builder not initialized")

        # Add basic treatment for totals
        self.analysis_builder.add_treatment("AddTotalCoverage", self.pred_dict)
