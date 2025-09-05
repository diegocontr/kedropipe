from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from ..segmentation import _format_edge, compute_percentile_bins
from .analyses import ModelAnalysis, register_analysis, resolve_model_config


class PDPAnalysis(ModelAnalysis):
    """Partial Dependence Plot (PDP) for a single model.

    Output columns: ["segment", "bin", "pdp", "prediction", "eval_value"]
    - segment: the segment name (e.g., "age_group", "income_level")
    - bin: the bin label (string)
    - eval_value: the numeric/category value used to evaluate the model (e.g., midpoint or category)
    - prediction: alias for pdp (kept for clarity; same as pdp)
    - pdp: weighted mean prediction with the feature fixed at eval_value
    """

    def __init__(self, config: Dict):
        """Initialize PDP analysis with configuration resolved for model usage."""
        super().__init__(config)
        self._cfg = resolve_model_config(config)

    @staticmethod
    def required_columns(config: Dict) -> List[str]:
        cfg = resolve_model_config(config)
        cols = set(cfg.feature_cols)
        if cfg.weight_col:
            cols.add(cfg.weight_col)
        return list(cols)

    def _weighted_mean(self, arr: np.ndarray, weights: Optional[np.ndarray]) -> float:
        if weights is None:
            return float(np.mean(arr))
        w_sum = float(np.sum(weights))
        if w_sum == 0:
            return float(np.mean(arr))
        return float(np.sum(arr * weights) / w_sum)

    @staticmethod
    def _format_interval_label(a: float, b: float) -> str:
        return f"[{a}, {b})"

    def _grid_from_segment(self, df: pd.DataFrame, segment) -> Optional[pd.DataFrame]:
        """Return a DataFrame with columns [feature, segment, value, bin] describing PDP grid."""
        # SegmentCustom (has bins and seg_col)
        if (
            hasattr(segment, "bins")
            and hasattr(segment, "seg_col")
            and hasattr(segment, "seg_name")
        ):
            col = segment.seg_col
            seg_name = segment.seg_name
            bins = segment.bins
            sig_digits = getattr(segment, "sig_digits", 3)
            round_bounds = getattr(segment, "round_bounds", False)

            if isinstance(bins, int):
                # Percentile-based bins with consistent label rounding
                edges, auto_labels = compute_percentile_bins(
                    df[col],
                    n_bins=bins,
                    sig_digits=sig_digits,
                    round_bounds=round_bounds,
                )
                edges_np = np.asarray(edges, dtype=float)
                if edges_np.size < 2:
                    return None
                mids = (edges_np[:-1] + edges_np[1:]) / 2.0
                # Prefer user-provided labels if present; else use auto labels
                if getattr(segment, "bin_labels", None) is not None:
                    labels = list(segment.bin_labels)
                    if len(labels) != len(mids):
                        raise ValueError(
                            "The number of bin_labels must match the number of intervals derived for PDP."
                        )
                else:
                    labels = auto_labels
            else:
                # explicit edges
                edges = np.asarray(bins, dtype=float)
                if len(edges) < 2:
                    return None
                mids = (edges[:-1] + edges[1:]) / 2.0
                if getattr(segment, "bin_labels", None):
                    labels = list(segment.bin_labels)
                else:
                    # Format labels using significant digits for readability
                    labels = [
                        f"[{_format_edge(float(edges[i]), sig_digits)}, {_format_edge(float(edges[i + 1]), sig_digits)})"
                        for i in range(len(mids))
                    ]

            return pd.DataFrame(
                {
                    "feature": col,
                    "segment": seg_name,
                    "value": mids,
                    "bin": labels,
                }
            )

        # SegmentCategorical
        if (
            hasattr(segment, "mapping")
            and hasattr(segment, "seg_col")
            and hasattr(segment, "seg_name")
        ):
            col = segment.seg_col
            seg_name = segment.seg_name
            cats = sorted(df[col].dropna().unique())
            return pd.DataFrame(
                {
                    "feature": col,
                    "segment": seg_name,
                    "value": cats,
                    "bin": [str(c) for c in cats],
                }
            )

        return None

    def run(self, df: pd.DataFrame, segments) -> None:
        model = self._cfg.model
        feature_cols = self._cfg.feature_cols
        weight_col = self._cfg.weight_col

        X_base = df[feature_cols].copy()
        weights = (
            df[weight_col].to_numpy()
            if weight_col and weight_col in df.columns
            else None
        )

        rows: List[Dict] = []

        for seg in segments:
            grid_df = self._grid_from_segment(df, seg)
            if grid_df is None:
                continue
            feat = grid_df["feature"].iloc[0]
            if feat not in feature_cols:
                continue

            for _, r in grid_df.iterrows():
                v = r["value"]
                X = X_base.copy()
                X[feat] = v
                preds = model.predict(X)
                preds = np.asarray(preds, dtype=float).reshape(-1)
                pdp_val = self._weighted_mean(preds, weights)
                rows.append(
                    {
                        "segment": r["segment"],
                        "bin": r["bin"],
                        "eval_value": v,
                        "prediction": pdp_val,
                        "pdp": pdp_val,
                    }
                )

        self._data = (
            pd.DataFrame(
                rows, columns=["segment", "bin", "eval_value", "prediction", "pdp"]
            )
            if rows
            else pd.DataFrame(
                columns=["segment", "bin", "eval_value", "prediction", "pdp"]
            )
        )


# Register
register_analysis("PDP", PDPAnalysis)
