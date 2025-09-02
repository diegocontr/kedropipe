from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Type

import pandas as pd


class GlobalAnalysis:
    """Base interface for global analyses.

    Usage pattern:
    - required_columns(config): class/staticmethod to list needed columns
    - instance = AnalysisClass(config)
    - instance.run(df): computes and stores internal results
    - instance.get_data(): returns the stored DataFrame result
    """

    def __init__(self, config: Dict):
        """Store analysis configuration and initialize result holder."""
        self.config: Dict = config
        self._data: Optional[pd.DataFrame] = None

    @staticmethod
    def required_columns(config: Dict) -> List[str]:
        raise NotImplementedError

    def run(self, df: pd.DataFrame) -> None:
        raise NotImplementedError

    def get_data(self) -> pd.DataFrame:
        if self._data is None:
            raise RuntimeError(
                "Analysis has not been executed yet. Call run(df) first."
            )
        return self._data

    def plot(self, ax, theme: Dict, color_provider, panel_cfg: Optional[Dict] = None):
        """Optional plotting method to be implemented by subclasses.

        Parameters:
        - ax: matplotlib Axes to draw on
        - theme: plotting theme dictionary (e.g., from _PLOT_CONFIG)
        - color_provider: callable n -> list of colors
        - panel_cfg: optional per-panel overrides
        """
        raise NotImplementedError("plot is not implemented for this analysis.")

    def plot_data(self, *args, **kwargs):
        """Deprecated. Use plot(...) instead."""
        raise NotImplementedError("plot_data is not implemented for this analysis.")


@dataclass
class _ResolvedColumns:
    default_target_col: Optional[str]
    default_weight_col: Optional[str]
    # (model_key, pred_col, weight_col, target_col, display_name)
    model_columns: List[Tuple[str, str, Optional[str], str, str]]


class _ColumnResolver:
    """Utility to resolve required columns from a configuration dictionary.

    Expected config schema (flexible, defaults applied):
    {
        'models': {
            'Model A': {
                'pred_col': 'prediction_A',
                'weight_col': 'weight' (optional),
                'target_col': 'target_A' (optional),
                'name': 'model A' (optional display/suffix)
            },
            'Model B': {'pred_col': 'prediction_A_comp'}
        },
        'observation': {
            'target_col': 'target_A',
            'weight_col': 'weight' (optional)
        },
        'weight_col': 'weight' (optional global default),
        'n_bins': 20 (optional),
        'ascending': True (optional)
    }
    """

    @staticmethod
    def resolve(config: Dict) -> _ResolvedColumns:
        if (
            "models" not in config
            or not isinstance(config["models"], dict)
            or len(config["models"]) == 0
        ):
            raise ValueError("config must contain a non-empty 'models' dictionary")
        obs = config.get("observation", {})
        default_target = obs.get("target_col")

        default_weight = config.get("weight_col", obs.get("weight_col"))
        model_columns: List[Tuple[str, str, Optional[str], str, str]] = []
        for model_key, spec in config["models"].items():
            if not isinstance(spec, dict) or "pred_col" not in spec:
                raise ValueError(
                    f"Each model spec must be a dict with 'pred_col'. Problem with model '{model_key}'."
                )
            pred_col = spec["pred_col"]
            model_w = spec.get("weight_col", default_weight)
            t_col = spec.get("target_col", default_target)
            if not isinstance(t_col, str) or not t_col:
                raise ValueError(
                    f"Target column not specified for model '{model_key}' and no default observation.target_col provided."
                )
            display_name = spec.get("name", model_key)
            model_columns.append((model_key, pred_col, model_w, t_col, display_name))

        return _ResolvedColumns(
            default_target_col=default_target,
            default_weight_col=default_weight,
            model_columns=model_columns,
        )

    @staticmethod
    def required_columns(config: Dict) -> List[str]:
        r = _ColumnResolver.resolve(config)
        cols = set()
        if r.default_target_col:
            cols.add(r.default_target_col)
        if r.default_weight_col:
            cols.add(r.default_weight_col)
        for _, pred_c, w_c, t_c, _ in r.model_columns:
            cols.add(pred_c)
            cols.add(t_c)
            if w_c:
                cols.add(w_c)
        return list(cols)


# Import concrete analyses and build registry
from .calibration import CalibrationCurveAnalysis
from .lorenz import LorenzCurveAnalysis
from .prediction import PredictionAnalysis

ANALYSIS_REGISTRY: Dict[str, Type[GlobalAnalysis]] = {
    "lorenz_curve": LorenzCurveAnalysis,
    "calibration_curve": CalibrationCurveAnalysis,
    "prediction_analysis": PredictionAnalysis,
}


def get_available_analyses() -> List[str]:
    return list(ANALYSIS_REGISTRY.keys())
