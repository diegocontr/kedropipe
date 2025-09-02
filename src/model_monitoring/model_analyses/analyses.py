from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Type

import pandas as pd


class ModelAnalysis:
    """Base interface for model-based analyses (e.g., PDP).

    Usage pattern:
    - required_columns(config): class/staticmethod to list needed columns
    - instance = AnalysisClass(config)
    - instance.run(df, segments): computes and stores internal results
    - instance.get_data(): returns the stored DataFrame result
    """

    def __init__(self, config: Dict):
        """Store analysis configuration and initialize result holder."""
        self.config: Dict = config
        self._data: Optional[pd.DataFrame] = None

    @staticmethod
    def required_columns(config: Dict) -> List[str]:
        raise NotImplementedError

    def run(self, df: pd.DataFrame, segments) -> None:
        raise NotImplementedError

    def get_data(self) -> pd.DataFrame:
        if self._data is None:
            raise RuntimeError(
                "Analysis has not been executed yet. Call run(df, segments) first."
            )
        return self._data


@dataclass
class _ResolvedModelCfg:
    model: object
    feature_cols: List[str]
    weight_col: Optional[str]
    target_col: Optional[str]
    name: str


def _get_feature_names_from_model(model) -> Optional[List[str]]:
    """Attempt to extract feature names from common model types (CatBoost, sklearn, etc.)."""
    # CatBoost stores feature names in feature_names_ when fitted from a DataFrame
    names = getattr(model, "feature_names_", None)
    if names and isinstance(names, (list, tuple)):
        return list(names)
    # Fallbacks
    return None


def resolve_model_config(config: Dict) -> _ResolvedModelCfg:
    if "model" not in config or not isinstance(config["model"], dict):
        raise ValueError(
            "config must contain a 'model' dictionary with at least the model instance under key 'model'"
        )

    m_cfg = config["model"]
    model = m_cfg.get("model")
    if model is None:
        raise ValueError(
            "config['model'] must contain the trained model under key 'model'"
        )

    # Feature columns: prefer explicit, otherwise try to infer from model
    feature_cols = m_cfg.get("feature_cols")
    if feature_cols is None:
        feature_cols = _get_feature_names_from_model(model)
    if not feature_cols:
        raise ValueError(
            "Feature columns could not be resolved. Provide config['model']['feature_cols'] or use a model that exposes feature names."
        )

    weight_col = config.get("weight_col")
    target_col = config.get("target_col")
    name = m_cfg.get("name", "model")

    return _ResolvedModelCfg(
        model=model,
        feature_cols=list(feature_cols),
        weight_col=weight_col,
        target_col=target_col,
        name=name,
    )


# Registry for analyses in this submodule
ANALYSIS_REGISTRY: Dict[str, Type[ModelAnalysis]] = {}


def register_analysis(name: str, cls: Type[ModelAnalysis]):
    if name in ANALYSIS_REGISTRY:
        raise ValueError(f"Analysis '{name}' already registered in model_analyses.")
    ANALYSIS_REGISTRY[name] = cls


def get_available_analyses() -> List[str]:
    return list(ANALYSIS_REGISTRY.keys())
