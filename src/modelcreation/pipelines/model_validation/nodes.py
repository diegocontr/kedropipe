from __future__ import annotations

from typing import Any, Optional

import numpy as np
import pandas as pd

# (ModelAnalysis now only used inside analysis_definition builders; no direct import needed here.)


def _extract_run_id(run_id: Optional[str], model_metrics: Optional[Any]) -> Optional[str]:
    """Best-effort extraction of run_id from explicit param or model_metrics JSON/text."""
    if run_id:
        return run_id
    if model_metrics is None:
        return None
    try:
        if isinstance(model_metrics, str):
            import json as _json

            data = _json.loads(model_metrics)
        elif isinstance(model_metrics, dict):
            data = model_metrics
        else:
            return None
        return data.get("mlflow_run_id")
    except Exception:
        return None


def _binarize_target(series, threshold: float) -> pd.Series:
    return (series > threshold).astype(int)


def run_global_analyses(
    *,
    train_df_path: str,
    test_df_path: str,
    target_column: str,
    prediction_column: str,
    old_model_column: Optional[str] = None,
    run_id: Optional[str] = None,
    model_metrics: Optional[Any] = None,
    model_validation_params: Optional[dict] = None,
) -> None:
    """Run global analyses using mandatory parquet paths (defined in parameters)."""
    from .analysis_definition.global_analysis import build_and_run_global_analyses

    if old_model_column is None and model_validation_params:
        old_model_column = model_validation_params.get("old_model_column")

    build_and_run_global_analyses(
        train_df_path=train_df_path,
        test_df_path=test_df_path,
        target_column=target_column,
        prediction_column=prediction_column,
        old_model_column=old_model_column,
        params=model_validation_params,
        run_id=run_id,
        resolved_run_extractor=_extract_run_id,
        model_metrics=model_metrics,
    )


def generate_predictions(
    trained_model,
    test_dataset: pd.DataFrame,
    train_dataset: pd.DataFrame,
    prediction_column: str,
    old_model_column: str | None = None,
    old_model_noise_factor: float | None = None,
    random_state: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Add prediction column to both train and test datasets.

    Optionally create a synthetic old model prediction column (old_model_column) for
    side-by-side comparison in analyses if it does not already exist.

    Synthetic old model logic:
      - If old_model_noise_factor > 0: old_pred = new_pred * (1 + Normal(0, noise_factor)).
      - Else: old_pred = new_pred * 0.9 (simple under-performing baseline).
    Values are clipped at 0 for non-negative targets (Poisson-like use case).
    """
    rng = np.random.default_rng(random_state)

    train_out = train_dataset.copy()
    test_out = test_dataset.copy()

    # Use only feature columns present in model if attribute exists
    try:
        feature_names = getattr(trained_model, 'feature_names_', None)
        if feature_names:
            train_pred_input = train_dataset[feature_names]
            test_pred_input = test_dataset[feature_names]
        else:
            train_pred_input = train_dataset
            test_pred_input = test_dataset
    except Exception:
        train_pred_input = train_dataset
        test_pred_input = test_dataset

    train_out[prediction_column] = trained_model.predict(train_pred_input)
    test_out[prediction_column] = trained_model.predict(test_pred_input)

    # Add weight column if missing (defaults to 1.0 for all rows)
    if "weight" not in train_out.columns:
        train_out["weight"] = 1.0
    if "weight" not in test_out.columns:
        test_out["weight"] = 1.0

    # Do NOT synthesize old model predictions automatically; only use if column already exists
    if old_model_column:
        missing = []
        if old_model_column not in train_out.columns:
            missing.append("train")
        if old_model_column not in test_out.columns:
            missing.append("test")
        if missing:
            print(
                f"Warning: old_model_column '{old_model_column}' missing in {', '.join(missing)} dataset(s). Global analyses will show only the new model."
            )

    print(f"[generate_predictions] Output train columns: {list(train_out.columns)}")
    print(f"[generate_predictions] Output test columns: {list(test_out.columns)}")
    if old_model_column:
        present_train = old_model_column in train_out.columns
        present_test = old_model_column in test_out.columns
        print(f"[generate_predictions] old_model_column='{old_model_column}' train_present={present_train} test_present={present_test}")
    return train_out, test_out
