from __future__ import annotations

from typing import Optional

from datetime import datetime
import logging
import mlflow
import numpy as np
import pandas as pd

# (ModelAnalysis now only used inside analysis_definition builders; no direct import needed here.)


def _extract_run_id(run_id: Optional[str]) -> Optional[str]:
    """Identity extractor kept for compatibility (metrics-based fallback removed)."""
    return run_id


def _binarize_target(series, threshold: float) -> pd.Series:
    return (series > threshold).astype(int)


def start_mlflow_run(experiment_name: str | None) -> tuple[str, object]:
    """Ensure an experiment is selected and return an active MLflow run id and saver.

    Rules:
      - If a run is already active (e.g. training kept it open or prior node), reuse it.
      - Otherwise start a new run in the selected experiment.
    This prevents duplicate mlflow.start_run() calls causing 'already active' errors.
    
    Returns:
        tuple: (run_id, mlflow_saver) - The run ID and MLflow artifact saver instance
    """
    from .mlflow_saver import MLflowArtifactSaver
    
    if not experiment_name:
        experiment_name = f"validation_experiment_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
    mlflow.set_experiment(experiment_name)
    active = mlflow.active_run()
    if active:
        run_id = active.info.run_id
    else:
        run = mlflow.start_run()
        run_id = run.info.run_id
    
    # Create MLflow saver with the run ID
    mlflow_saver = MLflowArtifactSaver(run_id=run_id)
    
    return run_id, mlflow_saver


def end_mlflow_run(run_id: str | None = None) -> None:  # run_id accepted for dependency but unused
    """End the active MLflow run if present."""
    active = mlflow.active_run()
    if active:
        try:
            mlflow.end_run()
        except Exception as exc:
            logging.debug("Failed to end MLflow run %s: %s", active.info.run_id, exc)


def run_global_analyses(
    *,
    mlflow_saver,  # MLflow saver passed from start_mlflow_run node
    train_df_path: str,
    test_df_path: str,
    # New consolidated feature configuration dict. Expected keys: target, weight (optional), old_model (optional), prediction (optional), model_features (optional)
    feat_conf: Optional[dict] = None,
    # Deprecated explicit params (kept for backward compatibility)
    target_column: Optional[str] = None,
    prediction_column: Optional[str] = None,
    old_model_column: Optional[str] = None,
    run_id: Optional[str] = None,
    global_analysis_config: Optional[dict] = None,
) -> None:
    """Run global analyses using mandatory parquet paths (defined in parameters)."""
    from .analysis_definition.global_analysis import build_and_run_global_analyses

    # Extract columns from feat_conf
    target_column = feat_conf.get("target", target_column)
    old_model_column = feat_conf.get("old_model", old_model_column)
    prediction_column = feat_conf.get("prediction", prediction_column)
    weight_column = feat_conf.get("weight") if feat_conf else None

    if target_column is None or prediction_column is None:
        raise ValueError("target_column and prediction_column must be provided via feat_conf or legacy params.")

    # Check if global analysis is enabled
    if not global_analysis_config.get("enabled", True):
        print("[run_global_analyses] Skipping global analysis - disabled in configuration")
        return

    # Merge all available parameters for the analysis
    merged_params = {**(global_analysis_config or {})}
    
    # Add model features from feat_conf if available
    if feat_conf and "model_features" in feat_conf:
        merged_params.setdefault("data_preparation", {})["feature_columns"] = feat_conf["model_features"]

    build_and_run_global_analyses(
        mlflow_saver=mlflow_saver,
        train_df_path=train_df_path,
        test_df_path=test_df_path,
        target_column=target_column,
        prediction_column=prediction_column,
        old_model_column=old_model_column,
        weight_column=weight_column,
        params=merged_params,
        run_id=run_id,
        resolved_run_extractor=_extract_run_id,
    )


def run_segmented_analyses(
    *,
    mlflow_saver,  # MLflow saver passed from start_mlflow_run node
    train_df_path: str,
    test_df_path: str,
    feat_conf: Optional[dict] = None,
    target_column: Optional[str] = None,
    prediction_column: Optional[str] = None,
    old_model_column: Optional[str] = None,
    run_id: Optional[str] = None,
    segmented_analysis_config: Optional[dict] = None,
) -> None:
    """Run segmented analyses using mandatory parquet paths (defined in parameters)."""
    from .analysis_definition.segmented_analysis import build_and_run_segmented_analyses

    # Extract columns from feat_conf
    target_column = feat_conf.get("target", target_column)
    old_model_column = feat_conf.get("old_model", old_model_column)
    prediction_column = feat_conf.get("prediction", prediction_column)
    weight_column = feat_conf.get("weight") if feat_conf else None

    if target_column is None or prediction_column is None:
        raise ValueError("target_column and prediction_column must be provided via feat_conf or legacy params.")

    # Check if segmented analysis is enabled
    if not segmented_analysis_config.get("enabled", True):
        print("[run_segmented_analyses] Skipping segmented analysis - disabled in configuration")
        return

    # Merge all available parameters for the analysis
    merged_params = {**(segmented_analysis_config or {})}
    
    # Add model features from feat_conf if available
    if feat_conf and "model_features" in feat_conf:
        merged_params.setdefault("data_preparation", {})["feature_columns"] = feat_conf["model_features"]
    
    # Add categorical features from feat_conf if available
    if feat_conf and "categorical_features" in feat_conf:
        merged_params["categorical_features"] = feat_conf["categorical_features"]

    build_and_run_segmented_analyses(
        mlflow_saver=mlflow_saver,
        train_df_path=train_df_path,
        test_df_path=test_df_path,
        target_column=target_column,
        prediction_column=prediction_column,
        old_model_column=old_model_column,
        weight_column=weight_column,
        params=merged_params,
        run_id=run_id,
        resolved_run_extractor=_extract_run_id,
    )


def run_pdp_analyses(
    *,
    mlflow_saver,  # MLflow saver passed from start_mlflow_run node
    train_df_path: str,
    test_df_path: str,
    feat_conf: Optional[dict] = None,
    target_column: Optional[str] = None,
    prediction_column: Optional[str] = None,
    trained_model,
    old_model_column: Optional[str] = None,
    run_id: Optional[str] = None,
    pdp_analysis_config: Optional[dict] = None,
) -> None:
    """Run PDP analyses using mandatory parquet paths and trained model."""
    from .analysis_definition.pdp_analysis import build_and_run_pdp_analyses

    # Extract columns from feat_conf
    target_column = feat_conf.get("target", target_column)
    old_model_column = feat_conf.get("old_model", old_model_column)
    prediction_column = feat_conf.get("prediction", prediction_column)
    weight_column = feat_conf.get("weight") if feat_conf else None

    if target_column is None or prediction_column is None:
        raise ValueError("target_column and prediction_column must be provided via feat_conf or legacy params.")

    # Check if PDP analysis is enabled
    if not pdp_analysis_config.get("enabled", True):
        print("[run_pdp_analyses] Skipping PDP analysis - disabled in configuration")
        return

    # Merge all available parameters for the analysis
    merged_params = {**(pdp_analysis_config or {})}
    
    # Add model features from feat_conf if available
    if feat_conf and "model_features" in feat_conf:
        merged_params.setdefault("data_preparation", {})["feature_columns"] = feat_conf["model_features"]

    build_and_run_pdp_analyses(
        mlflow_saver=mlflow_saver,
        train_df_path=train_df_path,
        test_df_path=test_df_path,
        target_column=target_column,
        prediction_column=prediction_column,
        trained_model=trained_model,
        old_model_column=old_model_column,
        weight_column=weight_column,
        params=merged_params,
        run_id=run_id,
        resolved_run_extractor=_extract_run_id,
    )


def generate_predictions(
    trained_model,
    test_dataset: pd.DataFrame,
    train_dataset: pd.DataFrame,
    prediction_column: str,
    old_model_column: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Add prediction column to both train and test datasets.

    For real datasets:
    - old_model_column should either exist in the dataset (historical predictions) or be None
    - No synthetic data generation needed - just use the model to predict on real features
    - Weight column will be added with default value 1.0 if not present
    
    For toy datasets (current implementation):
    - Synthetic old model predictions are generated for comparison purposes
    - This entire synthetic generation logic should be removed for real datasets
    
    Migration notes:
    - When moving to real datasets, remove all "TOY DATASET" blocks
    - The function will become much simpler - just model.predict() on features
    - Historical model comparisons (if needed) should use actual historical predictions
    """
    # ==================== TOY DATASET SYNTHETIC DATA GENERATION ====================
    # TODO: Remove this entire block when moving to real datasets
    # Real datasets should have actual historical predictions or no old model comparison
    old_model_noise_factor = None  # Hardcoded for toy dataset
    random_state = 42  # Hardcoded for toy dataset
    
    rng = np.random.default_rng(random_state)
    # =============================================================================
    
    train_out = train_dataset.copy()
    test_out = test_dataset.copy()

    feature_names = getattr(trained_model, 'feature_names_', None)
    train_pred_input = train_dataset[feature_names]
    test_pred_input = test_dataset[feature_names]

    train_out[prediction_column] = trained_model.predict(train_pred_input)
    test_out[prediction_column] = trained_model.predict(test_pred_input)



    # ==================== TOY DATASET OLD MODEL HANDLING ====================
    # TODO: Remove this block when moving to real datasets
    # For real datasets: either use existing old_model_column or skip old model comparison
    # Add weight column if missing (defaults to 1.0 for all rows)
    if "weight" not in train_out.columns:
        train_out["weight"] = 1.0
    if "weight" not in test_out.columns:
        test_out["weight"] = 1.0

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
    # =========================================================================

    print(f"[generate_predictions] Output train columns: {list(train_out.columns)}")
    print(f"[generate_predictions] Output test columns: {list(test_out.columns)}")
    if old_model_column:
        present_train = old_model_column in train_out.columns
        present_test = old_model_column in test_out.columns
        print(f"[generate_predictions] old_model_column='{old_model_column}' train_present={present_train} test_present={present_test}")
    return train_out, test_out
