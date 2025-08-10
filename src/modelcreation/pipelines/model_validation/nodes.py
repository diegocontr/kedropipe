from __future__ import annotations

from typing import Any, Optional

import numpy as np
import pandas as pd

try:
    from modelcreation.pipelines.model_validation.analysis_definition import global_analysis as global_funcs  # type: ignore
except Exception:  # pragma: no cover
    from .analysis_definition import global_analysis as global_funcs  # type: ignore

# isort: off
try:
    from modelcreation.pipelines.model_validation.analysis_class import ModelAnalysis  # type: ignore
except Exception:  # pragma: no cover
    from .analysis_class import ModelAnalysis  # type: ignore
# isort: on


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
    test_dataset,
    train_dataset,
    target_column: str,
    prediction_column: str,
    run_id: Optional[str] = None,
    model_metrics: Optional[Any] = None,
    model_validation_params: Optional[dict] = None,
) -> None:
    """Execute configured global analyses using config-defined params."""
    from .config import global_analysis  # local import for registry

    context_args = {
        "test_dataset": test_dataset,
        "train_dataset": train_dataset,
        "target_column": target_column,
        "prediction_column": prediction_column,
        "run_id": run_id,
        "model_metrics": model_metrics,
    }
    if model_validation_params:
        context_args.update(model_validation_params)
        if "calibration_bins" in model_validation_params and "n_bins" not in context_args:
            context_args["n_bins"] = model_validation_params["calibration_bins"]

    resolved_run = _extract_run_id(context_args.get("run_id"), context_args.get("model_metrics"))

    for item in global_analysis:
        gen_func = getattr(global_funcs, item.get("func_name", ""), None)
        if gen_func is None:
            continue
        title = item.get("title", item.get("name", "Analysis"))
        inputs_spec = item.get("inputs", {})
        dataset_keys = item.get("datasets", ["test_dataset"])  # default
        analysis_cfg = item.get("analysis_params", {})

        for ds_key in dataset_keys:
            dataset = context_args.get(ds_key)
            if dataset is None:
                continue
            run_subset_label = ds_key.replace("_dataset", "")

            kwargs = {}
            for arg_name, spec in inputs_spec.items():
                spec_type = spec.get("type")
                if spec_type == "column":
                    col_name_key = spec.get("source")
                    if col_name_key not in context_args:
                        continue
                    col_name = context_args[col_name_key]
                    if col_name not in dataset.columns:
                        continue
                    series = dataset[col_name]
                    thr_param = spec.get("binarize_threshold_param")
                    if thr_param and thr_param in context_args:
                        thr_val = context_args.get(thr_param) or 0.0
                        series = _binarize_target(series, thr_val)
                    kwargs[arg_name] = series
                elif spec_type == "param":
                    kwargs[arg_name] = context_args.get(spec.get("source"))
                elif spec_type == "literal":
                    kwargs[arg_name] = spec.get("value")
                else:
                    continue
            if not kwargs:
                continue
            analysis = ModelAnalysis(gen_func, title, config=analysis_cfg)
            analysis.run_analysis(run_subset_label, **kwargs)
            analysis.save_to_mlflow(identifier_run=resolved_run)


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

    train_out[prediction_column] = trained_model.predict(train_dataset)
    test_out[prediction_column] = trained_model.predict(test_dataset)

    if old_model_column:
        if old_model_column not in train_out.columns or old_model_column not in test_out.columns:
            noise_factor = float(old_model_noise_factor or 0.0)
            for df in (train_out, test_out):
                base = df[prediction_column].astype(float)
                if noise_factor > 0:
                    noise = rng.normal(0.0, noise_factor, size=len(base))
                    old_vals = base * (1.0 + noise)
                else:
                    old_vals = base * 0.9  # simple deterministic baseline
                # Clip negatives if target appears non-negative
                if (df.get('target_B') is not None) and (df['target_B'].min() >= 0):
                    old_vals = np.clip(old_vals, 0, None)
                df[old_model_column] = old_vals

    return train_out, test_out
