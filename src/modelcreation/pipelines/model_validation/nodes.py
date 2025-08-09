from __future__ import annotations

from typing import Any, Optional

import pandas as pd

try:
    from modelcreation.pipelines.model_validation.analysis_definition import global_analysis  # type: ignore
except Exception:  # pragma: no cover
    from .analysis_definition import global_analysis  # type: ignore

# isort: off
try:
    # When package is installed/available on sys.path
    from modelcreation.pipelines.model_validation.analysis_class import ModelAnalysis  # type: ignore
except Exception:  # pragma: no cover
    # Fallback for relative import during development/editing
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


def run_registered_analyses(
    *,
    validation_dataset,
    target_column: str,
    prediction_column: str,
    roc_threshold: Optional[float] = 0.0,
    n_bins: int = 10,
    run_id: Optional[str] = None,
    model_metrics: Optional[Any] = None,
) -> None:
    """Iterate through analysis registry and execute each analysis runner.

    This provides a single Kedro node entrypoint; registry is imported at call time
    to avoid circular imports.
    """
    from .config import analysis_to_run  # local import

    context_args = {
        "validation_dataset": validation_dataset,
        "target_column": target_column,
        "prediction_column": prediction_column,
        "roc_threshold": roc_threshold,
        "n_bins": n_bins,
        "run_id": run_id,
        "model_metrics": model_metrics,
    }

    dataset = context_args["validation_dataset"]
    resolved_run = _extract_run_id(context_args["run_id"], context_args["model_metrics"])

    for item in analysis_to_run:
        gen_func = getattr(global_analysis, item.get("func_name", ""), None)
        if gen_func is None:
            continue
        title = item.get("title", item.get("name", "Analysis"))
        inputs_spec = item.get("inputs", {})

        # Build kwargs dynamically
        kwargs = {}
        for arg_name, spec in inputs_spec.items():
            spec_type = spec.get("type")
            if spec_type == "column":
                col_name_key = spec.get("source")
                if col_name_key not in context_args:
                    continue
                col_name = context_args[col_name_key]
                series = dataset[col_name]
                # Optional binarization
                thr_param = spec.get("binarize_threshold_param")
                if thr_param:
                    thr_val = context_args.get(thr_param) or 0.0
                    series = _binarize_target(series, thr_val)
                kwargs[arg_name] = series
            elif spec_type == "param":
                src = spec.get("source")
                kwargs[arg_name] = context_args.get(src)
            elif spec_type == "literal":
                kwargs[arg_name] = spec.get("value")
            else:
                # Unsupported spec type; skip this arg
                continue

        analysis = ModelAnalysis(gen_func, title)
        analysis.run_analysis("validation_set", **kwargs)
        analysis.save_to_mlflow(identifier_run=resolved_run)
