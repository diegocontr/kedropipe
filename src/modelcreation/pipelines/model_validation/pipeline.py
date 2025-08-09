from kedro.pipeline import Pipeline, node

# isort: off
try:
    from modelcreation.pipelines.model_validation.nodes import run_registered_analyses  # type: ignore
except Exception:  # pragma: no cover
    from .nodes import run_registered_analyses  # type: ignore
# isort: on

def create_pipeline(**kwargs):
    return Pipeline(
        [
            node(
                func=run_registered_analyses,
                inputs={
                    "validation_dataset": "validation_dataset",
                    "target_column": "params:model_validation.target_column",
                    "prediction_column": "params:model_validation.prediction_column",
                    "roc_threshold": "params:model_validation.roc_threshold",
                    "n_bins": "params:model_validation.calibration_bins",
                    "run_id": "params:model_validation.mlflow_run_id",
                    "model_metrics": "model_metrics",
                },
                outputs=None,
                name="run_registered_analyses",
            ),
        ]
    )
