from kedro.pipeline import Pipeline, node

from .nodes import run_global_analyses, generate_predictions


def resolve_mlflow_run_id(run_id: str | None) -> str:
    return run_id or ""


def create_pipeline(**kwargs):
    return Pipeline(
        [
            node(
                func=resolve_mlflow_run_id,
                inputs="params:model_validation.mlflow_run_id",
                outputs="resolved_mlflow_run_id",
                name="resolve_mlflow_run_id",
                tags=["memory_only"],
            ),
            node(
                func=generate_predictions,
                inputs={
                    "trained_model": "trained_model",
                    "test_dataset": "test_dataset",
                    "train_dataset": "train_dataset",
                    "prediction_column": "params:model_validation.prediction_column",
                    "old_model_column": "params:model_validation.old_model_column",
                    "old_model_noise_factor": "params:model_validation.old_model_noise_factor",
                    "random_state": "params:model_training.random_state",
                },
                outputs=["train_dataset_with_preds", "test_dataset_with_preds"],
                name="generate_predictions",
            ),
            node(
                func=run_global_analyses,
                inputs={
                    "train_df_path": "params:model_validation.train_with_preds_path",
                    "test_df_path": "params:model_validation.test_with_preds_path",
                    "target_column": "params:model_validation.target_column",
                    "prediction_column": "params:model_validation.prediction_column",
                    "old_model_column": "params:model_validation.old_model_column",
                    "run_id": "resolved_mlflow_run_id",
                    "model_metrics": "model_metrics",
                    "model_validation_params": "params:model_validation",
                },
                outputs=None,
                name="run_global_analyses",
            ),
        ]
    )
