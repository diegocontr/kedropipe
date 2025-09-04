from kedro.pipeline import Pipeline, node

from .nodes import (
    generate_predictions,
    run_global_analyses,
    run_pdp_analyses,
    run_segmented_analyses,
    start_mlflow_run,
)

def create_pipeline(**kwargs):
    return Pipeline(
        [
            node(
                func=start_mlflow_run,
                inputs="params:model_validation.mlflow_experiment_name",
                outputs="resolved_mlflow_run_id",
                name="start_mlflow_run",
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
                    "feat_conf": "params:model_validation.feat_conf",
                    "run_id": "resolved_mlflow_run_id",
                    "model_validation_params": "params:model_validation",
                    "data_preparation_params": "params:data_preparation",
                },
                outputs=None,
                name="run_global_analyses",
            ),
            node(
                func=run_segmented_analyses,
                inputs={
                    "train_df_path": "params:model_validation.train_with_preds_path",
                    "test_df_path": "params:model_validation.test_with_preds_path",
                    "feat_conf": "params:model_validation.feat_conf",
                    "run_id": "resolved_mlflow_run_id",
                    "model_validation_params": "params:model_validation",
                    "data_preparation_params": "params:data_preparation",
                },
                outputs=None,
                name="run_segmented_analyses",
            ),
            node(
                func=run_pdp_analyses,
                inputs={
                    "train_df_path": "params:model_validation.train_with_preds_path",
                    "test_df_path": "params:model_validation.test_with_preds_path",
                    "feat_conf": "params:model_validation.feat_conf",
                    "trained_model": "trained_model",
                    "run_id": "resolved_mlflow_run_id",
                    "model_validation_params": "params:model_validation",
                    "data_preparation_params": "params:data_preparation",
                },
                outputs=None,
                name="run_pdp_analyses",
            ),
        ]
    )
