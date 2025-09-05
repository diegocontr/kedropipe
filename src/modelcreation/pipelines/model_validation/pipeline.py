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
                outputs=["resolved_mlflow_run_id", "mlflow_saver"],
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
                },
                outputs=["train_dataset_with_preds", "test_dataset_with_preds"],
                name="generate_predictions",
            ),
            node(
                func=run_global_analyses,
                inputs={
                    "mlflow_saver": "mlflow_saver",
                    "train_df_path": "params:model_validation.train_with_preds_path",
                    "test_df_path": "params:model_validation.test_with_preds_path",
                    "feat_conf": "params:model_validation.feat_conf",
                    "run_id": "resolved_mlflow_run_id",
                    "global_analysis_config": "params:global_analysis_config",
                },
                outputs=None,
                name="run_global_analyses",
            ),
            node(
                func=run_segmented_analyses,
                inputs={
                    "mlflow_saver": "mlflow_saver",
                    "train_df_path": "params:model_validation.train_with_preds_path",
                    "test_df_path": "params:model_validation.test_with_preds_path",
                    "feat_conf": "params:model_validation.feat_conf",
                    "run_id": "resolved_mlflow_run_id",
                    "segmented_analysis_config": "params:segmented_analysis_config",
                },
                outputs=None,
                name="run_segmented_analyses",
            ),
            node(
                func=run_pdp_analyses,
                inputs={
                    "mlflow_saver": "mlflow_saver",
                    "train_df_path": "params:model_validation.train_with_preds_path",
                    "test_df_path": "params:model_validation.test_with_preds_path",
                    "feat_conf": "params:model_validation.feat_conf",
                    "trained_model": "trained_model",
                    "run_id": "resolved_mlflow_run_id",
                    "pdp_analysis_config": "params:pdp_analysis_config",
                },
                outputs=None,
                name="run_pdp_analyses",
            ),
        ]
    )
