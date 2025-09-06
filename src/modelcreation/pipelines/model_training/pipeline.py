"""Model training pipeline for CatBoost models."""

from kedro.pipeline import Pipeline, node

from .nodes import (
    separate_features_target,
    split_data,
    train_catboost_model,
)


def create_pipeline(**kwargs) -> Pipeline:
    """Create the model training pipeline.

    Returns:
        A kedro ``Pipeline`` object.
    """
    return Pipeline(
        [
            node(
                func=split_data,
                inputs=["prepared_model_data", "params:model_training"],
                outputs=["train_dataset", "test_dataset"],
                name="split_full_data_node",
            ),
            node(
                func=separate_features_target,
                inputs=[
                    "train_dataset",
                    "test_dataset",
                    "params:data_preparation.target_column",
                    "feature_columns",
                ],
                outputs=["X_train", "X_test", "y_train", "y_test"],
                name="separate_features_target_node",
            ),
            node(
                func=train_catboost_model,
                inputs=[
                    "X_train",
                    "X_test",
                    "y_train",
                    "y_test",
                    "feature_columns",
                    "params:model_training",
                ],
                outputs=["trained_model", "model_metrics"],
                name="train_catboost_model_node",
            ),
        ]
    )
