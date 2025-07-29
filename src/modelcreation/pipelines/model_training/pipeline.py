"""Model training pipeline for CatBoost models."""

from kedro.pipeline import Pipeline, node

from .nodes import train_catboost_model, split_data


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
                outputs=["X_train", "X_test", "y_train", "y_test"],
                name="split_data_node",
            ),
            node(
                func=train_catboost_model,
                inputs=["X_train", "X_test", "y_train", "y_test", "feature_columns", "params:model_training"],
                outputs=["trained_model", "model_metrics"],
                name="train_catboost_model_node",
            ),
        ]
    )
