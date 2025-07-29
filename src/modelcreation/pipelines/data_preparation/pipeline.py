"""Data preparation pipeline for model training."""

from kedro.pipeline import Pipeline, node

from .nodes import prepare_model_data


def create_pipeline(**kwargs) -> Pipeline:
    """Create the data preparation pipeline.
    
    Returns:
        A kedro ``Pipeline`` object.
    """
    return Pipeline(
        [
            node(
                func=prepare_model_data,
                inputs=["raw_segmentation_data", "params:data_preparation"],
                outputs=["prepared_model_data", "feature_columns"],
                name="prepare_model_data_node",
            ),
        ]
    )
