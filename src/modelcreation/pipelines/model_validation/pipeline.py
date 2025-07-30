"""Model validation pipeline for analyzing model performance and comparison."""

from kedro.pipeline import Pipeline, node

from .nodes import (
    generate_model_predictions,
    prepare_validation_data,
    calculate_validation_metrics,
    compare_with_old_model,
    generate_validation_reports
)


def create_pipeline(**kwargs) -> Pipeline:
    """Create the model validation pipeline.
    
    This pipeline performs comprehensive model validation including:
    - Model predictions generation
    - Statistical analysis (S/PP ratios, Gini coefficients, etc.)
    - Comparison with optional old model
    - Visualization reports generation
    
    Returns:
        A kedro ``Pipeline`` object.
    """
    return Pipeline(
        [
            node(
                func=generate_model_predictions,
                inputs=["trained_model", "model_metrics", "X_test", "params:model_validation"],
                outputs="model_predictions",
                name="generate_predictions_node",
            ),
            node(
                func=prepare_validation_data,
                inputs=[
                    "X_test", 
                    "y_test", 
                    "model_predictions", 
                    "params:model_validation"
                ],
                outputs="validation_dataset",
                name="prepare_validation_data_node",
            ),
            node(
                func=calculate_validation_metrics,
                inputs=["validation_dataset", "params:model_validation"],
                outputs=["validation_metrics", "segmented_metrics"],
                name="calculate_metrics_node",
            ),
            node(
                func=compare_with_old_model,
                inputs=[
                    "validation_dataset", 
                    "validation_metrics", 
                    "segmented_metrics",
                    "params:model_validation"
                ],
                outputs="model_comparison_results",
                name="model_comparison_node",
            ),
            node(
                func=generate_validation_reports,
                inputs=[
                    "validation_dataset",
                    "validation_metrics", 
                    "segmented_metrics",
                    "model_comparison_results",
                    "model_metrics",
                    "params:model_validation"
                ],
                outputs="validation_report_paths",
                name="generate_reports_node",
            ),
        ]
    )
