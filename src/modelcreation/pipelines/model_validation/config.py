"""Analysis registry for model_validation pipeline.

Each entry defines:
    func: callable analysis runner (already handles logging to MLflow)
    required_args: ordered list of argument names the runner expects; these
        will be resolved from context built inside the orchestration node.
"""

analysis_to_run = [
    {
        "name": "roc",
        "func_name": "generate_roc_curve",
        "title": "ROC Curve",
        # Input specification describing how to construct the kwargs passed to the generate_* function.
        # Supported spec types:
        #   column: take a column from validation_dataset using the provided context key (source)
        #   param: take a parameter/context value directly (no dataset indexing)
        #   literal: fixed literal value supplied under 'value'
        # Optional keys:
        #   binarize_threshold_param: name of param in context whose value is used for binarization (only for column specs)
        "inputs": {
            "y_true": {
                "type": "column",
                "source": "target_column",
                "binarize_threshold_param": "roc_threshold",
            },
            "y_pred_proba": {"type": "column", "source": "prediction_column"},
        },
    },
    {
        "name": "calibration",
        "func_name": "generate_calibration_plot",
        "title": "Calibration",
        "inputs": {
            "y_true": {"type": "column", "source": "target_column"},
            "y_pred": {"type": "column", "source": "prediction_column"},
            "n_bins": {"type": "param", "source": "n_bins"},
        },
    },
]

__all__ = ["analysis_to_run"]