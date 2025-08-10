"""Analysis registry for model_validation pipeline.

Each entry defines:
    func: callable analysis runner (already handles logging to MLflow)
    required_args: ordered list of argument names the runner expects; these
        will be resolved from context built inside the orchestration node.
"""

global_analysis = [
    {
        "name": "lorenz",
        "func_name": "generate_lorenz_curve",
        "title": "Lorenz Curve",
        "datasets": ["test_dataset", "train_dataset"],
        "analysis_params": {
            "description": "Lorenz curve for regression predictions (inequality / concentration).",
        },
        # Inputs map directly to the generate_lorenz_curve signature
        "inputs": {
            "y_true": {"type": "column", "source": "target_column"},
            "y_pred": {"type": "column", "source": "prediction_column"},
            # Optional old model predictions column (parameter: old_model_column)
            "y_pred_old": {"type": "column", "source": "old_model_column"},
        },
    },
    {
        "name": "calibration",
        "func_name": "generate_calibration_plot",
        "title": "Calibration",
        "datasets": ["test_dataset", "train_dataset"],
        "analysis_params": {
            "description": "Observed / predicted ratio across bins",
        },
        "inputs": {
            "y_true": {"type": "column", "source": "target_column"},
            "y_pred": {"type": "column", "source": "prediction_column"},
            "y_pred_old": {"type": "column", "source": "old_model_column"},
            "n_bins": {"type": "param", "source": "n_bins"},
        },
    },
]

__all__ = ["global_analysis"]