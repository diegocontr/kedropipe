"""Nodes for model validation pipeline."""

import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mlflow
import mlflow.catboost

from model_monitoring.plotting import plot_segment_statistics, set_plot_theme
from .nodes_global import generate_global_analysis_plots


logger = logging.getLogger(__name__)



def load_model_from_mlflow(
    model_metrics: str,
    parameters: Dict[str, Any]
) -> Any:
    """Load model from MLflow using the run ID stored in model metrics.
    
    Args:
        model_metrics: JSON string containing model metrics with MLflow run ID
        parameters: Model validation parameters
        
    Returns:
        Loaded model from MLflow
    """
    logger.info("Loading model from MLflow")
    
    # Parse model metrics to get MLflow run ID
    metrics = json.loads(model_metrics)
    run_id = metrics.get('mlflow_run_id')
    
    if not run_id:
        raise ValueError("No MLflow run ID found in model metrics. Make sure the model was trained with MLflow.")
    
    logger.info(f"Loading model from MLflow run: {run_id}")
    
    # Load model from MLflow
    model_uri = f"runs:/{run_id}/catboost_model"
    model = mlflow.catboost.load_model(model_uri)
    
    logger.info(f"Successfully loaded model from MLflow: {model_uri}")
    return model


def generate_model_predictions(
    trained_model: Any,
    model_metrics: str,
    X_test: pd.DataFrame,
    parameters: Dict[str, Any]
) -> pd.DataFrame:
    """Generate predictions from the trained model, optionally loading from MLflow.
    
    Args:
        trained_model: The trained model object (fallback)
        model_metrics: JSON string containing model metrics with MLflow run ID
        X_test: Test features dataframe
        parameters: Model validation parameters
        
    Returns:
        DataFrame with predictions
    """
    logger.info("Generating model predictions")
    
    # Try to load model from MLflow first, fallback to trained_model
    use_mlflow = parameters.get('use_mlflow', True)
    
    if use_mlflow:
        try:
            model = load_model_from_mlflow(model_metrics, parameters)
            logger.info("Using model loaded from MLflow")
        except Exception as e:
            logger.warning(f"Failed to load model from MLflow: {e}")
            logger.info("Falling back to direct model object")
            model = trained_model
    else:
        logger.info("Using direct model object (MLflow disabled)")
        model = trained_model
    
    # Make predictions
    if hasattr(model, 'predict'):
        predictions = model.predict(X_test)
    else:
        raise ValueError(f"Model does not have a predict method. Model type: {type(model)}")
    
    # Create predictions dataframe
    predictions_df = pd.DataFrame({
        'prediction_new': predictions
    }, index=X_test.index)
    
    logger.info(f"Generated {len(predictions_df)} predictions")
    return predictions_df


def prepare_validation_data(
    X_test: pd.DataFrame,
    y_test: str,  # This comes as text from the catalog
    model_predictions: pd.DataFrame,
    parameters: Dict[str, Any]
) -> pd.DataFrame:
    """Prepare the validation dataset combining features, targets, and predictions.
    
    Args:
        X_test: Test features
        y_test: Test targets (as string from text dataset)
        model_predictions: Model predictions
        parameters: Validation parameters
        
    Returns:
        Combined validation dataset
    """
    logger.info("Preparing validation dataset")
    
    # Parse y_test from string format (the data is in simple format: column name on first line, then values)
    lines = y_test.strip().split('\n')
    target_col = lines[0]  # First line is the column name
    values = [int(val) for val in lines[1:]]  # Convert remaining lines to integers
    
    # Create DataFrame with proper index to match X_test
    y_test_df = pd.DataFrame({target_col: values}, index=X_test.index)
    
    logger.info(f"Target column: {target_col}")
    
    # Combine all data
    validation_data = X_test.copy()
    validation_data[target_col] = y_test_df[target_col]
    validation_data = validation_data.join(model_predictions)
    
    # Add weight column (default to 1.0 if not specified)
    weight_col = parameters.get('weight_column', 'weight')
    if weight_col not in validation_data.columns:
        validation_data[weight_col] = 1.0
    
    # Add old model predictions if specified
    old_model_col = parameters.get('old_model_column')
    if old_model_col and old_model_col in validation_data.columns:
        logger.info(f"Using existing column '{old_model_col}' as old model predictions")
    elif old_model_col:
        logger.warning(f"Old model column '{old_model_col}' not found in data")
        # Generate synthetic old model for demonstration
        np.random.seed(42)
        noise_factor = parameters.get('old_model_noise_factor', 0.1)
        validation_data[old_model_col] = (
            validation_data['prediction_new'] * 
            (1 + np.random.normal(0, noise_factor, len(validation_data)))
        )
        logger.info(f"Generated synthetic old model predictions with noise factor {noise_factor}")
    
    logger.info(f"Prepared validation dataset with {len(validation_data)} samples")
    logger.info(f"Columns: {list(validation_data.columns)}")
    
    return validation_data



def _save_plot_as_artifact(
    fig: plt.Figure,
    plot_name: str,
    run_id: str,
    use_mlflow: bool,
    artifact_dir: str,
    report_dir: Path,
    generated_files: Dict[str, str]
) -> None:
    """Save a plot as MLflow artifact or locally.
    
    Args:
        fig: Matplotlib figure to save
        plot_name: Name for the plot file
        run_id: MLflow run ID
        use_mlflow: Whether to use MLflow
        artifact_dir: MLflow artifact directory
        report_dir: Local report directory
        generated_files: Dictionary to store file paths
    """
    # Save to temporary file first
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
        tmp_path = tmp_file.name
        fig.savefig(tmp_path, dpi=300, bbox_inches='tight')
    
    # Save as MLflow artifact if run_id is available
    if run_id and use_mlflow:
        try:
            client = mlflow.tracking.MlflowClient()
            client.log_artifact(run_id, tmp_path, artifact_dir)
            
            filename = f"{plot_name}.png"
            artifact_path = f"{artifact_dir}/{filename}"
            generated_files[plot_name] = f"mlflow_artifact:{run_id}/{artifact_path}"
            logger.info(f"Saved plot as MLflow artifact: {artifact_path}")
            
        except Exception as e:
            logger.warning(f"Failed to save MLflow artifact: {e}")
            # Fallback to local save
            plot_path = report_dir / f"{plot_name}.png"
            fig.savefig(plot_path, dpi=300, bbox_inches='tight')
            generated_files[plot_name] = str(plot_path)
            logger.info(f"Saved plot locally: {plot_path}")
    else:
        # Save locally
        plot_path = report_dir / f"{plot_name}.png"
        fig.savefig(plot_path, dpi=300, bbox_inches='tight')
        generated_files[plot_name] = str(plot_path)
        logger.info(f"Saved plot locally: {plot_path}")
    
    # Clean up temporary file
    try:
        os.unlink(tmp_path)
    except Exception as e:
        logger.debug(f"Could not clean up temporary file {tmp_path}: {e}")
    
    # Close the figure to free memory
    plt.close(fig)


def _generate_segment_reports(
    segmented_metrics: Dict[str, pd.DataFrame],
    validation_metrics: pd.DataFrame,
    has_old_model: bool,
    parameters: Dict[str, Any],
    run_id: str,
    use_mlflow: bool,
    report_dir: Path,
    generated_files: Dict[str, str]
) -> None:
    """Generate segmented validation reports.
    
    Args:
        segmented_metrics: Segmented metrics
        validation_metrics: Overall metrics
        has_old_model: Whether old model is available
        parameters: Validation parameters
        run_id: MLflow run ID
        use_mlflow: Whether to use MLflow
        report_dir: Local report directory
        generated_files: Dictionary to store file paths
    """
    target_col = parameters.get('target_column', 'target_B')
    pred_col = parameters.get('prediction_column', 'prediction_new')
    old_model_col = parameters.get('old_model_column') if has_old_model else None
    
    # Define report panels - multiple curves in same plots
    # Check which prediction columns actually exist in the stats data
    available_pred_cols = []
    if pred_col in next(iter(segmented_metrics.values())).columns:
        available_pred_cols.append(pred_col)
    if has_old_model and old_model_col in next(iter(segmented_metrics.values())).columns:
        available_pred_cols.append(old_model_col)
    
    report_panels = [
        {
            "title": "Prediction vs. Target",
            "type": "pred_vs_target",
            "pred_col": available_pred_cols,
            "target_col": target_col,
            "plot_type": "line",
            "show_mean_line": "first",  # Changed from "all" to "first"
            "colors": ["#FF6B6B", "#4ECDC4"] if len(available_pred_cols) >= 2 else ["#FF6B6B"],  # Red for new, teal for old
        },
        {
            "title": "S/PP (Observed/Predicted Ratio)",
            "type": "spp", 
            "spp_col": ["S/PP"] + (["S/PP_old"] if has_old_model else []),
            "plot_type": "line",
            "show_mean_line": False,
        },
        {
            "title": "Exposure and Gini",
            "type": "exposure",
            "metric_col": "exposure(k)",
            "plot_type": "bar",
            "colors": ["#9F9DAA"],
        },
    ]
    
    # Add Gini as twin plot on the same axis as Exposure
    if has_old_model:
        report_panels.append({
            "title": "Exposure and Gini",  # Same title to make it a twin plot
            "type": "metric",
            "metric_col": ["gini", "gini_old"], 
            "plot_type": "line",
            "colors": ["#3D27B9", "#FF6B6B"],
        })
    else:
        report_panels.append({
            "title": "Exposure and Gini",  # Same title to make it a twin plot
            "type": "metric",
            "metric_col": "gini", 
            "plot_type": "line",
            "colors": ["#3D27B9"],
        })
    
    # Generate plots for each segment
    for segment_name, stats_df in segmented_metrics.items():
        logger.info(f"Generating report for segment: {segment_name}")
        
        try:
            # Generate the plot (this will create but not display the figure)
            fig = plot_segment_statistics(
                stats_df, 
                panel_configs=report_panels, 
                agg_stats=validation_metrics
            )
            
            if fig:
                # Turn off interactive mode to prevent display
                plt.ioff()
                
                plot_name = f"plot_{segment_name}"
                _save_plot_as_artifact(
                    fig=fig,
                    plot_name=f"validation_report_{segment_name}",
                    run_id=run_id,
                    use_mlflow=use_mlflow,
                    artifact_dir="validation_plots",
                    report_dir=report_dir,
                    generated_files=generated_files
                )
                # Update the key to match the original structure
                if f"validation_report_{segment_name}" in generated_files:
                    generated_files[plot_name] = generated_files.pop(f"validation_report_{segment_name}")
            
        except Exception as e:
            logger.error(f"Error generating plot for {segment_name}: {e!s}")


def _generate_global_reports(
    validation_dataset: pd.DataFrame,
    parameters: Dict[str, Any],
    run_id: str,
    use_mlflow: bool,
    report_dir: Path,
    generated_files: Dict[str, str]
) -> None:
    """Generate global analysis reports.
    
    Args:
        validation_dataset: Validation dataset
        parameters: Validation parameters
        run_id: MLflow run ID
        use_mlflow: Whether to use MLflow
        report_dir: Local report directory
        generated_files: Dictionary to store file paths
    """
    try:
        logger.info("Generating global analysis plots")
        global_plots = generate_global_analysis_plots(validation_dataset, parameters)
        
        for plot_category, figures in global_plots.items():
            for i, fig in enumerate(figures):
                plot_name = f"global_{plot_category}_{i}" if len(figures) > 1 else f"global_{plot_category}"
                
                _save_plot_as_artifact(
                    fig=fig,
                    plot_name=plot_name,
                    run_id=run_id,
                    use_mlflow=use_mlflow,
                    artifact_dir="global_analysis_plots",
                    report_dir=report_dir,
                    generated_files=generated_files
                )
                
    except Exception as e:
        logger.error(f"Error generating global analysis plots: {e!s}")


def _generate_summary_file(
    validation_metrics: pd.DataFrame,
    segmented_metrics: Dict[str, pd.DataFrame],
    has_old_model: bool,
    report_dir: Path,
    generated_files: Dict[str, str]
) -> None:
    """Generate summary statistics file.
    
    Args:
        validation_metrics: Overall metrics
        segmented_metrics: Segmented metrics
        has_old_model: Whether old model is available
        report_dir: Local report directory
        generated_files: Dictionary to store file paths
    """
    summary_path = report_dir / "validation_summary.txt"
    with open(summary_path, 'w') as f:
        f.write("MODEL VALIDATION SUMMARY\\n")
        f.write("=" * 50 + "\\n\\n")
        
        f.write("OVERALL METRICS:\\n")
        f.write(validation_metrics.to_string())
        f.write("\\n\\n")
        
        if has_old_model:
            f.write("OLD MODEL METRICS (when available):\\n")
            # Show old model specific metrics from validation_metrics
            old_metrics = validation_metrics[validation_metrics.index.str.contains('_old', na=False)]
            if not old_metrics.empty:
                f.write(old_metrics.to_string())
                f.write("\\n\\n")
        
        f.write("SEGMENT METRICS:\\n")
        for segment_name, stats_df in segmented_metrics.items():
            f.write(f"\\n{segment_name.upper()}:\\n")
            f.write(stats_df.to_string())
            f.write("\\n")
    
    generated_files['summary'] = str(summary_path)


def _generate_pdp_reports(
    pdp_plots: Dict[str, plt.Figure],
    run_id: str,
    use_mlflow: bool,
    report_dir: Path,
    generated_files: Dict[str, str]
) -> None:
    """Generate PDP reports.
    
    Args:
        pdp_plots: PDP figures
        run_id: MLflow run ID
        use_mlflow: Whether to use MLflow
        report_dir: Local report directory
        generated_files: Dictionary to store file paths
    """
    try:
        logger.info("Generating PDP analysis plots")
        
        for plot_name, fig in pdp_plots.items():
            _save_plot_as_artifact(
                fig=fig,
                plot_name=plot_name,
                run_id=run_id,
                use_mlflow=use_mlflow,
                artifact_dir="pdp_analysis_plots",
                report_dir=report_dir,
                generated_files=generated_files
            )
                
    except Exception as e:
        logger.error(f"Error generating PDP analysis plots: {e!s}")


def generate_validation_reports(
    validation_dataset: pd.DataFrame,
    validation_metrics: pd.DataFrame, 
    segmented_metrics: Dict[str, pd.DataFrame],
    model_metrics: str,
    parameters: Dict[str, Any]
) -> Dict[str, str]:
    """Generate validation reports and visualizations, saving plots as MLflow artifacts.
    
    Args:
        validation_dataset: Validation dataset
        validation_metrics: Overall metrics (includes has_old_model flag)
        segmented_metrics: Segmented metrics  
        model_metrics: JSON string containing model metrics with MLflow run ID
        parameters: Validation parameters
        
    Returns:
        Dictionary with paths to generated reports
    """
    logger.info("Generating validation reports with MLflow artifacts")
    
    # Get old model availability from metrics
    has_old_model = bool(validation_metrics.loc['has_old_model'].iloc[0]) if 'has_old_model' in validation_metrics.index else False
    
    # Get MLflow run ID for artifact logging
    use_mlflow = parameters.get('use_mlflow', True)
    run_id = None
    
    if use_mlflow:
        try:
            metrics = json.loads(model_metrics)
            run_id = metrics.get('mlflow_run_id')
            if run_id:
                logger.info(f"Will save artifacts to MLflow run: {run_id}")
            else:
                logger.warning("No MLflow run ID found, saving locally only")
        except Exception as e:
            logger.warning(f"Failed to parse MLflow run ID: {e}")
    
    # Create reporting directory for local fallback
    report_dir = Path("data/reporting")
    report_dir.mkdir(exist_ok=True)
    
    # Set plotting theme
    set_plot_theme(
        annotation_fontsize=14, 
        style="ggplot", 
        target_color="#1E1D25", 
        h_line_style=":"
    )
    
    generated_files = {}
    
    # Generate segmented reports
    _generate_segment_reports(
        segmented_metrics=segmented_metrics,
        validation_metrics=validation_metrics,
        has_old_model=has_old_model,
        parameters=parameters,
        run_id=run_id,
        use_mlflow=use_mlflow,
        report_dir=report_dir,
        generated_files=generated_files
    )
    
    # Generate global analysis reports
    _generate_global_reports(
        validation_dataset=validation_dataset,
        parameters=parameters,
        run_id=run_id,
        use_mlflow=use_mlflow,
        report_dir=report_dir,
        generated_files=generated_files
    )
    
    # Generate summary file
    _generate_summary_file(
        validation_metrics=validation_metrics,
        segmented_metrics=segmented_metrics,
        has_old_model=has_old_model,
        report_dir=report_dir,
        generated_files=generated_files
    )
    
    logger.info(f"Generated {len(generated_files)} report files")
    return generated_files
