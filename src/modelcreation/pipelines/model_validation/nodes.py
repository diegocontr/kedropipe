"""Nodes for model validation pipeline."""

import logging
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
import mlflow
import mlflow.catboost

from model_monitoring import (
    AnalysisDataBuilder,
    SegmentCustom,
    calculate_statistics,
)
from model_monitoring.plotting import plot_segment_statistics, set_plot_theme


logger = logging.getLogger(__name__)


def create_feature_segments(validation_dataset: pd.DataFrame, parameters: Dict[str, Any]) -> list:
    """Create segments for all feature columns based on configuration.
    
    Args:
        validation_dataset: Validation dataset with features
        parameters: Model validation parameters including feature_binning config
        
    Returns:
        List of SegmentCustom objects for all features
    """
    segments = []
    
    # Get binning configuration
    feature_binning = parameters.get('feature_binning', {})
    default_bins = feature_binning.get('default_bins', 5)
    
    # Get feature columns (exclude target, prediction, weight columns)
    exclude_cols = {
        parameters.get('target_column', 'target_B'),
        parameters.get('prediction_column', 'prediction_new'), 
        parameters.get('weight_column', 'weight'),
        parameters.get('old_model_column', 'prediction_A')
    }
    
    # Add any prediction columns that might exist
    prediction_cols = [col for col in validation_dataset.columns if col.startswith('prediction')]
    exclude_cols.update(prediction_cols)
    
    # Get feature columns
    feature_cols = [col for col in validation_dataset.columns 
                   if col not in exclude_cols and validation_dataset[col].dtype in ['int64', 'float64']]
    
    logger.info(f"Creating segments for features: {feature_cols}")
    
    for feature in feature_cols:
        # Get custom binning for this feature, or use default
        feature_config = feature_binning.get(feature, {})
        
        # Check if custom bins and labels are provided
        if 'bins' in feature_config and 'labels' in feature_config:
            # Custom bins with labels
            bins = feature_config['bins']
            labels = feature_config['labels']
            segment_name = f"{feature}_group"
            
            logger.info(f"Using custom binning for {feature}: bins={bins}, labels={labels}")
            
        elif 'bins' in feature_config and isinstance(feature_config['bins'], list):
            # Custom bin edges without labels
            bins = feature_config['bins']
            labels = None  # Will be auto-generated
            segment_name = f"{feature}_group"
            
            logger.info(f"Using custom bin edges for {feature}: bins={bins}")
            
        elif 'bins' in feature_config and isinstance(feature_config['bins'], int):
            # Number of quantile bins
            bins = feature_config['bins']
            labels = None
            segment_name = f"{feature}_level"
            
            logger.info(f"Using {bins} quantile bins for {feature}")
            
        else:
            # Use default binning
            bins = default_bins
            labels = None
            segment_name = f"{feature}_level"
            
            logger.info(f"Using default {bins} quantile bins for {feature}")
        
        # Create segment
        if labels:
            segment = SegmentCustom(
                seg_col=feature,
                seg_name=segment_name,
                bins=bins,
                bin_labels=labels
            )
        else:
            segment = SegmentCustom(
                seg_col=feature,
                seg_name=segment_name, 
                bins=bins
            )
            
        segments.append(segment)
    
    logger.info(f"Created {len(segments)} feature segments")
    return segments


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
    import json
    
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


def calculate_validation_metrics(
    validation_dataset: pd.DataFrame,
    parameters: Dict[str, Any]
) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
    """Calculate validation metrics using the model monitoring framework.
    
    Args:
        validation_dataset: Combined validation dataset
        parameters: Validation parameters
        
    Returns:
        Tuple of (overall_metrics, segmented_metrics)
    """
    logger.info("Calculating validation metrics")
    
    # Get column names from parameters
    target_col = parameters.get('target_column', 'target_B')
    pred_col = parameters.get('prediction_column', 'prediction_new')
    weight_col = parameters.get('weight_column', 'weight')
    
    # Ensure columns exist
    if target_col not in validation_dataset.columns:
        available_targets = [col for col in validation_dataset.columns if col.startswith('target_')]
        if available_targets:
            target_col = available_targets[0]
            logger.info(f"Using detected target column: {target_col}")
        else:
            raise ValueError(f"Target column not found. Available columns: {list(validation_dataset.columns)}")
    
    # Create temporary parquet file for AnalysisDataBuilder
    with tempfile.NamedTemporaryFile(suffix='.parquet', delete=False) as temp_file:
        temp_path = temp_file.name
        validation_dataset.to_parquet(temp_path)
    
    try:
        # Initialize AnalysisDataBuilder
        extra_cols = [weight_col, pred_col, target_col]
        old_model_col = parameters.get('old_model_column')
        if old_model_col and old_model_col in validation_dataset.columns:
            extra_cols.append(old_model_col)
            
        analysis = AnalysisDataBuilder(
            data=temp_path,
            extra_cols=extra_cols
        )
        
        # Define segments using the generic function
        segments = create_feature_segments(validation_dataset, parameters)
        
        # Add segments to analysis
        for segment in segments:
            analysis.add_segment(segment)
        
        # Load and process data
        analysis.load_data()
        analysis.apply_treatments()
        analysis.apply_segments()
        
        # Define statistics to calculate
        func_dict = {
            "aggregations": {
                pred_col: ("weighted_mean", [pred_col, weight_col]),
                target_col: ("observed_charge", [target_col, weight_col]),
                "gini": ("gini", [target_col, pred_col, weight_col]),
                "exposure(k)": (lambda df, e: df[e].sum() / 1000, [weight_col]),
            },
            "post_aggregations": {
                "S/PP": ("division", [target_col, pred_col]),
            },
        }
        
        # Add old model metrics if available
        if old_model_col and old_model_col in validation_dataset.columns:
            func_dict["aggregations"][old_model_col] = ("weighted_mean", [old_model_col, weight_col])
            func_dict["post_aggregations"]["S/PP_old"] = ("division", [target_col, old_model_col])
            func_dict["aggregations"]["gini_old"] = ("gini", [target_col, old_model_col, weight_col])
        
        # Calculate statistics
        dict_stats, agg_stats = calculate_statistics(
            analysis, 
            func_dict, 
            bootstrap=parameters.get('bootstrap', False)
        )
        
        logger.info("Validation metrics calculated successfully")
        logger.info(f"Overall metrics: {agg_stats.to_dict()}")
        logger.info(f"Segmented metrics for {len(dict_stats)} segments")
        
        # Convert Series to DataFrame for saving
        agg_stats_df = agg_stats.to_frame(name='value')
        
        return agg_stats_df, dict_stats
        
    finally:
        # Clean up temporary file
        if os.path.exists(temp_path):
            os.remove(temp_path)


def compare_with_old_model(
    validation_dataset: pd.DataFrame,
    validation_metrics: pd.DataFrame,
    segmented_metrics: Dict[str, pd.DataFrame],
    parameters: Dict[str, Any]
) -> Dict[str, Any]:
    """Compare new model with old model if available.
    
    Args:
        validation_dataset: Validation dataset
        validation_metrics: Overall validation metrics
        segmented_metrics: Segmented validation metrics
        parameters: Validation parameters
        
    Returns:
        Dictionary with comparison results
    """
    logger.info("Comparing models")
    
    old_model_col = parameters.get('old_model_column')
    comparison_results = {
        'has_old_model': old_model_col is not None and old_model_col in validation_dataset.columns,
        'comparison_summary': {}
    }
    
    if comparison_results['has_old_model']:
        logger.info(f"Comparing with old model column: {old_model_col}")
        
        # Overall comparison
        if 'S/PP' in validation_metrics.index and 'S/PP_old' in validation_metrics.index:
            spp_new = validation_metrics.loc['S/PP'].iloc[0]
            spp_old = validation_metrics.loc['S/PP_old'].iloc[0]
            comparison_results['comparison_summary']['S/PP_improvement'] = spp_new - spp_old
            
        if 'gini' in validation_metrics.index and 'gini_old' in validation_metrics.index:
            gini_new = validation_metrics.loc['gini'].iloc[0]
            gini_old = validation_metrics.loc['gini_old'].iloc[0]
            comparison_results['comparison_summary']['gini_improvement'] = gini_new - gini_old
        
        # Segment-wise comparison
        segment_comparisons = {}
        for segment_name, segment_data in segmented_metrics.items():
            if 'S/PP' in segment_data.columns and 'S/PP_old' in segment_data.columns:
                # Convert to dict and ensure keys are JSON serializable
                spp_diff = (segment_data['S/PP'] - segment_data['S/PP_old']).to_dict()
                spp_diff = {str(k): float(v) for k, v in spp_diff.items()}
                segment_comparisons[segment_name] = {
                    'S/PP_improvement': spp_diff,
                }
            if 'gini' in segment_data.columns and 'gini_old' in segment_data.columns:
                if segment_name not in segment_comparisons:
                    segment_comparisons[segment_name] = {}
                # Convert to dict and ensure keys are JSON serializable
                gini_diff = (segment_data['gini'] - segment_data['gini_old']).to_dict()
                gini_diff = {str(k): float(v) for k, v in gini_diff.items()}
                segment_comparisons[segment_name]['gini_improvement'] = gini_diff
        
        comparison_results['segment_comparisons'] = segment_comparisons
        
        logger.info("Model comparison completed")
        logger.info(f"Overall improvements: {comparison_results['comparison_summary']}")
    else:
        logger.info("No old model available for comparison")
    
    return comparison_results


def generate_validation_reports(
    validation_dataset: pd.DataFrame,
    validation_metrics: pd.DataFrame, 
    segmented_metrics: Dict[str, pd.DataFrame],
    model_comparison_results: Dict[str, Any],
    model_metrics: str,
    parameters: Dict[str, Any]
) -> Dict[str, str]:
    """Generate validation reports and visualizations, saving plots as MLflow artifacts.
    
    Args:
        validation_dataset: Validation dataset
        validation_metrics: Overall metrics
        segmented_metrics: Segmented metrics  
        model_comparison_results: Model comparison results
        model_metrics: JSON string containing model metrics with MLflow run ID
        parameters: Validation parameters
        
    Returns:
        Dictionary with paths to generated reports
    """
    import json
    import tempfile
    import matplotlib.pyplot as plt
    
    logger.info("Generating validation reports with MLflow artifacts")
    
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
    
    target_col = parameters.get('target_column', 'target_B')
    pred_col = parameters.get('prediction_column', 'prediction_new')
    old_model_col = parameters.get('old_model_column')
    
    # Define report panels - multiple curves in same plots
    report_panels = [
        {
            "title": "Prediction vs. Target",
            "type": "pred_vs_target",
            "pred_col": [pred_col] + ([old_model_col] if model_comparison_results['has_old_model'] else []),
            "target_col": target_col,
            "plot_type": "line",
            "show_mean_line": "all",
        },
        {
            "title": "S/PP (Observed/Predicted Ratio)",
            "type": "spp", 
            "spp_col": ["S/PP"] + (["S/PP_old"] if model_comparison_results['has_old_model'] else []),
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
    if model_comparison_results['has_old_model']:
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
    
    generated_files = {}
    
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
                
                # Save to temporary file first
                with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
                    tmp_path = tmp_file.name
                    fig.savefig(tmp_path, dpi=300, bbox_inches='tight')
                
                # Save as MLflow artifact if run_id is available
                if run_id and use_mlflow:
                    try:
                        # Create a client to log artifact to specific run
                        client = mlflow.tracking.MlflowClient()
                        artifact_dir = "validation_plots"
                        client.log_artifact(run_id, tmp_path, artifact_dir)
                        
                        # The artifact path will be artifact_dir/filename
                        filename = f"validation_report_{segment_name}.png"
                        artifact_path = f"{artifact_dir}/{filename}"
                        generated_files[f"plot_{segment_name}"] = f"mlflow_artifact:{run_id}/{artifact_path}"
                        logger.info(f"Saved plot as MLflow artifact: {artifact_path}")
                        
                    except Exception as e:
                        logger.warning(f"Failed to save MLflow artifact: {e}")
                        # Fallback to local save
                        plot_path = report_dir / f"validation_report_{segment_name}.png"
                        fig.savefig(plot_path, dpi=300, bbox_inches='tight')
                        generated_files[f"plot_{segment_name}"] = str(plot_path)
                        logger.info(f"Saved plot locally: {plot_path}")
                else:
                    # Save locally
                    plot_path = report_dir / f"validation_report_{segment_name}.png"
                    fig.savefig(plot_path, dpi=300, bbox_inches='tight')
                    generated_files[f"plot_{segment_name}"] = str(plot_path)
                    logger.info(f"Saved plot locally: {plot_path}")
                
                # Clean up temporary file
                try:
                    os.unlink(tmp_path)
                except Exception as e:
                    logger.debug(f"Could not clean up temporary file {tmp_path}: {e}")
                
                # Close the figure to free memory
                plt.close(fig)
            
        except Exception as e:
            logger.error(f"Error generating plot for {segment_name}: {e!s}")
    
    # Generate summary statistics file
    summary_path = report_dir / "validation_summary.txt"
    with open(summary_path, 'w') as f:
        f.write("MODEL VALIDATION SUMMARY\\n")
        f.write("=" * 50 + "\\n\\n")
        
        f.write("OVERALL METRICS:\\n")
        f.write(validation_metrics.to_string())
        f.write("\\n\\n")
        
        if model_comparison_results['has_old_model']:
            f.write("MODEL COMPARISON:\\n")
            for metric, value in model_comparison_results['comparison_summary'].items():
                f.write(f"{metric}: {value:.4f}\\n")
            f.write("\\n")
        
        f.write("SEGMENT METRICS:\\n")
        for segment_name, stats_df in segmented_metrics.items():
            f.write(f"\\n{segment_name.upper()}:\\n")
            f.write(stats_df.to_string())
            f.write("\\n")
    
    generated_files['summary'] = str(summary_path)
    
    logger.info(f"Generated {len(generated_files)} report files")
    return generated_files
